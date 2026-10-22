"""Initialization defaults must not overwrite module-owned parameter/buffer dtypes."""
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import PretrainedConfig, PreTrainedModel, Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.modeling.auto import AutoDraftModel


def config(rope_type="yarn"):
    rope = {"rope_type": rope_type, "rope_theta": 10000.0}
    if rope_type != "default":
        rope["factor"] = 16.0
    if rope_type == "yarn":
        rope["original_max_position_embeddings"] = 128
    return Qwen3Config(
        architectures=["DSparkDraftModel"],
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_target_layers=3,
        head_dim=64,
        vocab_size=64,
        block_size=4,
        max_position_embeddings=2048,
        layer_types=["full_attention"],
        rope_parameters=rope,
        dflash_config={
            "projector_type": "dspark",
            "target_layer_ids": [1],
            "mask_token_id": 63,
            "markov_head_type": "vanilla",
            "markov_rank": 8,
            "enable_confidence_head": False,
        },
        attn_implementation="sdpa",
    )


class ExplicitPrecisionModel(PreTrainedModel):
    config_class = PretrainedConfig

    def __init__(self, config):
        super().__init__(config)
        self.default_layer = nn.Linear(4, 4)
        self.precise = nn.Parameter(torch.tensor([1.000123], dtype=torch.float32))
        self.register_buffer('frequency', torch.tensor([.000123], dtype=torch.float32))


class DraftPrecisionTest(unittest.TestCase):
    def test_initialization_default_preserves_explicit_module_dtypes(self):
        original_default = torch.get_default_dtype()
        with patch.object(AutoDraftModel, '_model_cls_from_config',
                          return_value=ExplicitPrecisionModel):
            model = AutoDraftModel.from_config(PretrainedConfig(), torch_dtype=torch.bfloat16)
        self.assertEqual(torch.get_default_dtype(), original_default)
        self.assertEqual(model.default_layer.weight.dtype, torch.bfloat16)
        self.assertEqual(model.precise.dtype, torch.float32)
        self.assertEqual(model.frequency.dtype, torch.float32)
        torch.testing.assert_close(model.precise, torch.tensor([1.000123]), rtol=0, atol=0)
        torch.testing.assert_close(model.frequency, torch.tensor([.000123]), rtol=0, atol=0)

    def test_default_dtype_restored_after_constructor_failure(self):
        class FailingModel(ExplicitPrecisionModel):
            def __init__(self, config):
                super().__init__(config)
                raise RuntimeError('constructor failed')
        original_default = torch.get_default_dtype()
        with patch.object(AutoDraftModel, '_model_cls_from_config', return_value=FailingModel):
            with self.assertRaisesRegex(RuntimeError, 'constructor failed'):
                AutoDraftModel.from_config(PretrainedConfig(), torch_dtype=torch.bfloat16)
        self.assertEqual(torch.get_default_dtype(), original_default)

    def test_native_rope_precision_and_dynamic_updates(self):
        for rope_type in ('default', 'yarn', 'dynamic'):
            with self.subTest(rope_type=rope_type):
                cfg = config(rope_type)
                draft = AutoDraftModel.from_config(cfg, torch_dtype=torch.bfloat16)
                self.assertIs(type(draft.rotary_emb), Qwen3RotaryEmbedding)
                self.assertTrue(all(p.dtype == torch.bfloat16 for p in draft.parameters()))
                reference = Qwen3RotaryEmbedding(cfg)
                x = torch.zeros(1, 1, 64, dtype=torch.bfloat16)
                for position in (4096, 8192, 20):
                    pos = torch.tensor([[position]])
                    actual = draft.rotary_emb(x, pos)
                    expected = reference(x, pos)
                    torch.testing.assert_close(draft.rotary_emb.inv_freq,
                                               reference.inv_freq, rtol=0, atol=0)
                    for a, b in zip(actual, expected):
                        self.assertEqual(a.dtype, x.dtype)
                        torch.testing.assert_close(a, b, rtol=0, atol=0)

    def test_full_draft_export_reload_and_backward(self):
        draft = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
        frequencies = draft.rotary_emb.inv_freq.clone()
        self.assertEqual(frequencies.dtype, torch.float32)
        noise = torch.randn(1, 4, 128, dtype=torch.bfloat16, requires_grad=True)
        features = torch.randn(1, 3, 128, dtype=torch.bfloat16, requires_grad=True)
        args = dict(noise_embedding=noise, target_hidden=features,
                    position_ids=torch.arange(7).unsqueeze(0))
        result = draft(**args)
        self.assertEqual(result.dtype, torch.bfloat16)
        result.float().square().mean().backward()
        for grad in (noise.grad, features.grad, draft.fc.weight.grad):
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all() and grad.abs().sum() > 0)
        with tempfile.TemporaryDirectory() as directory:
            draft.save_pretrained(directory)
            restored = AutoDraftModel.from_pretrained(directory, dtype=torch.bfloat16)
            torch.testing.assert_close(restored.rotary_emb.inv_freq, frequencies, rtol=0, atol=0)
            torch.testing.assert_close(restored(**args), result, rtol=0, atol=0)
            self.assertFalse(any('inv_freq' in key for key in restored.state_dict()))

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required')
    def test_device_only_moves_preserve_native_buffers(self):
        draft = AutoDraftModel.from_config(config(), torch_dtype=torch.bfloat16)
        original = draft.rotary_emb.inv_freq.clone()
        for device in ('cuda', 'cpu', 'cuda'):
            draft.to(device=device)
            self.assertEqual(draft.rotary_emb.inv_freq.dtype, torch.float32)
            self.assertEqual(draft.fc.weight.dtype, torch.bfloat16)
            torch.testing.assert_close(draft.rotary_emb.inv_freq.cpu(), original, rtol=0, atol=0)
        reference = Qwen3RotaryEmbedding(draft.config, device='cuda')
        x = torch.zeros(1, 3, 64, dtype=torch.bfloat16, device='cuda')
        pos = torch.tensor([[0, 1024, 65535]], device='cuda')
        with torch.autocast('cuda', dtype=torch.bfloat16):
            actual = draft.rotary_emb(x, pos)
        for a, b in zip(actual, reference(x, pos)):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
