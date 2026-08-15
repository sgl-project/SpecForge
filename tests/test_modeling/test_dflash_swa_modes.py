import copy
import unittest

import torch
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import (
    create_dflash_block_mask,
    create_dflash_sdpa_mask,
)
from specforge.modeling.draft.dflash import (
    DFlashDraftModel,
    Qwen3DFlashAttention,
    Qwen3DFlashMLAAttention,
    resolve_dflash_attention_mode,
    resolve_dflash_attention_modes,
)
from specforge.modeling.draft.dspark import DSparkDraftModel


def _swa_config(
    *,
    attention_modes=("gqa", "mla", "gqa", "mla"),
    layer_types=(
        "sliding_attention",
        "sliding_attention",
        "full_attention",
        "full_attention",
    ),
    architecture="DFlashDraftModel",
    implementation="sdpa",
):
    dflash_config = {
        "attention_modes": list(attention_modes),
        "target_layer_ids": [1],
        "mla_rope_interleaved": True,
        "mla_use_output_gate": False,
    }
    if architecture == "DSparkDraftModel":
        dflash_config.update(
            {
                "projector_type": "dspark",
                "markov_rank": 4,
                "enable_confidence_head": False,
            }
        )
    config = Qwen3Config(
        architectures=[architecture],
        block_size=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=len(layer_types),
        num_target_layers=4,
        head_dim=16,
        q_lora_rank=16,
        kv_lora_rank=8,
        qk_nope_head_dim=12,
        qk_rope_head_dim=4,
        v_head_dim=16,
        max_position_embeddings=64,
        vocab_size=64,
        layer_types=list(layer_types),
        sliding_window=4,
        use_sliding_window=True,
        attention_bias=False,
        attention_dropout=0.0,
        dflash_config=dflash_config,
    )
    config._attn_implementation = implementation
    return config


def _masks(*, device, flex):
    factory = create_dflash_block_mask if flex else create_dflash_sdpa_mask
    kwargs = {
        "anchor_positions": torch.tensor([[4]], device=device),
        "block_keep_mask": torch.tensor([[True]], device=device),
        "S": 6,
        "block_size": 2,
        "device": device,
    }
    return {
        "full_attention": factory(**kwargs),
        "sliding_attention": factory(**kwargs, sliding_window=4),
    }


class TestDFlashSWAProjectionModes(unittest.TestCase):
    def test_config_selects_swa_gqa_and_swa_mla_per_layer(self):
        model = DFlashDraftModel(_swa_config())

        self.assertEqual(model.attention_mode, "mixed")
        self.assertEqual(model.attention_modes, ("gqa", "mla", "gqa", "mla"))
        expected_classes = (
            Qwen3DFlashAttention,
            Qwen3DFlashMLAAttention,
            Qwen3DFlashAttention,
            Qwen3DFlashMLAAttention,
        )
        for index, expected_class in enumerate(expected_classes):
            with self.subTest(index=index):
                attention = model.layers[index].self_attn
                self.assertIsInstance(attention, expected_class)
                expected_window = 4 if index < 2 else None
                self.assertEqual(attention.sliding_window, expected_window)

    def test_mixed_swa_gqa_mla_forward_backward(self):
        torch.manual_seed(17)
        model = DFlashDraftModel(_swa_config()).train()
        noise_embedding = torch.randn(1, 2, 32, requires_grad=True)
        target_hidden = torch.randn(1, 6, 32, requires_grad=True)
        output = model(
            position_ids=torch.arange(8).unsqueeze(0),
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=_masks(device=torch.device("cpu"), flex=False),
        )

        self.assertEqual(output.shape, (1, 2, 32))
        self.assertTrue(torch.isfinite(output).all())
        output.square().mean().backward()
        for tensor in (noise_embedding, target_hidden):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
        for index in range(4):
            attention = model.layers[index].self_attn
            projection = (
                attention.q_a_proj
                if isinstance(attention, Qwen3DFlashMLAAttention)
                else attention.q_proj
            )
            self.assertIsNotNone(projection.weight.grad, index)
            self.assertTrue(torch.isfinite(projection.weight.grad).all(), index)

    def test_dspark_uses_the_same_mixed_swa_backbone(self):
        config = _swa_config(architecture="DSparkDraftModel")
        model = DSparkDraftModel(config)

        self.assertEqual(model.attention_modes, ("gqa", "mla", "gqa", "mla"))
        self.assertIsInstance(model.layers[0].self_attn, Qwen3DFlashAttention)
        self.assertIsInstance(model.layers[1].self_attn, Qwen3DFlashMLAAttention)
        self.assertEqual(
            model.config.dflash_config["attention_modes"],
            ["gqa", "mla", "gqa", "mla"],
        )

    def test_uniform_attention_mode_shorthand_remains_compatible(self):
        config = _swa_config(attention_modes=("mla",) * 4)
        config.dflash_config.pop("attention_modes")
        config.dflash_config["attention_mode"] = "MLA"

        self.assertEqual(resolve_dflash_attention_modes(config), ("mla",) * 4)
        self.assertEqual(resolve_dflash_attention_mode(config), "mla")

        config.dflash_config.pop("attention_mode")
        self.assertEqual(resolve_dflash_attention_modes(config), ("gqa",) * 4)
        self.assertEqual(resolve_dflash_attention_mode(config), "gqa")

    def test_rejects_invalid_per_layer_attention_modes(self):
        cases = (
            ("not-a-list", "per-layer list"),
            (["gqa"], "num_hidden_layers"),
            (["gqa", "mla", "latent-ish", "gqa"], "latent-ish"),
        )
        for modes, message in cases:
            with self.subTest(modes=modes):
                config = copy.deepcopy(_swa_config())
                config.dflash_config["attention_modes"] = modes
                with self.assertRaisesRegex(ValueError, message):
                    resolve_dflash_attention_modes(config)

        config = _swa_config()
        config.dflash_config["attention_mode"] = "gqa"
        with self.assertRaisesRegex(ValueError, "only one"):
            resolve_dflash_attention_modes(config)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_mixed_swa_gqa_mla_flex_forward_backward_cuda(self):
        config = _swa_config(
            attention_modes=("gqa", "mla"),
            layer_types=("sliding_attention", "sliding_attention"),
            implementation="flex_attention",
        )
        model = DFlashDraftModel(config).to(device="cuda", dtype=torch.bfloat16).train()
        noise_embedding = torch.randn(
            1,
            2,
            32,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        target_hidden = torch.randn(
            1,
            6,
            32,
            device="cuda",
            dtype=torch.bfloat16,
            requires_grad=True,
        )
        output = model(
            position_ids=torch.arange(8, device="cuda").unsqueeze(0),
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            attention_mask=_masks(device=torch.device("cuda"), flex=True),
        )

        self.assertEqual(output.shape, (1, 2, 32))
        self.assertTrue(torch.isfinite(output).all())
        output.float().square().mean().backward()
        for tensor in (noise_embedding, target_hidden):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
