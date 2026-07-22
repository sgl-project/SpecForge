import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaForCausalLMEagle3,
    LlamaMLP,
    LlamaRMSNorm,
)

# from model_module import LlamaForCausalLMEagle3


class TestLlamaForCausalLMEagle3Loading(unittest.TestCase):

    def setUp(self):
        """Set up the test environment before each test."""
        self.temp_dir = tempfile.mkdtemp()

        config_dict = {
            "architectures": ["LlamaForCausalLM"],
            "bos_token_id": 128000,
            "eos_token_id": 128001,
            "hidden_act": "silu",
            "hidden_size": 4096,
            "initializer_range": 0.02,
            "intermediate_size": 14336,
            "max_position_embeddings": 2048,
            "model_type": "llama",
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-05,
            "tie_word_embeddings": False,
            "torch_dtype": "float16",
            "transformers_version": "4.28.1",
            "use_cache": True,
            "vocab_size": 128256,
            "draft_vocab_size": 32000,
        }

        self.config = LlamaConfig(**config_dict)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_model_initialization(self):
        model = LlamaForCausalLMEagle3(self.config)

        self.assertIsInstance(model.midlayer.self_attn, LlamaAttention)
        self.assertIsInstance(model.midlayer.mlp, LlamaMLP)
        self.assertIsInstance(model.midlayer.hidden_norm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.input_layernorm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer.post_attention_layernorm, LlamaRMSNorm)
        self.assertEqual(model.midlayer.hidden_size, self.config.hidden_size)

    def test_save_pretrained(self):
        """Test the model's save_pretrained functionality."""
        model = LlamaForCausalLMEagle3(self.config)

        self.config.save_pretrained(self.temp_dir)

        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)

        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        """mock"""
        mock_model = LlamaForCausalLMEagle3(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = LlamaForCausalLMEagle3.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, LlamaForCausalLMEagle3)

    def test_model_forward_pass(self):
        """forward"""
        model = LlamaForCausalLMEagle3(self.config)
        model.eval()

        batch_size = 2
        seq_len = 10

        input_emb = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 3)
        attention_mask = torch.ones(batch_size, seq_len)

        with torch.no_grad():
            outputs = model(
                inputs_embeds=input_emb,
                hidden_states=hidden_states,
                attention_mask=attention_mask,
            )

        self.assertEqual(outputs.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_state_dict_compatibility(self):
        model1 = LlamaForCausalLMEagle3(self.config)
        model2 = LlamaForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()

        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_rotary_buffers_absent_from_state_dict(self):
        """The rotary buffers are non-persistent, so a checkpoint never stores them.

        inv_freq / cos_cached / sin_cached are registered with persistent=False, so
        they are absent from state_dict() and thus from any saved checkpoint. This is
        the property that makes warm-starting safe (see the next test): there is
        nothing in the checkpoint that could overwrite the live model's correctly
        initialized rotary buffers. Historically the opposite was dangerous — loading
        a meta-device model *as* the live model left these buffers uninitialized and
        training diverged to loss=NaN.
        """
        model = LlamaForCausalLMEagle3(self.config)
        keys = model.state_dict().keys()
        for buf in ("inv_freq", "cos_cached", "sin_cached"):
            self.assertFalse(
                any(k.endswith(buf) for k in keys),
                f"{buf} unexpectedly present in state_dict (should be persistent=False)",
            )

    def test_warm_start_preserves_rotary_buffers(self):
        """Warm-starting from a checkpoint must not disturb the rotary buffers.

        The training warm-start path builds the live draft model via from_config
        (so __init__ runs _init_rope and the rotary buffers are valid) and then loads
        only the checkpoint's draft weights with load_state_dict(strict=False). Since
        the checkpoint state_dict contains no rotary buffers (previous test), the live
        model's finite buffers must survive untouched. This pins the invariant against
        a future regression that goes back to using a from_pretrained model as the
        live model, which would reintroduce the uninitialized-buffer loss=NaN.
        """
        # A checkpoint's draft weights: exactly what warm-start loads. Being
        # persistent=False, it carries no rotary buffers.
        checkpoint_state = LlamaForCausalLMEagle3(self.config).state_dict()

        # The live model, built the normal way, with valid rotary buffers.
        live = LlamaForCausalLMEagle3(self.config)
        rot = live.midlayer.self_attn.rotary_emb
        before = {
            name: getattr(rot, name).clone()
            for name in ("inv_freq", "cos_cached", "sin_cached")
        }

        result = live.load_state_dict(checkpoint_state, strict=False)

        # The rotary buffers are not tracked by state_dict at all, so they appear in
        # neither the checkpoint nor the missing/unexpected key sets.
        for buf in ("inv_freq", "cos_cached", "sin_cached"):
            self.assertFalse(any(k.endswith(buf) for k in result.missing_keys))
        rot = live.midlayer.self_attn.rotary_emb
        for name, ref in before.items():
            val = getattr(rot, name)
            self.assertTrue(
                torch.isfinite(val).all(), f"{name} became non-finite after warm start"
            )
            self.assertTrue(torch.equal(val, ref), f"{name} was modified by warm start")

    def test_config_validation(self):
        # A dimensionally-valid config (hidden_size divisible by num_attention_heads
        # so it passes transformers' strict config validation) that is still missing
        # the required `draft_vocab_size` attribute. Building the Eagle3 model from it
        # should raise AttributeError.
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        with self.assertRaises(AttributeError):
            LlamaForCausalLMEagle3(invalid_config)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestLlamaForCausalLMEagle3Loading))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
