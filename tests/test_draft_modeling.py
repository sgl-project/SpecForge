import os
import shutil
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn
from transformers import LlamaConfig

from sgl_spec.modeling.draft.llama3_eagle import (
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

        self.assertIsInstance(model.self_attn, LlamaAttention)
        self.assertIsInstance(model.mlp, LlamaMLP)
        self.assertIsInstance(model.hidden_norm, LlamaRMSNorm)
        self.assertIsInstance(model.input_layernorm, LlamaRMSNorm)
        self.assertIsInstance(model.post_attention_layernorm, LlamaRMSNorm)

        self.assertEqual(model.hidden_size, self.config.hidden_size)

    def test_save_pretrained(self):
        """save"""
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
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        cache_hidden = [[], []]

        attention_mask = torch.ones(batch_size, 1, seq_len, seq_len)
        attention_mask = attention_mask.triu(diagonal=1)
        attention_mask = attention_mask.masked_fill(
            attention_mask.bool(), float("-inf")
        )

        position_ids = torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1)

        with torch.no_grad():
            outputs = model(
                input_emb=input_emb,
                hidden_states=hidden_states,
                cache_hidden=cache_hidden,
                attention_mask=attention_mask,
                position_ids=position_ids,
            )

        self.assertEqual(len(outputs), 2)
        hidden_output, return_hidden = outputs

        self.assertEqual(
            hidden_output.shape, (batch_size, seq_len, self.config.hidden_size)
        )
        self.assertEqual(
            return_hidden.shape, (batch_size, seq_len, self.config.hidden_size * 2)
        )

    def test_state_dict_compatibility(self):
        model1 = LlamaForCausalLMEagle3(self.config)
        model2 = LlamaForCausalLMEagle3(self.config)

        state_dict = model1.state_dict()

        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    def test_config_validation(self):
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=127,
            num_attention_heads=4,
            num_key_value_heads=2,
        )

        with self.assertRaises(ValueError):
            LlamaForCausalLMEagle3(invalid_config)

    def test_different_last_parameter(self):
        model_last_true = LlamaForCausalLMEagle3(self.config, last=True)
        model_last_false = LlamaForCausalLMEagle3(self.config, last=False)

        self.assertTrue(model_last_true.last)
        self.assertFalse(model_last_false.last)

        self.assertTrue(model_last_true.mlp.last)
        self.assertFalse(model_last_false.mlp.last)


if __name__ == "__main__":
    suite = unittest.TestSuite()

    suite.addTest(unittest.makeSuite(TestLlamaForCausalLMEagle3Loading))

    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)
