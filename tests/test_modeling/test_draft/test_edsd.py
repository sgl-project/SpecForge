import os
import shutil
import tempfile
import unittest
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling.draft.edsd import EDFuse, EdsdDecoderLayer, EdsdDraftModel
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3, LlamaRMSNorm


class TestEdsdDraftModel(unittest.TestCase):

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

        # Small config mirroring qwen3-8b-edsd.json structure
        # target_layer_ids=[1,2] means 2 layers concatenated -> hidden_size * 2
        self.config_dict = {
            "architectures": ["EdsdDraftModel"],
            "bos_token_id": 1,
            "eos_token_id": 2,
            "hidden_act": "silu",
            "hidden_size": 64,
            "initializer_range": 0.02,
            "intermediate_size": 128,
            "max_position_embeddings": 512,
            "model_type": "llama",
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "num_hidden_layers": 1,
            "pad_token_id": 0,
            "rms_norm_eps": 1e-5,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "use_cache": True,
            "vocab_size": 256,
            "draft_vocab_size": 64,
            "target_layer_ids": [1, 2],
        }
        self.config = LlamaConfig(**self.config_dict)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def test_model_initialization(self):
        model = EdsdDraftModel(self.config)

        # EDSD-specific modules
        self.assertIsInstance(model.edfuse, EDFuse)
        self.assertIsInstance(model.embnorm, LlamaRMSNorm)
        self.assertIsInstance(model.midlayer, EdsdDecoderLayer)

        # Inherited modules
        self.assertIsInstance(model.embed_tokens, torch.nn.Embedding)
        self.assertIsInstance(model.lm_head, torch.nn.Linear)
        self.assertIsInstance(model.norm, LlamaRMSNorm)

    def test_edsd_decoder_layer_qkv_input_dim(self):
        """EdsdDecoderLayer replaces q/k/v input dim from hidden_size*2 to hidden_size.

        The output dims are preserved from the original GQA layout:
        - q_proj: hidden_size -> num_heads * head_dim
        - k_proj: hidden_size -> num_key_value_heads * head_dim
        - v_proj: hidden_size -> num_key_value_heads * head_dim
        """
        model = EdsdDraftModel(self.config)
        attn = model.midlayer.self_attn

        head_dim = self.config.hidden_size // self.config.num_attention_heads
        q_out = self.config.num_attention_heads * head_dim
        kv_out = self.config.num_key_value_heads * head_dim

        for name, expected_out in [("q_proj", q_out), ("k_proj", kv_out), ("v_proj", kv_out)]:
            proj = getattr(attn, name)
            self.assertIsInstance(proj, torch.nn.Linear)
            self.assertEqual(proj.in_features, self.config.hidden_size)
            self.assertEqual(proj.out_features, expected_out)
            self.assertIsNone(proj.bias)

    def test_fc_input_dim_matches_target_layer_count(self):
        """fc input dim must equal hidden_size * num_target_layers (2 for EDSD)."""
        model = EdsdDraftModel(self.config)
        # EdsdDraftModel.project_hidden_states asserts hidden_size * 2,
        # so fc.in_features must also be hidden_size * 2.
        self.assertEqual(model.fc.in_features, self.config.hidden_size * 2)
        self.assertEqual(model.fc.out_features, self.config.hidden_size)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def test_edfuse_forward(self):
        model = EdsdDraftModel(self.config)
        model.eval()
        batch_size, seq_len = 2, 10
        x = torch.randn(batch_size, seq_len, self.config.hidden_size)
        embed_tokens = torch.randn(batch_size, seq_len, self.config.hidden_size)

        with torch.no_grad():
            out = model.edfuse(x, embed_tokens)

        self.assertEqual(out.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_project_hidden_states(self):
        model = EdsdDraftModel(self.config)
        model.eval()
        batch_size, seq_len = 2, 10
        # EDSD uses 2 target layers -> hidden_size * 2
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 2)

        with torch.no_grad():
            projected = model.project_hidden_states(hidden_states)

        self.assertEqual(projected.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_backbone_forward(self):
        model = EdsdDraftModel(self.config)
        model.eval()
        batch_size, seq_len = 2, 10
        input_embeds = torch.randn(batch_size, seq_len, self.config.hidden_size)
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size)
        # backbone() passes attention_mask directly to SDPA, which requires
        # 4D format (batch, num_heads, q_len, kv_len) or None for causal.
        # Passing None triggers is_causal=True in the attention layer.
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            output = model.backbone(
                input_embeds=input_embeds,
                hidden_states=hidden_states,
                cache_hidden=None,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )

        self.assertEqual(output.shape, (batch_size, seq_len, self.config.hidden_size))

    def test_full_forward_pass(self):
        """End-to-end: embed -> project -> backbone -> logits."""
        model = EdsdDraftModel(self.config)
        model.eval()
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, self.config.vocab_size, (batch_size, seq_len))
        hidden_states = torch.randn(batch_size, seq_len, self.config.hidden_size * 2)
        position_ids = torch.arange(seq_len).unsqueeze(0).expand(batch_size, -1)

        with torch.no_grad():
            input_embeds = model.embed_input_ids(input_ids)
            projected = model.project_hidden_states(hidden_states)
            output = model.backbone(
                input_embeds=input_embeds,
                hidden_states=projected,
                cache_hidden=None,
                attention_mask=None,
                position_ids=position_ids,
                past_key_values=None,
                use_cache=False,
            )
            logits = model.compute_logits(output)

        self.assertEqual(
            logits.shape, (batch_size, seq_len, self.config.draft_vocab_size)
        )

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def test_save_pretrained(self):
        model = EdsdDraftModel(self.config)
        self.config.save_pretrained(self.temp_dir)
        model_path = os.path.join(self.temp_dir, "pytorch_model.bin")
        torch.save(model.state_dict(), model_path)
        self.assertTrue(os.path.exists(os.path.join(self.temp_dir, "config.json")))
        self.assertTrue(os.path.exists(model_path))

    def test_state_dict_compatibility(self):
        model1 = EdsdDraftModel(self.config)
        model2 = EdsdDraftModel(self.config)
        state_dict = model1.state_dict()
        model2.load_state_dict(state_dict)

        for name, param1 in model1.named_parameters():
            param2 = dict(model2.named_parameters())[name]
            self.assertTrue(torch.equal(param1, param2))

    @patch("transformers.modeling_utils.PreTrainedModel.from_pretrained")
    def test_from_pretrained_mock(self, mock_from_pretrained):
        mock_model = EdsdDraftModel(self.config)
        mock_from_pretrained.return_value = mock_model

        loaded_model = EdsdDraftModel.from_pretrained(self.temp_dir)
        mock_from_pretrained.assert_called_once_with(self.temp_dir)
        self.assertIsInstance(loaded_model, EdsdDraftModel)

    # ------------------------------------------------------------------
    # Config validation
    # ------------------------------------------------------------------

    def test_config_validation(self):
        invalid_config = LlamaConfig(
            vocab_size=1000,
            hidden_size=127,
            num_attention_heads=4,
            num_key_value_heads=2,
        )
        with self.assertRaises(AssertionError):
            EdsdDraftModel(invalid_config)


if __name__ == "__main__":
    suite = unittest.TestSuite()
    suite.addTest(unittest.makeSuite(TestEdsdDraftModel))
    runner = unittest.TextTestRunner(verbosity=2)
    runner.run(suite)