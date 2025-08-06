import unittest

import torch
import torch.nn as nn
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlashAttention,
    prepare_decoder_attention_mask,
)
from specforge.utils import padding


class TestLlamaAttention(unittest.TestCase):
    """Comprehensive test suite for LlamaAttention with simulated inputs."""

    def setUp(self):
        """Set up test configurations and common parameters."""
        # Basic configuration
        self.config_dict = {
            "hidden_size": 512,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 2048,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 1376,
            "hidden_act": "silu",
        }
        self.config = LlamaConfig(**self.config_dict)

        # Test parameters
        self.ttt_length = 7
        self.batch_size = 2
        self.seq_len = 2048
        self.padding_start_index = self.seq_len // 3

        if not torch.cuda.is_available():
            self.skipTest("CUDA is not available, required for Flash Attention")

    def create_attention_inputs(self):
        """Create consistent input tensors for attention testing"""
        torch.manual_seed(42)  # For reproducibility

        batch_size = self.batch_size
        seq_len = self.seq_len
        hidden_size = self.config.hidden_size
        padding_start_index = self.padding_start_index
        position_ids = (
            torch.arange(seq_len, device="cuda").unsqueeze(0).repeat(batch_size, 1)
        )
        attention_mask = torch.ones(batch_size, seq_len)
        # Simulate one item in the batch is masked and not taking a full block.
        attention_mask[1, padding_start_index:] = False
        loss_mask = attention_mask[..., None].to("cuda")
        input_embeds = torch.randn(batch_size, seq_len, hidden_size)
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        ).to("cuda")
        hidden_states = torch.randn(
            batch_size, seq_len, hidden_size * 2, requires_grad=True, device="cuda"
        )
        return hidden_states, position_ids, decoder_attention_mask, loss_mask

    def test_forward_pass(self):
        """Test forward pass."""
        attention = LlamaAttention(self.config).to("cuda")
        flash_attention = LlamaFlashAttention(self.config).to("cuda")

        # Ensure same weights
        flash_attention.load_state_dict(attention.state_dict())

        attention.eval()
        flash_attention.eval()

        cache_hidden = [[], []]
        flash_cache_hidden = [[], []]

        hidden_states, position_ids, decoder_attention_mask, loss_mask = (
            self.create_attention_inputs()
        )
        flash_hidden_states = hidden_states.clone()

        for idx in range(self.ttt_length):
            is_last = idx == self.ttt_length - 1

            with torch.no_grad():
                output = attention(
                    hidden_states=hidden_states,
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_hidden=cache_hidden,
                )
            with torch.no_grad():
                output_flash = flash_attention(
                    hidden_states=flash_hidden_states,
                    position_ids=position_ids,
                    cache_hidden=flash_cache_hidden,
                )
            torch.testing.assert_close(output[0], output_flash[0], atol=1e-3, rtol=1e-3)
            torch.testing.assert_close(
                output[1][: self.padding_start_index],
                output_flash[1][: self.padding_start_index],
                atol=1e-3,
                rtol=1e-3,
            )
            if not is_last:
                # Step 5.7: we need to update the loss mask
                loss_mask = padding(loss_mask, left=False)
                hidden_states = padding(hidden_states, left=False)
                flash_hidden_states = padding(flash_hidden_states, left=False)
            # Check output shape
            self.assertEqual(output.shape, output_flash.shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(output_flash).any())
            self.assertFalse(torch.isinf(output_flash).any())

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlashAttention."""
        attention = LlamaAttention(self.config).to("cuda")
        flash_attention = LlamaFlashAttention(self.config).to("cuda")
        head = nn.Linear(
            self.config.hidden_size, self.config.vocab_size, bias=False, device="cuda"
        )
        target = torch.randn(
            self.batch_size, self.seq_len, self.config.hidden_size, device="cuda"
        )
        with torch.no_grad():
            target_head = head(target)
            target_p = nn.Softmax(dim=2)(target_head)
        # Ensure same weights
        flash_attention.load_state_dict(attention.state_dict())

        cache_hidden = [[], []]
        flash_cache_hidden = [[], []]

        hidden_states, position_ids, decoder_attention_mask, loss_mask = (
            self.create_attention_inputs()
        )
        flash_hidden_states = hidden_states.clone().detach().requires_grad_(True)

        # Create input tensors that require gradients
        loss_list = []
        loss_flash_list = []

        for idx in range(self.ttt_length):
            is_last = idx == self.ttt_length - 1

            output = attention(
                hidden_states=hidden_states,
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
            )
            output_flash = flash_attention(
                hidden_states=flash_hidden_states,
                position_ids=position_ids,
                cache_hidden=flash_cache_hidden,
            )
            logits = head(output)
            logits_flash = head(output_flash)
            out_logp = nn.LogSoftmax(dim=2)(logits)
            out_logp_flash = nn.LogSoftmax(dim=2)(logits_flash)
            plogp = target_p * out_logp
            plogp_flash = target_p * out_logp_flash

            # Apply loss mask on calculation over batch
            loss = -torch.sum(loss_mask * plogp, 2).mean()
            loss_flash = -torch.sum(loss_mask * plogp_flash, 2).mean()
            torch.testing.assert_close(loss, loss_flash, atol=1e-3, rtol=1e-3)
            loss_list.append(loss)
            loss_flash_list.append(loss_flash)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                loss_mask = padding(loss_mask, left=False)
                hidden_states = padding(hidden_states, left=False)
                flash_hidden_states = padding(flash_hidden_states, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flash = sum(loss_flash_list) / len(loss_flash_list)
        mean_loss.backward()
        mean_loss_flash.backward()
        projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in projections:
            torch.testing.assert_close(
                getattr(attention, proj_name).weight.grad,
                getattr(flash_attention, proj_name).weight.grad,
                atol=1e-3,
                rtol=1e-3,
                msg=f"Gradients should be similar between LlamaAttention and LlamaFlashAttention for {proj_name}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
