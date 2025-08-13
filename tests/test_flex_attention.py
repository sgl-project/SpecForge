import sys
import unittest

import torch
import torch._dynamo as dynamo
import torch.nn as nn
from transformers import LlamaConfig
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft.flex_attention import (
    compile_friendly_create_block_mask,
    compile_friendly_flex_attention,
    create_block_mask,
    generate_eagle3_mask,
)
from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlexAttention,
    prepare_decoder_attention_mask,
)
from specforge.utils import padding

dynamo.config.recompile_limit = 64
TTT_LENGTH = 7


class TestFlexAttention(unittest.TestCase):
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
            "num_hidden_layers": 1,
            "torch_dtype": "float32",
        }
        self.config = LlamaConfig(**self.config_dict)

        self.seq_lengths = [128, 256, 300, 512, 800, 1024, 2048]
        self.dtype = torch.float32

    def test_forward_pass_comparison(self):
        """Test forward pass comparison between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward_pass_comparison_for_seq_len(seq_len)

    def _test_forward_pass_comparison_for_seq_len(self, seq_len):
        """Helper method to test forward pass comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        attention.eval()
        flex_attention.eval()
        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############

        position_ids = (
            torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        )
        cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = seq_len - min(
            200, seq_len // 3
        )  # Adjust padding based on seq_len
        attention_mask[1, padding_start_index:] = False
        input_embeds = torch.randn(
            batch_size, seq_len, self.config.hidden_size, dtype=self.dtype
        ).to("cuda")
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        past_key_values = DynamicCache()
        for idx in range(TTT_LENGTH):
            is_last = idx == TTT_LENGTH - 1
            with torch.no_grad():
                output = attention(
                    hidden_states=hidden_states_list[idx],
                    attention_mask=decoder_attention_mask,
                    position_ids=position_ids,
                    cache_hidden=cache_hidden,
                    output_attentions=False,
                    use_cache=True,
                )
            with torch.no_grad():
                output_flex = flex_attention(
                    hidden_states=flex_hidden_states_list[idx],
                    attention_mask=attention_mask,
                    position_ids=flex_position_ids,
                    past_key_values=past_key_values,
                )

            torch.testing.assert_close(
                output[0][: -1 - idx], output_flex[0][: -1 - idx], atol=1e-3, rtol=1e-3
            )
            torch.testing.assert_close(
                output[1][: padding_start_index - idx],
                output_flex[1][: padding_start_index - idx],
                atol=1e-3,
                rtol=1e-3,
            )
            if not is_last:
                # Step 5.7: we need to update the loss mask
                ind = torch.arange(seq_len, device=decoder_attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_len - idx]
                decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(
                    decoder_attention_mask.dtype
                ).min
            # Check output shape
            expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
            self.assertEqual(output_flex.shape, expected_output_shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(output_flex).any())
            self.assertFalse(torch.isinf(output_flex).any())

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlexAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward_pass_gradient_comparison_for_seq_len(seq_len)

    def _test_backward_pass_gradient_comparison_for_seq_len(self, seq_len):
        """Helper method to test backward pass gradient comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flex_attention = LlamaFlexAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flex_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flex_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flex_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flex_attention.o_proj.weight.copy_(attention.o_proj.weight)

        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############
        position_ids = (
            torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        )
        cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        # padding_start_index = seq_len - 50
        # attention_mask[1, padding_start_index:] = False
        input_embeds = torch.randn(
            batch_size, seq_len, self.config.hidden_size, dtype=self.dtype
        ).to("cuda")
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flex Attention Inputs ##############
        flex_position_ids = position_ids.clone()
        ttt_length = TTT_LENGTH
        past_key_values = DynamicCache()
        loss_mask = torch.ones(
            batch_size, seq_len, dtype=self.dtype, requires_grad=False
        ).to("cuda")

        # Create input tensors that require gradients
        loss_list = []
        loss_flex_list = []
        hidden_states_list = []
        flex_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = torch.randn(
                batch_size, seq_len, hidden_size, device="cuda", dtype=self.dtype
            )
            flex_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flex_hidden_states_list.append(flex_hidden_states)

        for idx in range(TTT_LENGTH):
            is_last = idx == TTT_LENGTH - 1
            output = attention(
                hidden_states=hidden_states_list[idx],
                attention_mask=decoder_attention_mask,
                position_ids=position_ids,
                cache_hidden=cache_hidden,
                output_attentions=False,
                use_cache=True,
            )
            output_flex = flex_attention(
                hidden_states=flex_hidden_states_list[idx],
                attention_mask=attention_mask,
                position_ids=flex_position_ids,
                past_key_values=past_key_values,
            )
            # Apply loss mask on calculation over batch
            loss = (output * loss_mask[..., None]).sum().mean()
            loss_flex = (output_flex * loss_mask[..., None]).sum().mean()
            torch.testing.assert_close(loss, loss_flex, atol=1e-3, rtol=1e-3)
            loss_list.append(loss)
            loss_flex_list.append(loss_flex)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                ind = torch.arange(seq_len, device=decoder_attention_mask.device)
                ind0 = ind[idx:]
                ind1 = ind[: seq_len - idx]
                decoder_attention_mask[:, :, ind0, ind1] = torch.finfo(
                    decoder_attention_mask.dtype
                ).min
                loss_mask = padding(loss_mask, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flex = sum(loss_flex_list) / len(loss_flex_list)
        mean_loss.backward()
        mean_loss_flex.backward()
        projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in projections:
            torch.testing.assert_close(
                getattr(attention, proj_name).weight.grad,
                getattr(flex_attention, proj_name).weight.grad,
                atol=1e-3,
                rtol=1e-3,
            )


class TestEagle3FlexMask(unittest.TestCase):

    def test_eagle3_flex_mask(self):
        B = 1
        H = 1
        S = 128 * 8
        D = 128
        Q_LEN = S
        KV_LEN = S * 3
        data_type = torch.bfloat16
        query = torch.randn(
            B, H, S, D, device="cuda", dtype=data_type, requires_grad=True
        )
        key_cache = torch.randn(
            B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True
        )
        value_cache = torch.randn(
            B, H, KV_LEN, D, device="cuda", dtype=data_type, requires_grad=True
        )
        seq_lengths = torch.tensor([S], device="cuda", dtype=torch.int32)
        block_mask = compile_friendly_create_block_mask(
            mask_mod=generate_eagle3_mask(
                seq_lengths=seq_lengths, Q_LEN=Q_LEN, KV_LEN=KV_LEN, shift_left=128 * 2
            ),
            B=1,
            H=1,
            Q_LEN=Q_LEN,
            KV_LEN=KV_LEN,
            device=query.device,
        )
        # fmt: off
        expected_mask = torch.tensor([[[
            [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
            [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
            [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0],
            [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1],
        ]]], dtype=torch.int32).to(query.device)
        # fmt: on
        dense_mask = block_mask.to_dense()
        assert torch.allclose(dense_mask, expected_mask)
        output = compile_friendly_flex_attention(
            query, key_cache, value_cache, block_mask=block_mask
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
