import unittest

import torch
import torch.nn.functional as F
from transformers import LlamaConfig

from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaFlashAttention,
    prepare_decoder_attention_mask,
)
from specforge.utils import padding
from tests.test_utils.utils import norm_tensor

TTT_LENGTH = 7
torch.manual_seed(0)


def assert_similar(ref, out):
    # We are looser with the checks since we are comparing bf16 backends
    ref = ref.to(torch.float32)
    out = out.to(torch.float32)
    similarity = F.cosine_similarity(ref.flatten(), out.flatten(), dim=0)
    norm_ratio = torch.linalg.norm(ref) / torch.linalg.norm(out)
    assert similarity >= 0.975, f"{similarity=}"
    assert abs(1 - norm_ratio) <= 0.025, f"{norm_ratio=}"


class TestFlashAttention(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(0)
        self.config_dict = {
            "hidden_size": 128,
            "num_attention_heads": 8,
            "num_key_value_heads": 2,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-05,
            "vocab_size": 32000,
            "intermediate_size": 688,
            "hidden_act": "silu",
            "num_hidden_layers": 1,
            "torch_dtype": "bfloat16",
        }
        self.config = LlamaConfig(**self.config_dict)

        self.seq_lengths = [128, 200, 256, 300, 512, 800, 1024, 2048]
        self.dtype = torch.bfloat16

    def test_forward_pass_comparison(self):
        """Test forward pass comparison between LlamaAttention and LlamaFlashAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_forward_pass_comparison_for_seq_len(seq_len)

    def _test_forward_pass_comparison_for_seq_len(self, seq_len):
        """Helper method to test forward pass comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flash_attention = LlamaFlashAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flash_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flash_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flash_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flash_attention.o_proj.weight.copy_(attention.o_proj.weight)

        attention.eval()
        flash_attention.eval()
        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############

        position_ids = (
            torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        )
        cache_hidden = [[], []]  # [cache_k, cache_v]
        flash_cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=self.dtype).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        padding_start_index = seq_len - min(
            200, seq_len // 3
        )  # Adjust padding based on seq_len
        attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )
        hidden_states_list = []
        flash_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flash_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flash_hidden_states_list.append(flash_hidden_states)

        ############### Flash Attention Inputs ##############
        flash_position_ids = position_ids.clone()
        for idx in range(TTT_LENGTH):
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
                output_flash = flash_attention(
                    hidden_states=flash_hidden_states_list[idx],
                    position_ids=flash_position_ids,
                    cache_hidden=flash_cache_hidden,
                )
            assert_similar(output[0][: -1 - idx], output_flash[0][: -1 - idx])
            assert_similar(
                output[1][: padding_start_index - idx],
                output_flash[1][: padding_start_index - idx],
            )
            # Check output shape
            expected_output_shape = (batch_size, seq_len, self.config.hidden_size)
            self.assertEqual(output_flash.shape, expected_output_shape)
            # Check output is not NaN or Inf
            self.assertFalse(torch.isnan(output_flash).any())
            self.assertFalse(torch.isinf(output_flash).any())

    def test_backward_pass_gradient_comparison(self):
        """Test backward pass comparing gradients between LlamaAttention and LlamaFlashAttention."""
        for seq_len in self.seq_lengths:
            with self.subTest(seq_len=seq_len):
                self._test_backward_pass_gradient_comparison_for_seq_len(seq_len)

    def _test_backward_pass_gradient_comparison_for_seq_len(self, seq_len):
        """Helper method to test backward pass gradient comparison for a specific sequence length."""
        attention = LlamaAttention(self.config).to("cuda").to(self.dtype)
        flash_attention = LlamaFlashAttention(self.config).to("cuda").to(self.dtype)

        # Ensure same weights
        with torch.no_grad():
            flash_attention.q_proj.weight.copy_(attention.q_proj.weight)
            flash_attention.k_proj.weight.copy_(attention.k_proj.weight)
            flash_attention.v_proj.weight.copy_(attention.v_proj.weight)
            flash_attention.o_proj.weight.copy_(attention.o_proj.weight)

        batch_size = 2
        hidden_size = self.config.hidden_size * 2

        ############### Attention Inputs ##############
        position_ids = (
            torch.arange(seq_len).unsqueeze(0).repeat(batch_size, 1).to("cuda")
        )
        cache_hidden = [[], []]  # [cache_k, cache_v]
        flash_cache_hidden = [[], []]  # [cache_k, cache_v]
        attention_mask = torch.ones(batch_size, seq_len, dtype=torch.bool).to("cuda")
        # Simulate one item in the batch is masked and not taking a full block.
        # padding_start_index = seq_len - 50
        # attention_mask[1, padding_start_index:] = False
        input_embeds = norm_tensor(
            (batch_size, seq_len, self.config.hidden_size),
            device="cuda",
            dtype=self.dtype,
        )
        decoder_attention_mask = prepare_decoder_attention_mask(
            attention_mask=attention_mask,
            input_shape=(batch_size, seq_len),
            inputs_embeds=input_embeds,
            past_key_values_length=0,
        )

        ############### Flash Attention Inputs ##############
        flash_position_ids = position_ids.clone()
        loss_mask = torch.ones(
            batch_size, seq_len, dtype=self.dtype, requires_grad=False
        ).to("cuda")

        # Create input tensors that require gradients
        loss_list = []
        loss_flash_list = []
        hidden_states_list = []
        flash_hidden_states_list = []
        for idx in range(TTT_LENGTH):
            hidden_states = norm_tensor(
                (batch_size, seq_len, hidden_size), device="cuda", dtype=self.dtype
            )
            flash_hidden_states = hidden_states.clone().detach()
            hidden_states_list.append(hidden_states)
            flash_hidden_states_list.append(flash_hidden_states)

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
            output_flash = flash_attention(
                hidden_states=flash_hidden_states_list[idx],
                position_ids=flash_position_ids,
                cache_hidden=flash_cache_hidden,
            )
            # Apply loss mask on calculation over batch
            loss = (output * loss_mask[..., None]).sum().mean()
            loss_flash = (output_flash * loss_mask[..., None]).sum().mean()
            loss_list.append(loss)
            loss_flash_list.append(loss_flash)
            # Compare gradients

            if not is_last:
                # Step 5.7: we need to update the loss mask
                loss_mask = padding(loss_mask, left=False)
        mean_loss = sum(loss_list) / len(loss_list)
        mean_loss_flash = sum(loss_flash_list) / len(loss_flash_list)
        mean_loss.backward()
        mean_loss_flash.backward()
        projections = ["q_proj", "k_proj", "v_proj", "o_proj"]
        for proj_name in projections:
            assert_similar(
                getattr(attention, proj_name).weight.grad,
                getattr(flash_attention, proj_name).weight.grad,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
