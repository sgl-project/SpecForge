import unittest

import torch
from torch import nn
from torch.testing import assert_close
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import create_dflash_sdpa_mask
from specforge.modeling.draft.dflash import Qwen3DFlashAttention
from specforge.modeling.draft.dflash_kernels import DFlashKernels


def _make_attention(layer_type, implementation, sliding_window):
    config = Qwen3Config(
        hidden_size=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=1,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=[layer_type],
        sliding_window=sliding_window,
        use_sliding_window=sliding_window is not None,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = implementation
    kernels = DFlashKernels(
        make_rms_norm=lambda *_: nn.Identity(),
        make_mlp=lambda *_: nn.Identity(),
    )
    return Qwen3DFlashAttention(config, layer_idx=0, kernels=kernels).eval()


def _forward(attention, hidden_states, target_hidden, attention_mask):
    total_length = target_hidden.shape[1] + hidden_states.shape[1]
    position_embeddings = (
        hidden_states.new_ones(1, total_length, attention.head_dim),
        hidden_states.new_zeros(1, total_length, attention.head_dim),
    )
    return attention(
        hidden_states=hidden_states,
        target_hidden=target_hidden,
        position_embeddings=position_embeddings,
        attention_mask=attention_mask,
    )


class TestDFlashEagerAttentionMasking(unittest.TestCase):
    def test_eager_matches_sdpa_for_full_and_sliding_masks(self):
        for layer_type, sliding_window in (
            ("full_attention", None),
            ("sliding_attention", 2),
        ):
            with self.subTest(layer_type=layer_type):
                torch.manual_seed(17)
                eager = _make_attention(layer_type, "eager", sliding_window)
                sdpa = _make_attention(layer_type, "sdpa", sliding_window)
                sdpa.load_state_dict(eager.state_dict())

                mask = create_dflash_sdpa_mask(
                    anchor_positions=torch.tensor([[2, 4]]),
                    block_keep_mask=torch.tensor([[True, False]]),
                    S=4,
                    block_size=2,
                    device=torch.device("cpu"),
                    sliding_window=sliding_window,
                )
                eager_hidden = torch.randn(1, 4, 8, requires_grad=True)
                eager_target = torch.randn(1, 4, 8, requires_grad=True)
                sdpa_hidden = eager_hidden.detach().clone().requires_grad_(True)
                sdpa_target = eager_target.detach().clone().requires_grad_(True)

                eager_output, eager_weights = _forward(
                    eager,
                    eager_hidden,
                    eager_target,
                    mask,
                )
                sdpa_output, _ = _forward(
                    sdpa,
                    sdpa_hidden,
                    sdpa_target,
                    mask,
                )
                assert_close(eager_output, sdpa_output, rtol=1e-5, atol=1e-6)

                allowed = mask.expand_as(eager_weights)
                forbidden_weights = eager_weights.masked_select(~allowed)
                assert_close(
                    forbidden_weights,
                    torch.zeros_like(forbidden_weights),
                    rtol=0,
                    atol=0,
                )
                invalid_rows = ~mask.any(dim=-1).squeeze(1)
                assert_close(
                    eager_output[invalid_rows],
                    torch.zeros_like(eager_output[invalid_rows]),
                    rtol=0,
                    atol=0,
                )

                output_grad = torch.randn_like(eager_output)
                eager_grads = torch.autograd.grad(
                    (eager_output * output_grad).sum(),
                    (eager_hidden, eager_target),
                )
                sdpa_grads = torch.autograd.grad(
                    (sdpa_output * output_grad).sum(),
                    (sdpa_hidden, sdpa_target),
                )
                for eager_grad, sdpa_grad in zip(eager_grads, sdpa_grads):
                    assert_close(eager_grad, sdpa_grad, rtol=1e-5, atol=1e-6)


if __name__ == "__main__":
    unittest.main(verbosity=2)
