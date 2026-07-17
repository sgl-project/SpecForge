import copy
import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import Qwen3DFlashAttention


class _SplitLinear(nn.Module):
    """Reference adapter for the former context/noise projection calls."""

    def __init__(self, linear: nn.Linear, context_length: int) -> None:
        super().__init__()
        self.linear = linear
        self.context_length = context_length

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        context = self.linear(values[:, : self.context_length])
        noise = self.linear(values[:, self.context_length :])
        return torch.cat((context, noise), dim=1)


class TestQwen3DFlashAttention(unittest.TestCase):
    def test_combined_kv_projection_preserves_forward_and_gradients(self):
        torch.manual_seed(17)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=4,
            attention_bias=True,
            attention_dropout=0.0,
            vocab_size=32,
        )
        config._attn_implementation = "eager"
        combined = Qwen3DFlashAttention(config, layer_idx=0).double()
        split = copy.deepcopy(combined)
        context_length = 3
        split.k_proj = _SplitLinear(split.k_proj, context_length)
        split.v_proj = _SplitLinear(split.v_proj, context_length)

        combined_hidden = torch.randn(2, 4, 16, dtype=torch.float64, requires_grad=True)
        combined_target = torch.randn(
            2, context_length, 16, dtype=torch.float64, requires_grad=True
        )
        split_hidden = combined_hidden.detach().clone().requires_grad_(True)
        split_target = combined_target.detach().clone().requires_grad_(True)
        positions = context_length + combined_hidden.shape[1]
        cos = torch.randn(2, positions, config.head_dim, dtype=torch.float64)
        sin = torch.randn_like(cos)

        projection_inputs = {"k": [], "v": []}
        hooks = [
            combined.k_proj.register_forward_pre_hook(
                lambda _module, args: projection_inputs["k"].append(args[0])
            ),
            combined.v_proj.register_forward_pre_hook(
                lambda _module, args: projection_inputs["v"].append(args[0])
            ),
        ]
        try:
            combined_output, _ = combined(
                hidden_states=combined_hidden,
                target_hidden=combined_target,
                position_embeddings=(cos, sin),
                attention_mask=None,
            )
        finally:
            for hook in hooks:
                hook.remove()
        split_output, _ = split(
            hidden_states=split_hidden,
            target_hidden=split_target,
            position_embeddings=(cos, sin),
            attention_mask=None,
        )

        expected_input = torch.cat((combined_target, combined_hidden), dim=1)
        self.assertEqual(len(projection_inputs["k"]), 1)
        self.assertEqual(len(projection_inputs["v"]), 1)
        torch.testing.assert_close(projection_inputs["k"][0], expected_input)
        torch.testing.assert_close(projection_inputs["v"][0], expected_input)
        torch.testing.assert_close(
            combined_output, split_output, rtol=1e-12, atol=1e-12
        )

        probe = torch.randn_like(combined_output)
        (combined_output * probe).sum().backward()
        (split_output * probe).sum().backward()
        torch.testing.assert_close(
            combined_hidden.grad, split_hidden.grad, rtol=1e-11, atol=1e-12
        )
        torch.testing.assert_close(
            combined_target.grad, split_target.grad, rtol=1e-11, atol=1e-12
        )

        split_parameters = dict(split.named_parameters())
        for name, parameter in combined.named_parameters():
            reference_name = name.replace("k_proj.", "k_proj.linear.").replace(
                "v_proj.", "v_proj.linear."
            )
            torch.testing.assert_close(
                parameter.grad,
                split_parameters[reference_name].grad,
                rtol=1e-11,
                atol=1e-12,
                msg=lambda message, name=name: f"{name}: {message}",
            )


if __name__ == "__main__":
    unittest.main()
