# coding=utf-8
"""MoELayer: Qwen3.5-MoE sparse-block parity, optional shared expert, dtype round trip."""

import unittest

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    Qwen3_5MoeSparseMoeBlock,
)

from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe

E, H, I, S, K = 4, 64, 32, 48, 2


def _config(**overrides):
    values = dict(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        norm_topk_prob=True,
    )
    return Qwen3_5MoeTextConfig(**{**values, **overrides})


def _forward_and_grads(module, x):
    x = x.clone().requires_grad_()
    out = module(x)
    params = dict(module.named_parameters())
    grads = torch.autograd.grad(out.square().sum(), [x, *params.values()])
    return out, dict(zip(["x", *params], grads))


class MoELayerTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(2, 5, H)

    def test_matches_transformers_qwen3_5_moe_sparse_block_with_gradients(self):
        # transformers' Qwen3.5-MoE router always renormalizes the top-k probs.
        layer = build_moe(_config())
        reference = Qwen3_5MoeSparseMoeBlock(_config())
        reference.load_state_dict(layer.state_dict())  # strict: same names
        out, grads = _forward_and_grads(layer, self.x)
        ref_out, ref_grads = _forward_and_grads(reference, self.x)
        torch.testing.assert_close(out, ref_out)
        self.assertEqual(grads.keys(), ref_grads.keys())
        for name, grad in grads.items():
            torch.testing.assert_close(grad, ref_grads[name], rtol=1e-4, atol=1e-6)

    def test_zero_shared_width_builds_routed_experts_only(self):
        layer = build_moe(_config(shared_expert_intermediate_size=0))
        self.assertIsNone(layer.shared_expert)
        self.assertIsNone(layer.shared_expert_gate)
        x = self.x.reshape(-1, H)
        _, indices, weights = layer.gate(x)
        routed = layer.experts(x, indices, weights).view_as(self.x)
        torch.testing.assert_close(layer(self.x), routed)

    def test_reset_parameters_resets_the_router_and_experts(self):
        layer = build_moe(_config())
        params = (
            layer.gate.weight,
            layer.experts.gate_up_proj,
            layer.experts.down_proj,
        )
        with torch.no_grad():
            for param in params:
                param.fill_(1e30)
        layer.reset_parameters(0.02)
        for param in params:
            self.assertAlmostEqual(param.std().item(), 0.02, delta=0.004)

    def test_bf16_in_bf16_out(self):
        layer = build_moe(_config()).bfloat16()
        self.assertEqual(layer(self.x.bfloat16()).dtype, torch.bfloat16)


if __name__ == "__main__":
    unittest.main(verbosity=2)
