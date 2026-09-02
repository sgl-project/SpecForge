# coding=utf-8
"""Sparse MoE draft FFN: routing, balancing bias, dispatch, state-dict naming."""

import unittest
from types import SimpleNamespace

import torch
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe import (
    GroupedExperts,
    SparseMoE,
    stack_grouped_expert_state_dict,
    unstack_grouped_expert_state_dict,
)

CUDA = torch.cuda.is_available()


def _moe_config(**overrides):
    fields = dict(
        hidden_size=32,
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        dflash_config={},
    )
    fields.update(overrides)
    return SimpleNamespace(**fields)


def _dflash_moe_config():
    return Qwen3Config(
        architectures=["DFlashDraftModel"],
        block_size=2,
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_target_layers=6,
        head_dim=16,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention", "full_attention"],
        n_routed_experts=8,
        n_shared_experts=1,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        scoring_func="sqrtsoftplus",
        routed_scaling_factor=1.5,
        swiglu_limit=10.0,
        dflash_config={"attention_mode": "gqa", "moe_bias_update_rate": 0.005},
        _attn_implementation="sdpa",
    )


class TestSparseMoE(unittest.TestCase):
    def _moe(self, **overrides):
        torch.manual_seed(0)
        moe = SparseMoE(_moe_config(**overrides))
        for p in moe.parameters():
            torch.nn.init.normal_(p, std=0.05)
        return moe

    def test_forward_stashes_counts_and_update_moves_bias(self):
        moe = self._moe().train()
        moe.bias_update_rate = 1e-3
        x = torch.randn(6, 32)
        moe(x)
        self.assertIsNotNone(moe._pending_counts)
        before = moe.gate.bias.clone()
        moe.apply_pending_balance_update()
        self.assertIsNone(moe._pending_counts)
        self.assertFalse(torch.equal(before, moe.gate.bias))
        # No pending counts: a second apply is a no-op.
        after = moe.gate.bias.clone()
        moe.apply_pending_balance_update()
        self.assertTrue(torch.equal(after, moe.gate.bias))

    def test_eval_never_stashes_counts(self):
        moe = self._moe().eval()
        moe.bias_update_rate = 1e-3
        moe(torch.randn(6, 32))
        self.assertIsNone(moe._pending_counts)

    def test_gate_bias_stays_fp32_through_dtype_casts(self):
        moe = self._moe().to(torch.bfloat16)
        self.assertEqual(moe.gate.bias.dtype, torch.float32)
        y = moe(torch.randn(4, 32, dtype=torch.bfloat16))
        self.assertEqual(y.dtype, torch.bfloat16)

    def test_state_dict_unstack_stack_roundtrip(self):
        moe = self._moe()
        state = moe.state_dict()
        self.assertIn("experts.w1", state)
        official = unstack_grouped_expert_state_dict(state)
        self.assertIn("experts.0.w1.weight", official)
        self.assertNotIn("experts.w1", official)
        # The module loads the official per-expert naming directly...
        fresh = self._moe()
        fresh.load_state_dict(official)
        self.assertTrue(torch.equal(fresh.experts.w1, moe.experts.w1))
        # ...and stack() is the exact inverse for wrapper-level loads.
        restacked = stack_grouped_expert_state_dict(official)
        self.assertTrue(torch.equal(restacked["experts.w1"], state["experts.w1"]))

    def test_stack_rejects_missing_expert_indices(self):
        official = unstack_grouped_expert_state_dict(self._moe().state_dict())
        del official["experts.3.w2.weight"]
        with self.assertRaises(KeyError):
            stack_grouped_expert_state_dict(official)

    def test_grouped_experts_match_linear_init(self):
        torch.manual_seed(0)
        grouped = GroupedExperts(4, 8, 16, swiglu_limit=0.0)
        self.assertEqual(tuple(grouped.w1.shape), (4, 16, 8))
        self.assertTrue(torch.isfinite(grouped.w1).all())

    @unittest.skipUnless(
        CUDA and hasattr(torch, "_grouped_mm"), "requires CUDA grouped GEMM"
    )
    def test_grouped_dispatch_matches_sorted_loop(self):
        # Enough experts vs tokens that some experts are guaranteed empty.
        torch.manual_seed(3)
        moe = SparseMoE(_moe_config(n_routed_experts=16))
        for p in moe.parameters():
            torch.nn.init.normal_(p, std=0.05)
        moe = moe.to("cuda", torch.bfloat16).train()
        moe.bias_update_rate = 0.0
        x = (torch.randn(6, 32, device="cuda") * 0.5).to(torch.bfloat16)

        results = {}
        for grouped in (False, True):
            moe.grouped_dispatch = grouped
            moe.zero_grad(set_to_none=True)
            xg = x.clone().requires_grad_(True)
            y = moe(xg)
            y.float().square().sum().backward()
            results[grouped] = (
                y.detach().clone(),
                xg.grad.clone(),
                {
                    n: p.grad.clone()
                    for n, p in moe.named_parameters()
                    if p.grad is not None
                },
            )

        y0, dx0, g0 = results[False]
        y1, dx1, g1 = results[True]
        # Same math batched into grouped GEMMs: bf16-rounding-level equal.
        self.assertTrue(torch.allclose(y0.float(), y1.float(), rtol=2e-2, atol=2e-2))
        self.assertTrue(torch.allclose(dx0.float(), dx1.float(), rtol=2e-2, atol=2e-2))
        # The loop leaves inactive experts' grads None; grouped GEMM emits
        # explicit zeros for them (an FSDP-friendly superset).
        self.assertLessEqual(set(g0), set(g1))
        for name in g1:
            if name in g0:
                self.assertTrue(
                    torch.allclose(
                        g0[name].float(), g1[name].float(), rtol=2e-2, atol=2e-2
                    ),
                    name,
                )
            else:
                self.assertEqual(int(torch.count_nonzero(g1[name])), 0, name)


class TestDFlashMoEVariant(unittest.TestCase):
    def test_layers_use_sparse_moe_when_configured(self):
        model = DFlashDraftModel(_dflash_moe_config())
        for layer in model.layers:
            self.assertIsInstance(layer.mlp, SparseMoE)
            self.assertEqual(layer.mlp.bias_update_rate, 0.005)
        # The gate's bare Parameter is covered by _init_weights.
        gate = model.layers[0].mlp.gate
        self.assertTrue(torch.isfinite(gate.weight).all())
        self.assertTrue(torch.equal(gate.bias, torch.zeros_like(gate.bias)))

    def test_dense_config_keeps_dense_mlp(self):
        config = _dflash_moe_config()
        config.n_routed_experts = 0
        model = DFlashDraftModel(config)
        for layer in model.layers:
            self.assertNotIsInstance(layer.mlp, SparseMoE)

    def test_model_state_dict_roundtrips_through_official_naming(self):
        model = DFlashDraftModel(_dflash_moe_config())
        official = unstack_grouped_expert_state_dict(model.state_dict())
        stacked_keys = [k for k in official if k.endswith(".experts.w1")]
        self.assertEqual(stacked_keys, [])
        model.load_state_dict(stack_grouped_expert_state_dict(official))


if __name__ == "__main__":
    unittest.main(verbosity=2)
