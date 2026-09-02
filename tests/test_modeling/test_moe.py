# coding=utf-8
"""MoE FFN skeleton: config resolution and the component contracts, with stub
components so the contracts are pinned independently of any target-family
implementation."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.moe import (
    BALANCE_CONTROLLERS,
    EXPERTS_BACKENDS,
    MOE_PRESETS,
    ROUTERS,
    SCORE_FUNCTIONS,
    SHARED_EXPERTS,
    BalanceController,
    RoutedExperts,
    Router,
    RoutingResult,
    SharedExpert,
    register_balance_controller,
    register_experts_backend,
    register_moe_preset,
    register_router,
    register_score_function,
    register_shared_expert,
    resolve_moe_config,
)

PRESET = "_test_family"


# --- stub components: a minimal but real top-k reference for the contracts ---


class _StubBalance(BalanceController):
    def __init__(self, cfg, n_experts):
        super().__init__(cfg, n_experts)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.observed = []
        self.applied = 0

    def adjust_selection_scores(self, scores):
        return scores + self.bias

    def observe(self, counts):
        self.observed.append(counts.detach().clone())

    def apply_pending_update(self):
        self.applied += 1

    def aux_loss(self):
        if self.cfg.aux_loss_coeff <= 0:
            return None
        return torch.tensor(self.cfg.aux_loss_coeff)

    def metrics(self):
        return {"stub_metric": 1.0}


class _StubRouter(Router):
    def __init__(self, cfg, hidden_size, balance):
        super().__init__(cfg, hidden_size, balance)
        self.weight = nn.Parameter(torch.empty(self.n_experts, hidden_size))
        self.score_fn = SCORE_FUNCTIONS.get(cfg.scoring_func)

    def reset_parameters(self, std):
        nn.init.normal_(self.weight, std=std)

    def forward(self, x):
        scores = self.score_fn(x.float() @ self.weight.float().t())
        indices = self.balance.adjust_selection_scores(scores).topk(self.topk).indices
        weights = scores.gather(1, indices)
        if self.cfg.norm_topk_prob:
            weights = weights / weights.sum(-1, keepdim=True)
        weights = weights * self.cfg.routed_scaling_factor
        counts = torch.zeros(
            self.n_experts, dtype=torch.long, device=x.device
        ).scatter_add_(0, indices.flatten(), torch.ones_like(indices.flatten()))
        return RoutingResult(weights=weights, indices=indices, counts=counts)


class _StubExperts(RoutedExperts):
    def __init__(self, cfg, hidden_size):
        super().__init__(cfg, hidden_size)
        self.w = nn.Parameter(torch.empty(self.n_experts, hidden_size, hidden_size))

    def reset_parameters(self, std):
        nn.init.normal_(self.w, std=std)

    def forward(self, x, routing):
        # dense gather: fine for tiny tests, pins the [T, k] -> [T, H] contract
        per_choice = torch.einsum(
            "td,tkdo->tko", x.float(), self.w[routing.indices].float()
        )
        return (routing.weights.unsqueeze(-1) * per_choice).sum(1).to(x.dtype)


class _StubShared(SharedExpert):
    def __init__(self, cfg, hidden_size):
        super().__init__(cfg, hidden_size)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(self, x):
        return self.proj(x)


def setUpModule():
    register_score_function("_test_softplus")(torch.nn.functional.softplus)
    register_router("_test_router")(_StubRouter)
    register_balance_controller("_test_balance")(_StubBalance)
    register_experts_backend("_test_experts")(_StubExperts)
    register_shared_expert("_test_shared")(_StubShared)
    register_moe_preset(
        PRESET,
        scoring_func="_test_softplus",
        router="_test_router",
        balance="_test_balance",
        experts_backend="_test_experts",
        shared_expert="_test_shared",
        routed_scaling_factor=1.5,
    )


def tearDownModule():
    SCORE_FUNCTIONS.unregister("_test_softplus")
    ROUTERS.unregister("_test_router")
    BALANCE_CONTROLLERS.unregister("_test_balance")
    EXPERTS_BACKENDS.unregister("_test_experts")
    SHARED_EXPERTS.unregister("_test_shared")
    MOE_PRESETS.unregister(PRESET)


def _moe_json(**overrides):
    payload = dict(
        hidden_size=16,
        moe_preset=PRESET,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dflash_config={},
    )
    payload.update(overrides)
    return payload


def _dflash_config(moe=True, **overrides):
    fields = dict(
        architectures=["DFlashDraftModel"],
        block_size=2,
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=2,
        num_key_value_heads=1,
        num_hidden_layers=2,
        num_target_layers=6,
        head_dim=8,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention", "full_attention"],
        dflash_config={"attention_mode": "gqa"},
    )
    if moe:
        fields.update(_moe_json(hidden_size=16))
        fields["dflash_config"] = {
            "attention_mode": "gqa",
            "moe_bias_update_rate": 0.005,
        }
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


class TestMoEConfig(unittest.TestCase):
    def test_dense_config_resolves_to_none(self):
        self.assertIsNone(resolve_moe_config({"hidden_size": 16}))
        self.assertIsNone(resolve_moe_config(SimpleNamespace(n_routed_experts=0)))

    def test_preset_is_required_for_moe(self):
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            resolve_moe_config(_moe_json(moe_preset=None))
        with self.assertRaisesRegex(KeyError, "available"):
            resolve_moe_config(_moe_json(moe_preset="no-such-family"))

    def test_preset_defaults_and_explicit_overrides(self):
        cfg = resolve_moe_config(_moe_json())
        self.assertEqual(cfg.preset, PRESET)
        self.assertEqual(cfg.scoring_func, "_test_softplus")
        self.assertEqual(cfg.routed_scaling_factor, 1.5)
        self.assertEqual(
            cfg.shared_expert_intermediate_size, 8
        )  # defaults to moe width
        cfg = resolve_moe_config(
            _moe_json(routed_scaling_factor=2.0, shared_expert_intermediate_size=4)
        )
        self.assertEqual(cfg.routed_scaling_factor, 2.0)
        self.assertEqual(cfg.shared_expert_intermediate_size, 4)
        # Works on attribute-style configs (HF PretrainedConfig) as well.
        self.assertEqual(resolve_moe_config(_dflash_config()).n_routed_experts, 8)

    def test_training_knobs_come_from_dflash_config(self):
        cfg = resolve_moe_config(
            _moe_json(
                dflash_config={
                    "moe_bias_update_rate": 0.01,
                    "moe_dispatch": "grouped_mm",
                }
            )
        )
        self.assertEqual(cfg.bias_update_rate, 0.01)
        self.assertEqual(cfg.dispatch, "grouped_mm")
        with self.assertRaisesRegex(ValueError, "unknown MoE training keys"):
            resolve_moe_config(_moe_json(dflash_config={"moe_bias_udpate_rate": 1}))

    def test_validation(self):
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            resolve_moe_config(_moe_json(num_experts_per_tok=9))
        with self.assertRaisesRegex(ValueError, "n_shared_experts"):
            resolve_moe_config(_moe_json(n_shared_experts=2))
        with self.assertRaisesRegex(ValueError, "n_group"):
            resolve_moe_config(_moe_json(n_group=3))
        cfg = resolve_moe_config(_moe_json(n_group=4, topk_group=2))
        self.assertTrue(cfg.group_limited)
        self.assertFalse(resolve_moe_config(_moe_json()).group_limited)

    def test_presets_cannot_set_per_run_or_training_fields(self):
        with self.assertRaisesRegex(ValueError, "per-run/training"):
            register_moe_preset("_bad", n_routed_experts=4)
        with self.assertRaisesRegex(ValueError, "unknown MoEConfig fields"):
            register_moe_preset("_bad", nonsense=1)
        self.assertNotIn("_bad", MOE_PRESETS)

    def test_registry_errors_name_the_kind_and_choices(self):
        with self.assertRaisesRegex(KeyError, "MoE router.*available"):
            ROUTERS.get("missing")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_router("_test_router")(_StubRouter.__mro__[1])
        self.assertIn("none", BALANCE_CONTROLLERS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
