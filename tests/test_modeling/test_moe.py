# coding=utf-8
"""MoE FFN: config resolution, the router and experts, composition, and the
model/trainer/checkpoint seams. The test family registers its own score
function and a stub balance controller, so the contracts are pinned
independently of any target family."""

import unittest
from types import SimpleNamespace

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3MLP

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe import (
    BALANCE_CONTROLLERS,
    EXPERTS_BACKENDS,
    MOE_PRESETS,
    ROUTERS,
    SCORE_FUNCTIONS,
    SHARED_EXPERTS,
    BalanceController,
    MoELayer,
    Router,
    apply_pending_balance_updates,
    build_ffn,
    collect_moe_aux_loss,
    collect_moe_metrics,
    from_checkpoint_state_dict,
    iter_moe_layers,
    plan_warm_start,
    register_balance_controller,
    register_experts_backend,
    register_moe_preset,
    register_router,
    register_score_function,
    register_shared_expert,
    register_state_dict_converter,
    resolve_moe_config,
    select_target_experts,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.experts import GroupedExperts, swiglu
from specforge.modeling.draft.moe.router import TopKRouter
from specforge.modeling.draft.moe.shared import SwiGLUSharedExpert
from specforge.modeling.draft.moe.state_dict import unregister_state_dict_converter
from specforge.training.strategies.base import _moe_metrics

PRESET = "_test_family"


# --- stub balance controller: the contract the router and the layer call ---


class _StubBalance(BalanceController):
    def __init__(self, cfg, n_experts):
        super().__init__(cfg, n_experts)
        self.register_buffer("bias", torch.zeros(n_experts))
        self.observed = []
        self.applied = 0

    def adjust_selection_scores(self, scores):
        return scores + self.bias

    def observe(self, routing):
        self.observed.append(routing.counts.detach().clone())
        self.saw_scores = routing.scores is not None

    def apply_pending_update(self):
        self.applied += 1

    def aux_loss(self):
        if self.cfg.aux_loss_coeff <= 0:
            return None
        return torch.tensor(self.cfg.aux_loss_coeff)

    def metrics(self):
        return {"stub_metric": 1.0}


def setUpModule():
    register_score_function("_test_softplus")(torch.nn.functional.softplus)
    register_balance_controller("_test_balance")(_StubBalance)
    register_moe_preset(
        PRESET,
        scoring_func="_test_softplus",
        balance="_test_balance",
        routed_scaling_factor=1.5,
    )


def tearDownModule():
    SCORE_FUNCTIONS.unregister("_test_softplus")
    BALANCE_CONTROLLERS.unregister("_test_balance")
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
        fields["dflash_config"] = {"attention_mode": "gqa"}
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


class TestMoEConfig(unittest.TestCase):
    def test_dense_configs_resolve_to_none(self):
        self.assertIsNone(resolve_moe_config({"hidden_size": 16}))
        self.assertIsNone(resolve_moe_config({"n_routed_experts": 0}))

    def test_preset_defaults_and_explicit_overrides(self):
        cfg = resolve_moe_config(_moe_json())
        self.assertEqual(cfg.preset, PRESET)
        self.assertEqual(cfg.scoring_func, "_test_softplus")
        self.assertEqual(cfg.routed_scaling_factor, 1.5)
        self.assertEqual(cfg.shared_expert_intermediate_size, 8)  # the moe width
        cfg = resolve_moe_config(_moe_json(routed_scaling_factor=2.0))
        self.assertEqual(cfg.routed_scaling_factor, 2.0)
        # Works on attribute-style configs (HF PretrainedConfig) as well.
        self.assertEqual(resolve_moe_config(_dflash_config()).n_routed_experts, 8)

    def test_training_keys_live_under_dflash_config(self):
        cfg = resolve_moe_config(
            _moe_json(dflash_config={"moe_aux_loss_coeff": 0.01, "moe_dispatch": "x"})
        )
        self.assertEqual(cfg.aux_loss_coeff, 0.01)
        self.assertEqual(cfg.dispatch, "x")
        with self.assertRaisesRegex(ValueError, "unknown MoE training keys"):
            resolve_moe_config(_moe_json(dflash_config={"moe_typo": 1}))

    def test_expert_counts_must_be_positive_integers(self):
        for bad in (-1, 0, 0.5, "8", True):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                resolve_moe_config(_moe_json(n_routed_experts=bad))
        self.assertEqual(
            resolve_moe_config(_moe_json(n_routed_experts=8.0)).n_routed_experts, 8
        )
        with self.assertRaisesRegex(ValueError, "num_experts_per_tok"):
            resolve_moe_config(_moe_json(num_experts_per_tok=9))

    def test_routed_experts_without_a_preset_never_fall_back_to_dense(self):
        config = Qwen3Config(hidden_size=16, intermediate_size=32)
        config.n_routed_experts = 8
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            build_ffn(config, dense=lambda c: nn.Identity())
        dense = Qwen3Config(hidden_size=16, intermediate_size=32)
        self.assertIsInstance(
            build_ffn(dense, dense=lambda c: nn.Identity()), nn.Identity
        )

    def test_per_run_sizes_are_required(self):
        payload = _moe_json()
        del payload["moe_intermediate_size"]
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size"):
            resolve_moe_config(payload)

    def test_presets_may_not_set_per_run_or_training_fields(self):
        with self.assertRaisesRegex(ValueError, "per-run/training"):
            register_moe_preset("_bad", n_routed_experts=4)
        with self.assertRaisesRegex(ValueError, "per-run/training"):
            register_moe_preset("_bad", aux_loss_coeff=0.1)
        with self.assertRaisesRegex(ValueError, "unknown MoEConfig"):
            register_moe_preset("_bad", not_a_field=1)

    def test_registry_errors_name_the_kind_and_choices(self):
        with self.assertRaisesRegex(KeyError, "MoE router.*available"):
            ROUTERS.get("missing")
        with self.assertRaisesRegex(ValueError, "already registered"):
            register_router("topk")(Router)
        self.assertIs(BALANCE_CONTROLLERS.get("none"), BalanceController)


class TestMoELayerComposition(unittest.TestCase):
    def _layer(self, **overrides):
        torch.manual_seed(0)
        cfg = resolve_moe_config(_moe_json(**overrides))
        layer = MoELayer(cfg, 16)
        layer.reset_parameters(std=0.02)
        return layer

    def test_dense_config_uses_the_dense_factory_verbatim(self):
        sentinel = nn.Identity()
        self.assertIs(
            build_ffn(SimpleNamespace(hidden_size=16), lambda c: sentinel), sentinel
        )

    def test_moe_config_builds_official_attribute_layout(self):
        layer = build_ffn(SimpleNamespace(**_moe_json()), lambda c: nn.Identity())
        self.assertIsInstance(layer, MoELayer)
        names = {name for name, _ in layer.named_children()}
        self.assertEqual(names, {"gate", "experts", "shared_experts"})
        self.assertIsInstance(layer.gate.balance, _StubBalance)
        self.assertIs(layer.balance, layer.gate.balance)

    def test_forward_preserves_shape_and_adds_shared_expert(self):
        layer = self._layer()
        x = torch.randn(2, 3, 16)
        y = layer(x)
        self.assertEqual(y.shape, x.shape)
        routed_only = self._layer(n_shared_experts=0)
        self.assertIsNone(routed_only.shared_experts)
        routed_only.load_state_dict(
            {k: v for k, v in layer.state_dict().items() if "shared" not in k}
        )
        shared = layer.shared_experts(x.reshape(-1, 16)).view_as(x)
        self.assertTrue(torch.allclose(y, routed_only(x) + shared, atol=1e-5))

    def test_training_observes_counts_and_eval_does_not(self):
        layer = self._layer().train()
        x = torch.randn(5, 16)
        layer(x)
        self.assertEqual(len(layer.balance.observed), 1)
        self.assertEqual(int(layer.balance.observed[0].sum()), 5 * 2)
        self.assertTrue(torch.equal(layer.last_counts, layer.balance.observed[0]))
        self.assertTrue(layer.balance.saw_scores)
        layer.eval()
        layer(x)
        self.assertEqual(len(layer.balance.observed), 1)

    def test_model_hooks_delegate_to_the_controller(self):
        layer = self._layer(dflash_config={"moe_aux_loss_coeff": 0.25}).train()
        layer(torch.randn(4, 16))
        layer.apply_pending_balance_update()
        self.assertEqual(layer.balance.applied, 1)
        self.assertAlmostEqual(float(layer.aux_loss()), 0.25)
        metrics = layer.metrics()
        self.assertEqual(
            set(metrics),
            {"load_max_ratio", "load_min_ratio", "experts_unused_frac", "stub_metric"},
        )
        self.assertGreaterEqual(float(metrics["load_max_ratio"]), 1.0)
        self.assertLessEqual(float(metrics["load_min_ratio"]), 1.0)

    def test_selection_bias_changes_choice_not_weights(self):
        layer = self._layer().eval()
        x = torch.randn(1, 16)
        before = layer.gate(x)
        layer.balance.bias[:] = -1e3
        favored = int((before.indices[0, 0] + 1) % 8)
        layer.balance.bias[favored] = 0.0
        after = layer.gate(x)
        self.assertIn(favored, after.indices[0].tolist())
        # combine weights come from raw scores: still normalized and scaled
        self.assertAlmostEqual(float(after.weights.detach().sum()), 1.5, places=5)


class TestRouter(unittest.TestCase):
    def test_score_functions(self):
        logits = torch.randn(4, 6)
        cfg = resolve_moe_config(_moe_json(scoring_func="softmax"))
        router = TopKRouter(cfg, 16, BalanceController(cfg, 8))
        self.assertTrue(torch.allclose(router.score_fn(logits), logits.softmax(-1)))


def _reference_routed(x, routing, w1, w2, w3):
    """Per-token, per-choice reference of the routed sum in fp32."""
    out = torch.zeros(x.shape, dtype=torch.float32)
    for t in range(x.shape[0]):
        for j in range(routing.topk):
            e = int(routing.indices[t, j])
            h = swiglu(x[t] @ w1[e].t(), x[t] @ w3[e].t())
            out[t] += routing.weights[t, j] * (h.to(x.dtype) @ w2[e].t()).float()
    return out


class TestExpertsAndLayer(unittest.TestCase):
    def _layer(self, **overrides):
        torch.manual_seed(0)
        layer = MoELayer(resolve_moe_config(_moe_json(**overrides)), 16)
        layer.reset_parameters(std=0.2)
        return layer

    def test_grouped_experts_match_the_per_token_reference_and_return_fp32(self):
        layer = self._layer()
        x = torch.randn(9, 16) * 4
        routing = layer.gate(x)
        y = layer.experts(x, routing)
        self.assertEqual(y.dtype, torch.float32)
        e = layer.experts
        expected = _reference_routed(x, routing, e.w1, e.w2, e.w3)
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_grouped_mm_matches_the_loop(self):
        loop = self._layer().cuda().to(torch.bfloat16)
        grouped = self._layer(dflash_config={"moe_dispatch": "grouped_mm"}).cuda()
        grouped.load_state_dict(loop.state_dict())
        grouped.to(torch.bfloat16)
        x = torch.randn(64, 16, device="cuda", dtype=torch.bfloat16)
        a, b = loop(x).float(), grouped(x).float()
        self.assertLess(float((a - b).norm() / a.norm()), 1e-2)

    def test_layer_accumulates_routed_and_shared_in_fp32_and_casts_once(self):
        layer = self._layer().to(torch.bfloat16)
        x = torch.randn(2, 5, 16, dtype=torch.bfloat16)
        y = layer.eval()(x)
        flat = x.reshape(-1, 16)
        routing = layer.gate(flat)
        expected = layer.experts(flat, routing) + layer.shared_experts(flat).float()
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(y, expected.to(torch.bfloat16).view_as(x)))

    def test_layer_builds_the_registered_implementations(self):
        layer = self._layer()
        self.assertIsInstance(layer.experts, GroupedExperts)
        self.assertIsInstance(layer.shared_experts, SwiGLUSharedExpert)

        class _Experts(GroupedExperts):
            pass

        class _Shared(SwiGLUSharedExpert):
            pass

        register_experts_backend("_test_experts")(_Experts)
        self.addCleanup(EXPERTS_BACKENDS.unregister, "_test_experts")
        register_shared_expert("_test_shared")(_Shared)
        self.addCleanup(SHARED_EXPERTS.unregister, "_test_shared")
        layer = self._layer(
            experts_backend="_test_experts", shared_expert="_test_shared"
        )
        self.assertIs(type(layer.experts), _Experts)
        self.assertIs(type(layer.shared_experts), _Shared)


class TestHooks(unittest.TestCase):
    def _tree(self, **overrides):
        cfg = resolve_moe_config(_moe_json(**overrides))
        a, b = MoELayer(cfg, 16), MoELayer(cfg, 16)
        for layer in (a, b):
            layer.reset_parameters(std=0.02)
        return nn.Sequential(nn.Linear(16, 16), a, nn.Linear(16, 16), b), (a, b)

    def test_iteration_updates_and_aggregation(self):
        tree, (a, b) = self._tree(dflash_config={"moe_aux_loss_coeff": 0.5})
        self.assertEqual(list(iter_moe_layers(tree)), [a, b])
        tree.train()(torch.randn(3, 16))
        apply_pending_balance_updates(tree)
        self.assertEqual((a.balance.applied, b.balance.applied), (1, 1))
        self.assertAlmostEqual(float(collect_moe_aux_loss(tree)), 1.0)
        metrics = collect_moe_metrics(tree)
        self.assertTrue(all(key.startswith("moe/") for key in metrics))
        self.assertAlmostEqual(float(metrics["moe/stub_metric"]), 1.0)

    def test_dense_trees_are_inert(self):
        dense = nn.Sequential(nn.Linear(16, 16))
        apply_pending_balance_updates(dense)
        self.assertIsNone(collect_moe_aux_loss(dense))
        self.assertEqual(collect_moe_metrics(dense), {})
        self.assertEqual(_moe_metrics(SimpleNamespace()), {})

    def test_strategy_metrics_read_the_wrapped_draft(self):
        tree, _ = self._tree()
        tree.train()(torch.randn(3, 16))
        metrics = _moe_metrics(SimpleNamespace(draft_model=tree))
        self.assertIn("moe/load_max_ratio", metrics)


class TestStateDictBoundary(unittest.TestCase):
    def test_dense_state_passes_through_unchanged(self):
        state = {"a": torch.zeros(1), "layers.0.mlp.gate_proj.weight": torch.ones(1)}
        for convert in (to_checkpoint_state_dict, from_checkpoint_state_dict):
            out = convert(dict(state))
            self.assertEqual(set(out), set(state))
            for key in state:
                self.assertIs(out[key], state[key])

    def test_registered_converters_apply_in_both_directions(self):
        def to_ckpt(state):
            return {k.replace("native.", "official."): v for k, v in state.items()}

        def from_ckpt(state):
            return {k.replace("official.", "native."): v for k, v in state.items()}

        register_state_dict_converter(
            "_test", to_checkpoint=to_ckpt, from_checkpoint=from_ckpt
        )
        try:
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_state_dict_converter(
                    "_test", to_checkpoint=to_ckpt, from_checkpoint=from_ckpt
                )
            official = to_checkpoint_state_dict({"native.w": 1, "other": 2})
            self.assertEqual(official, {"official.w": 1, "other": 2})
            self.assertEqual(
                from_checkpoint_state_dict(official), {"native.w": 1, "other": 2}
            )
        finally:
            unregister_state_dict_converter("_test")
        self.assertEqual(to_checkpoint_state_dict({"native.w": 1}), {"native.w": 1})


class TestWarmStartPlan(unittest.TestCase):
    def test_selection_strategies(self):
        self.assertEqual(select_target_experts(256, 4), (0, 64, 128, 192))
        self.assertEqual(select_target_experts(8, 3, "strided"), (0, 2, 5))
        self.assertEqual(select_target_experts(8, 3, "first"), (0, 1, 2))
        self.assertEqual(len(set(select_target_experts(256, 64))), 64)
        with self.assertRaises(ValueError):
            select_target_experts(4, 8)
        with self.assertRaises(ValueError):
            select_target_experts(8, 2, "random")

    def test_plan_follows_the_moe_config(self):
        cfg = resolve_moe_config(_moe_json(n_shared_experts=0))
        plan = plan_warm_start(cfg, n_target_experts=64)
        self.assertEqual(plan.n_draft_experts, 8)
        self.assertFalse(plan.copy_shared_expert)
        self.assertTrue(plan.copy_gate_rows)


class TestDFlashWiring(unittest.TestCase):
    def _forward(self, model):
        return model(
            position_ids=torch.arange(6).unsqueeze(0),
            noise_embedding=torch.randn(1, 2, 16),
            target_hidden=torch.randn(1, 4, 2 * 16),
        )

    def test_dense_draft_is_unchanged(self):
        model = DFlashDraftModel(_dflash_config(moe=False))
        for layer in model.layers:
            self.assertIsInstance(layer.mlp, Qwen3MLP)
        self.assertEqual(list(iter_moe_layers(model)), [])

    def test_moe_draft_layers_init_and_apply_balance_updates(self):
        model = DFlashDraftModel(_dflash_config())
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertIs(layer, [m for m in model.layers if m.mlp is layer][0].mlp)
            # _init_weights reached the bare Parameters (no uninitialized memory)
            self.assertTrue(torch.isfinite(layer.gate.weight).all())
            self.assertGreater(float(layer.gate.weight.abs().sum()), 0.0)
            self.assertTrue(torch.isfinite(layer.experts.w1).all())
        model.train()
        self._forward(model)
        self._forward(model)
        self.assertEqual([layer.balance.applied for layer in layers], [2, 2])
        self.assertEqual([len(layer.balance.observed) for layer in layers], [2, 2])
        model.eval()
        self._forward(model)
        self.assertEqual([layer.balance.applied for layer in layers], [2, 2])

    def test_state_dict_names_follow_the_official_layout(self):
        model = DFlashDraftModel(_dflash_config())
        keys = set(model.state_dict())
        self.assertIn("layers.0.mlp.gate.weight", keys)
        self.assertIn("layers.0.mlp.shared_experts.w1.weight", keys)
        self.assertTrue(any(k.startswith("layers.0.mlp.experts.") for k in keys))


if __name__ == "__main__":
    unittest.main(verbosity=2)
