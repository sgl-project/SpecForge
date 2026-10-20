# coding=utf-8
"""MoE FFN framework tests; the file registers its own preset and layout."""

import os
import tempfile
import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe import (
    MoELayer,
    Router,
    apply_pending_balance_updates,
    build_ffn,
    collect_moe_aux_loss,
    collect_moe_metrics,
    from_checkpoint_state_dict,
    iter_moe_layers,
    register_balance_controller,
    register_experts_backend,
    register_layout,
    register_moe_preset,
    register_router,
    register_shared_expert,
    resolve_moe_config,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.balance import BALANCE_CONTROLLERS, BalanceController
from specforge.modeling.draft.moe.checkpoint import LAYOUTS, layout_for
from specforge.modeling.draft.moe.config import MOE_PRESETS
from specforge.modeling.draft.moe.experts import (
    EXPERTS_BACKENDS,
    GroupedExperts,
    swiglu,
)
from specforge.modeling.draft.moe.router import ROUTERS, TopKRouter
from specforge.modeling.draft.moe.shared import SHARED_EXPERTS, SwiGLUSharedExpert

H = 16
PRESET = "_test_family"
LAYOUT = "_test_per_expert"


# -- a per-expert test layout, as a family would register it -------------------


def _to_per_expert(layer):
    out = {}
    for key, value in layer.items():
        if key.startswith("experts.w") and value.dim() == 3:
            for i in range(value.shape[0]):
                out[f"experts.{i}.{key[len('experts.'):]}.weight"] = value[i]
        else:
            out[key] = value
    return out


def _from_per_expert(layer):
    groups, out = {}, {}
    for key, value in layer.items():
        parts = key.split(".")
        if key.startswith("experts.") and parts[1].isdigit():
            groups.setdefault(parts[2], {})[int(parts[1])] = value
        else:
            out[key] = value
    for name, members in groups.items():
        out[f"experts.{name}"] = torch.stack([members[i] for i in sorted(members)])
    return out


def setUpModule():
    register_layout(LAYOUT, to_file=_to_per_expert, from_file=_from_per_expert)
    register_moe_preset(PRESET, routed_scaling_factor=1.5, layout=LAYOUT)


def tearDownModule():
    MOE_PRESETS.unregister(PRESET)
    LAYOUTS.unregister(LAYOUT)


def _moe_json(**overrides):
    payload = dict(
        hidden_size=H,
        moe_preset=PRESET,
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dflash_config={},
    )
    payload.update(overrides)
    return payload


def _layer(seed=0, **overrides):
    torch.manual_seed(seed)
    layer = MoELayer(resolve_moe_config(_moe_json(**overrides)), H)
    layer.reset_parameters(std=0.2)
    return layer


# -- config -----------------------------------------------------------------------


class TestConfig(unittest.TestCase):
    def test_dense_configs_resolve_to_none(self):
        self.assertIsNone(resolve_moe_config({"hidden_size": H}))
        self.assertIsNone(resolve_moe_config({"n_routed_experts": 0}))

    def test_preset_defaults_and_explicit_overrides(self):
        self.assertEqual(resolve_moe_config(_moe_json()).routed_scaling_factor, 1.5)
        cfg = resolve_moe_config(_moe_json(routed_scaling_factor=2.0))
        self.assertEqual(cfg.routed_scaling_factor, 2.0)
        self.assertEqual(cfg.layout, LAYOUT)
        self.assertEqual(cfg.shared_expert_intermediate_size, 8)

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
        config = Qwen3Config(hidden_size=H, intermediate_size=32)
        config.n_routed_experts = 8
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            build_ffn(config, dense=lambda c: nn.Identity())
        dense = Qwen3Config(hidden_size=H, intermediate_size=32)
        self.assertIsInstance(
            build_ffn(dense, dense=lambda c: nn.Identity()), nn.Identity
        )

    def test_per_run_sizes_are_required(self):
        payload = _moe_json()
        del payload["moe_intermediate_size"]
        with self.assertRaisesRegex(ValueError, "moe_intermediate_size"):
            resolve_moe_config(payload)

    def test_layout_override_and_unknown_layout(self):
        self.assertEqual(
            resolve_moe_config(_moe_json(moe_layout="native")).layout, "native"
        )
        with self.assertRaisesRegex(KeyError, "checkpoint layout"):
            resolve_moe_config(_moe_json(moe_layout="_nope"))

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


# -- router -----------------------------------------------------------------------


class _FixedBias(BalanceController):
    """A selection bias the test sets directly."""

    def __init__(self, cfg, n_experts):
        super().__init__(cfg, n_experts)
        self.bias = torch.zeros(n_experts)

    def adjust_selection_scores(self, scores):
        return scores + self.bias


class TestRouter(unittest.TestCase):
    def test_score_functions(self):
        logits = torch.randn(4, 6)
        cfg = resolve_moe_config(_moe_json(scoring_func="softmax"))
        router = TopKRouter(cfg, H, BalanceController(cfg, 8))
        self.assertTrue(torch.allclose(router.score_fn(logits), logits.softmax(-1)))

    def test_selection_bias_changes_choices_not_weights(self):
        cfg = resolve_moe_config(_moe_json())
        torch.manual_seed(0)
        router = TopKRouter(cfg, H, _FixedBias(cfg, 8))
        router.reset_parameters(0.2)
        x = torch.randn(10, H)
        before = router(x)
        router.balance.bias.fill_(-100.0)
        router.balance.bias[[1, 5]] = 100.0
        after = router(x)
        self.assertTrue((after.indices.sort(-1).values == torch.tensor([1, 5])).all())
        scores = (x @ router.weight.t()).softmax(-1)
        expected = scores[:, [1, 5]]
        expected = 1.5 * expected / expected.sum(-1, keepdim=True)
        self.assertTrue(
            torch.allclose(
                after.weights.sort(-1).values, expected.sort(-1).values, atol=1e-6
            )
        )
        self.assertTrue(torch.allclose(after.weights.sum(-1), torch.full((10,), 1.5)))
        self.assertEqual(int(after.counts.sum()), 20)
        self.assertFalse(torch.equal(before.indices, after.indices))


# -- experts, shared expert, layer ---------------------------------------------------


class _CountingBalance(BalanceController):
    """Records the TrainingState calls the layer and the model make."""

    def __init__(self, cfg, n_experts):
        super().__init__(cfg, n_experts)
        self.observed = []
        self.applied = 0

    def observe(self, routing):
        self.observed.append(routing.counts.detach().clone())

    def apply_pending_update(self):
        self.applied += 1


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
    def test_grouped_experts_match_the_per_token_reference_and_return_fp32(self):
        layer = _layer()
        x = torch.randn(9, H) * 4
        routing = layer.gate(x)
        y = layer.experts(x, routing)
        self.assertEqual(y.dtype, torch.float32)
        e = layer.experts
        expected = _reference_routed(x, routing, e.w1, e.w2, e.w3)
        self.assertTrue(torch.allclose(y, expected, atol=1e-5))

    @unittest.skipUnless(torch.cuda.is_available(), "grouped_mm needs CUDA")
    def test_grouped_mm_matches_the_loop(self):
        loop = _layer().cuda().to(torch.bfloat16)
        grouped = _layer(dflash_config={"moe_dispatch": "grouped_mm"}).cuda()
        grouped.load_state_dict(loop.state_dict())
        grouped.to(torch.bfloat16)
        x = torch.randn(64, H, device="cuda", dtype=torch.bfloat16)
        a, b = loop(x).float(), grouped(x).float()
        self.assertLess(float((a - b).norm() / a.norm()), 1e-2)

    def test_layer_accumulates_routed_and_shared_in_fp32_and_casts_once(self):
        layer = _layer().to(torch.bfloat16)
        x = torch.randn(2, 5, H, dtype=torch.bfloat16)
        y = layer.eval()(x)
        flat = x.reshape(-1, H)
        routing = layer.gate(flat)
        expected = layer.experts(flat, routing) + layer.shared_experts(flat).float()
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(y, expected.to(torch.bfloat16).view_as(x)))

    def test_layer_without_shared_expert(self):
        layer = _layer(n_shared_experts=0)
        self.assertIsNone(layer.shared_experts)
        self.assertEqual(len(layer.components()), 3)

    def test_layer_builds_the_registered_implementations(self):
        layer = _layer()
        self.assertIsInstance(layer.gate, TopKRouter)
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
        layer = _layer(experts_backend="_test_experts", shared_expert="_test_shared")
        self.assertIs(type(layer.experts), _Experts)
        self.assertIs(type(layer.shared_experts), _Shared)
        with self.assertRaisesRegex(ValueError, "unknown MoE dispatch"):
            _layer(dflash_config={"moe_dispatch": "x"})

    def test_build_ffn_gives_the_layer_with_the_contract_attribute_names(self):
        config = Qwen3Config(intermediate_size=32, **_moe_json())
        layer = build_ffn(config, dense=lambda c: nn.Identity())
        self.assertIsInstance(layer, MoELayer)
        names = {name for name, _ in layer.named_children()}
        self.assertEqual(names, {"gate", "experts", "shared_experts"})
        self.assertIs(layer.balance, layer.gate.balance)

    def test_training_observes_counts_and_eval_does_not(self):
        register_balance_controller("_test_counting")(_CountingBalance)
        self.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_counting")
        layer = _layer(balance="_test_counting").train()
        x = torch.randn(5, H)
        layer(x)
        self.assertEqual(len(layer.balance.observed), 1)
        self.assertEqual(int(layer.balance.observed[0].sum()), 5 * 2)
        self.assertTrue(torch.equal(layer.last_counts, layer.balance.observed[0]))
        layer.eval()(x)
        self.assertEqual(len(layer.balance.observed), 1)

    def test_components_run_through_one_training_state_loop(self):
        register_balance_controller("_test_grad_balance")(_GradBalance)
        self.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_grad_balance")
        layer = _layer(
            balance="_test_grad_balance", dflash_config={"moe_aux_loss_coeff": 0.1}
        ).train()
        layer(torch.randn(2, 4, H))
        self.assertTrue(torch.allclose(layer.aux_loss(), layer.balance.aux_loss()))
        metrics = layer.metrics()
        for key in ("load_max_ratio", "experts_unused_frac", "grad_balance"):
            self.assertIn(key, metrics)


class TestHooks(unittest.TestCase):
    def test_hooks_walk_any_module_tree(self):
        register_balance_controller("_test_grad_balance")(_GradBalance)
        self.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_grad_balance")
        balanced = dict(
            balance="_test_grad_balance", dflash_config={"moe_aux_loss_coeff": 0.1}
        )
        tree = nn.Sequential(
            _layer(0, **balanced), nn.Linear(H, H), _layer(1, **balanced)
        ).train()
        tree(torch.randn(3, H))
        self.assertEqual(len(list(iter_moe_layers(tree))), 2)
        aux = collect_moe_aux_loss(tree)
        self.assertTrue(torch.allclose(aux, tree[0].aux_loss() + tree[2].aux_loss()))
        metrics = collect_moe_metrics(tree)
        self.assertIn("moe/load_max_ratio", metrics)
        self.assertIn("moe/experts_unused_frac", metrics)
        apply_pending_balance_updates(tree)
        dense = nn.Linear(H, H)
        self.assertIsNone(collect_moe_aux_loss(dense))
        self.assertEqual(collect_moe_metrics(dense), {})

    def test_pending_updates_reach_every_layer(self):
        register_balance_controller("_test_counting")(_CountingBalance)
        self.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_counting")
        tree = nn.Sequential(
            _layer(0, balance="_test_counting"), _layer(1, balance="_test_counting")
        )
        apply_pending_balance_updates(tree)
        self.assertEqual([layer.balance.applied for layer in tree], [1, 1])


# -- checkpoint boundary --------------------------------------------------------------


class TestCheckpointBoundary(unittest.TestCase):
    def _tree(self):
        # "layers.1." is a prefix of "layers.11.": the split must keep them apart.
        tree = nn.ModuleDict(
            {"1": _layer(1), "11": _layer(11), "dense": nn.Linear(H, H)}
        )
        return nn.ModuleDict({"layers": tree})

    def test_round_trip_per_layer_with_prefix_disambiguation(self):
        model = self._tree()
        state = model.state_dict()
        files = to_checkpoint_state_dict(state, model)
        self.assertIn("layers.11.experts.7.w2.weight", files)
        self.assertIn("layers.dense.weight", files)
        self.assertNotIn("layers.1.experts.w1", files)
        back = from_checkpoint_state_dict(files, model)
        self.assertEqual(sorted(back), sorted(state))
        for key in state:
            self.assertTrue(torch.equal(back[key], state[key]), key)

    def test_owner_may_be_the_model_or_its_config(self):
        model = self._tree()
        self.assertIs(layout_for(model), LAYOUTS.get(LAYOUT))
        self.assertIs(layout_for(_moe_json()), LAYOUTS.get(LAYOUT))
        self.assertIsNone(layout_for(nn.Linear(H, H)))
        self.assertIsNone(layout_for({"hidden_size": H}))
        state = {"a": torch.ones(1)}
        self.assertEqual(to_checkpoint_state_dict(state, nn.Linear(H, H)), state)

    def test_native_layout_is_the_identity(self):
        layer = _layer(moe_layout="native")
        state = layer.state_dict()
        self.assertEqual(sorted(to_checkpoint_state_dict(state, layer)), sorted(state))


# -- trainer seam ---------------------------------------------------------------------


class _GradBalance(BalanceController):
    """Auxiliary loss built from the router scores, so it has a gradient."""

    def observe(self, routing):
        self._aux = self.cfg.aux_loss_coeff * routing.scores.pow(2).mean()

    def aux_loss(self):
        return self._aux if self.cfg.aux_loss_coeff > 0 else None

    def metrics(self):
        return {"grad_balance": 1.0}


# -- draft model, loading, export ---------------------------------------------------


def _dflash_config(moe=True, **overrides):
    fields = dict(
        architectures=["DFlashDraftModel"],
        block_size=2,
        hidden_size=H,
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
        fields.update(_moe_json())
        fields["dflash_config"] = {"attention_mode": "gqa"}
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


def _forward(model, steps=1):
    model.train()
    for _ in range(steps):
        out = model(
            position_ids=torch.arange(6).unsqueeze(0),
            noise_embedding=torch.randn(1, 2, H),
            target_hidden=torch.randn(1, 4, model.fc.in_features),
        )
    return out


class TestDraftModelSeams(unittest.TestCase):
    def test_dense_draft_is_unaffected(self):
        model = DFlashDraftModel(_dflash_config(moe=False))
        self.assertEqual(list(iter_moe_layers(model)), [])

    def test_moe_draft_initializes_and_trains(self):
        torch.manual_seed(0)
        model = DFlashDraftModel(_dflash_config())
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        gate = layers[0].gate.weight
        self.assertTrue(torch.isfinite(gate).all() and gate.std() > 0)
        self.assertTrue(torch.isfinite(layers[0].experts.w1).all())
        out = _forward(model, steps=2)
        out.float().pow(2).mean().backward()
        self.assertGreater(float(layers[0].experts.w1.grad.abs().sum()), 0.0)

    def test_training_forward_applies_pending_updates_first(self):
        register_balance_controller("_test_counting")(_CountingBalance)
        self.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_counting")
        model = DFlashDraftModel(_dflash_config(balance="_test_counting"))
        layers = list(iter_moe_layers(model))
        _forward(model, steps=2)
        self.assertEqual([layer.balance.applied for layer in layers], [2, 2])
        self.assertEqual([len(layer.balance.observed) for layer in layers], [2, 2])
        model.eval()
        model(
            position_ids=torch.arange(6).unsqueeze(0),
            noise_embedding=torch.randn(1, 2, H),
            target_hidden=torch.randn(1, 4, model.fc.in_features),
        )
        self.assertEqual([layer.balance.applied for layer in layers], [2, 2])

    def test_pretrained_round_trip_through_the_layout(self):
        from specforge.modeling.auto import AutoDraftModel

        torch.manual_seed(0)
        config = _dflash_config()
        model = DFlashDraftModel(config)
        with tempfile.TemporaryDirectory() as tmp:
            model.save_pretrained(
                tmp, state_dict=to_checkpoint_state_dict(model.state_dict(), model)
            )
            from safetensors import safe_open

            files = [f for f in os.listdir(tmp) if f.endswith(".safetensors")]
            with safe_open(os.path.join(tmp, files[0]), "pt") as handle:
                keys = set(handle.keys())
            self.assertIn("layers.0.mlp.experts.3.w1.weight", keys)
            reloaded = AutoDraftModel.from_pretrained(tmp, config=config)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(reloaded.state_dict()[key], value), key)


if __name__ == "__main__":
    unittest.main()
