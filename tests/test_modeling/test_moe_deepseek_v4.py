# coding=utf-8
"""DeepSeek-V4 MoE preset: routing, aux-loss-free balancing, grouped experts,
official checkpoint naming, warm start, and DFlash integration."""

import json
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe import (
    MOE_PRESETS,
    MoELayer,
    apply_pending_balance_updates,
    apply_warm_start,
    collect_moe_metrics,
    from_checkpoint_state_dict,
    get_score_function,
    iter_moe_layers,
    plan_warm_start,
    resolve_moe_config,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.grouped_experts import (
    GroupedExperts,
    stack_grouped_expert_state_dict,
    swiglu_clamped,
    unstack_grouped_expert_state_dict,
)
from specforge.modeling.draft.moe.noaux_tc import NoAuxTCController
from specforge.modeling.draft.moe.swiglu_shared import SwiGLUSharedExpert
from specforge.modeling.draft.moe.topk_router import TopKRouter, group_limited_mask

REPO_ROOT = Path(__file__).resolve().parents[2]
CUDA = torch.cuda.is_available()


def _json(**overrides):
    payload = dict(
        moe_preset="deepseek_v4",
        n_routed_experts=8,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        dflash_config={"moe_bias_update_rate": 1e-3},
    )
    payload.update(overrides)
    return payload


def _layer(**overrides) -> MoELayer:
    torch.manual_seed(0)
    layer = MoELayer(resolve_moe_config(_json(**overrides)), 32)
    layer.reset_parameters(std=0.05)
    for p in layer.shared_experts.parameters():
        nn.init.normal_(p, std=0.05)
    return layer


def _dflash_config(**overrides):
    fields = dict(
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
        initializer_range=0.02,
        **_json(dflash_config={"attention_mode": "gqa", "moe_bias_update_rate": 0.005}),
    )
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


def _reference_forward(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """Per-token dense reference of the routed + shared FFN."""
    routing = layer.gate(x)
    e = layer.experts
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(routing.topk):
            i = int(routing.indices[t, k])
            h = swiglu_clamped(x[t] @ e.w1[i].t(), x[t] @ e.w3[i].t(), e.swiglu_limit)
            out[t] += routing.weights[t, k] * (h.to(x.dtype) @ e.w2[i].t()).float()
    return (out + layer.shared_experts(x).float()).to(x.dtype)


class TestPresetAndConfig(unittest.TestCase):
    def test_preset_matches_deepseek_v4_recipe(self):
        self.assertIn("deepseek_v4", MOE_PRESETS)
        cfg = resolve_moe_config(_json())
        self.assertEqual(cfg.scoring_func, "sqrtsoftplus")
        self.assertEqual(cfg.balance, "noaux_tc")
        self.assertEqual(cfg.routed_scaling_factor, 1.5)
        self.assertTrue(cfg.norm_topk_prob)
        self.assertEqual(cfg.swiglu_limit, 10.0)
        self.assertEqual(cfg.n_shared_experts, 1)
        self.assertFalse(cfg.group_limited)

    def test_checked_in_draft_config_resolves(self):
        payload = json.loads(
            (REPO_ROOT / "configs" / "deepseek-v4-flash-dspark-moe.json").read_text()
        )
        cfg = resolve_moe_config(payload)
        self.assertEqual(
            (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size),
            (64, 6, 2048),
        )
        self.assertEqual(cfg.dispatch, "grouped_mm")
        self.assertEqual(cfg.bias_update_rate, 1e-3)
        dense = json.loads(
            (REPO_ROOT / "configs" / "deepseek-v4-flash-dspark.json").read_text()
        )
        moe_only = {
            "moe_preset",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "n_shared_experts",
        }
        self.assertEqual(set(payload) - set(dense), moe_only)
        for key in dense:
            if key != "dflash_config":
                self.assertEqual(payload[key], dense[key], key)

    def test_score_functions(self):
        logits = torch.tensor([[0.0, 2.0, -3.0]])
        self.assertTrue(
            torch.allclose(
                get_score_function("sqrtsoftplus")(logits),
                torch.nn.functional.softplus(logits).sqrt(),
            )
        )
        self.assertAlmostEqual(
            float(get_score_function("softmax")(logits).sum()), 1.0, places=5
        )
        self.assertTrue(
            torch.allclose(get_score_function("sigmoid")(logits), logits.sigmoid())
        )


class TestTopKRouter(unittest.TestCase):
    def test_weights_are_renormalized_and_scaled(self):
        layer = _layer()
        routing = layer.gate(torch.randn(5, 32))
        self.assertIsInstance(layer.gate, TopKRouter)
        self.assertTrue(torch.allclose(routing.weights.sum(-1), torch.full((5,), 1.5)))
        self.assertEqual(int(routing.counts.sum()), 10)
        for row in routing.indices.tolist():
            self.assertEqual(len(set(row)), 2)

    def test_group_limited_routing_stays_within_selected_groups(self):
        selection = torch.randn(64, 16)
        masked = group_limited_mask(selection, n_group=4, topk_group=1)
        finite = torch.isfinite(masked).view(64, 4, 4)
        self.assertTrue((finite.all(-1).sum(-1) == 1).all())
        layer = _layer(
            n_routed_experts=16, n_group=4, topk_group=1, num_experts_per_tok=3
        )
        routing = layer.gate(torch.randn(10, 32))
        groups = routing.indices // 4
        self.assertTrue((groups == groups[:, :1]).all())


class TestNoAuxTC(unittest.TestCase):
    def test_deferred_bias_update_semantics(self):
        layer = _layer().train()
        ctrl = layer.balance
        self.assertIsInstance(ctrl, NoAuxTCController)
        layer(torch.randn(6, 32))
        self.assertIsNotNone(ctrl._pending_counts)
        before = ctrl.bias.clone()
        layer.apply_pending_balance_update()
        self.assertIsNone(ctrl._pending_counts)
        self.assertFalse(torch.equal(before, ctrl.bias))
        self.assertTrue(((ctrl.bias - before).abs() <= 1e-3 + 1e-7).all())
        after = ctrl.bias.clone()
        layer.apply_pending_balance_update()  # nothing pending: no-op
        self.assertTrue(torch.equal(after, ctrl.bias))
        layer.eval()
        layer(torch.randn(6, 32))
        self.assertIsNone(ctrl._pending_counts)

    def test_bias_stays_fp32_and_moves_selection_only(self):
        layer = _layer().to(torch.bfloat16).eval()
        self.assertEqual(layer.balance.bias.dtype, torch.float32)
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        self.assertEqual(layer(x).dtype, torch.bfloat16)
        layer.balance.bias[:] = -100.0
        layer.balance.bias[3] = 0.0
        routing = layer.gate(x)
        self.assertTrue((routing.indices == 3).any(dim=-1).all())
        self.assertTrue(torch.allclose(routing.weights.sum(-1), torch.full((4,), 1.5)))

    def test_aux_balance_loss_is_differentiable_and_uniform_at_balance(self):
        layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.5}).train()
        x = torch.randn(64, 32)
        layer(x)
        aux = layer.aux_loss()
        self.assertIsNotNone(aux)
        self.assertTrue(aux.requires_grad)
        aux.backward()
        self.assertIsNotNone(layer.gate.weight.grad)
        self.assertGreater(float(layer.gate.weight.grad.abs().sum()), 0.0)
        # perfectly uniform routing and affinities give exactly coeff * 1
        routing = layer.gate(x)
        n_experts = layer.cfg.n_routed_experts
        uniform_scores = torch.full((64, n_experts), 0.25, requires_grad=True)
        counts = torch.full((n_experts,), 64 * routing.topk // n_experts)
        layer.balance.observe(
            type(routing)(routing.weights, routing.indices, counts, uniform_scores)
        )
        self.assertAlmostEqual(float(layer.balance.aux_loss()), 0.5, places=5)
        self.assertIn("aux_loss", layer.balance.metrics())
        # disabled by default, and never built without a gradient signal
        layer = _layer().train()
        layer(torch.randn(8, 32))
        self.assertIsNone(layer.aux_loss())
        with torch.no_grad():
            aux_layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.5}).train()
            aux_layer(torch.randn(8, 32))
        self.assertIsNone(aux_layer.aux_loss())

    def test_metrics_include_bias_and_global_load(self):
        layer = _layer().train()
        layer(torch.randn(6, 32))
        layer.apply_pending_balance_update()
        metrics = collect_moe_metrics(nn.Sequential(layer))
        for key in (
            "moe/load_max_ratio",
            "moe/bias_abs_max",
            "moe/global_load_max_ratio",
        ):
            self.assertIn(key, metrics)


class TestGroupedExperts(unittest.TestCase):
    def test_layout_init_and_dense_reference(self):
        layer = _layer().eval()
        e = layer.experts
        self.assertIsInstance(e, GroupedExperts)
        self.assertIsInstance(layer.shared_experts, SwiGLUSharedExpert)
        self.assertEqual(tuple(e.w1.shape), (8, 16, 32))
        self.assertEqual(tuple(e.w2.shape), (8, 32, 16))
        self.assertAlmostEqual(float(e.w1.std()), 0.05, delta=0.01)
        x = torch.randn(7, 32)
        self.assertTrue(
            torch.allclose(layer(x), _reference_forward(layer, x), atol=1e-5)
        )

    def test_swiglu_clamp(self):
        gate = torch.tensor([50.0, -50.0])
        up = torch.tensor([50.0, -50.0])
        clamped = swiglu_clamped(gate, up, 10.0)
        # gate clamps to max 10, up to [-10, 10]: silu(10)*10 and silu(-50)*-10
        expected = torch.nn.functional.silu(torch.tensor([10.0, -50.0])) * torch.tensor(
            [10.0, -10.0]
        )
        self.assertTrue(torch.allclose(clamped, expected, atol=1e-6))
        self.assertGreater(float(swiglu_clamped(gate, up, 0.0)[0]), 100.0)

    def test_unknown_dispatch_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "dispatch"):
            _layer(dflash_config={"moe_dispatch": "magic"})

    @unittest.skipUnless(
        CUDA and hasattr(torch, "_grouped_mm"), "needs CUDA grouped GEMM"
    )
    def test_grouped_mm_matches_sorted_loop(self):
        torch.manual_seed(3)
        layer = _layer(n_routed_experts=16).to("cuda", torch.bfloat16).train()
        x = (torch.randn(6, 32, device="cuda") * 0.5).to(torch.bfloat16)
        results = {}
        for grouped in (False, True):
            layer.experts.grouped_mm = grouped
            layer.zero_grad(set_to_none=True)
            xg = x.clone().requires_grad_(True)
            y = layer(xg)
            y.float().square().sum().backward()
            results[grouped] = (
                y.detach().clone(),
                xg.grad.clone(),
                {
                    n: p.grad.clone()
                    for n, p in layer.named_parameters()
                    if p.grad is not None
                },
            )
        (y0, dx0, g0), (y1, dx1, g1) = results[False], results[True]
        self.assertTrue(torch.allclose(y0.float(), y1.float(), rtol=2e-2, atol=2e-2))
        self.assertTrue(torch.allclose(dx0.float(), dx1.float(), rtol=2e-2, atol=2e-2))
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


class TestCheckpointNaming(unittest.TestCase):
    def test_layer_roundtrip_through_official_naming(self):
        layer = _layer()
        native = layer.state_dict()
        self.assertIn("experts.w1", native)
        self.assertIn("gate.balance.bias", native)
        official = to_checkpoint_state_dict(native)
        self.assertIn("experts.0.w1.weight", official)
        self.assertIn("gate.bias", official)
        self.assertIn("shared_experts.w1.weight", official)
        self.assertFalse(
            any("balance" in k or k.endswith("experts.w1") for k in official)
        )
        fresh = _layer(dflash_config={"moe_bias_update_rate": 0.0})
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        self.assertTrue(torch.equal(fresh.experts.w2, layer.experts.w2))
        # both directions are idempotent
        self.assertEqual(set(to_checkpoint_state_dict(official)), set(official))
        self.assertEqual(set(from_checkpoint_state_dict(native)), set(native))

    def test_dense_gate_bias_is_left_alone(self):
        state = {
            "head.gate.bias": torch.zeros(1),
            "head.gate.weight": torch.zeros(1, 1),
        }
        self.assertEqual(set(from_checkpoint_state_dict(state)), set(state))

    def test_stack_rejects_missing_expert_indices(self):
        official = unstack_grouped_expert_state_dict(_layer().state_dict())
        del official["experts.3.w2.weight"]
        with self.assertRaises(KeyError):
            stack_grouped_expert_state_dict(official)


class TestWarmStart(unittest.TestCase):
    def test_apply_plan_copies_selected_experts_gate_rows_and_shared(self):
        layer = _layer()
        n_target = 16
        source = {
            "gate.weight": torch.randn(n_target, 32),
            "gate.bias": torch.randn(n_target),
        }
        for j in range(n_target):
            source[f"experts.{j}.w1.weight"] = torch.randn(16, 32)
            source[f"experts.{j}.w2.weight"] = torch.randn(32, 16)
            source[f"experts.{j}.w3.weight"] = torch.randn(16, 32)
        for w, shape in (("w1", (16, 32)), ("w2", (32, 16)), ("w3", (16, 32))):
            source[f"shared_experts.{w}.weight"] = torch.randn(*shape)
        plan = plan_warm_start(layer.cfg, n_target_experts=n_target)
        self.assertEqual(plan.target_expert_ids, (0, 2, 4, 6, 8, 10, 12, 14))
        loaded = apply_warm_start(layer, plan, source)
        self.assertIn("experts.w1", loaded)
        for i, j in enumerate(plan.target_expert_ids):
            self.assertTrue(
                torch.equal(layer.experts.w1[i], source[f"experts.{j}.w1.weight"])
            )
            self.assertTrue(torch.equal(layer.gate.weight[i], source["gate.weight"][j]))
            self.assertEqual(
                float(layer.balance.bias[i]), float(source["gate.bias"][j])
            )
        self.assertTrue(
            torch.equal(
                layer.shared_experts.w2.weight, source["shared_experts.w2.weight"]
            )
        )
        with self.assertRaises(ValueError):
            apply_warm_start(_layer(n_routed_experts=4), plan, source)


class TestServingExport(unittest.TestCase):
    def test_serving_fields_carry_the_resolved_recipe(self):
        fields = resolve_moe_config(_json()).serving_fields()
        self.assertEqual(fields["topk_method"], "noaux_tc")
        self.assertEqual(fields["scoring_func"], "sqrtsoftplus")
        self.assertEqual(fields["routed_scaling_factor"], 1.5)
        self.assertEqual(fields["swiglu_limit"], 10.0)
        self.assertEqual((fields["n_group"], fields["topk_group"]), (1, 1))
        # a disabled clamp is omitted rather than exported as a clamp at 0
        self.assertNotIn(
            "swiglu_limit", resolve_moe_config(_json(swiglu_limit=0)).serving_fields()
        )

    def test_hf_export_reloads_and_carries_serving_config(self):
        import os
        import tempfile

        from specforge.export import export_to_hf
        from specforge.modeling.auto import AutoDraftModel

        torch.manual_seed(1)
        config = _dflash_config()
        model = DFlashDraftModel(config).to(torch.bfloat16)
        for layer in iter_moe_layers(model):
            layer.balance.bias.uniform_(-1.0, 1.0)
        workdir = tempfile.mkdtemp(prefix="moe_export_")
        config_path = os.path.join(workdir, "draft.json")
        config.save_pretrained(workdir)
        os.replace(os.path.join(workdir, "config.json"), config_path)
        ckpt_dir = os.path.join(workdir, "run-step1")
        os.makedirs(ckpt_dir)
        torch.save(
            {
                "draft_state_dict": to_checkpoint_state_dict(model.state_dict()),
                "strategy": "dflash",
                "global_step": 1,
            },
            os.path.join(ckpt_dir, "training_state.pt"),
        )
        out = export_to_hf(ckpt_dir, config_path, os.path.join(workdir, "hf"))
        exported = json.loads((Path(out) / "config.json").read_text())
        self.assertEqual(exported["topk_method"], "noaux_tc")
        self.assertEqual(exported["scoring_func"], "sqrtsoftplus")
        self.assertEqual(exported["n_routed_experts"], 8)
        from safetensors import safe_open

        with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:
            keys = set(f.keys())
        self.assertIn("layers.0.mlp.experts.0.w1.weight", keys)
        self.assertIn("layers.0.mlp.gate.bias", keys)
        # HF from_pretrained assigns tensors by key; SpecForge's loader must
        # convert the official naming back into the stacked module layout.
        reloaded = AutoDraftModel.from_pretrained(out, torch_dtype=torch.bfloat16)
        fresh = reloaded.state_dict()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.float(), fresh[key].float()), key)


class TestDeepseekV4TargetDequant(unittest.TestCase):
    def test_fp4_and_fp8_dequant_conventions(self):
        from specforge.modeling.draft.moe.deepseek_v4_target import (
            FP4_TABLE,
            dequant_fp4_packed,
            dequant_fp8_block,
            dequantize_ffn_tensors,
        )

        # one row of 32 fp4 values: nibbles 0..15 twice; low nibble = even index
        codes = torch.arange(16, dtype=torch.uint8)
        packed = (
            (codes | (codes << 4)).repeat(2).view(1, 32).to(torch.int8)
        )  # 64 values
        scale = torch.tensor([[2.0, 0.5]], dtype=torch.float32)  # two groups of 32
        out = dequant_fp4_packed(packed, scale)
        self.assertEqual(tuple(out.shape), (1, 64))
        expected = FP4_TABLE[codes.long()].repeat_interleave(2).repeat(2)
        expected[:32] *= 2.0
        expected[32:] *= 0.5
        self.assertTrue(torch.equal(out.float()[0], expected))
        w8 = torch.full((128, 256), 1.0).to(torch.float8_e4m3fn)
        s8 = torch.tensor([[1.0, 4.0]])
        d8 = dequant_fp8_block(w8, s8).float()
        self.assertTrue((d8[:, :128] == 1.0).all() and (d8[:, 128:] == 4.0).all())
        raw = {
            "layers.3.ffn.gate.weight": torch.randn(4, 8),
            "layers.3.ffn.gate.bias": torch.randn(4),
            "layers.3.ffn.experts.0.w1.weight": packed.repeat(2, 1),
            "layers.3.ffn.experts.0.w1.scale": scale.repeat(2, 1),
            "layers.3.ffn.shared_experts.w1.weight": w8,
            "layers.3.ffn.shared_experts.w1.scale": s8,
        }
        rel = dequantize_ffn_tensors(raw, "layers.3.ffn.")
        self.assertEqual(
            set(rel),
            {
                "gate.weight",
                "gate.bias",
                "experts.0.w1.weight",
                "shared_experts.w1.weight",
            },
        )
        self.assertEqual(rel["gate.bias"].dtype, torch.float32)
        self.assertEqual(rel["experts.0.w1.weight"].dtype, torch.bfloat16)
        with self.assertRaisesRegex(ValueError, "hash-routed"):
            dequantize_ffn_tensors(
                {
                    "layers.1.ffn.gate.tid2eid": torch.zeros(4),
                    "layers.1.ffn.gate.weight": torch.zeros(4, 8),
                },
                "layers.1.ffn.",
            )


class TestDFlashIntegration(unittest.TestCase):
    def _forward(self, model):
        return model(
            position_ids=torch.arange(6).unsqueeze(0),
            noise_embedding=torch.randn(1, 2, 32),
            target_hidden=torch.randn(1, 4, 2 * 32),
        )

    def test_layers_train_and_balance_through_the_model(self):
        model = DFlashDraftModel(_dflash_config())
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertIsInstance(layer.experts, GroupedExperts)
            self.assertAlmostEqual(float(layer.experts.w1.std()), 0.02, delta=0.005)
            self.assertAlmostEqual(float(layer.gate.weight.std()), 0.02, delta=0.005)
            self.assertTrue(torch.equal(layer.balance.bias, torch.zeros(8)))
        model.train()
        out = self._forward(model)
        out.float().square().mean().backward()
        for layer in layers:
            self.assertIsNotNone(layer.experts.w2.grad)
            self.assertIsNotNone(layer.gate.weight.grad)
            self.assertTrue(torch.equal(layer.balance.bias, torch.zeros(8)))  # deferred
        self._forward(model)  # applies the pending update before routing
        self.assertTrue(any(layer.balance.bias.abs().sum() > 0 for layer in layers))

    def test_model_checkpoint_uses_official_naming_and_reloads(self):
        model = DFlashDraftModel(_dflash_config())
        official = to_checkpoint_state_dict(model.state_dict())
        self.assertIn("layers.0.mlp.experts.0.w1.weight", official)
        self.assertIn("layers.0.mlp.gate.bias", official)
        self.assertIn("layers.1.mlp.shared_experts.w3.weight", official)
        self.assertFalse(
            any(".balance." in k or k.endswith(".experts.w1") for k in official)
        )
        fresh = DFlashDraftModel(_dflash_config())
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        self.assertTrue(
            torch.equal(fresh.layers[1].mlp.experts.w1, model.layers[1].mlp.experts.w1)
        )

    def test_dense_config_is_unaffected(self):
        config = _dflash_config()
        for key in (
            "moe_preset",
            "n_routed_experts",
            "num_experts_per_tok",
            "moe_intermediate_size",
            "n_shared_experts",
        ):
            if hasattr(config, key):
                delattr(config, key)
        model = DFlashDraftModel(config)
        self.assertEqual(list(iter_moe_layers(model)), [])
        apply_pending_balance_updates(model)


if __name__ == "__main__":
    unittest.main(verbosity=2)
