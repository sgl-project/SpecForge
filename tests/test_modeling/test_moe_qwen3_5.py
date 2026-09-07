# coding=utf-8
"""Qwen3.5/3.8 MoE preset: softmax top-k routing, gated SwiGLU shared expert,
auxiliary load-balancing loss (== transformers), Qwen checkpoint naming
(SGLang ``Qwen2MoeSparseMoeBlock``), warm start, and DSpark integration."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe import (
    MOE_PRESETS,
    MoELayer,
    apply_warm_start,
    collect_moe_aux_loss,
    collect_moe_metrics,
    from_checkpoint_state_dict,
    iter_moe_layers,
    plan_warm_start,
    resolve_moe_config,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.aux_loss import AuxLossController, load_balancing_loss
from specforge.modeling.draft.moe.grouped_experts import GroupedExperts, swiglu_clamped
from specforge.modeling.draft.moe.qwen_layout import from_qwen_layout, to_qwen_layout
from specforge.modeling.draft.moe.swiglu_shared import SwiGLUSharedExpert
from specforge.modeling.draft.moe.topk_router import (
    TopKRouter,
    fold_router_centering,
    routing_diagnostics,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
HIDDEN = 32


def _json(**overrides):
    payload = dict(
        moe_preset="qwen3_5_moe",
        n_routed_experts=8,
        num_experts_per_tok=3,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        dflash_config={"moe_aux_loss_coeff": 0.01},
    )
    payload.update(overrides)
    return payload


def _layer(**overrides) -> MoELayer:
    torch.manual_seed(0)
    layer = MoELayer(resolve_moe_config(_json(**overrides)), HIDDEN)
    layer.reset_parameters(std=0.05)
    if layer.shared_experts is not None:
        for p in layer.shared_experts.parameters():
            nn.init.normal_(p, std=0.05)
    return layer


def _draft_config(architecture="DFlashDraftModel", **overrides):
    dflash_config = {"attention_mode": "gqa", "moe_aux_loss_coeff": 0.01}
    if architecture == "DSparkDraftModel":
        dflash_config.update(
            {
                "projector_type": "dspark",
                "markov_rank": 4,
                "enable_confidence_head": True,
                "confidence_head_with_markov": True,
                "target_layer_ids": [1, 3],
            }
        )
    fields = dict(
        architectures=[architecture],
        block_size=3,
        hidden_size=HIDDEN,
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
        **_json(dflash_config=dflash_config),
    )
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "sdpa"
    return config


def _reference_forward(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """Per-token dense reference: Qwen3_5MoeSparseMoeBlock semantics."""
    logits = x.float() @ layer.gate.weight.float().t()
    probs = logits.softmax(-1)
    top, idx = probs.topk(layer.cfg.num_experts_per_tok, dim=-1)
    top = top / top.sum(-1, keepdim=True)
    e = layer.experts
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(idx.shape[1]):
            i = int(idx[t, k])
            h = torch.nn.functional.silu(x[t] @ e.w1[i].t()) * (x[t] @ e.w3[i].t())
            out[t] += top[t, k] * (h @ e.w2[i].t())
    s = layer.shared_experts
    shared = s.w2(torch.nn.functional.silu(s.w1(x)) * s.w3(x))
    shared = torch.sigmoid(s.gate(x)) * shared
    return (out + shared.float()).to(x.dtype)


class TestPresetAndConfig(unittest.TestCase):
    def test_preset_matches_qwen3_5_moe_recipe(self):
        self.assertIn("qwen3_5_moe", MOE_PRESETS)
        cfg = resolve_moe_config(_json())
        self.assertEqual(cfg.scoring_func, "softmax")
        self.assertTrue(cfg.norm_topk_prob)
        self.assertEqual(cfg.routed_scaling_factor, 1.0)
        self.assertEqual(cfg.n_shared_experts, 1)
        self.assertEqual(cfg.shared_expert, "swiglu")
        self.assertEqual(cfg.shared_expert_gate, "sigmoid")
        self.assertEqual(cfg.router, "topk")
        self.assertEqual(cfg.balance, "aux_loss")
        self.assertEqual(cfg.swiglu_limit, 0.0)
        self.assertEqual(cfg.experts_backend, "grouped")
        self.assertFalse(cfg.group_limited)
        self.assertEqual(cfg.shared_expert_intermediate_size, 24)
        self.assertEqual(cfg.aux_loss_coeff, 0.01)
        # the DeepSeek preset is untouched
        ds = MOE_PRESETS.get("deepseek_v4")
        self.assertEqual(
            (ds["balance"], ds["shared_expert_gate"]), ("noaux_tc", "none")
        )

    def test_checked_in_draft_config_resolves(self):
        payload = json.loads(
            (REPO_ROOT / "configs" / "qwen3.8-27b-dspark-moe.json").read_text()
        )
        cfg = resolve_moe_config(payload)
        self.assertEqual(cfg.preset, "qwen3_5_moe")
        self.assertEqual(
            (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size),
            (512, 10, 512),
        )
        self.assertEqual(cfg.n_shared_experts, 1)
        self.assertEqual(cfg.shared_expert_intermediate_size, 2048)
        self.assertEqual(cfg.aux_loss_coeff, 0.001)
        self.assertEqual(cfg.dispatch, "grouped_mm")
        self.assertEqual(cfg.bias_update_rate, 0.0)
        # the dense Qwen3.8-27B DSpark geometry is preserved
        self.assertEqual(payload["architectures"], ["DSparkDraftModel"])
        self.assertEqual(payload["block_size"], 7)
        self.assertEqual(payload["hidden_size"], 5120)
        self.assertEqual(payload["vocab_size"], 248320)
        self.assertEqual(payload["num_hidden_layers"], 5)
        self.assertEqual(payload["layer_types"], ["full_attention"] * 5)
        self.assertEqual(payload["rope_parameters"]["rope_type"], "yarn")
        self.assertEqual(payload["dflash_config"]["mask_token_id"], 248070)
        self.assertEqual(
            payload["dflash_config"]["target_layer_ids"], [5, 19, 33, 47, 61]
        )

    def test_serving_fields_are_greedy_softmax(self):
        fields = resolve_moe_config(_json()).serving_fields()
        self.assertEqual(fields["topk_method"], "greedy")
        self.assertEqual(fields["scoring_func"], "softmax")
        self.assertTrue(fields["norm_topk_prob"])
        self.assertEqual(fields["routed_scaling_factor"], 1.0)
        self.assertEqual(fields["n_shared_experts"], 1)
        self.assertNotIn("swiglu_limit", fields)

    def test_unknown_shared_expert_gate_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "shared_expert_gate"):
            _layer(shared_expert_gate="tanh")


class TestRouterAndSharedExpert(unittest.TestCase):
    def test_router_is_softmax_topk_renormalized_unscaled(self):
        layer = _layer().eval()
        self.assertIsInstance(layer.gate, TopKRouter)
        x = torch.randn(6, HIDDEN)
        routing = layer.gate(x)
        probs = (x @ layer.gate.weight.t()).softmax(-1)
        top, idx = probs.topk(3, dim=-1)
        self.assertTrue(torch.equal(routing.indices, idx))
        self.assertTrue(
            torch.allclose(routing.weights, top / top.sum(-1, keepdim=True))
        )
        self.assertTrue(torch.allclose(routing.weights.sum(-1), torch.ones(6)))
        self.assertTrue(torch.allclose(routing.scores, probs))

    def test_gated_shared_expert_matches_reference_formula(self):
        layer = _layer()
        shared = layer.shared_experts
        self.assertIsInstance(shared, SwiGLUSharedExpert)
        self.assertTrue(shared.gated)
        self.assertEqual(tuple(shared.gate.weight.shape), (1, HIDDEN))
        self.assertIsNone(shared.gate.bias)
        x = torch.randn(5, HIDDEN)
        expected = torch.sigmoid(shared.gate(x)) * shared.w2(
            torch.nn.functional.silu(shared.w1(x)) * shared.w3(x)
        )
        self.assertTrue(torch.allclose(shared(x), expected, atol=1e-6))
        # the gate is a real per-token scalar in (0, 1)
        nn.init.constant_(shared.gate.weight, 0.0)
        ungated = shared.w2(torch.nn.functional.silu(shared.w1(x)) * shared.w3(x))
        self.assertTrue(torch.allclose(shared(x), 0.5 * ungated, atol=1e-6))

    def test_deepseek_shared_expert_stays_ungated(self):
        layer = MoELayer(
            resolve_moe_config(
                dict(
                    moe_preset="deepseek_v4",
                    n_routed_experts=4,
                    num_experts_per_tok=2,
                    moe_intermediate_size=8,
                )
            ),
            HIDDEN,
        )
        self.assertFalse(layer.shared_experts.gated)
        self.assertNotIn("shared_experts.gate.weight", layer.state_dict())

    def test_layer_forward_matches_dense_reference(self):
        layer = _layer().eval()
        self.assertIsInstance(layer.experts, GroupedExperts)
        self.assertEqual(layer.experts.swiglu_limit, 0.0)
        x = torch.randn(7, HIDDEN)
        self.assertTrue(
            torch.allclose(layer(x), _reference_forward(layer, x), atol=1e-5)
        )
        self.assertEqual(layer(torch.randn(2, 3, HIDDEN)).shape, (2, 3, HIDDEN))

    def test_bf16_forward_keeps_dtype(self):
        layer = _layer().to(torch.bfloat16).eval()
        y = layer(torch.randn(4, HIDDEN, dtype=torch.bfloat16))
        self.assertEqual(y.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(y.float()).all())


class TestAuxLoss(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(1)
        for n_experts, top_k, tokens in ((8, 3, 40), (16, 2, 7), (64, 10, 128)):
            logits = torch.randn(tokens, n_experts)
            probs = logits.softmax(-1)
            idx = probs.topk(top_k, -1).indices
            counts = torch.zeros(n_experts, dtype=torch.long).scatter_add_(
                0, idx.flatten(), torch.ones_like(idx.flatten())
            )
            ours = load_balancing_loss(counts, probs, tokens)
            ref = load_balancing_loss_func((logits,), n_experts, top_k)
            self.assertAlmostEqual(float(ours), float(ref), delta=1e-5)

    def test_controller_emits_the_scaled_transformers_loss(self):
        layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.5}).train()
        self.assertIsInstance(layer.balance, AuxLossController)
        x = torch.randn(50, HIDDEN)
        layer(x)
        aux = layer.aux_loss()
        self.assertIsNotNone(aux)
        self.assertTrue(aux.requires_grad)
        logits = x @ layer.gate.weight.detach().t()
        ref = load_balancing_loss_func((logits,), 8, 3)
        self.assertAlmostEqual(float(aux), 0.5 * float(ref), delta=1e-5)
        aux.backward()
        self.assertIsNotNone(layer.gate.weight.grad)
        self.assertGreater(float(layer.gate.weight.grad.abs().sum()), 0.0)
        # experts get no gradient from the balance term alone
        self.assertIsNone(layer.experts.w1.grad)

    def test_uniform_routing_gives_coeff_times_topk(self):
        # transformers' f_e sums over the k slots, so balance == top_k
        counts = torch.full((8,), 16 * 3 // 8)  # 16 tokens, k=3: 48 slots / 8
        scores = torch.full((16, 8), 1.0 / 8, requires_grad=True)
        self.assertAlmostEqual(float(load_balancing_loss(counts, scores, 16)), 3.0, 5)

    def test_disabled_without_coefficient_or_gradient(self):
        layer = _layer(dflash_config={"moe_aux_loss_coeff": 0.0}).train()
        layer(torch.randn(8, HIDDEN))
        self.assertIsNone(layer.aux_loss())
        layer = _layer().train()
        with torch.no_grad():
            layer(torch.randn(8, HIDDEN))
        self.assertIsNone(layer.aux_loss())
        layer.eval()
        layer(torch.randn(8, HIDDEN))
        self.assertIsNone(layer.aux_loss())

    def test_observe_overwrites_and_update_is_a_noop(self):
        layer = _layer().train()
        layer(torch.randn(8, HIDDEN))
        first = float(layer.aux_loss())
        layer(torch.randn(8, HIDDEN))
        self.assertNotAlmostEqual(first, float(layer.aux_loss()))
        state = {k: v.clone() for k, v in layer.state_dict().items()}
        layer.apply_pending_balance_update()
        for k, v in layer.state_dict().items():
            self.assertTrue(torch.equal(v, state[k]), k)
        self.assertNotIn("bias", dict(layer.balance.named_buffers()))

    def test_metrics_expose_aux_loss_and_load_fractions(self):
        layer = _layer().train()
        layer(torch.randn(16, HIDDEN))
        metrics = collect_moe_metrics(nn.Sequential(layer))
        for key in (
            "moe/load_max_ratio",
            "moe/aux_loss",
            "moe/load_frac_max",
            "moe/load_frac_mean",
        ):
            self.assertIn(key, metrics)
        self.assertAlmostEqual(float(metrics["moe/load_frac_mean"]), 1 / 8, places=6)
        self.assertGreaterEqual(float(metrics["moe/load_frac_max"]), 1 / 8)
        self.assertLessEqual(float(metrics["moe/load_frac_max"]), 1 / 3 + 1e-6)


class TestQwenCheckpointNaming(unittest.TestCase):
    def test_layer_roundtrip_through_qwen_naming(self):
        layer = _layer()
        native = layer.state_dict()
        self.assertIn("experts.w1", native)
        self.assertIn("shared_experts.gate.weight", native)
        official = to_checkpoint_state_dict(native)
        expected = {"gate.weight", "shared_expert_gate.weight"}
        for proj in ("gate_proj", "up_proj", "down_proj"):
            expected.add(f"shared_expert.{proj}.weight")
            expected.update(f"experts.{i}.{proj}.weight" for i in range(8))
        self.assertEqual(set(official), expected)
        self.assertEqual(
            tuple(official["shared_expert_gate.weight"].shape), (1, HIDDEN)
        )
        self.assertEqual(
            tuple(official["experts.2.gate_proj.weight"].shape), (16, HIDDEN)
        )
        self.assertEqual(
            tuple(official["experts.2.down_proj.weight"].shape), (HIDDEN, 16)
        )
        self.assertTrue(
            torch.equal(official["experts.2.gate_proj.weight"], native["experts.w1"][2])
        )
        self.assertTrue(
            torch.equal(official["experts.2.up_proj.weight"], native["experts.w3"][2])
        )
        self.assertTrue(
            torch.equal(official["experts.2.down_proj.weight"], native["experts.w2"][2])
        )
        self.assertTrue(
            torch.equal(
                official["shared_expert.up_proj.weight"],
                native["shared_experts.w3.weight"],
            )
        )
        fresh = _layer()
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        for key, value in layer.state_dict().items():
            self.assertTrue(torch.equal(value, fresh.state_dict()[key]), key)
        # both directions are idempotent
        self.assertEqual(set(to_checkpoint_state_dict(official)), set(official))
        self.assertEqual(set(from_checkpoint_state_dict(native)), set(native))

    def test_deepseek_layout_is_untouched(self):
        ds = {
            "layers.0.mlp.experts.0.w1.weight": torch.zeros(1),
            "layers.0.mlp.shared_experts.w1.weight": torch.zeros(1),
            "layers.0.mlp.gate.bias": torch.zeros(1),
        }
        self.assertEqual(set(to_qwen_layout(dict(ds))), set(ds))
        self.assertEqual(set(from_qwen_layout(dict(ds))), set(ds))
        dense = {"layers.0.mlp.gate_proj.weight": torch.zeros(1)}
        self.assertEqual(set(to_qwen_layout(dict(dense))), set(dense))
        self.assertEqual(set(from_qwen_layout(dict(dense))), set(dense))

    def test_model_checkpoint_uses_qwen_naming_and_reloads(self):
        model = DFlashDraftModel(_draft_config())
        official = to_checkpoint_state_dict(model.state_dict())
        self.assertIn("layers.0.mlp.gate.weight", official)
        self.assertIn("layers.0.mlp.experts.0.gate_proj.weight", official)
        self.assertIn("layers.1.mlp.experts.7.down_proj.weight", official)
        self.assertIn("layers.1.mlp.shared_expert.up_proj.weight", official)
        self.assertIn("layers.0.mlp.shared_expert_gate.weight", official)
        self.assertFalse(
            any(
                ".shared_experts." in k or k.endswith(".experts.w1") or ".w1." in k
                for k in official
            )
        )
        fresh = DFlashDraftModel(_draft_config())
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value, fresh.state_dict()[key]), key)

    def test_hf_export_uses_qwen_naming_and_reloads(self):
        from safetensors import safe_open

        from specforge.export import export_to_hf
        from specforge.modeling.auto import AutoDraftModel

        torch.manual_seed(1)
        config = _draft_config()
        model = DFlashDraftModel(config).to(torch.bfloat16)
        workdir = tempfile.mkdtemp(prefix="moe_qwen_export_")
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
        self.assertEqual(exported["topk_method"], "greedy")
        self.assertEqual(exported["scoring_func"], "softmax")
        self.assertEqual(exported["shared_expert_intermediate_size"], 24)
        with safe_open(os.path.join(out, "model.safetensors"), "pt") as f:
            keys = set(f.keys())
        self.assertIn("layers.0.mlp.experts.0.gate_proj.weight", keys)
        self.assertIn("layers.0.mlp.shared_expert.down_proj.weight", keys)
        self.assertIn("layers.0.mlp.shared_expert_gate.weight", keys)
        reloaded = AutoDraftModel.from_pretrained(out, torch_dtype=torch.bfloat16)
        fresh = reloaded.state_dict()
        for key, value in model.state_dict().items():
            self.assertTrue(torch.equal(value.float(), fresh[key].float()), key)


class TestWarmStart(unittest.TestCase):
    def test_apply_plan_from_a_qwen_named_source(self):
        layer = _layer()
        n_target = 16
        source = {
            "gate.weight": torch.randn(n_target, HIDDEN),
            "shared_expert_gate.weight": torch.randn(1, HIDDEN),
        }
        for j in range(n_target):
            source[f"experts.{j}.gate_proj.weight"] = torch.randn(16, HIDDEN)
            source[f"experts.{j}.up_proj.weight"] = torch.randn(16, HIDDEN)
            source[f"experts.{j}.down_proj.weight"] = torch.randn(HIDDEN, 16)
        source["shared_expert.gate_proj.weight"] = torch.randn(24, HIDDEN)
        source["shared_expert.up_proj.weight"] = torch.randn(24, HIDDEN)
        source["shared_expert.down_proj.weight"] = torch.randn(HIDDEN, 24)
        plan = plan_warm_start(layer.cfg, n_target_experts=n_target)
        loaded = apply_warm_start(layer, plan, source)
        self.assertIn("shared_experts.gate.weight", loaded)
        for i, j in enumerate(plan.target_expert_ids):
            self.assertTrue(
                torch.equal(
                    layer.experts.w1[i], source[f"experts.{j}.gate_proj.weight"]
                )
            )
            self.assertTrue(
                torch.equal(layer.experts.w3[i], source[f"experts.{j}.up_proj.weight"])
            )
            self.assertTrue(
                torch.equal(
                    layer.experts.w2[i], source[f"experts.{j}.down_proj.weight"]
                )
            )
            self.assertTrue(torch.equal(layer.gate.weight[i], source["gate.weight"][j]))
        self.assertTrue(
            torch.equal(
                layer.shared_experts.w3.weight, source["shared_expert.up_proj.weight"]
            )
        )
        self.assertTrue(
            torch.equal(
                layer.shared_experts.gate.weight, source["shared_expert_gate.weight"]
            )
        )
        with self.assertRaises(KeyError):
            apply_warm_start(_layer(), plan, {"gate.weight": source["gate.weight"]})


class TestDSparkIntegration(unittest.TestCase):
    def _forward(self, model, batch=2):
        hidden = model.config.hidden_size
        block = model.config.block_size
        anchors = 5
        noise = torch.randn(batch, block, hidden, requires_grad=True)
        target_hidden = torch.randn(batch, anchors, model.fc.in_features)
        out = model(
            position_ids=torch.arange(anchors + block).expand(batch, -1),
            noise_embedding=noise,
            target_hidden=target_hidden,
            attention_mask=torch.ones(
                batch, 1, block, anchors + block, dtype=torch.bool
            ),
        )
        return out, noise

    def test_dspark_forward_backward_with_aux_loss(self):
        torch.manual_seed(7)
        model = DSparkDraftModel(_draft_config("DSparkDraftModel")).train()
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertIsInstance(layer.balance, AuxLossController)
            self.assertTrue(layer.shared_experts.gated)
            self.assertTrue(torch.isfinite(layer.gate.weight).all())
            self.assertTrue(torch.isfinite(layer.experts.w1).all())
            self.assertTrue(torch.isfinite(layer.shared_experts.gate.weight).all())
        self.assertIsNotNone(model.markov_head)
        self.assertIsNotNone(model.confidence_head)
        out, noise = self._forward(model)
        self.assertTrue(torch.isfinite(out).all())
        aux = collect_moe_aux_loss(model)
        self.assertIsNotNone(aux)
        self.assertTrue(torch.isfinite(aux))
        self.assertGreater(float(aux), 0.0)
        loss = out.float().square().mean() + aux
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(noise.grad).all())
        for layer in layers:
            for name in (
                "gate.weight",
                "experts.w1",
                "experts.w2",
                "shared_experts.gate.weight",
            ):
                grad = dict(layer.named_parameters())[name].grad
                self.assertIsNotNone(grad, name)
                self.assertTrue(torch.isfinite(grad).all(), name)
        metrics = collect_moe_metrics(model)
        self.assertIn("moe/aux_loss", metrics)
        self.assertIn("moe/load_frac_max", metrics)
        # the aux-loss policy has no deferred state: a second step is a plain forward
        self._forward(model)
        self.assertIsNotNone(collect_moe_aux_loss(model))

    def test_checked_in_config_builds_on_meta_at_the_expected_scale(self):
        from transformers import AutoConfig

        config = AutoConfig.from_pretrained(
            str(REPO_ROOT / "configs" / "qwen3.8-27b-dspark-moe.json")
        )
        config._attn_implementation = "sdpa"
        with torch.device("meta"):
            model = DSparkDraftModel(config)
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 5)
        total = sum(p.numel() for p in model.parameters())
        routed = sum(p.numel() for l in layers for p in l.experts.parameters())
        per_expert = 3 * 5120 * 512
        self.assertEqual(routed, 5 * 512 * per_expert)
        active = total - routed + 5 * 10 * per_expert
        self.assertAlmostEqual(total / 1e9, 20.8, delta=0.2)
        self.assertAlmostEqual(active / 1e9, 1.08, delta=0.05)
        official = to_checkpoint_state_dict(model.state_dict())
        self.assertIn("layers.4.mlp.experts.511.gate_proj.weight", official)
        self.assertIn("layers.4.mlp.shared_expert_gate.weight", official)


class TestRouterRegularizers(unittest.TestCase):
    """Knobs for from-scratch drafters whose router inputs share a dominant
    token-independent component (every token picks the same top-k)."""

    @staticmethod
    def _collapsed_inputs(tokens=64, common_scale=200.0, seed=3):
        torch.manual_seed(seed)
        common = torch.randn(HIDDEN)
        return common_scale * common + torch.randn(tokens, HIDDEN)

    def test_config_knobs_and_validation(self):
        cfg = resolve_moe_config(
            _json(
                dflash_config={
                    "moe_router_noise_std": 0.5,
                    "moe_router_z_loss_coeff": 1e-3,
                    "moe_router_init_std": 0.1,
                    "moe_router_center": "ema",
                    "moe_router_center_momentum": 0.9,
                }
            )
        )
        self.assertEqual(
            (
                cfg.router_noise_std,
                cfg.router_z_loss_coeff,
                cfg.router_init_std,
                cfg.router_center,
                cfg.router_center_momentum,
            ),
            (0.5, 1e-3, 0.1, "ema", 0.9),
        )
        defaults = resolve_moe_config(_json())
        self.assertEqual(defaults.router_noise_std, 0.0)
        self.assertEqual(defaults.router_center, "none")
        for bad in (
            {"moe_router_noise_std": -1},
            {"moe_router_z_loss_coeff": -1},
            {"moe_router_init_std": -1},
            {"moe_router_center": "running"},
            {"moe_router_center_momentum": 1.0},
        ):
            with self.assertRaises(ValueError):
                resolve_moe_config(_json(dflash_config=bad))

    def test_diagnostics_detect_collapse_and_diversity(self):
        x = self._collapsed_inputs()
        layer = _layer().train()
        layer(x)
        m = layer.metrics()
        for key in (
            "router_input_cos",
            "logit_common_frac",
            "logit_common_std",
            "logit_token_std",
            "route_entropy_frac",
            "top1_mode_frac",
        ):
            self.assertIn(key, m)
        # a dominant common direction: near-parallel inputs, common logits
        self.assertGreater(float(m["router_input_cos"]), 0.9)
        self.assertGreater(float(m["logit_common_frac"]), 0.9)
        self.assertGreater(float(m["logit_common_std"]), float(m["logit_token_std"]))
        self.assertAlmostEqual(float(m["top1_mode_frac"]), 1.0, places=6)
        # every token picks the same k: entropy log(k)/log(E)
        import math

        self.assertAlmostEqual(
            float(m["route_entropy_frac"]), math.log(3) / math.log(8), places=4
        )
        self.assertAlmostEqual(float(m["experts_unused_frac"]), 5 / 8, places=6)
        # diverse inputs: low cosine, common fraction small, entropy high
        layer(torch.randn(64, HIDDEN))
        m = layer.metrics()
        self.assertLess(float(m["router_input_cos"]), 0.2)
        self.assertLess(float(m["logit_common_frac"]), 0.3)
        self.assertGreater(float(m["route_entropy_frac"]), 0.9)
        # eval forwards neither compute nor clear diagnostics
        layer.eval()
        layer(torch.randn(4, HIDDEN))
        self.assertIn("router_input_cos", layer.metrics())

    def test_diagnostics_are_exact_on_a_toy_case(self):
        x = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        logits = torch.tensor([[1.0, 3.0, 2.0], [1.0, 3.0, 2.0]])
        indices = torch.tensor([[1], [1]])
        counts = torch.tensor([0, 2, 0])
        d = routing_diagnostics(x, logits, indices, counts)
        self.assertAlmostEqual(float(d["router_input_cos"]), 0.0, places=6)
        self.assertAlmostEqual(float(d["logit_common_frac"]), 1.0, places=6)
        self.assertAlmostEqual(float(d["logit_token_std"]), 0.0, places=6)
        self.assertAlmostEqual(float(d["route_entropy_frac"]), 0.0, places=6)
        self.assertAlmostEqual(float(d["top1_mode_frac"]), 1.0, places=6)

    def test_per_layer_metrics_are_reported(self):
        model = DSparkDraftModel(_draft_config("DSparkDraftModel")).train()
        TestDSparkIntegration._forward(self, model)
        metrics = collect_moe_metrics(model)
        for i in range(2):
            self.assertIn(f"moe/layer{i}/experts_unused_frac", metrics)
            self.assertIn(f"moe/layer{i}/logit_common_frac", metrics)
        self.assertIn("moe/router_input_cos", metrics)

    def test_noise_is_training_only_and_diversifies_collapsed_routing(self):
        x = self._collapsed_inputs()
        plain = _layer().train()
        plain(x)
        self.assertAlmostEqual(float(plain.metrics()["experts_unused_frac"]), 5 / 8)
        noisy = _layer(dflash_config={"moe_router_noise_std": 50.0}).train()
        noisy.load_state_dict(plain.state_dict())
        torch.manual_seed(0)
        noisy(x)
        self.assertLess(float(noisy.metrics()["experts_unused_frac"]), 0.2)
        # combine weights come from the jittered scores in training ...
        torch.manual_seed(0)
        y_train = noisy(x)
        torch.manual_seed(1)
        self.assertFalse(torch.allclose(y_train, noisy(x)))
        # ... and the eval forward is deterministic and equals the plain router
        noisy.eval()
        plain.eval()
        self.assertTrue(torch.allclose(noisy(x), plain(x)))

    def test_z_loss_is_added_to_the_layer_aux_loss(self):
        layer = _layer(
            dflash_config={"moe_aux_loss_coeff": 0.0, "moe_router_z_loss_coeff": 0.1}
        ).train()
        x = torch.randn(16, HIDDEN)
        layer(x)
        z = layer.aux_loss()
        self.assertIsNotNone(z)
        logits = x @ layer.gate.weight.detach().t()
        ref = 0.1 * torch.logsumexp(logits, -1).square().mean()
        self.assertAlmostEqual(float(z), float(ref), places=5)
        self.assertIn("z_loss", layer.metrics())
        z.backward()
        self.assertIsNotNone(layer.gate.weight.grad)
        # with the balance loss on, both terms are summed
        both = _layer(
            dflash_config={"moe_aux_loss_coeff": 0.5, "moe_router_z_loss_coeff": 0.1}
        ).train()
        both(x)
        self.assertGreater(float(both.aux_loss()), float(both.balance.aux_loss()))
        # off in eval / without the coefficient
        layer.eval()
        layer(x)
        self.assertIsNone(layer.aux_loss())
        self.assertIsNone(_layer().train().gate.aux_loss())

    def test_router_init_std_overrides_the_model_default(self):
        torch.manual_seed(0)
        layer = MoELayer(
            resolve_moe_config(_json(dflash_config={"moe_router_init_std": 1.0})),
            HIDDEN,
        )
        layer.reset_parameters(std=0.02)
        self.assertGreater(float(layer.gate.weight.std()), 0.5)
        layer = MoELayer(resolve_moe_config(_json()), HIDDEN)
        layer.reset_parameters(std=0.02)
        self.assertLess(float(layer.gate.weight.std()), 0.05)

    def test_ema_centering_removes_the_common_mode(self):
        x = self._collapsed_inputs()
        layer = _layer(dflash_config={"moe_router_center": "ema"}).train()
        self.assertIn("input_mean", dict(layer.gate.named_buffers()))
        self.assertEqual(layer.gate.input_mean.dtype, torch.float32)
        # first forward: mean is still zero -> collapsed like the plain router
        layer(x)
        self.assertAlmostEqual(float(layer.metrics()["experts_unused_frac"]), 5 / 8)
        self.assertTrue(torch.equal(layer.gate.input_mean, torch.zeros(HIDDEN)))
        # the update happens outside the forward (deferred, like a balance bias)
        layer.apply_pending_balance_update()
        self.assertTrue(torch.allclose(layer.gate.input_mean, x.mean(0)))
        self.assertEqual(int(layer.gate.input_mean_steps), 1)
        layer(x)
        m = layer.metrics()
        self.assertLess(float(m["experts_unused_frac"]), 0.2)
        self.assertLess(float(m["logit_common_frac"]), 0.3)
        self.assertGreater(float(m["input_mean_norm"]), 0.0)
        # second update is an EMA step
        layer.apply_pending_balance_update()
        self.assertTrue(torch.allclose(layer.gate.input_mean, x.mean(0), atol=1e-5))
        y = self._collapsed_inputs(seed=4)
        layer(y)
        layer.apply_pending_balance_update()
        expected = 0.99 * x.mean(0) + 0.01 * y.mean(0)
        self.assertTrue(torch.allclose(layer.gate.input_mean, expected, atol=1e-4))
        # eval uses the same centering; no pending state is created
        layer.eval()
        layer(x)
        self.assertIsNone(layer.gate._pending_input_mean)
        # the buffer round-trips through the checkpoint boundary
        state = to_checkpoint_state_dict(layer.state_dict())
        self.assertIn("gate.input_mean", state)
        fresh = _layer(dflash_config={"moe_router_center": "ema"})
        fresh.load_state_dict(from_checkpoint_state_dict(state))
        self.assertTrue(torch.equal(fresh.gate.input_mean, layer.gate.input_mean))
        # the plain router has no such buffer and a no-op update
        plain = _layer().train()
        plain(x)
        plain.apply_pending_balance_update()
        self.assertNotIn("input_mean", dict(plain.gate.named_buffers()))
        # bf16 casts keep the statistics fp32
        layer.to(torch.bfloat16)
        self.assertEqual(layer.gate.input_mean.dtype, torch.float32)

    def test_batch_centering_removes_the_common_mode_from_the_first_forward(self):
        x = self._collapsed_inputs()
        layer = _layer(dflash_config={"moe_router_center": "batch"}).train()
        layer(x)
        m = layer.metrics()
        # exact: the token-mean logit vector carries (almost) no energy
        self.assertLess(float(m["logit_common_frac"]), 0.05)
        self.assertLess(float(m["experts_unused_frac"]), 0.2)
        # the EMA lags (still zero) -> a large eval-time logit error is reported
        self.assertIn("logit_lag_std", m)
        self.assertGreater(float(m["logit_lag_std"]), float(m["logit_token_std"]))
        layer.apply_pending_balance_update()
        self.assertTrue(torch.allclose(layer.gate.input_mean, x.mean(0)))
        layer(x)
        self.assertLess(float(layer.metrics()["logit_lag_std"]), 1e-3)
        # gradient reaches the gate weight through the centered logits
        layer(x).float().square().mean().backward()
        self.assertIsNotNone(layer.gate.weight.grad)
        # eval uses the EMA buffer, like "ema" mode
        layer.eval()
        ema = _layer(dflash_config={"moe_router_center": "ema"}).eval()
        ema.load_state_dict(layer.state_dict())
        self.assertTrue(torch.allclose(layer(x), ema(x)))
        # the plain and ema routers report no lag metric
        plain = _layer().train()
        plain(x)
        self.assertNotIn("logit_lag_std", plain.metrics())

    def test_normalize_keeps_logit_scale_and_folds_at_export(self):
        with self.assertRaises(ValueError):
            resolve_moe_config(_json(dflash_config={"moe_router_normalize": True}))
        knobs = {"moe_router_center": "batch", "moe_router_normalize": True}
        layer = _layer(dflash_config=knobs).train()
        self.assertIn("input_rms", dict(layer.gate.named_buffers()))
        x = self._collapsed_inputs()
        layer(x)
        std_full = float(layer.metrics()["logit_token_std"])
        # shrinking the token-specific part 10x leaves the logit scale intact
        common = x.mean(0, keepdim=True)
        x_small = common + 0.1 * (x - common)
        layer(x_small)
        m = layer.metrics()
        self.assertAlmostEqual(float(m["logit_token_std"]), std_full, delta=0.05 * std_full)
        # ... while the plain batch-centered router's logits shrink with it
        plain = _layer(dflash_config={"moe_router_center": "batch"}).train()
        plain.load_state_dict(layer.state_dict(), strict=False)
        plain(x)
        s1 = float(plain.metrics()["logit_token_std"])
        plain(x_small)
        self.assertLess(float(plain.metrics()["logit_token_std"]), 0.2 * s1)
        # EMA statistics move only in the deferred update
        self.assertEqual(float(layer.gate.input_rms), 1.0)
        layer.apply_pending_balance_update()
        centered = x_small - x_small.mean(0)
        self.assertAlmostEqual(
            float(layer.gate.input_rms), float(centered.square().mean().sqrt()), places=5
        )
        self.assertIn("input_rms", layer.metrics())
        # eval: EMA mean and rms; export fold reproduces the eval logits exactly
        layer.eval()
        state = to_checkpoint_state_dict(layer.state_dict())
        folded = fold_router_centering(state)
        self.assertNotIn("gate.input_rms", folded)
        eval_logits = (
            (x - layer.gate.input_mean) / layer.gate.input_rms
        ) @ layer.gate.weight.t()
        folded_logits = x @ folded["gate.weight"].t() + folded["gate.bias"]
        torch.testing.assert_close(eval_logits, folded_logits, rtol=1e-4, atol=1e-3)
        # bf16 casts keep the statistics fp32
        layer.to(torch.bfloat16)
        self.assertEqual(layer.gate.input_rms.dtype, torch.float32)

    def test_export_fold_turns_centering_into_a_gate_bias(self):
        x = self._collapsed_inputs()
        layer = _layer(dflash_config={"moe_router_center": "ema"}).train()
        layer(x)
        layer.apply_pending_balance_update()
        layer.eval()
        state = to_checkpoint_state_dict(layer.state_dict())
        folded = fold_router_centering(state)
        self.assertNotIn("gate.input_mean", folded)
        self.assertNotIn("gate.input_mean_steps", folded)
        self.assertIn("gate.bias", folded)
        self.assertEqual(folded["gate.bias"].shape, (8,))
        # W (x - mu) == W x + bias for every token
        logits_centered = (x - layer.gate.input_mean) @ layer.gate.weight.t()
        logits_folded = x @ folded["gate.weight"].t() + folded["gate.bias"]
        self.assertTrue(torch.allclose(logits_centered, logits_folded, atol=1e-4))
        # nested prefixes and dense dicts
        nested = {f"layers.3.mlp.{k}": v for k, v in state.items()}
        self.assertIn("layers.3.mlp.gate.bias", fold_router_centering(nested))
        plain = {"gate.weight": torch.zeros(8, HIDDEN)}
        self.assertEqual(fold_router_centering(plain), plain)

    def test_dspark_model_applies_router_updates_before_routing(self):
        config = _draft_config("DSparkDraftModel")
        config.dflash_config["moe_router_center"] = "ema"
        model = DSparkDraftModel(config).train()
        gates = [layer.gate for layer in iter_moe_layers(model)]
        TestDSparkIntegration._forward(self, model)
        for gate in gates:
            self.assertIsNotNone(gate._pending_input_mean)
            self.assertEqual(int(gate.input_mean_steps), 0)
        TestDSparkIntegration._forward(self, model)
        for gate in gates:
            self.assertEqual(int(gate.input_mean_steps), 1)
            self.assertGreater(float(gate.input_mean.norm()), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
