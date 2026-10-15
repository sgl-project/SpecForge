# coding=utf-8
"""Qwen3 MoE preset: softmax top-k routing without a shared expert, the Qwen
``num_experts`` spelling of the expert count, and the Qwen3.8-27B DFlash2
dense-vs-MoE ablation pair (draft JSON + recipe)."""

import difflib
import json
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.moe import (
    MOE_PRESETS,
    MoELayer,
    from_checkpoint_state_dict,
    is_moe_config,
    iter_moe_layers,
    resolve_moe_config,
    to_checkpoint_state_dict,
)
from specforge.modeling.draft.moe.config import routed_expert_count
from specforge.modeling.draft.moe.grouped_experts import GroupedExperts
from specforge.modeling.draft.moe.noaux_tc import NoAuxTCController
from specforge.modeling.draft.moe.topk_router import TopKRouter

REPO_ROOT = Path(__file__).resolve().parents[2]
DENSE_CONFIG = REPO_ROOT / "configs" / "qwen3.8-27b-dflash2.json"
MOE_CONFIG = REPO_ROOT / "configs" / "qwen3.8-27b-dflash2-moe.json"
RECIPES = (
    REPO_ROOT / "examples" / "configs" / "online" / "disaggregated" / "managed-local"
)
DENSE_RECIPE = RECIPES / "qwen3.8-27b-dflash2-4server-dp4-disaggregated.yaml"
MOE_RECIPE = RECIPES / "qwen3.8-27b-dflash2-moe-4server-dp4-disaggregated.yaml"

#: Draft-JSON keys that turn the dense Qwen3.8-27B drafter into the MoE arm.
MOE_ONLY_KEYS = {
    "moe_preset",
    "num_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "n_shared_experts",
}
MOE_ONLY_TRAINING_KEYS = {"moe_bias_update_rate", "moe_dispatch"}


def _json(**overrides):
    payload = dict(
        moe_preset="qwen3",
        num_experts=8,
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
    return layer


def _dflash2_config(**overrides):
    method_config = {
        "block_size": 4,
        "conv_group_size": 4,
        "conv_kernel_size": 2,
        "mask_token_id": 31,
        "selector_rank": 4,
        "selector_top_k": 3,
        "target_layer_ids": [1],
        "moe_bias_update_rate": 0.005,
    }
    fields = dict(
        architectures=["DFlash2DraftModel"],
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=2,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=32,
        layer_types=["full_attention", "full_attention"],
        initializer_range=0.02,
        moe_preset="qwen3",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        dflash_config=method_config,
    )
    fields.update(overrides)
    config = Qwen3Config(**fields)
    config._attn_implementation = "eager"
    return config


def _reference_forward(layer: MoELayer, x: torch.Tensor) -> torch.Tensor:
    """Per-token dense reference of the routed FFN: plain SwiGLU, no shared expert."""
    routing = layer.gate(x)
    e = layer.experts
    out = torch.zeros_like(x, dtype=torch.float32)
    for t in range(x.shape[0]):
        for k in range(routing.topk):
            i = int(routing.indices[t, k])
            h = F.silu(x[t] @ e.w1[i].t()) * (x[t] @ e.w3[i].t())
            out[t] += routing.weights[t, k] * (h.to(x.dtype) @ e.w2[i].t()).float()
    return out.to(x.dtype)


def _dflash2_forward(model: DFlash2DraftModel, context_length: int = 3):
    hidden = model.config.hidden_size
    noise = torch.randn(1, model.block_size, hidden)
    target_hidden = torch.randn(1, context_length, hidden)
    position_ids = torch.arange(context_length + model.block_size).unsqueeze(0)
    attention_mask = torch.ones(
        1, 1, model.block_size, context_length + model.block_size, dtype=torch.bool
    )
    return model(
        position_ids=position_ids,
        attention_mask=attention_mask,
        noise_embedding=noise,
        target_hidden=target_hidden,
    )


class TestPresetAndConfig(unittest.TestCase):
    def test_preset_matches_qwen3_moe_recipe(self):
        self.assertIn("qwen3", MOE_PRESETS)
        cfg = resolve_moe_config(_json())
        self.assertEqual(cfg.preset, "qwen3")
        self.assertEqual(cfg.scoring_func, "softmax")
        self.assertTrue(cfg.norm_topk_prob)
        self.assertEqual(cfg.routed_scaling_factor, 1.0)
        self.assertEqual(cfg.n_shared_experts, 0)
        self.assertEqual(cfg.swiglu_limit, 0.0)
        self.assertEqual(cfg.balance, "noaux_tc")
        self.assertEqual(cfg.experts_backend, "grouped")
        self.assertFalse(cfg.group_limited)

    def test_num_experts_is_an_accepted_spelling_of_the_expert_count(self):
        self.assertEqual(routed_expert_count(_json()), 8)
        self.assertTrue(is_moe_config(_json()))
        qwen_spelling = resolve_moe_config(_json())
        deepseek_spelling = resolve_moe_config(
            _json(num_experts=None, n_routed_experts=8)
        )
        self.assertEqual(qwen_spelling.n_routed_experts, 8)
        self.assertEqual(qwen_spelling, deepseek_spelling)
        # both spellings may be present when they agree
        self.assertEqual(
            resolve_moe_config(_json(n_routed_experts=8)).n_routed_experts, 8
        )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            resolve_moe_config(_json(n_routed_experts=4))
        # dense configs under either spelling
        self.assertFalse(is_moe_config({}))
        self.assertFalse(is_moe_config({"num_experts": 0}))
        self.assertIsNone(resolve_moe_config({"num_experts": 0, "moe_preset": "qwen3"}))
        # a positive count still needs the preset
        with self.assertRaisesRegex(ValueError, "moe_preset"):
            resolve_moe_config({"num_experts": 8})

    def test_checked_in_draft_config_is_the_dense_config_plus_moe_keys(self):
        payload = json.loads(MOE_CONFIG.read_text())
        dense = json.loads(DENSE_CONFIG.read_text())
        cfg = resolve_moe_config(payload)
        self.assertEqual(cfg.preset, "qwen3")
        self.assertEqual(
            (cfg.n_routed_experts, cfg.num_experts_per_tok, cfg.moe_intermediate_size),
            (16, 4, 4352),
        )
        self.assertEqual(cfg.n_shared_experts, 0)
        self.assertEqual(cfg.dispatch, "grouped_mm")
        self.assertEqual(cfg.bias_update_rate, 1e-3)
        self.assertEqual(cfg.aux_loss_coeff, 0.0)
        # iso-activated-width ablation: top-k x expert width == the dense MLP width
        self.assertEqual(
            payload["num_experts_per_tok"] * payload["moe_intermediate_size"],
            dense["intermediate_size"],
        )
        self.assertEqual(set(payload) - set(dense), MOE_ONLY_KEYS)
        for key in dense:
            if key != "dflash_config":
                self.assertEqual(payload[key], dense[key], key)
        self.assertEqual(
            set(payload["dflash_config"]) - set(dense["dflash_config"]),
            MOE_ONLY_TRAINING_KEYS,
        )
        for key, value in dense["dflash_config"].items():
            self.assertEqual(payload["dflash_config"][key], value, key)
        self.assertIsNone(resolve_moe_config(dense))

    def test_recipe_differs_from_the_dense_arm_only_in_draft_and_run_names(self):
        def lines(path):
            return [
                line
                for line in path.read_text().splitlines()
                if line.strip() and not line.lstrip().startswith("#")
            ]

        changed = [
            line
            for line in difflib.unified_diff(
                lines(DENSE_RECIPE), lines(MOE_RECIPE), n=0
            )
            if line[:1] in "+-" and not line.startswith(("+++", "---"))
        ]
        self.assertTrue(changed)
        allowed = (
            "draft_model_config:",
            "cache_dir:",
            "run_id:",
            "output_dir:",
            "control_dir:",
            "consumer_state_dir:",
        )
        for line in changed:
            self.assertTrue(line[1:].strip().startswith(allowed), line)
        self.assertIn(
            "draft_model_config: configs/qwen3.8-27b-dflash2-moe.json",
            MOE_RECIPE.read_text(),
        )


class TestQwen3Routing(unittest.TestCase):
    def test_softmax_topk_weights_renormalize_to_one(self):
        layer = _layer()
        self.assertIsInstance(layer.gate, TopKRouter)
        x = torch.randn(5, 32)
        routing = layer.gate(x)
        self.assertTrue(
            torch.allclose(routing.scores.sum(-1), torch.ones(5), atol=1e-5)
        )
        self.assertTrue(
            torch.allclose(routing.weights.sum(-1), torch.ones(5), atol=1e-5)
        )
        self.assertEqual(int(routing.counts.sum()), 10)
        for row in routing.indices.tolist():
            self.assertEqual(len(set(row)), 2)

    def test_layer_has_no_shared_expert_and_matches_dense_reference(self):
        layer = _layer().eval()
        self.assertIsNone(layer.shared_experts)
        self.assertIsInstance(layer.experts, GroupedExperts)
        self.assertEqual(tuple(layer.experts.w1.shape), (8, 16, 32))
        self.assertEqual(tuple(layer.experts.w2.shape), (8, 32, 16))
        x = torch.randn(7, 32)
        self.assertTrue(
            torch.allclose(layer(x), _reference_forward(layer, x), atol=1e-5)
        )
        # the FFN keeps the caller's shape
        self.assertEqual(layer(torch.randn(2, 3, 32)).shape, (2, 3, 32))

    def test_balance_bias_moves_selection_but_not_combine_weights(self):
        layer = _layer().to(torch.bfloat16).eval()
        self.assertIsInstance(layer.balance, NoAuxTCController)
        self.assertEqual(layer.balance.bias.dtype, torch.float32)
        x = torch.randn(4, 32, dtype=torch.bfloat16)
        self.assertEqual(layer(x).dtype, torch.bfloat16)
        layer.balance.bias[:] = -100.0
        layer.balance.bias[3] = 0.0
        routing = layer.gate(x)
        self.assertTrue((routing.indices == 3).any(dim=-1).all())
        self.assertTrue(
            torch.allclose(routing.weights.sum(-1), torch.ones(4), atol=1e-5)
        )

    def test_deferred_balance_update_in_training(self):
        layer = _layer().train()
        layer(torch.randn(6, 32))
        before = layer.balance.bias.clone()
        layer.apply_pending_balance_update()
        self.assertFalse(torch.equal(before, layer.balance.bias))
        self.assertTrue(((layer.balance.bias - before).abs() <= 1e-3 + 1e-7).all())
        self.assertIsNone(layer.aux_loss())  # aux-loss-free by default


class TestCheckpointNaming(unittest.TestCase):
    def test_roundtrip_through_official_naming_without_shared_expert_keys(self):
        layer = _layer()
        official = to_checkpoint_state_dict(layer.state_dict())
        self.assertIn("experts.0.w1.weight", official)
        self.assertIn("experts.7.w3.weight", official)
        self.assertIn("gate.bias", official)
        self.assertIn("gate.weight", official)
        self.assertFalse(any("shared_experts" in k for k in official))
        self.assertFalse(
            any(".balance." in k or k.endswith("experts.w1") for k in official)
        )
        fresh = _layer(dflash_config={"moe_bias_update_rate": 0.0})
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        self.assertTrue(torch.equal(fresh.experts.w2, layer.experts.w2))

    def test_serving_fields_carry_the_qwen3_recipe(self):
        fields = resolve_moe_config(_json()).serving_fields()
        self.assertEqual(fields["n_routed_experts"], 8)
        self.assertEqual(fields["n_shared_experts"], 0)
        self.assertEqual(fields["scoring_func"], "softmax")
        self.assertTrue(fields["norm_topk_prob"])
        self.assertEqual(fields["routed_scaling_factor"], 1.0)
        self.assertEqual(fields["topk_method"], "noaux_tc")
        self.assertNotIn("swiglu_limit", fields)


class TestDFlash2Integration(unittest.TestCase):
    def test_layers_train_and_balance_through_the_model(self):
        torch.manual_seed(0)
        model = DFlash2DraftModel(_dflash2_config())
        layers = list(iter_moe_layers(model))
        self.assertEqual(len(layers), 2)
        for layer in layers:
            self.assertIsInstance(layer.experts, GroupedExperts)
            self.assertIsNone(layer.shared_experts)
            self.assertAlmostEqual(
                float(layer.experts.w1.detach().std()), 0.02, delta=0.005
            )
            self.assertAlmostEqual(
                float(layer.gate.weight.detach().std()), 0.02, delta=0.005
            )
            self.assertTrue(torch.equal(layer.balance.bias, torch.zeros(4)))
        # DFlash2's post_init zeroes the dynamic conv branch; MoE init must not undo it.
        for layer in model.layers:
            self.assertTrue(
                torch.equal(
                    layer.mlp_conv.kernel_projection.weight,
                    torch.zeros_like(layer.mlp_conv.kernel_projection.weight),
                )
            )
        model.train()
        out = _dflash2_forward(model)
        self.assertEqual(out.shape, (1, model.block_size, 16))
        out.float().square().mean().backward()
        for layer in layers:
            self.assertIsNotNone(layer.experts.w2.grad)
            self.assertIsNotNone(layer.gate.weight.grad)
            self.assertTrue(torch.equal(layer.balance.bias, torch.zeros(4)))  # deferred
        self.assertIsNotNone(model.layers[0].mlp_conv.kernel_projection.weight.grad)
        _dflash2_forward(model)  # applies the pending update before routing
        self.assertTrue(any(layer.balance.bias.abs().sum() > 0 for layer in layers))

    def test_model_checkpoint_uses_official_naming_and_reloads(self):
        model = DFlash2DraftModel(_dflash2_config())
        official = to_checkpoint_state_dict(model.state_dict())
        self.assertIn("layers.0.mlp.experts.0.w1.weight", official)
        self.assertIn("layers.1.mlp.experts.3.w2.weight", official)
        self.assertIn("layers.0.mlp.gate.bias", official)
        self.assertIn("layers.0.mlp_conv.base_kernel", official)
        self.assertIn("candidate_selector.predecessor_codebook", official)
        self.assertFalse(
            any(
                ".balance." in k or k.endswith(".experts.w1") or "shared_experts" in k
                for k in official
            )
        )
        fresh = DFlash2DraftModel(_dflash2_config())
        fresh.load_state_dict(from_checkpoint_state_dict(official), strict=True)
        self.assertTrue(
            torch.equal(fresh.layers[1].mlp.experts.w1, model.layers[1].mlp.experts.w1)
        )

    def test_dense_dflash2_config_is_unaffected(self):
        config = _dflash2_config()
        for key in MOE_ONLY_KEYS:
            if hasattr(config, key):
                delattr(config, key)
        config.dflash_config = {
            k: v for k, v in config.dflash_config.items() if not k.startswith("moe_")
        }
        model = DFlash2DraftModel(config)
        self.assertEqual(list(iter_moe_layers(model)), [])
        self.assertEqual(type(model.layers[0].mlp).__name__, "Qwen3MLP")


if __name__ == "__main__":
    unittest.main(verbosity=2)
