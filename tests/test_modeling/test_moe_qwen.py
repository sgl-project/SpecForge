# coding=utf-8
"""Qwen3.5/3.8 MoE family: preset vs transformers' block, layout, serving, export."""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

import torch
from transformers.models.qwen3_5_moe import modeling_qwen3_5_moe as hf
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)

from specforge.modeling.auto import AutoDraftModelConfig
from specforge.modeling.draft.moe import (
    MoELayer,
    collect_moe_aux_loss,
    exported_config_updates,
    from_checkpoint_state_dict,
    register_balance_controller,
    resolve_moe_config,
    serving_architecture,
    serving_config,
    to_checkpoint_state_dict,
    to_serving_state_dict,
)
from specforge.modeling.draft.moe.balance import BALANCE_CONTROLLERS, BalanceController

REPO = Path(__file__).resolve().parents[2]
CONFIG = REPO / "configs" / "qwen3.8-27b-dspark-moe.json"
H, E, K, I, S = 16, 8, 2, 8, 12


class _BiasBalance(BalanceController):
    """A selection-bias buffer, as an aux-loss-free policy would keep one."""

    def __init__(self, cfg, n_experts):
        super().__init__(cfg, n_experts)
        self.register_buffer("bias", torch.zeros(n_experts))


def _register_bias_balance(test):
    register_balance_controller("_test_bias")(_BiasBalance)
    test.addCleanup(BALANCE_CONTROLLERS.unregister, "_test_bias")


def _moe(**overrides):
    payload = dict(
        hidden_size=H,
        moe_preset="qwen3_5_moe",
        n_routed_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        dflash_config={},
    )
    payload.update(overrides)
    return payload


def _hf_block():
    config = Qwen3_5MoeTextConfig(
        hidden_size=H,
        num_experts=E,
        num_experts_per_tok=K,
        moe_intermediate_size=I,
        shared_expert_intermediate_size=S,
        norm_topk_prob=True,
        hidden_act="silu",
    )
    block = hf.Qwen3_5MoeSparseMoeBlock(config).float()
    torch.manual_seed(0)
    for parameter in block.parameters():
        torch.nn.init.normal_(parameter, std=0.2)
    return block


def _copy_hf_into(layer, block):
    """transformers' fused [E, 2I, H] gate_up / [E, H, I] down -> module layout."""
    with torch.no_grad():
        layer.gate.weight.copy_(block.gate.weight)
        layer.experts.w1.copy_(block.experts.gate_up_proj[:, :I])
        layer.experts.w3.copy_(block.experts.gate_up_proj[:, I:])
        layer.experts.w2.copy_(block.experts.down_proj)
        shared = layer.shared_experts
        shared.w1.weight.copy_(block.shared_expert.gate_proj.weight)
        shared.w3.weight.copy_(block.shared_expert.up_proj.weight)
        shared.w2.weight.copy_(block.shared_expert.down_proj.weight)
        shared.gate.weight.copy_(block.shared_expert_gate.weight)


class TestQwenPreset(unittest.TestCase):
    def test_recipe_and_checked_in_config(self):
        cfg = resolve_moe_config(_moe())
        self.assertEqual(
            (cfg.scoring_func, cfg.balance, cfg.shared_expert_gate, cfg.layout),
            ("softmax", "none", "sigmoid", "qwen"),
        )
        self.assertEqual(cfg.routed_scaling_factor, 1.0)
        checked_in = resolve_moe_config(AutoDraftModelConfig.from_file(str(CONFIG)))
        self.assertEqual(
            (
                checked_in.n_routed_experts,
                checked_in.num_experts_per_tok,
                checked_in.moe_intermediate_size,
                checked_in.shared_expert_intermediate_size,
            ),
            (512, 10, 512, 2048),
        )
        self.assertEqual(checked_in.router_center, "sample")
        self.assertEqual(checked_in.aux_loss_coeff, 0.001)

    def test_matches_transformers_qwen3_5_moe_block(self):
        block = _hf_block()
        layer = MoELayer(resolve_moe_config(_moe()), H).float().eval()
        _copy_hf_into(layer, block)
        x = torch.randn(2, 5, H)
        with torch.no_grad():
            expected = block(x)
            actual = layer(x)
        self.assertLess(float((actual - expected).abs().max()), 1e-5)

    def test_aux_loss_matches_transformers(self):
        block = _hf_block()
        layer = MoELayer(
            resolve_moe_config(_moe(dflash_config={"moe_aux_loss_coeff": 0.01})), H
        ).float()
        _copy_hf_into(layer, block)
        x = torch.randn(10, H)
        layer.train()(x)
        logits, _, _ = block.gate(x)
        expected = 0.01 * hf.load_balancing_loss_func((logits,), E, K)
        self.assertTrue(
            torch.allclose(collect_moe_aux_loss(layer), expected, atol=1e-6)
        )


class TestQwenLayout(unittest.TestCase):
    def test_file_names_and_round_trip(self):
        layer = MoELayer(resolve_moe_config(_moe()), H)
        layer.reset_parameters(0.1)
        state = layer.state_dict()
        files = to_checkpoint_state_dict(state, layer)
        expected = {
            "gate.weight",
            "shared_expert.gate_proj.weight",
            "shared_expert.up_proj.weight",
            "shared_expert.down_proj.weight",
            "shared_expert_gate.weight",
        } | {
            f"experts.{i}.{p}.weight"
            for i in range(E)
            for p in ("gate_proj", "up_proj", "down_proj")
        }
        self.assertEqual(set(files), expected)
        self.assertTrue(
            torch.equal(files["experts.3.up_proj.weight"], layer.experts.w3[3])
        )
        back = from_checkpoint_state_dict(files, layer)
        self.assertEqual(set(back), set(state))
        for key in state:
            self.assertTrue(torch.equal(back[key], state[key]), key)

    def test_missing_experts_and_selection_bias_are_rejected(self):
        layer = MoELayer(resolve_moe_config(_moe()), H)
        files = to_checkpoint_state_dict(layer.state_dict(), layer)
        del files["experts.2.down_proj.weight"]
        with self.assertRaisesRegex(KeyError, "missing expert indices"):
            from_checkpoint_state_dict(files, layer)
        _register_bias_balance(self)
        biased = MoELayer(resolve_moe_config(_moe(balance="_test_bias")), H)
        with self.assertRaisesRegex(ValueError, "selection bias"):
            to_checkpoint_state_dict(biased.state_dict(), biased)


class TestQwenServing(unittest.TestCase):
    def test_export_folds_centering_into_a_router_bias(self):
        layer = MoELayer(
            resolve_moe_config(_moe(dflash_config={"moe_router_center": "sample"})), H
        )
        layer.reset_parameters(0.1)
        layer.train()(torch.randn(2, 4, H) + 2.0)
        layer.apply_pending_update()
        served = to_serving_state_dict(layer.state_dict(), layer)
        self.assertIn("gate.bias", served)
        self.assertEqual(served["gate.bias"].dtype, torch.float32)
        self.assertNotIn("gate.input_mean", served)
        self.assertIn("shared_expert_gate.weight", served)
        fields = serving_config(layer)
        self.assertTrue(fields["moe_router_bias"])
        self.assertEqual(fields["num_experts"], E)
        self.assertEqual(fields["shared_expert_intermediate_size"], S)
        self.assertEqual(
            serving_architecture(layer, "Qwen3DSparkModel"), "Qwen3MoeDSparkModel"
        )
        # The exported config describes the folded draft, so it reloads.
        updates = exported_config_updates(layer)
        exported = _moe(
            **{k: v for k, v in updates.items() if k != "dflash_config"},
            dflash_config=updates["dflash_config"],
        )
        reloaded = MoELayer(resolve_moe_config(exported), H).eval()
        reloaded.load_state_dict(from_checkpoint_state_dict(served, reloaded))
        x = torch.randn(16, H) * 2.0 + 2.0
        with torch.no_grad():
            want, got = layer.eval().gate(x), reloaded.gate(x)
        self.assertTrue(torch.equal(want.indices, got.indices))
        self.assertTrue(torch.allclose(want.weights, got.weights, atol=1e-5))

    def test_unservable_recipes_are_refused(self):
        _register_bias_balance(self)
        for override in (
            {"scoring_func": "sigmoid"},
            {"balance": "_test_bias"},
            {"shared_expert_gate": "none"},
        ):
            with (
                self.subTest(**override),
                self.assertRaisesRegex(ValueError, "Qwen3MoeDSparkModel cannot serve"),
            ):
                serving_config(_moe(**override))

    def test_normalizer_names_the_moe_serving_class(self):
        spec = importlib.util.spec_from_file_location(
            "normalize_dflash_export",
            REPO / "scripts" / "gates" / "normalize_dflash_export.py",
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        normalize_export = module.normalize_export

        exported = json.loads(CONFIG.read_text())
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "config.json")
            Path(path).write_text(json.dumps(exported))
            config = normalize_export(path, expected_block_size=7)
            self.assertEqual(config["architectures"], ["Qwen3MoeDSparkModel"])
            self.assertEqual(config["num_experts"], 512)
            self.assertTrue(config["moe_router_bias"])
            dense = {k: v for k, v in exported.items() if k != "moe_preset"}
            Path(path).write_text(json.dumps(dense))
            config = normalize_export(path, expected_block_size=7)
            self.assertEqual(config["architectures"], ["Qwen3DSparkModel"])
            self.assertNotIn("num_experts", config)


if __name__ == "__main__":
    unittest.main()
