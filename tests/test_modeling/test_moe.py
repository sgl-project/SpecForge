# coding=utf-8
"""MoE FFN skeleton: config resolution and the preset registry."""

import unittest
from types import SimpleNamespace

from transformers import Qwen3Config

from specforge.modeling.draft.moe import (
    MOE_PRESETS,
    register_moe_preset,
    resolve_moe_config,
)

PRESET = "_test_family"


def setUpModule():
    register_moe_preset(
        PRESET,
        scoring_func="_test_softplus",
        routed_scaling_factor=1.5,
    )


def tearDownModule():
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
