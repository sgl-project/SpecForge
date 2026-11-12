"""MoE FFN nodes: validation, codec round trips and use inside a draft configuration."""

import dataclasses
import json
import unittest

from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import (
    DenseFFNSpec,
    MoEFFNSpec,
    MoERouterSpec,
    MoESharedExpertSpec,
)
from specforge.config.draft.registry import component_schema
from tests.test_config.draft_fixtures import draft_config

ROUTER = {
    "scoring": "softmax",
    "normalize_top_k": False,
    "route_scale": 1.0,
    "expert_bias": True,
}
MOE = {
    "type": "moe",
    "num_experts": 8,
    "num_experts_per_tok": 1,
    "intermediate_size": 96,
    "hidden_act": "silu",
    "shared_expert": None,
    "shared_expert_gate": False,
    "router": ROUTER,
}
# Qwen/DeepSeek's gated FFN and Nemotron's non-gated relu2 MLP as shared experts.
GATED = {"intermediate_size": 48, "hidden_act": "silu", "gated": True}
RELU2 = {"intermediate_size": 48, "hidden_act": "relu2", "gated": False}


def moe_fields():
    return {name: value for name, value in MOE.items() if name != "type"}


def moe_spec(**overrides):
    router = MoERouterSpec(**{**ROUTER, **overrides.pop("router", {})})
    return MoEFFNSpec(**{**moe_fields(), "router": router, **overrides})


def moe_draft_config(**overrides):
    """A dense layer followed by a MoE layer, each referencing its own node."""
    values = {
        "mlp_configs": {
            "dense": DenseFFNSpec(intermediate_size=64, hidden_act="silu"),
            "moe": moe_spec(),
        },
        "layer_configs": {
            "full_dense": LayerSpec(attention="full", mlp="dense"),
            "full_moe": LayerSpec(attention="full", mlp="moe"),
        },
        "layer_types": ("full_dense", "full_moe"),
    }
    return draft_config(**{**values, **overrides})


class MoESpecTest(unittest.TestCase):
    def test_node_is_registered_frozen_and_hashable(self):
        node = moe_spec()
        self.assertIs(component_schema("mlp", "moe"), MoEFFNSpec)
        self.assertEqual(node, moe_spec())
        self.assertEqual(hash(node), hash(moe_spec()))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.num_experts = 2
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.router.expert_bias = False

    def test_invalid_settings_fail_at_construction(self):
        for changes, message in (
            ({"num_experts_per_tok": 9}, "exceeds num_experts"),
            ({"num_experts": 0}, "num_experts must be a positive integer"),
            ({"num_experts_per_tok": True}, "num_experts_per_tok"),
            ({"intermediate_size": -1}, "intermediate_size"),
            ({"hidden_act": "gelu_new"}, "hidden_act"),
            ({"shared_expert_gate": True}, "requires a shared expert"),
            ({"router": {"scoring": "relu"}}, "scoring"),
            ({"router": {"route_scale": 0}}, "route_scale"),
            ({"router": {"route_scale": float("inf")}}, "route_scale"),
            ({"router": {"normalize_top_k": 1}}, "normalize_top_k"),
        ):
            with self.subTest(changes=changes):
                with self.assertRaisesRegex(ValueError, message):
                    moe_spec(**changes)
        for changes, message in (
            ({"intermediate_size": 0}, "intermediate_size"),
            ({"hidden_act": "gelu_new"}, "shared expert hidden_act"),
            ({"gated": 1}, "gated"),
        ):
            with self.subTest(shared_expert=changes):
                with self.assertRaisesRegex(ValueError, message):
                    MoESharedExpertSpec(**{**GATED, **changes})
        for router in (None, ROUTER):
            with self.subTest(router=router):
                with self.assertRaisesRegex(TypeError, "router"):
                    MoEFFNSpec(**{**moe_fields(), "router": router})
        with self.assertRaisesRegex(TypeError, "shared_expert"):
            moe_spec(shared_expert=GATED)

    def test_component_codec_round_trips_and_stays_strict(self):
        qwen = {
            **MOE,
            "shared_expert": GATED,
            "shared_expert_gate": True,
            "router": {
                "scoring": "sigmoid",
                "normalize_top_k": True,
                "route_scale": 2.5,
                "expert_bias": False,
            },
        }
        nemotron = {**MOE, "shared_expert": RELU2}
        for payload in (MOE, qwen, nemotron):
            with self.subTest(shared_expert=payload["shared_expert"]):
                node = decode_component("mlp", payload)
                self.assertIsInstance(node, MoEFFNSpec)
                if payload["shared_expert"] is not None:
                    self.assertEqual(
                        node.shared_expert,
                        MoESharedExpertSpec(**payload["shared_expert"]),
                    )
                self.assertEqual(encode_component("mlp", node), payload)
                restored = decode_component("mlp", json.loads(json.dumps(payload)))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
        for payload, message in (
            ({**MOE, "norm_topk_prob": False}, "unknown fields"),
            ({**MOE, "router": {**ROUTER, "top_k": 1}}, "router: unknown fields"),
            ({k: v for k, v in MOE.items() if k != "router"}, "missing fields"),
            ({**MOE, "router": {**ROUTER, "route_scale": "1"}}, "expected a number"),
            ({**MOE, "shared_expert": {"intermediate_size": 48}}, "shared_expert"),
            (
                {**MOE, "shared_expert": {**GATED, "gated": "yes"}},
                "gated: expected bool",
            ),
            ({**MOE, "shared_expert_gate": True}, "requires a shared expert"),
            ({**MOE, "num_experts_per_tok": 9}, "exceeds num_experts"),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    decode_component("mlp", payload)

    def test_draft_config_with_moe_layers_round_trips(self):
        config = moe_draft_config()
        self.assertEqual(config.num_hidden_layers, 2)
        self.assertIs(config.mlp_configs["moe"], config.mlp_configs["moe"])
        payload = encode_config(config)
        self.assertEqual(payload["mlp_configs"]["moe"], MOE)
        self.assertEqual(payload["layer_configs"]["full_moe"]["mlp"], "moe")
        restored = decode_config(json.loads(json.dumps(payload)))
        self.assertEqual(restored, config)
        self.assertEqual(encode_config(restored), payload)

    def test_cross_node_validation_covers_moe_references(self):
        with self.assertRaisesRegex(ValueError, "unused"):
            moe_draft_config(layer_types=("full_dense", "full_dense"))
        with self.assertRaisesRegex(ValueError, "unknown mlp definition"):
            moe_draft_config(
                layer_configs={
                    "full_dense": LayerSpec(attention="full", mlp="dense"),
                    "full_moe": LayerSpec(attention="full", mlp="experts"),
                }
            )

    def test_yaml_shorthand_expands_a_moe_mlp(self):
        canonical = encode_config(draft_config())
        shorthand = {
            key: value
            for key, value in canonical.items()
            if key
            not in (
                "attention_configs",
                "mlp_configs",
                "conv_configs",
                "layer_configs",
                "layer_types",
            )
        }
        shorthand["attention"] = canonical["attention_configs"]["full"]
        shorthand["mlp"] = MOE
        config = parse_draft(shorthand, target_config=None)
        self.assertEqual(dict(config.mlp_configs), {"default": moe_spec()})
        self.assertEqual(config.layer_types, ("default", "default"))
        self.assertEqual(
            dict(config.layer_configs),
            {"default": LayerSpec(attention="default", mlp="default")},
        )
        self.assertEqual(decode_config(encode_config(config)), config)


if __name__ == "__main__":
    unittest.main()
