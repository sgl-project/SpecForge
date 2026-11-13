"""MoE FFN nodes: validation, codec round trips and use inside a draft configuration."""

import json
import unittest

from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import (
    DenseFFNSpec,
    MoEExpertSpec,
    MoEFFNSpec,
    MoERouterSpec,
)
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
    "experts": {"intermediate_size": 96, "hidden_act": "silu", "gated": True},
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
    experts = MoEExpertSpec(**{**MOE["experts"], **overrides.pop("experts", {})})
    return MoEFFNSpec(
        **{**moe_fields(), "experts": experts, "router": router, **overrides}
    )


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
    def test_invalid_settings_fail_at_construction(self):
        for changes, message in (
            ({"num_experts_per_tok": 9}, "exceeds num_experts"),
            ({"experts": {"gated": False}}, "gated FFNs only"),
            ({"num_experts": 0}, "num_experts must be a positive integer"),
            ({"num_experts_per_tok": True}, "num_experts_per_tok"),
            ({"experts": {"intermediate_size": -1}}, "intermediate_size"),
            ({"experts": {"hidden_act": "gelu_new"}}, "hidden_act"),
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
            ({"hidden_act": "gelu_new"}, "expert hidden_act"),
            ({"gated": 1}, "gated"),
        ):
            with self.subTest(shared_expert=changes):
                with self.assertRaisesRegex(ValueError, message):
                    MoEExpertSpec(**{**GATED, **changes})
        for router in (None, ROUTER):
            with self.subTest(router=router):
                with self.assertRaisesRegex(TypeError, "router"):
                    MoEFFNSpec(
                        **{
                            **moe_fields(),
                            "experts": MoEExpertSpec(**MOE["experts"]),
                            "router": router,
                        }
                    )
        with self.assertRaisesRegex(TypeError, "shared_expert"):
            moe_spec(shared_expert=GATED)
        with self.assertRaisesRegex(TypeError, "experts"):
            MoEFFNSpec(
                **{**moe_fields(), "experts": GATED, "router": MoERouterSpec(**ROUTER)}
            )

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
                self.assertEqual(node.experts, MoEExpertSpec(**payload["experts"]))
                if payload["shared_expert"] is not None:
                    self.assertEqual(
                        node.shared_expert,
                        MoEExpertSpec(**payload["shared_expert"]),
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
            ({**MOE, "experts": {"intermediate_size": 96}}, "experts: missing"),
            ({**MOE, "shared_expert": {"intermediate_size": 48}}, "shared_expert"),
            (
                {**MOE, "shared_expert": {**GATED, "gated": "yes"}},
                "gated: expected bool",
            ),
        ):
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    decode_component("mlp", payload)


if __name__ == "__main__":
    unittest.main()
