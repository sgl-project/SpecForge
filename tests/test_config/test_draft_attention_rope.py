"""Per-attention rotary overrides preserve explicit inheritance and disabling."""

import copy
import dataclasses
import json
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.attention import AttentionSpec
from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.rope import DefaultRoPESpec, LinearRoPESpec, NoRoPESpec
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mla_spec import MLA


class AttentionRoPETest(unittest.TestCase):
    def test_inherit_override_and_disable_are_distinct(self):
        base = draft_config()
        attention = base.attention_configs["full"]
        scaled = LinearRoPESpec(rope_theta=1000, partial_rotary_factor=0.5, factor=4)
        node = dataclasses.replace(
            base,
            attention_configs={
                "inherit": attention,
                "scaled": dataclasses.replace(attention, rope_parameters=scaled),
                "disabled": dataclasses.replace(
                    attention, rope_parameters=NoRoPESpec()
                ),
            },
            layer_configs={
                name: LayerSpec(attention=name, mlp="dense")
                for name in ("inherit", "scaled", "disabled")
            },
            layer_types=("inherit", "scaled", "disabled", "inherit"),
        )
        effective = [
            node.attention_configs[name].effective_rope(node.rope_parameters)
            for name in ("inherit", "scaled", "disabled")
        ]
        self.assertIs(effective[0], base.rope_parameters)
        self.assertIs(effective[1], scaled)
        self.assertEqual([rope.rotary_dim(8) for rope in effective], [8, 4, 0])
        payload = encode_config(node)
        self.assertIsNone(payload["attention_configs"]["inherit"]["rope_parameters"])
        self.assertEqual(
            payload["attention_configs"]["scaled"]["rope_parameters"]["rope_type"],
            "linear",
        )
        self.assertEqual(
            payload["attention_configs"]["disabled"]["rope_parameters"],
            {"rope_type": "none"},
        )
        restored = decode_config(json.loads(json.dumps(payload)))
        self.assertEqual(restored, node)
        self.assertEqual(hash(restored), hash(node))

    def test_geometry_uses_effective_rope_per_attention(self):
        base = draft_config()
        attention = base.attention_configs["full"]
        odd_dimension = DefaultRoPESpec(rope_theta=10000, partial_rotary_factor=0.375)
        with self.assertRaisesRegex(
            ValueError, "attention_configs.full.*rotary dimension"
        ):
            dataclasses.replace(
                base,
                attention_configs={
                    "full": dataclasses.replace(
                        attention, rope_parameters=odd_dimension
                    )
                },
            )
        disabled = dataclasses.replace(attention, rope_parameters=NoRoPESpec())
        node = dataclasses.replace(
            base, rope_parameters=odd_dimension, attention_configs={"full": disabled}
        )
        self.assertEqual(
            node.attention_configs["full"]
            .effective_rope(node.rope_parameters)
            .rotary_dim(8),
            0,
        )
        with self.assertRaisesRegex(ValueError, "rotary dimension"):
            dataclasses.replace(node, attention_configs={"full": attention})

    def test_mla_rotates_only_its_rotary_subspace(self):
        payload = {
            **MLA,
            "rope_parameters": {
                "rope_type": "default",
                "rope_theta": 10000,
                "partial_rotary_factor": 0.5,
            },
        }
        node = decode_component("attention", payload)
        rope = node.effective_rope(NoRoPESpec())
        self.assertEqual(rope.rotary_dim(node.rotary_input_dim), 4)
        self.assertEqual(node.qk_head_dim, 24)
        self.assertEqual(encode_component("attention", node), payload)

    def test_canonical_choice_is_required_but_yaml_can_inherit(self):
        payload = encode_config(draft_config())
        payload["attention_configs"]["full"].pop("rope_parameters")
        original = copy.deepcopy(payload)
        with self.assertRaisesRegex(ValueError, "rope_parameters"):
            decode_config(payload)
        self.assertEqual(parse_draft(payload, target_config=None), draft_config())
        self.assertEqual(payload, original)
        attention = payload.pop("attention_configs")["full"]
        payload.pop("conv_configs")
        payload.pop("layer_configs")
        payload.pop("layer_types")
        payload["attention"] = attention
        payload["mlp"] = payload.pop("mlp_configs")["dense"]
        self.assertIsNone(
            parse_draft(payload, target_config=None)
            .attention_configs["default"]
            .rope_parameters
        )

    def test_malformed_overrides_do_not_fall_back_to_model_settings(self):
        for value in (
            False,
            "none",
            {},
            {"rope_type": "unknown"},
            {"rope_type": "none", "rope_theta": 10000},
        ):
            payload = encode_config(draft_config())
            payload["attention_configs"]["full"]["rope_parameters"] = value
            with self.assertRaisesRegex(ValueError, "rope_parameters"):
                decode_config(payload)
        attention = draft_config().attention_configs["full"]
        with self.assertRaisesRegex(ValueError, "rope_parameters"):
            dataclasses.replace(attention, rope_parameters={"rope_type": "none"})
        with self.assertRaisesRegex(ValueError, "rope_parameters"):
            attention.effective_rope(None)

    def test_unregistered_rope_nodes_are_rejected(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class UnregisteredRoPE(DefaultRoPESpec):
            rope_type = "test_unregistered"

        base = draft_config()
        rope = UnregisteredRoPE(rope_theta=10000, partial_rotary_factor=1.0)
        with self.assertRaisesRegex(ValueError, "^rope_parameters: unknown rope type"):
            dataclasses.replace(base, rope_parameters=rope)
        attention = dataclasses.replace(
            base.attention_configs["full"], rope_parameters=rope
        )
        with self.assertRaisesRegex(
            ValueError, "full.rope_parameters: unknown rope type"
        ):
            dataclasses.replace(base, attention_configs={"full": attention})

    def test_rope_override_on_a_non_rotary_kind_is_refused(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class NonRotaryAttention(AttentionSpec):
            type: ClassVar[str] = "test_non_rotary_attention"

            def capabilities(self) -> frozenset[str]:
                return frozenset({"target_context_kv"})
            num_heads: int

            @property
            def rotary_input_dim(self):
                return None

        with patch.dict(registry._SCHEMAS):
            registry.register_component_schema("attention", NonRotaryAttention)
            rope = LinearRoPESpec(rope_theta=1e4, partial_rotary_factor=1.0, factor=2.0)
            draft_config(
                attention_configs={
                    "full": NonRotaryAttention(rope_parameters=None, num_heads=4)
                }
            )
            with self.assertRaisesRegex(ValueError, "no effect"):
                draft_config(
                    attention_configs={
                        "full": NonRotaryAttention(rope_parameters=rope, num_heads=4)
                    }
                )

    def test_yaml_explicit_overrides_survive_parsing(self):
        payload = encode_config(draft_config())
        payload["attention_configs"]["full"]["rope_parameters"] = {"rope_type": "none"}
        node = parse_draft(payload, target_config=None)
        self.assertEqual(node.attention_configs["full"].rope_parameters, NoRoPESpec())
        payload["attention"] = payload.pop("attention_configs")["full"]
        payload["mlp"] = payload.pop("mlp_configs")["dense"]
        for name in ("conv_configs", "layer_configs", "layer_types"):
            payload.pop(name)
        node = parse_draft(payload, target_config=None)
        self.assertEqual(
            node.attention_configs["default"].rope_parameters, NoRoPESpec()
        )


if __name__ == "__main__":
    unittest.main()
