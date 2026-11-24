"""Type dispatch stays in configuration; layers only hold name references."""

import copy
import dataclasses
import json
import subprocess
import sys
import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.codec.components import (
    decode_component,
    decode_layer,
    encode_component,
    encode_layer,
)
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.mlp import FFNSpec
from specforge.config.draft.registry import component_schema, register_component_schema
from specforge.config.draft.rope import DefaultRoPESpec, RopeSpec

COMPONENTS = {
    "attention": {
        "qk_norm_position": "before_rope",
        "qk_norm_scope": "head",
        "qk_norm_config": None,
        "type": "gqa",
        "rope_parameters": None,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 16,
        "qk_norm": True,
        "attention_bias": False,
        "output_gate": False,
        "is_causal": True,
        "sliding_window": None,
        "attention_dropout": 0.0,
    },
    "mlp": {"type": "dense", "intermediate_size": 64, "hidden_act": "silu"},
    "conv": {"type": "grouped_causal", "kernel_size": 3, "group_size": 8},
    "norm": {"has_weight": True, "type": "rms", "eps": 1e-6, "weight_offset": 1},
    "residual": {"type": "standard"},
    "rope": {"rope_type": "none"},
    "logits": {"type": "mup", "multiplier": 8.0},
}


@dataclass(frozen=True, kw_only=True)
class RouterSettings:
    top_k: int


@dataclass(frozen=True, kw_only=True)
class NestedFFN(FFNSpec):
    type: ClassVar[str] = "test_nested_ffn"
    router: RouterSettings
    rope: RopeSpec


class ComponentCodecTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(registry._SCHEMAS))
        self.enterContext(patch.dict(registry._VALIDATORS))

    def test_dense_activation_is_validated_during_decode(self):
        for activation in ("gelu_new", "prelu", "unknown"):
            with self.subTest(activation=activation):
                with self.assertRaisesRegex(ValueError, "hidden_act"):
                    decode_component(
                        "mlp", COMPONENTS["mlp"] | {"hidden_act": activation}
                    )

    def test_nested_component_must_match_its_annotated_type(self):
        @dataclass(frozen=True, kw_only=True)
        class NarrowFFN(FFNSpec):
            type: ClassVar[str] = "test_narrow_ffn"
            rope: DefaultRoPESpec

        register_component_schema("mlp", NarrowFFN)
        payload = {"type": "test_narrow_ffn", "rope": {"rope_type": "none"}}
        with self.assertRaisesRegex(ValueError, "rope: expected DefaultRoPESpec"):
            decode_component("mlp", payload)

    def test_nested_nodes_and_components_use_the_same_codec(self):
        register_component_schema("mlp", NestedFFN)
        payload = {
            "type": "test_nested_ffn",
            "router": {"top_k": 2},
            "rope": {"rope_type": "none"},
        }
        node = decode_component("mlp", payload)
        self.assertEqual(encode_component("mlp", node), payload)
        self.assertEqual(decode_component("mlp", json.loads(json.dumps(payload))), node)
        with self.assertRaisesRegex(ValueError, "router.top_k"):
            decode_component("mlp", payload | {"router": {"top_k": True}})
        with self.assertRaisesRegex(ValueError, "rope.rope_type"):
            decode_component("mlp", payload | {"rope": {"rope_type": "unknown"}})

    def test_components_round_trip_through_json(self):
        for kind, payload in COMPONENTS.items():
            with self.subTest(kind=kind):
                source = copy.deepcopy(payload)
                node = decode_component(kind, source)
                encoded = encode_component(kind, node)
                self.assertEqual(encoded, payload)
                restored = decode_component(kind, json.loads(json.dumps(encoded)))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(source, payload)
                encoded["type"] = "unregistered"
                self.assertEqual(encode_component(kind, node), payload)
        mha = {**COMPONENTS["attention"], "type": "mha", "num_key_value_heads": 8}
        self.assertEqual(
            encode_component("attention", decode_component("attention", mha)), mha
        )

    def test_missing_unknown_and_malformed_fields(self):
        for kind, payload in COMPONENTS.items():
            for name in payload:
                with self.subTest(kind=kind, name=name):
                    with self.assertRaisesRegex(ValueError, name):
                        decode_component(
                            kind,
                            {
                                key: value
                                for key, value in payload.items()
                                if key != name
                            },
                        )
            with self.assertRaisesRegex(ValueError, "typo"):
                decode_component(kind, {**payload, "typo": 1})
            for source in (None, [], "dense", 3):
                with self.assertRaisesRegex(ValueError, "object"):
                    decode_component(kind, source)
        with self.assertRaisesRegex(ValueError, "attention/gqa.*is_causal"):
            decode_component(
                "attention", {**COMPONENTS["attention"], "is_causal": None}
            )
        with self.assertRaisesRegex(ValueError, "sliding_window"):
            decode_component(
                "attention", {**COMPONENTS["attention"], "sliding_window": "128"}
            )

    def test_unknown_kinds_and_types(self):
        for name in (None, [], {}, 3, "target", "typo"):
            with self.assertRaisesRegex(ValueError, "unknown attention type"):
                decode_component("attention", {"type": name})
        for kind in ("layer", "target", "ffn", None, []):
            with self.assertRaisesRegex(ValueError, "unknown component kind"):
                component_schema(kind, "decoder")

    def test_layer_is_a_fixed_reference_schema(self):
        payload = {"attention": "full", "mlp": "dense", "sublayer_conv": "conv"}
        node = decode_layer(payload)
        self.assertEqual(node, LayerSpec(**payload))
        self.assertEqual(encode_layer(node), payload)
        without_conv = decode_layer({"attention": "full", "mlp": "dense"})
        self.assertIsNone(without_conv.sublayer_conv)
        self.assertEqual(decode_layer(encode_layer(without_conv)), without_conv)
        for field, value in (
            ("type", "decoder"),
            ("type", "target"),
            ("hidden_size", 64),
            ("residual", "standard"),
        ):
            with self.assertRaisesRegex(ValueError, field):
                decode_layer({**payload, field: value})
        with self.assertRaisesRegex(ValueError, "attention"):
            decode_layer({"attention": COMPONENTS["attention"], "mlp": "dense"})
        with self.assertRaisesRegex(ValueError, "object"):
            decode_layer([])

    def test_schema_registration_is_separate_from_builders(self):
        @dataclass(frozen=True, kw_only=True)
        class CustomFFN(FFNSpec):
            type: ClassVar[str] = "test_ffn"
            width: int

            def __post_init__(self):
                if type(self.width) is not int or self.width <= 0:
                    raise ValueError("width must be positive")

        @dataclass(frozen=True, kw_only=True)
        class ConflictingFFN(FFNSpec):
            type: ClassVar[str] = "test_ffn"

        with patch.dict(registry._SCHEMAS):
            register_component_schema("mlp", CustomFFN)
            register_component_schema("mlp", CustomFFN)
            self.assertIs(component_schema("mlp", "test_ffn"), CustomFFN)
            payload = {"type": "test_ffn", "width": 32}
            self.assertEqual(
                encode_component("mlp", decode_component("mlp", payload)), payload
            )
            with self.assertRaisesRegex(ValueError, "width"):
                decode_component("mlp", {"type": "test_ffn", "width": 0})
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_component_schema("mlp", ConflictingFFN)
            with self.assertRaisesRegex(TypeError, "AttentionSpec"):
                register_component_schema("attention", CustomFFN)
            with self.assertRaisesRegex(ValueError, "unknown component kind"):
                register_component_schema("layer", LayerSpec)
            with self.assertRaisesRegex(ValueError, "type name"):
                register_component_schema("mlp", FFNSpec)

    def test_encode_checks_schema_identity(self):
        @dataclass(frozen=True, kw_only=True)
        class Impostor(FFNSpec):
            type: ClassVar[str] = "dense"

        with self.assertRaisesRegex(ValueError, "expected DenseFFNSpec"):
            encode_component("mlp", Impostor())
        with self.assertRaisesRegex(ValueError, "expected LayerSpec"):
            encode_layer({"attention": "full", "mlp": "dense"})

    def test_registration_rejects_implicit_defaults_and_undeclared_fields(self):
        class Undecorated(FFNSpec):
            type = "test_undecorated"
            width: int

        @dataclass(frozen=True, kw_only=True)
        class WithDefault(FFNSpec):
            type: ClassVar[str] = "test_default"
            width: int = 32

        @dataclass(frozen=True, kw_only=True)
        class MutableType(FFNSpec):
            type: str = "test_mutable_type"

        for schema, error in (
            (Undecorated, "frozen dataclass"),
            (WithDefault, "no defaults"),
            (MutableType, "ClassVar"),
        ):
            with self.subTest(schema=schema), self.assertRaisesRegex(TypeError, error):
                register_component_schema("mlp", schema)

    def test_registration_rejects_nodes_that_would_not_read_back_as_values(self):
        @dataclass(frozen=True, kw_only=True, eq=False)
        class IdentityEqual(FFNSpec):
            type: ClassVar[str] = "test_identity_equal"
            width: int

        @dataclass(frozen=True, kw_only=True)
        class Router:
            top_k: int = 2

        @dataclass(frozen=True, kw_only=True)
        class NestedDefault(FFNSpec):
            type: ClassVar[str] = "test_nested_default"
            router: Router

        @dataclass(frozen=True, kw_only=True)
        class Unresolvable(FFNSpec):
            type: ClassVar[str] = "test_unresolvable"
            router: "MissingRouter"

        for schema, error in (
            (IdentityEqual, "compare by value"),
            (NestedDefault, r"mlp\.router\.top_k: structural fields must have no"),
            (Unresolvable, "unresolvable annotation"),
        ):
            with (
                self.subTest(schema=schema.__name__),
                self.assertRaisesRegex(TypeError, error),
            ):
                register_component_schema("mlp", schema)

    def test_decoding_rejects_mutable_extension_values(self):
        @dataclass(frozen=True, kw_only=True)
        class MutableValue(FFNSpec):
            type: ClassVar[str] = "test_mutable_value"
            widths: list

        with patch.dict(registry._SCHEMAS):
            register_component_schema("mlp", MutableValue)
            with self.assertRaisesRegex(
                ValueError, "widths: unsupported field annotation"
            ):
                decode_component("mlp", {"type": "test_mutable_value", "widths": [32]})

    def test_number_errors_name_the_value_received(self):
        # PyYAML (YAML 1.1) loads `eps: 1e-6` as a string
        with self.assertRaisesRegex(
            ValueError, r"norm/rms\.eps: expected a number, got str '1e-6'"
        ):
            decode_component(
                "norm",
                {"has_weight": True, "type": "rms", "eps": "1e-6", "weight_offset": 0},
            )
        with self.assertRaisesRegex(
            ValueError, r"attention/gqa\.head_dim: expected int, got str '128'"
        ):
            decode_component(
                "attention", {**COMPONENTS["attention"], "head_dim": "128"}
            )

    def test_derived_fields_are_not_serialized(self):
        @dataclass(frozen=True, kw_only=True)
        class Window:
            size: int
            half: int = dataclasses.field(init=False)

            def __post_init__(self):
                object.__setattr__(self, "half", self.size // 2)

        @dataclass(frozen=True, kw_only=True)
        class DerivedFFN(FFNSpec):
            type: ClassVar[str] = "test_derived_ffn"
            window: Window
            width: int
            doubled: int = dataclasses.field(init=False)

            def __post_init__(self):
                object.__setattr__(self, "doubled", 2 * self.width)

        payload = {"type": "test_derived_ffn", "window": {"size": 4}, "width": 3}
        with patch.dict(registry._SCHEMAS):
            register_component_schema("mlp", DerivedFFN)
            node = decode_component("mlp", payload)
            self.assertEqual((node.doubled, node.window.half), (6, 2))
            saved = json.loads(json.dumps(encode_component("mlp", node)))
            self.assertEqual(saved, payload)
            self.assertEqual(decode_component("mlp", saved), node)

    def test_registry_and_codec_import_without_model_dependencies(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from specforge.config.draft.codec.components import decode_component
decode_component('norm', {'type': 'rms', 'has_weight': True, 'eps': 1e-6, 'weight_offset': 0})
for name in ('torch', 'transformers', 'pydantic', 'specforge.modeling',
             'specforge.algorithms', 'specforge.runtime', 'specforge.config.schema'):
    assert name not in sys.modules, name
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
