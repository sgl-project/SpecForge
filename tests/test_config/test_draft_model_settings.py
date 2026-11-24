"""Model-level numeric settings are explicit and independent of HF defaults."""

import dataclasses
import json
import subprocess
import sys
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.config.draft.registry import component_schema, register_component_schema
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec


@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomRoPESpec(DefaultRoPESpec):
    rope_type: ClassVar[str] = "test_custom_rope"
    bands: tuple[float, ...]


@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomLogitSpec(MuPLogitTransformSpec):
    type: ClassVar[str] = "test_custom_logits"


class ModelSettingsTest(unittest.TestCase):
    def setUp(self):
        self.enterContext(patch.dict(registry._SCHEMAS))
        self.enterContext(patch.dict(registry._VALIDATORS))

    def test_registered_logit_settings_round_trip(self):
        register_component_schema("logits", CustomLogitSpec)
        payload = {"type": "test_custom_logits", "multiplier": 3.0}
        node = decode_component("logits", payload)
        self.assertIs(type(node), CustomLogitSpec)
        self.assertEqual(encode_component("logits", node), payload)
        with self.assertRaisesRegex(ValueError, "multiplier"):
            decode_component("logits", {**payload, "multiplier": True})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            decode_component("logits", {**payload, "typo": 1})

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Replacement(CustomLogitSpec):
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_component_schema("logits", Replacement)
        with self.assertRaisesRegex(ValueError, "expected CustomLogitSpec"):
            encode_component("logits", Replacement(multiplier=3.0))

    def test_registered_rope_round_trip_and_validation(self):
        register_component_schema("rope", CustomRoPESpec)
        register_component_schema("rope", CustomRoPESpec)
        payload = {
            "interleaved": False,
            "rope_type": "test_custom_rope",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "bands": [1.0, 2.0],
        }
        node = decode_component("rope", payload)
        self.assertIs(type(node), component_schema("rope", payload["rope_type"]))
        self.assertEqual(node.bands, (1.0, 2.0))
        self.assertEqual(encode_component("rope", node), payload)
        with self.assertRaisesRegex(ValueError, "bands"):
            decode_component("rope", {**payload, "bands": [True]})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            decode_component("rope", {**payload, "typo": 1})

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Replacement(CustomRoPESpec):
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_component_schema("rope", Replacement)
        with self.assertRaisesRegex(ValueError, "expected CustomRoPESpec"):
            encode_component("rope", Replacement(**dataclasses.asdict(node)))

    def test_rope_and_logit_settings_round_trip(self):
        cases = (
            ("rope", {"rope_type": "none"}),
            (
                "rope",
                {
                    "interleaved": False,
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25,
                },
            ),
            (
                "rope",
                {
                    "interleaved": False,
                    "rope_type": "linear",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "factor": 8.0,
                },
            ),
            ("logits", {"type": "none"}),
            (
                "logits",
                {"type": "mup", "multiplier": 8.0},
            ),
        )
        for kind, payload in cases:
            with self.subTest(payload=payload):
                node = decode_component(kind, payload)
                self.assertEqual(encode_component(kind, node), payload)
                restored = decode_component(
                    kind, json.loads(json.dumps(encode_component(kind, node)))
                )
                self.assertEqual(node, restored)
                self.assertEqual(hash(node), hash(restored))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    node.new_setting = 1
                with self.assertRaisesRegex(ValueError, "unknown fields"):
                    decode_component(kind, {**payload, "typo": True})
                for field in payload:
                    with self.assertRaisesRegex(ValueError, field):
                        decode_component(
                            kind, {k: v for k, v in payload.items() if k != field}
                        )

    def test_rotary_geometry_is_derived_and_validated(self):
        full = DefaultRoPESpec(
            interleaved=False, rope_theta=10000, partial_rotary_factor=1.0
        )
        partial = dataclasses.replace(full, partial_rotary_factor=0.25)
        self.assertEqual(full.rotary_dim(128), 128)
        self.assertEqual(partial.rotary_dim(128), 32)
        self.assertEqual(NoRoPESpec().rotary_dim(128), 0)
        for head_dim in (True, 0, -1, 3.0):
            with self.assertRaisesRegex(ValueError, "head_dim"):
                full.rotary_dim(head_dim)
        for head_dim in (3, 12):
            with self.assertRaisesRegex(ValueError, "even rotary dimension"):
                partial.rotary_dim(head_dim)

    def test_numeric_values_are_not_coerced(self):
        for value in (True, 0, -1, "8", float("nan"), float("inf"), 10**400):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "multiplier"):
                    MuPLogitTransformSpec(multiplier=value)
                with self.assertRaisesRegex(ValueError, "rope_theta"):
                    DefaultRoPESpec(
                        interleaved=False, rope_theta=value, partial_rotary_factor=1.0
                    )
                with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
                    DefaultRoPESpec(
                        interleaved=False, rope_theta=10000, partial_rotary_factor=value
                    )
                with self.assertRaisesRegex(ValueError, "factor"):
                    decode_component(
                        "rope",
                        {
                            "interleaved": False,
                            "rope_type": "linear",
                            "rope_theta": 10000,
                            "partial_rotary_factor": 1.0,
                            "factor": value,
                        },
                    )
        with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
            DefaultRoPESpec(
                interleaved=False, rope_theta=10000, partial_rotary_factor=1.1
            )

    def test_unknown_formats_fail_instead_of_defaulting(self):
        for kind in ("rope", "logits"):
            for value in (
                None,
                [],
                "default",
                {},
                {"type": []},
                {"rope_type": "unknown"},
            ):
                with self.assertRaises(ValueError):
                    decode_component(kind, value)
        with self.assertRaisesRegex(ValueError, "unsupported node"):
            encode_component("rope", {"rope_type": "none"})
        with self.assertRaisesRegex(ValueError, "unsupported node"):
            encode_component("logits", NoRoPESpec())

    def test_settings_import_without_compute_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                """
import sys
from specforge.config.draft.codec.components import decode_component
decode_component("rope", {'rope_type': 'none'})
decode_component("logits", {'type': 'none'})
assert not any(x in sys.modules for x in ('torch', 'transformers', 'pydantic'))
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
