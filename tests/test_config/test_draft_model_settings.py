"""Model-level numeric settings are explicit and independent of HF defaults."""

import dataclasses
import json
import subprocess
import sys
import unittest
from typing import ClassVar

from specforge.config.draft.codec.settings import (
    decode_rope,
    decode_target_logits,
    encode_rope,
    encode_target_logits,
)
from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.config.draft.registry import component_schema, register_component_schema
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec


@dataclasses.dataclass(frozen=True, kw_only=True)
class CustomRoPESpec(DefaultRoPESpec):
    rope_type: ClassVar[str] = "test_custom_rope"
    bands: tuple[float, ...]


class ModelSettingsTest(unittest.TestCase):
    def test_registered_rope_round_trip_and_validation(self):
        register_component_schema("rope", CustomRoPESpec)
        register_component_schema("rope", CustomRoPESpec)
        payload = {
            "rope_type": "test_custom_rope",
            "rope_theta": 10000.0,
            "partial_rotary_factor": 1.0,
            "bands": [1.0, 2.0],
        }
        node = decode_rope(payload)
        self.assertIs(type(node), component_schema("rope", payload["rope_type"]))
        self.assertEqual(node.bands, (1.0, 2.0))
        self.assertEqual(encode_rope(node), payload)
        with self.assertRaisesRegex(ValueError, "bands"):
            decode_rope({**payload, "bands": [True]})
        with self.assertRaisesRegex(ValueError, "unknown fields"):
            decode_rope({**payload, "typo": 1})

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class Replacement(CustomRoPESpec):
            pass

        with self.assertRaisesRegex(ValueError, "already registered"):
            register_component_schema("rope", Replacement)
        with self.assertRaisesRegex(ValueError, "registered configuration node"):
            encode_rope(Replacement(**dataclasses.asdict(node)))

    def test_rope_and_logit_settings_round_trip(self):
        cases = (
            (decode_rope, encode_rope, {"rope_type": "none"}),
            (
                decode_rope,
                encode_rope,
                {
                    "rope_type": "default",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 0.25,
                },
            ),
            (
                decode_rope,
                encode_rope,
                {
                    "rope_type": "linear",
                    "rope_theta": 10000.0,
                    "partial_rotary_factor": 1.0,
                    "factor": 8.0,
                },
            ),
            (decode_target_logits, encode_target_logits, {"type": "none"}),
            (
                decode_target_logits,
                encode_target_logits,
                {"type": "mup", "multiplier": 8.0},
            ),
        )
        for decode, encode, payload in cases:
            with self.subTest(payload=payload):
                node = decode(payload)
                self.assertEqual(encode(node), payload)
                restored = decode(json.loads(json.dumps(encode(node))))
                self.assertEqual(node, restored)
                self.assertEqual(hash(node), hash(restored))
                with self.assertRaises(dataclasses.FrozenInstanceError):
                    node.new_setting = 1
                with self.assertRaisesRegex(ValueError, "unknown fields"):
                    decode({**payload, "typo": True})
                for field in payload:
                    with self.assertRaisesRegex(ValueError, field):
                        decode({k: v for k, v in payload.items() if k != field})

    def test_rotary_geometry_is_derived_and_validated(self):
        full = DefaultRoPESpec(rope_theta=10000, partial_rotary_factor=1.0)
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
                    DefaultRoPESpec(rope_theta=value, partial_rotary_factor=1.0)
                with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
                    DefaultRoPESpec(rope_theta=10000, partial_rotary_factor=value)
                with self.assertRaisesRegex(ValueError, "factor"):
                    decode_rope(
                        {
                            "rope_type": "linear",
                            "rope_theta": 10000,
                            "partial_rotary_factor": 1.0,
                            "factor": value,
                        }
                    )
        with self.assertRaisesRegex(ValueError, "partial_rotary_factor"):
            DefaultRoPESpec(rope_theta=10000, partial_rotary_factor=1.1)

    def test_unknown_formats_fail_instead_of_defaulting(self):
        for decode in (decode_rope, decode_target_logits):
            for value in (
                None,
                [],
                "default",
                {},
                {"type": []},
                {"rope_type": "unknown"},
            ):
                with self.assertRaises(ValueError):
                    decode(value)
        with self.assertRaisesRegex(ValueError, "unsupported node"):
            encode_rope({"rope_type": "none"})
        with self.assertRaisesRegex(ValueError, "unsupported node"):
            encode_target_logits(NoRoPESpec())

    def test_settings_import_without_compute_packages(self):
        result = subprocess.run(
            [
                sys.executable,
                "-S",
                "-c",
                """
import sys
from specforge.config.draft.codec.settings import decode_rope, decode_target_logits
decode_rope({'rope_type': 'none'})
decode_target_logits({'type': 'none'})
assert not any(x in sys.modules for x in ('torch', 'transformers', 'pydantic'))
""",
            ],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
