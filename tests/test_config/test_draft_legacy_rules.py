"""Historical defaults and aliases are isolated from canonical decoding."""

import copy
import unittest

from specforge.config.draft.codec.components import decode_component
from specforge.config.draft.codec.legacy.rules import (
    resolve_alias,
    resolve_attention_layout,
    resolve_rope,
)
from specforge.config.draft.rope import DefaultRoPESpec, LinearRoPESpec


class LegacyRulesTest(unittest.TestCase):
    def test_legacy_rope_rejects_interleaved_instead_of_overwriting_it(self):
        for field in ("rope_parameters", "rope_scaling"):
            for value in (True, False):
                with self.assertRaisesRegex(ValueError, "interleaved"):
                    resolve_rope({field: {"interleaved": value}})

    def test_aliases_reject_conflicts_instead_of_choosing_a_winner(self):
        self.assertEqual(
            resolve_alias("block_size", {"block_size": 8}, {"block_size": 8}), 8
        )
        self.assertEqual(
            resolve_alias(
                "block_size", {"block_size": None}, {"block_size": 8}, ignore_none=True
            ),
            8,
        )
        self.assertEqual(resolve_alias("block_size", {}, default=8), 8)
        for value in (16, True):
            with self.assertRaisesRegex(ValueError, "conflicting.*block_size"):
                resolve_alias("block_size", {"block_size": 1}, {"block_size": value})
        with self.assertRaisesRegex(ValueError, "missing block_size"):
            resolve_alias("block_size", {})

    def test_old_and_new_rotary_spellings_resolve_identically(self):
        expected = LinearRoPESpec(
            interleaved=False, rope_theta=1000000, partial_rotary_factor=1, factor=2
        )
        old = {"rope_theta": 1000000, "rope_scaling": {"type": "linear", "factor": 2}}
        new = {
            "rope_parameters": {
                "rope_theta": 1000000,
                "rope_type": "linear",
                "factor": 2,
            }
        }
        original = copy.deepcopy(old)
        for payload in (old, new, old | new):
            self.assertEqual(resolve_rope(payload), expected)
        self.assertEqual(old, original)
        self.assertEqual(
            resolve_rope({}),
            DefaultRoPESpec(
                interleaved=False, rope_theta=10000, partial_rotary_factor=1
            ),
        )
        self.assertEqual(
            resolve_rope({"partial_rotary_factor": 0.5}).partial_rotary_factor, 0.5
        )
        with self.assertRaisesRegex(ValueError, "missing fields"):
            decode_component("rope", {"rope_type": "default"})

    def test_rotary_conflicts_and_unsupported_settings_fail(self):
        for payload in (
            {"rope_theta": 10000, "rope_parameters": {"rope_theta": 1000000}},
            {
                "partial_rotary_factor": 0.5,
                "rope_parameters": {"partial_rotary_factor": 1},
            },
            {
                "rope_scaling": {"type": "linear", "factor": 2},
                "rope_parameters": {"rope_type": "default"},
            },
            {"rope_scaling": {"type": "linear", "rope_type": "default", "factor": 2}},
        ):
            with self.assertRaisesRegex(ValueError, "conflicting"):
                resolve_rope(payload)
        for payload in (
            {"rope_parameters": []},
            {"rope_scaling": False},
            {"rope_scaling": {"type": "unsupported", "factor": 2}},
            {"rope_parameters": {"rope_type": "default", "typo": 1}},
        ):
            with self.assertRaises(ValueError):
                resolve_rope(payload)

    def test_unset_direction_resolves_once_per_layer_kind(self):
        payload = {
            "layer_types": ["sliding_attention", "full_attention", "sliding_attention"],
            "use_sliding_window": True,
            "sliding_window": 16,
        }
        original = copy.deepcopy(payload)
        for causal in (None, True, False):
            layers, settings = resolve_attention_layout(
                payload | {"is_causal": causal}, 3
            )
            self.assertEqual(layers, tuple(payload["layer_types"]))
            self.assertEqual(
                settings["full_attention"], (False if causal is None else causal, None)
            )
            self.assertEqual(
                settings["sliding_attention"], (True if causal is None else causal, 16)
            )
        self.assertEqual(payload, original)

    def test_qwen3_layout_defaults_are_expanded(self):
        layers, settings = resolve_attention_layout({}, 2)
        self.assertEqual(layers, ("full_attention", "full_attention"))
        self.assertEqual(settings, {"full_attention": (False, None)})
        layers, settings = resolve_attention_layout(
            {"use_sliding_window": True, "max_window_layers": 1}, 3
        )
        self.assertEqual(
            layers, ("full_attention", "sliding_attention", "sliding_attention")
        )
        self.assertEqual(settings["sliding_attention"], (True, 4096))

    def test_invalid_layouts_do_not_fall_back(self):
        for payload in (
            {"layer_types": []},
            {"layer_types": "full_attention"},
            {"layer_types": ["unknown"]},
            {"layer_types": [[]]},
            {"layer_types": ["sliding_attention"]},
            {"use_sliding_window": 1},
            {"is_causal": 1},
            {"max_window_layers": True},
            {"max_window_layers": -1},
            {"use_sliding_window": True, "sliding_window": 0},
        ):
            with self.assertRaises(ValueError):
                resolve_attention_layout(payload, 1)


if __name__ == "__main__":
    unittest.main()
