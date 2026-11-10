"""MLA carries its own projection geometry rather than synthetic GQA fields."""

import dataclasses
import json
import unittest

from specforge.config.draft.attention import MLASpec
from specforge.config.draft.codec.components import decode_component, encode_component

MLA = {
    "type": "mla",
    "rope_parameters": None,
    "num_attention_heads": 8,
    "q_lora_rank": None,
    "kv_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 8,
    "v_head_dim": 16,
    "attention_bias": False,
    "is_causal": False,
    "sliding_window": None,
    "attention_dropout": 0.0,
}


class MLASpecTest(unittest.TestCase):
    def test_projection_variants_round_trip(self):
        for rank in (None, 16):
            for interleaved in (False, True):
                payload = {
                    **MLA,
                    "q_lora_rank": rank,
                    "rope_parameters": {
                        "rope_type": "default",
                        "rope_theta": 10000.0,
                        "partial_rotary_factor": 1.0,
                        "interleaved": interleaved,
                    },
                }
                with self.subTest(rank=rank, interleaved=interleaved):
                    node = decode_component("attention", payload)
                    self.assertIsInstance(node, MLASpec)
                    self.assertEqual(node.qk_head_dim, 24)
                    self.assertFalse(hasattr(node, "head_dim"))
                    self.assertFalse(hasattr(node, "num_key_value_heads"))
                    self.assertEqual(encode_component("attention", node), payload)
                    restored = decode_component(
                        "attention", json.loads(json.dumps(payload))
                    )
                    self.assertEqual(node, restored)
                    self.assertEqual(hash(node), hash(restored))
                    with self.assertRaises(dataclasses.FrozenInstanceError):
                        node.rope_parameters.interleaved = not interleaved

    def test_rotary_and_nonrotary_dimensions(self):
        node = decode_component("attention", {**MLA, "qk_nope_head_dim": 0})
        self.assertEqual(node.qk_head_dim, node.qk_rope_head_dim)
        for value in (-1, True, 2.0, "2", None):
            with self.assertRaisesRegex(ValueError, "qk_nope_head_dim"):
                decode_component("attention", {**MLA, "qk_nope_head_dim": value})
        with self.assertRaisesRegex(ValueError, "qk_rope_head_dim must be even"):
            decode_component("attention", {**MLA, "qk_rope_head_dim": 3})

    def test_invalid_ranks_and_dimensions(self):
        for name in (
            "num_attention_heads",
            "q_lora_rank",
            "kv_lora_rank",
            "qk_rope_head_dim",
            "v_head_dim",
        ):
            for value in (False, 0, -1, 2.0, "2"):
                with self.subTest(name=name, value=value):
                    with self.assertRaisesRegex(ValueError, name):
                        decode_component("attention", {**MLA, name: value})

    def test_visibility_and_dropout_validation(self):
        for name in ("attention_bias", "is_causal"):
            for value in (None, 0, "false"):
                with self.assertRaisesRegex(ValueError, name):
                    decode_component("attention", {**MLA, name: value})
        for value in (True, 0, -1, "128"):
            with self.assertRaisesRegex(ValueError, "sliding_window"):
                decode_component("attention", {**MLA, "sliding_window": value})
        for value in (True, -0.1, 1, float("nan"), float("inf")):
            with self.assertRaisesRegex(ValueError, "attention_dropout"):
                decode_component("attention", {**MLA, "attention_dropout": value})
        for causal in (False, True):
            for window in (None, 128):
                node = decode_component(
                    "attention", {**MLA, "is_causal": causal, "sliding_window": window}
                )
                self.assertIs(node.is_causal, causal)
                self.assertEqual(node.sliding_window, window)

    def test_is_causal_is_declared_as_the_causal_capability(self):
        for causal in (False, True):
            node = decode_component("attention", {**MLA, "is_causal": causal})
            self.assertIn("target_context_kv", node.capabilities())
            self.assertIs("causal" in node.capabilities(), causal)

    def test_no_implicit_projection_or_rotation_defaults(self):
        for field in MLA:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, field):
                decode_component(
                    "attention",
                    {name: value for name, value in MLA.items() if name != field},
                )
        for field, value in (
            ("head_dim", 24),
            ("num_key_value_heads", 2),
            ("qk_norm", True),
            ("output_gate", True),
        ):
            with self.assertRaisesRegex(ValueError, field):
                decode_component("attention", {**MLA, field: value})


if __name__ == "__main__":
    unittest.main()
