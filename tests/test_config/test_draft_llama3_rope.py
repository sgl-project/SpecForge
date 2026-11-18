"""Explicit Llama3 RoPE round-trips and preserves old/new field spellings."""

import dataclasses
import json
import unittest

from specforge.config.draft.codec import decode_config, encode_config
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.rope import Llama3RoPESpec
from tests.test_config.draft_fixtures import draft_config


def llama3_rope():
    return Llama3RoPESpec(
        interleaved=False,
        rope_theta=500000,
        partial_rotary_factor=1.0,
        factor=8,
        low_freq_factor=1,
        high_freq_factor=4,
        original_max_position_embeddings=8192,
    )


class Llama3RoPEConfigTest(unittest.TestCase):
    def test_bundled_recipe_rotary_settings_round_trip(self):
        rope = llama3_rope()
        self.assertIsInstance(rope, Llama3RoPESpec)
        self.assertEqual(rope.factor, 8)
        self.assertEqual(rope.rope_theta, 500000)
        self.assertEqual(rope.rotary_dim(128), 128)
        node = draft_config(rope_parameters=rope)
        restored = decode_config(json.loads(json.dumps(encode_config(node))))
        self.assertEqual(restored, node)
        self.assertEqual(hash(restored), hash(node))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rope.factor = 2

    def test_per_attention_override_and_partial_rotation(self):
        rope = dataclasses.replace(llama3_rope(), partial_rotary_factor=0.5)
        base = draft_config()
        attention = dataclasses.replace(
            base.attention_configs["full"], rope_parameters=rope
        )
        node = dataclasses.replace(base, attention_configs={"full": attention})
        self.assertEqual(
            attention.effective_rope(node.rope_parameters).rotary_dim(8), 4
        )
        self.assertEqual(decode_config(encode_config(node)), node)
        self.assertEqual(node.rope_parameters, base.rope_parameters)

    def test_required_fields_and_frequency_boundaries(self):
        payload = encode_component("rope", llama3_rope())
        for name in payload:
            with self.assertRaisesRegex(ValueError, name):
                decode_component(
                    "rope",
                    {key: value for key, value in payload.items() if key != name},
                )
        for name, values in {
            "factor": (0, True, None, float("nan")),
            "low_freq_factor": (0, -1, float("inf")),
            "high_freq_factor": (0, 0.5, False),
            "original_max_position_embeddings": (0, 1.5, True),
        }.items():
            for value in values:
                with (
                    self.subTest(name=name, value=value),
                    self.assertRaisesRegex(ValueError, name),
                ):
                    decode_component("rope", payload | {name: value})
        step = decode_component("rope", payload | {"high_freq_factor": 1})
        self.assertEqual(step.high_freq_factor, step.low_freq_factor)
        with self.assertRaisesRegex(ValueError, "attention_factor"):
            decode_component("rope", payload | {"attention_factor": 1})


if __name__ == "__main__":
    unittest.main()
