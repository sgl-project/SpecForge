"""Multi-axis sections and dynamic NTK thresholds are explicit structure."""

import dataclasses
import json
import unittest

from specforge.config.draft.codec import decode_config, encode_config
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.rope import DynamicNTKRoPESpec, MultiAxisRoPESpec
from tests.test_config.draft_fixtures import draft_config


class AdditionalRoPETest(unittest.TestCase):
    def test_multiaxis_sections_round_trip_as_arrays(self):
        source = {
            "interleaved": False,
            "rope_type": "mrope",
            "rope_theta": 10000,
            "partial_rotary_factor": 1,
            "mrope_section": [1, 1, 2],
            "mrope_interleaved": False,
        }
        rope = decode_component("rope", source)
        source["mrope_section"][0] = 3
        self.assertEqual(rope.mrope_section, (1, 1, 2))
        node = draft_config(rope_parameters=rope)
        self.assertEqual(
            decode_config(json.loads(json.dumps(encode_config(node)))), node
        )
        self.assertEqual(encode_component("rope", rope)["mrope_section"], [1, 1, 2])
        self.assertEqual(rope.rotary_dim(8), 8)

    def test_multiaxis_geometry_and_element_types_are_checked(self):
        for section in ((1, 2), (0, 0, 0), (-1, 2, 3), (True, 1, 2), (1.5, 1, 2)):
            with self.assertRaisesRegex(ValueError, "mrope_section"):
                MultiAxisRoPESpec(
                    interleaved=False,
                    mrope_interleaved=False,
                    rope_theta=10000,
                    partial_rotary_factor=1,
                    mrope_section=section,
                )
        rope = MultiAxisRoPESpec(
            interleaved=False,
            mrope_interleaved=False,
            rope_theta=10000,
            partial_rotary_factor=1,
            mrope_section=(0, 0, 4),
        )
        self.assertEqual(rope.rotary_dim(8), 8)
        with self.assertRaisesRegex(ValueError, "mrope_section"):
            draft_config(
                rope_parameters=dataclasses.replace(rope, mrope_section=(1, 1, 1))
            )

    def test_dynamic_threshold_and_partial_geometry(self):
        rope = DynamicNTKRoPESpec(
            interleaved=False,
            rope_theta=10000,
            partial_rotary_factor=1,
            factor=2,
            original_max_position_embeddings=128,
        )
        self.assertAlmostEqual(rope.scaled_theta(8), 10000 * 3 ** (8 / 6))
        node = draft_config(rope_parameters=rope)
        self.assertEqual(decode_config(encode_config(node)), node)
        with self.assertRaisesRegex(ValueError, "greater than two"):
            dataclasses.replace(rope, partial_rotary_factor=0.25).rotary_dim(8)
        for field, value in (
            ("factor", 0),
            ("factor", True),
            ("original_max_position_embeddings", 0),
        ):
            with self.assertRaisesRegex(ValueError, field):
                dataclasses.replace(rope, **{field: value})

    def test_new_rope_fields_are_required_and_unknown_settings_fail(self):
        nodes = [
            MultiAxisRoPESpec(
                interleaved=False,
                mrope_interleaved=False,
                rope_theta=10000,
                partial_rotary_factor=1,
                mrope_section=(1, 1, 2),
            ),
            DynamicNTKRoPESpec(
                interleaved=False,
                rope_theta=10000,
                partial_rotary_factor=1,
                factor=2,
                original_max_position_embeddings=128,
            ),
        ]
        for node in nodes:
            payload = encode_component("rope", node)
            for field in payload:
                with self.assertRaisesRegex(ValueError, field):
                    decode_component(
                        "rope",
                        {key: value for key, value in payload.items() if key != field},
                    )
            with self.assertRaisesRegex(ValueError, "unknown"):
                decode_component("rope", payload | {"unknown": True})


if __name__ == "__main__":
    unittest.main()
