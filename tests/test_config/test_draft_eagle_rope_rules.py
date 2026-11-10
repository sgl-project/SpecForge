"""EAGLE rotary conversion preserves effective settings instead of HF defaults."""

import copy
import json
import math
import unittest

from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.codec.legacy.eagle_rope import resolve_eagle_rope
from tests.test_config.test_draft_legacy_dflash import CONFIGS


def yarn_settings(**overrides):
    return (
        dict(
            rope_type="yarn",
            factor=40,
            original_max_position_embeddings=4096,
            beta_fast=32,
            beta_slow=1,
            mscale=0.707,
            mscale_all_dim=0.707,
        )
        | overrides
    )


class EagleRoPERulesTest(unittest.TestCase):
    def test_all_bundled_eagle_rotary_settings_round_trip(self):
        count = 0
        for path in CONFIGS.glob("*.json"):
            payload = json.loads(path.read_text())
            if payload.get("architectures") != ["LlamaForCausalLMEagle3"]:
                continue
            count += 1
            before = copy.deepcopy(payload)
            node = resolve_eagle_rope(payload)
            node.rotary_dim(
                payload.get(
                    "head_dim", payload["hidden_size"] // payload["num_attention_heads"]
                )
            )
            self.assertEqual(
                decode_component(
                    "rope", json.loads(json.dumps(encode_component("rope", node)))
                ),
                node,
            )
            self.assertEqual(payload, before)
        self.assertGreaterEqual(count, 27)

    def test_yarn_ratio_and_unused_fields_follow_eagle(self):
        values = yarn_settings() | {"mscale": 0, "mscale_all_dim": 0}
        node = resolve_eagle_rope({"rope_scaling": values})
        self.assertEqual(node.attention_factor, 1)
        values.update(mscale=1, mscale_all_dim=0, attention_factor=99, truncate=False)
        node = resolve_eagle_rope(
            {"rope_scaling": values, "partial_rotary_factor": 0.5}
        )
        self.assertAlmostEqual(node.attention_factor, 1 + 0.1 * math.log(40))
        self.assertEqual(node.mla_qk_scale_multiplier, 1)
        self.assertTrue(node.truncate)
        self.assertEqual(node.partial_rotary_factor, 1)

    def test_copied_multimodal_fields_follow_effective_rope_type(self):
        for params in (
            dict(
                rope_type="default",
                rope_theta=1e7,
                partial_rotary_factor=0.25,
                mrope_section=[11, 11, 10],
                mrope_interleaved=True,
            ),
            dict(
                type="mrope",
                rope_type="default",
                rope_theta=1e6,
                mrope_section=[16, 24, 24],
            ),
        ):
            node = resolve_eagle_rope({"rope_parameters": params})
            self.assertEqual(node.rope_type, "default")
            self.assertEqual(node.rope_theta, params["rope_theta"])
            self.assertEqual(node.partial_rotary_factor, 1)
        for extra in ({"mrope_interleaved": True}, {"factor": 2}):
            with self.assertRaisesRegex(ValueError, "does not support"):
                resolve_eagle_rope(
                    {
                        "rope_scaling": dict(
                            rope_type="mrope", mrope_section=[16, 24, 24]
                        )
                        | extra
                    }
                )

    def test_context_defaults_and_unsupported_dynamic(self):
        for params in (
            dict(rope_type="llama3", factor=8, low_freq_factor=1, high_freq_factor=4),
            {
                k: v
                for k, v in yarn_settings().items()
                if k != "original_max_position_embeddings"
            },
        ):
            payload = {"max_position_embeddings": 64, "rope_scaling": params}
            node = resolve_eagle_rope(payload)
            self.assertEqual(node.rope_type, params["rope_type"])
            self.assertEqual(node.original_max_position_embeddings, 64)
            self.assertEqual(
                node,
                resolve_eagle_rope(
                    payload
                    | {
                        "rope_scaling": params
                        | {"original_max_position_embeddings": 64}
                    }
                ),
            )
        with self.assertRaisesRegex(ValueError, "unsupported.*dynamic"):
            resolve_eagle_rope({"rope_scaling": {"type": "dynamic", "factor": 2}})
        for params in (
            {"rope_type": "llama3", "factor": 8},
            {
                "rope_type": "llama3",
                "low_freq_factor": 1,
                "high_freq_factor": 4,
            },
        ):
            with self.assertRaisesRegex(ValueError, "requires"):
                resolve_eagle_rope({"rope_scaling": params})

    def test_invalid_executed_fields_and_alias_conflicts_fail(self):
        for settings in (
            {"rope_type": "linear"},
            {"rope_type": "other"},
            {"rope_type": "default", "typo": 1},
            yarn_settings() | {"beta_fast": 0},
            yarn_settings() | {"mscale": None},
        ):
            with self.assertRaises(ValueError):
                resolve_eagle_rope({"rope_scaling": settings})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            resolve_eagle_rope(
                {
                    "rope_scaling": {
                        "type": "linear",
                        "rope_type": "dynamic",
                        "factor": 2,
                    }
                }
            )
        with self.assertRaisesRegex(ValueError, "conflicting"):
            resolve_eagle_rope(
                {
                    "rope_scaling": {"rope_type": "linear", "factor": 2},
                    "rope_parameters": {"rope_type": "linear", "factor": 4},
                }
            )


if __name__ == "__main__":
    unittest.main()
