"""YaRN migrations write all effective settings, with no implicit HF defaults."""

import copy
import dataclasses
import json
import math
import unittest

from specforge.config.draft.codec import decode_config, encode_config
from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.codec.legacy.rules import resolve_rope
from tests.test_config.draft_fixtures import draft_config


def yarn_payload(**overrides):
    return {
        "rope_parameters": {
            "rope_type": "yarn",
            "rope_theta": 10000,
            "factor": 16,
            "original_max_position_embeddings": 65536,
            **overrides,
        },
        "max_position_embeddings": 1048576,
    }


class YaRNConfigTest(unittest.TestCase):
    def test_legacy_expands_all_fields_and_canonical_round_trips(self):
        source = yarn_payload()
        before = copy.deepcopy(source)
        rope = resolve_rope(source)
        self.assertEqual(rope.attention_factor, 1 + 0.1 * math.log(16))
        self.assertEqual((rope.beta_fast, rope.beta_slow, rope.truncate), (32, 1, True))
        self.assertEqual(rope.mla_qk_scale_multiplier, 1)
        node = dataclasses.replace(draft_config(), rope_parameters=rope)
        self.assertEqual(
            decode_config(json.loads(json.dumps(encode_config(node)))), node
        )
        self.assertEqual(source, before)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            rope.factor = 2

    def test_mscale_correction_and_explicit_attention_factor_are_distinct(self):
        rope = resolve_rope(yarn_payload(mscale=0.707, mscale_all_dim=0.707))
        self.assertEqual(rope.attention_factor, 1)
        self.assertAlmostEqual(
            rope.mla_qk_scale_multiplier, (1 + 0.1 * 0.707 * math.log(16)) ** 2
        )
        explicit = resolve_rope(
            yarn_payload(mscale=0.707, mscale_all_dim=0.707, attention_factor=2)
        )
        self.assertEqual(explicit.attention_factor, 2)
        self.assertEqual(explicit.mla_qk_scale_multiplier, rope.mla_qk_scale_multiplier)
        self.assertNotIn("mscale", encode_component("rope", rope))
        self.assertEqual(
            resolve_rope(
                yarn_payload(factor=1, mscale_all_dim=1)
            ).mla_qk_scale_multiplier,
            1,
        )

    def test_historical_derived_factor_and_beta_defaults(self):
        rope = resolve_rope(yarn_payload(factor=None, beta_fast=0, beta_slow=None))
        self.assertEqual(rope.factor, 16)
        self.assertEqual((rope.beta_fast, rope.beta_slow), (32, 1))
        payload = yarn_payload()
        payload["rope_parameters"].pop("original_max_position_embeddings")
        self.assertEqual(
            resolve_rope(payload).original_max_position_embeddings, 1048576
        )

    def test_canonical_fields_are_all_required_and_validated(self):
        payload = encode_component("rope", resolve_rope(yarn_payload()))
        for name in payload:
            with self.assertRaisesRegex(ValueError, name):
                decode_component(
                    "rope", {k: v for k, v in payload.items() if k != name}
                )
        for name, values in {
            "factor": (None, 0, True),
            "attention_factor": (0, float("nan")),
            "beta_fast": (0, -1),
            "beta_slow": (None, 0),
            "truncate": (None, 1),
            "mscale_all_dim": (-1, float("inf")),
            "original_max_position_embeddings": (0, 1.5),
            "rope_theta": (1, 0.5),
        }.items():
            for value in values:
                with self.assertRaisesRegex(ValueError, name):
                    decode_component("rope", payload | {name: value})
        for name in ("mscale", "mscale_all_dim", "beta_fast", "beta_slow"):
            with self.assertRaisesRegex(ValueError, name):
                resolve_rope(yarn_payload(**{name: False}))


if __name__ == "__main__":
    unittest.main()
