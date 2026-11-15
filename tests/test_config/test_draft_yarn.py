"""YaRN migrations write all effective settings, with no implicit HF defaults."""

import unittest

from specforge.config.draft.codec.components import decode_component, encode_component
from specforge.config.draft.rope import YaRNSpec


class YaRNConfigTest(unittest.TestCase):

    def test_canonical_fields_are_all_required_and_validated(self):
        payload = encode_component("rope", yarn_node())
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


def yarn_node():
    return YaRNSpec(
        interleaved=False,
        rope_theta=10000.0,
        partial_rotary_factor=1.0,
        factor=16.0,
        original_max_position_embeddings=4096,
        attention_factor=1.0,
        beta_fast=32.0,
        beta_slow=1.0,
        truncate=True,
        mscale_all_dim=0.707,
    )
