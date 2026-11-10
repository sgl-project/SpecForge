"""Resolved YaRN produces the same frequencies and scales as legacy settings."""

import copy
import unittest

import torch
from transformers import Qwen3Config
from transformers.modeling_rope_utils import _compute_yarn_parameters

from specforge.config.draft.codec.components import encode_component
from specforge.config.draft.codec.legacy.rules import resolve_rope
from tests.test_config.test_draft_yarn import yarn_payload


class YaRNCodecParityTest(unittest.TestCase):
    def test_legacy_and_resolved_settings_compute_identically(self):
        for settings in (
            {},
            {"mscale": 0.707, "mscale_all_dim": 0.707},
            {"attention_factor": 2, "mscale": 0.5, "mscale_all_dim": 0.7},
            {"mscale": 0, "mscale_all_dim": 0.7},
            {"truncate": False},
            {"factor": None},
            {"beta_fast": 0, "beta_slow": None},
            {"partial_rotary_factor": 0.5},
            {"factor": 1},
        ):
            with self.subTest(settings=settings):
                payload = yarn_payload(**settings)
                legacy = Qwen3Config(head_dim=16, **copy.deepcopy(payload))
                expected_freq, expected_scale = _compute_yarn_parameters(
                    legacy, device=torch.device("cpu")
                )
                node = resolve_rope(payload)
                resolved = Qwen3Config(
                    head_dim=16,
                    max_position_embeddings=payload["max_position_embeddings"],
                    rope_parameters=encode_component("rope", node),
                )
                actual_freq, actual_scale = _compute_yarn_parameters(
                    resolved, device=torch.device("cpu")
                )
                torch.testing.assert_close(actual_freq, expected_freq, rtol=0, atol=0)
                self.assertEqual(actual_scale, expected_scale)


if __name__ == "__main__":
    unittest.main()
