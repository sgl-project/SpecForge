"""Llama3 settings preserve both EAGLE and HF rotary computations."""

import copy
import unittest

import torch
from transformers import LlamaConfig
from transformers.modeling_rope_utils import _compute_llama3_parameters

from specforge.config.draft.codec.components import encode_component
from specforge.config.draft.codec.legacy.rules import resolve_rope
from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding
from tests.test_config.test_draft_llama3_rope import llama3_payload


class Llama3RoPECodecParityTest(unittest.TestCase):
    def test_eagle_frequencies_and_cached_rotations_are_bit_identical(self):
        for overrides in ({}, {"factor": 1}, {"high_freq_factor": 1}):
            with self.subTest(overrides=overrides):
                payload = llama3_payload()
                payload["rope_scaling"].update(overrides)
                raw = payload["rope_scaling"]
                expected = LlamaRotaryEmbedding(
                    dim=128,
                    max_position_embeddings=64,
                    base=payload["rope_theta"],
                    scaling_factor=raw["factor"],
                    low_freq_factor=raw["low_freq_factor"],
                    high_freq_factor=raw["high_freq_factor"],
                    orig_max_position=raw["original_max_position_embeddings"],
                )
                node = resolve_rope(payload)
                actual = LlamaRotaryEmbedding(
                    dim=node.rotary_dim(128),
                    max_position_embeddings=64,
                    base=node.rope_theta,
                    scaling_factor=node.factor,
                    low_freq_factor=node.low_freq_factor,
                    high_freq_factor=node.high_freq_factor,
                    orig_max_position=node.original_max_position_embeddings,
                )
                for name in ("inv_freq", "cos_cached", "sin_cached"):
                    torch.testing.assert_close(
                        getattr(actual, name), getattr(expected, name), rtol=0, atol=0
                    )

    def test_hf_frequencies_preserve_partial_and_full_rotary_dimensions(self):
        for partial in (1.0, 0.5):
            payload = llama3_payload()
            payload["partial_rotary_factor"] = partial
            legacy = LlamaConfig.from_dict(copy.deepcopy(payload))
            node = resolve_rope(payload)
            resolved = LlamaConfig(
                head_dim=128, rope_parameters=encode_component("rope", node)
            )
            expected, expected_scale = _compute_llama3_parameters(
                legacy, device=torch.device("cpu")
            )
            actual, actual_scale = _compute_llama3_parameters(
                resolved, device=torch.device("cpu")
            )
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(actual_scale, expected_scale)


if __name__ == "__main__":
    unittest.main()
