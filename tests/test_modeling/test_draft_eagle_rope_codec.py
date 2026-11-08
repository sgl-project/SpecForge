"""EAGLE's resolved rotary nodes reproduce its existing numerical primitives."""

import unittest

import torch

from specforge.config.draft.codec.legacy.eagle_rope import resolve_eagle_rope
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.llama3_eagle import (
    LlamaAttention,
    LlamaDynamicNTKScalingRotaryEmbedding,
    LlamaLinearScalingRotaryEmbedding,
    LlamaMutiRotaryEmbedding,
    LlamaRotaryEmbedding,
    LlamaYarnRotaryEmbedding,
    apply_multimodal_rotary_pos_emb,
)
from tests.test_config.test_draft_eagle_rope_rules import yarn_settings


def rotary_from_node(node, head_dim=128, max_positions=64):
    common = dict(
        dim=node.rotary_dim(head_dim),
        max_position_embeddings=max_positions,
        base=node.rope_theta,
    )
    if node.rope_type == "default":
        return LlamaRotaryEmbedding(**common)
    if node.rope_type == "linear":
        return LlamaLinearScalingRotaryEmbedding(**common, scaling_factor=node.factor)
    if node.rope_type == "dynamic":
        common["max_position_embeddings"] = node.original_max_position_embeddings
        return LlamaDynamicNTKScalingRotaryEmbedding(
            **common, scaling_factor=node.factor
        )
    if node.rope_type == "llama3":
        return LlamaRotaryEmbedding(
            **common,
            scaling_factor=node.factor,
            low_freq_factor=node.low_freq_factor,
            high_freq_factor=node.high_freq_factor,
            orig_max_position=node.original_max_position_embeddings,
        )
    if node.rope_type == "mrope":
        return LlamaMutiRotaryEmbedding(**common)
    if node.rope_type == "yarn":
        module = LlamaYarnRotaryEmbedding(
            **common,
            scaling_factor=node.factor,
            original_max_position_embeddings=node.original_max_position_embeddings,
            beta_fast=node.beta_fast,
            beta_slow=node.beta_slow,
            mscale=0,
            mscale_all_dim=0,
        )
        module.cos_cached *= node.attention_factor
        module.sin_cached *= node.attention_factor
        return module
    raise AssertionError(node.rope_type)


class EagleRoPECodecParityTest(unittest.TestCase):
    def test_inverse_frequencies_caches_and_axis_selection_match(self):
        for scaling in (
            None,
            {"rope_type": "linear", "factor": 2},
            {"rope_type": "dynamic", "factor": 2},
            {
                "rope_type": "llama3",
                "factor": 8,
                "low_freq_factor": 1,
                "high_freq_factor": 4,
                "original_max_position_embeddings": 4096,
            },
            {"type": "mrope", "mrope_section": [16, 24, 24]},
            yarn_settings(),
            yarn_settings(mscale=0, mscale_all_dim=0),
            yarn_settings(mscale=1, mscale_all_dim=0),
        ):
            with self.subTest(scaling=scaling):
                payload = dict(
                    architectures=["LlamaForCausalLMEagle3"],
                    hidden_size=32,
                    num_attention_heads=2,
                    num_key_value_heads=1,
                    head_dim=128,
                    max_position_embeddings=64,
                    rope_scaling=scaling,
                    rope_theta=10000,
                    partial_rotary_factor=0.5,
                )
                expected = LlamaAttention(AutoDraftConfig.from_dict(payload)).rotary_emb
                node = resolve_eagle_rope(payload)
                actual = rotary_from_node(node)
                for field in ("inv_freq", "cos_cached", "sin_cached"):
                    torch.testing.assert_close(
                        getattr(actual, field), getattr(expected, field), rtol=0, atol=0
                    )
                if node.rope_type == "dynamic":
                    for module in (expected, actual):
                        module._set_cos_sin_cache(
                            128, torch.device("cpu"), torch.float32
                        )
                    torch.testing.assert_close(
                        actual.inv_freq, expected.inv_freq, rtol=0, atol=0
                    )
                    frequencies = 1 / (
                        node.scaled_theta(128, 128)
                        ** (torch.arange(0, 128, 2).float() / 128)
                    )
                    torch.testing.assert_close(
                        actual.inv_freq, frequencies, rtol=0, atol=0
                    )
                if node.rope_type == "mrope":
                    query, key = torch.randn(1, 2, 4, 128), torch.randn(1, 1, 4, 128)
                    positions = torch.arange(12).reshape(3, 1, 4)
                    left = apply_multimodal_rotary_pos_emb(
                        query,
                        key,
                        *expected(query, positions),
                        scaling["mrope_section"],
                    )
                    right = apply_multimodal_rotary_pos_emb(
                        query, key, *actual(query, positions), list(node.mrope_section)
                    )
                    torch.testing.assert_close(left, right, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
