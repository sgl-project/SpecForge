"""SGLang dynamic NTK is fixed at factor times the original context length."""

import unittest

import torch

from specforge.config.draft.rope import DynamicNTKRoPESpec
from specforge.modeling.draft.layers.rope import build_rotary


class DynamicRotaryTest(unittest.TestCase):
    def test_fixed_frequencies_survive_short_long_and_meta_calls(self):
        node = DynamicNTKRoPESpec(
            interleaved=False,
            rope_theta=10000.0,
            partial_rotary_factor=0.5,
            factor=2.5,
            original_max_position_embeddings=128,
        )
        with torch.device("meta"):
            module = build_rotary(node, head_dim=128)
        module.to_empty(device="cpu").bfloat16()
        hidden = torch.empty(1, 3, 128, dtype=torch.bfloat16)
        positions = torch.tensor([[0, 127, 65535]])
        expected = module(hidden, positions)
        module(hidden, torch.tensor([[131071]]))
        torch.testing.assert_close(module(hidden, positions), expected, rtol=0, atol=0)
        self.assertTrue(all(t.dtype == torch.float32 for t in expected))
        self.assertEqual(module.state_dict(), {})
        self.assertEqual(module(hidden, positions[:, :0])[0].shape, (1, 0, 64))
        with self.assertRaisesRegex(ValueError, "position_ids"):
            module(hidden, positions.flatten())
