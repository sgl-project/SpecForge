"""Attention-node constraints cannot widen the algorithm's visibility policy."""

import dataclasses
import unittest

import torch

from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    PaddingMask,
    dense_attention_mask,
)


class AttentionMaskTest(unittest.TestCase):
    def test_block_layout_matches_existing_causal_and_window_rules(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device in devices:
            anchors = torch.tensor([[2, 6], [1, 4]], device=device)
            keep = torch.tensor([[True, True], [True, False]], device=device)
            positions = (
                anchors.unsqueeze(-1) + torch.arange(3, device=device)
            ).flatten(1)
            context = torch.arange(8, device=device).expand(2, -1)
            base = create_dflash_sdpa_mask(anchors, keep, 8, 3, device, is_causal=False)
            before = base.clone()
            layout = AttentionLayout(
                query_positions=positions,
                key_positions=torch.cat((context, positions), dim=1),
                visibility=base,
            )
            for causal in (False, True):
                for window in (None, 1, 3, 8):
                    with self.subTest(device=device, causal=causal, window=window):
                        actual = dense_attention_mask(
                            layout, is_causal=causal, sliding_window=window
                        )
                        expected = create_dflash_sdpa_mask(
                            anchors,
                            keep,
                            8,
                            3,
                            device,
                            is_causal=causal,
                            sliding_window=window,
                        )
                        self.assertTrue(torch.equal(actual, expected))
                        self.assertFalse((actual & ~base).any())
            self.assertTrue(torch.equal(base, before))
            with self.assertRaises(dataclasses.FrozenInstanceError):
                layout.visibility = before

    def test_positions_not_tensor_indices_control_causality_and_windows(self):
        layout = AttentionLayout(
            query_positions=torch.tensor([[10, 14]]),
            key_positions=torch.tensor([[14, 9, 10, 15]]),
            visibility=torch.ones(1, 1, 2, 4, dtype=torch.bool),
        )
        causal = dense_attention_mask(layout, is_causal=True, sliding_window=2)
        self.assertEqual(
            causal.tolist(),
            [[[[False, True, True, False], [True, False, False, False]]]],
        )
        both = dense_attention_mask(layout, is_causal=False, sliding_window=2)
        self.assertEqual(
            both.tolist(), [[[[False, True, True, False], [True, False, False, True]]]]
        )

    def test_head_specific_visibility_and_batch_broadcast_are_preserved(self):
        positions = torch.tensor([[0, 1], [2, 1]])
        visibility = torch.ones(1, 2, 2, 2, dtype=torch.bool)
        visibility[:, 1, :, 0] = False
        layout = AttentionLayout(
            query_positions=positions, key_positions=positions, visibility=visibility
        )
        actual = dense_attention_mask(layout, is_causal=True, sliding_window=None)
        self.assertEqual(actual.shape, (2, 2, 2, 2))
        self.assertFalse(actual[:, 1, :, 0].any())
        self.assertTrue(actual[1, 0, 0, 1])
        self.assertFalse(actual[0, 0, 0, 1])

    def test_shared_positions_broadcast_with_per_example_visibility(self):
        positions = torch.arange(3).unsqueeze(0)
        visibility = torch.ones(2, 1, 3, 3, dtype=torch.bool)
        visibility[1, :, :, 0] = False
        for q, k in (
            (positions, positions),
            (positions.expand(2, -1), positions),
            (positions, positions.expand(2, -1)),
        ):
            layout = AttentionLayout(
                query_positions=q, key_positions=k, visibility=visibility
            )
            actual = dense_attention_mask(layout, is_causal=True, sliding_window=None)
            expected = visibility & torch.ones(3, 3, dtype=torch.bool).tril()
            torch.testing.assert_close(actual, expected)

    def test_padding_layout_validation_and_head_sharing(self):
        positions = torch.arange(3)[None]
        valid = torch.ones(1, 3, dtype=torch.bool)

        def layout(query, key):
            return AttentionLayout(
                query_positions=positions,
                key_positions=positions,
                visibility=PaddingMask(query, key),
                active_queries=valid,
            )

        for query, key in (
            (valid[:, :2], valid),
            (valid, torch.ones(1, 4, dtype=torch.bool)),
            (valid.to("meta"), valid.to("meta")),
        ):
            with self.assertRaisesRegex(ValueError, "lengths and device"):
                layout(query, key)
        with self.assertRaisesRegex(ValueError, "batch"):
            layout(valid, valid.expand(2, -1)).validate_dimensions(4, 8)
        mask = dense_attention_mask(
            layout(valid, valid),
            batch_size=4,
            num_heads=8,
            is_causal=False,
            sliding_window=None,
        )
        self.assertEqual(mask.shape, (4, 1, 3, 3))

    def test_invalid_layouts_and_ambiguous_flags_are_rejected(self):
        values = dict(
            query_positions=torch.zeros(1, 2, dtype=torch.long),
            key_positions=torch.zeros(1, 3, dtype=torch.long),
            visibility=torch.ones(1, 1, 2, 3, dtype=torch.bool),
        )
        for override in (
            {"query_positions": torch.zeros(1, 2)},
            {
                "query_positions": torch.zeros(2, 2, dtype=torch.long),
                "key_positions": torch.zeros(3, 3, dtype=torch.long),
            },
            {"visibility": torch.ones(1, 1, 2, 3)},
            {"visibility": torch.ones(1, 1, 1, 3, dtype=torch.bool)},
            {"visibility": torch.ones(1, 1, 2, 3, dtype=torch.bool, device="meta")},
        ):
            with self.assertRaises(ValueError):
                AttentionLayout(**(values | override))
        layout = AttentionLayout(**values)
        for causal, window in ((None, None), (1, None), (True, 0), (False, True)):
            with self.assertRaises(ValueError):
                dense_attention_mask(layout, is_causal=causal, sliding_window=window)


if __name__ == "__main__":
    unittest.main()
