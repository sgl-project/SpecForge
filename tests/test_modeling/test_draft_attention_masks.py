"""Attention-node constraints cannot widen the algorithm's visibility policy."""

import dataclasses
import unittest
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask, create_mask

from specforge.modeling.draft.layers.attention.backends import (
    AttentionPreparationScope,
    EagerAttentionBackend,
    SDPAAttentionBackend,
)
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    ContextBlocks,
    PaddingMask,
    dense_attention_mask,
)


def create_dflash_sdpa_mask(
    anchors, keep, context, size, device, *, is_causal, sliding_window=None
):
    batch, blocks = anchors.shape
    result = torch.zeros(
        batch,
        1,
        blocks * size,
        context + blocks * size,
        dtype=torch.bool,
        device=device,
    )
    for b in range(batch):
        for q in range(blocks * size):
            position = int(anchors[b, q // size]) + q % size
            for k in range(context + blocks * size):
                key = (
                    k
                    if k < context
                    else int(anchors[b, (k - context) // size]) + (k - context) % size
                )
                visible = (
                    k < int(anchors[b, q // size])
                    if k < context
                    else (k - context) // size == q // size
                )
                visible = visible and bool(keep[b, q // size])
                visible = visible and (not is_causal or key <= position)
                visible = visible and (
                    sliding_window is None or abs(key - position) < sliding_window
                )
                result[b, 0, q, k] = visible
    return result


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

    def test_windows_at_or_beyond_the_position_dtype_limit_keep_every_key(self):
        """A window at or past the position dtype's limit must not overflow or wrap."""
        valid = torch.ones(1, 4, dtype=torch.bool)
        for dtype in (torch.int32, torch.int64):
            positions = torch.arange(4, dtype=dtype)[None]
            for visibility, active in (
                (torch.ones(1, 1, 4, 4, dtype=torch.bool), None),
                (PaddingMask(valid, valid), valid),
            ):
                layout = AttentionLayout(
                    query_positions=positions,
                    key_positions=positions,
                    visibility=visibility,
                    active_queries=active,
                )
                for window in (torch.iinfo(dtype).max, torch.iinfo(dtype).max + 1):
                    with self.subTest(
                        dtype=dtype, mask_mod=active is not None, window=window
                    ):
                        masks = [
                            dense_attention_mask(
                                layout,
                                is_causal=True,
                                sliding_window=value,
                                batch_size=1,
                                num_heads=1,
                            )
                            for value in (window, None)
                        ]
                        self.assertTrue(torch.equal(*masks))

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


def prepare_blocks_visibility(anchors, keep, padding, size, offset):
    """The mask `prepare_blocks` built inline before ContextBlocks existed."""
    batch, count = anchors.shape
    rows = padding.shape[1]
    context = torch.arange(rows).expand(batch, -1)
    context_visible = context[:, None, None, :] < (anchors + offset)[..., None, None]
    context_visible = context_visible & padding[:, None, None, :].bool()
    context_visible = context_visible.expand(-1, -1, size, -1).reshape(batch, -1, rows)
    block_ids = torch.arange(count).repeat_interleave(size)
    draft_visible = (block_ids[:, None] == block_ids[None, :]).expand(batch, -1, -1)
    visible = torch.cat((context_visible, draft_visible), dim=-1)
    visible &= keep.repeat_interleave(size, dim=1)[..., None]
    return visible[:, None], keep.repeat_interleave(size, dim=1)


class ContextBlocksTest(unittest.TestCase):
    def test_block_layout_equals_the_inline_prepare_blocks_mask(self):
        generator = torch.Generator().manual_seed(0)
        for trial in range(40):
            batch = 1 + trial % 3
            rows = 1 + int(torch.randint(1, 40, (1,), generator=generator))
            count = 1 + trial % 5
            size = 1 + trial % 4
            # DMTP shifts the visible prefix by input and context offsets.
            offset = (-3, 0, 1, 2, 5)[trial % 5]
            anchors = torch.randint(0, rows, (batch, count), generator=generator)
            keep = torch.rand(batch, count, generator=generator) < 0.7
            valid = torch.randint(1, rows + 1, (batch, 1), generator=generator)
            padding = torch.arange(rows) < valid
            if trial % 4 == 0:
                padding = torch.rand(batch, rows, generator=generator) < 0.8
            with self.subTest(trial=trial):
                queries = (anchors[..., None] + torch.arange(size)).flatten(1)
                blocks = ContextBlocks(
                    block_size=size,
                    context_lengths=(anchors + offset).clamp(0, rows),
                    context_valid=padding,
                    active_blocks=keep,
                )
                layout = AttentionLayout(
                    query_positions=queries,
                    key_positions=torch.cat(
                        (torch.arange(rows).expand(batch, -1), queries), dim=1
                    ),
                    context_blocks=blocks,
                )
                visible, active = prepare_blocks_visibility(
                    anchors, keep, padding, size, offset
                )
                self.assertIs(layout.visibility, blocks)
                self.assertTrue(
                    torch.equal(
                        dense_attention_mask(
                            layout, is_causal=False, sliding_window=None
                        ),
                        visible,
                    )
                )
                self.assertTrue(
                    torch.equal(
                        create_mask(
                            layout.mask_mod(is_causal=False, sliding_window=None),
                            batch,
                            layout.mask_heads(4),
                            queries.shape[1],
                            rows + queries.shape[1],
                            device=queries.device,
                        ),
                        visible,
                    )
                )
                self.assertTrue(torch.equal(layout.active_queries, active))

    def test_block_layouts_derive_visibility_from_one_description(self):
        blocks = ContextBlocks(
            block_size=2,
            context_lengths=torch.tensor([[1, 3]]),
            context_valid=torch.ones(1, 4, dtype=torch.bool),
            active_blocks=torch.tensor([[True, False]]),
        )
        queries = torch.tensor([[1, 2, 3, 4]])
        keys = torch.cat((torch.arange(4)[None], queries), dim=1)
        layout = AttentionLayout(
            query_positions=queries, key_positions=keys, context_blocks=blocks
        )
        self.assertIs(layout.context_blocks, blocks)
        for override in (
            {"visibility": layout.visibility},
            {"active_queries": layout.active_queries},
        ):
            with (
                self.subTest(override=sorted(override)),
                self.assertRaisesRegex(ValueError, "context_blocks determine"),
            ):
                AttentionLayout(
                    query_positions=queries,
                    key_positions=keys,
                    context_blocks=blocks,
                    **override,
                )
        with self.assertRaisesRegex(ValueError, "context-then-query"):
            AttentionLayout(
                query_positions=queries,
                key_positions=keys[:, 1:],
                context_blocks=blocks,
            )
        with self.assertRaisesRegex(ValueError, "visibility or context_blocks"):
            AttentionLayout(query_positions=queries, key_positions=keys)
        with self.assertRaisesRegex(ValueError, "active_blocks"):
            dataclasses.replace(
                blocks, active_blocks=torch.ones(1, 3, dtype=torch.bool)
            )
        with self.assertRaisesRegex(ValueError, "context_lengths"):
            dataclasses.replace(blocks, context_lengths=torch.ones(1, 2))

    def test_context_blocks_stays_lazy_until_dense_materialization(self):
        blocks = ContextBlocks(
            block_size=3,
            context_lengths=torch.tensor([[2, 5], [1, 4]]),
            context_valid=torch.tensor([[True, True, False, True, True, True]]),
            active_blocks=torch.tensor([[True, True], [True, False]]),
        )
        queries = torch.tensor([[2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]])
        keys = torch.cat((torch.arange(6).expand(2, -1), queries), dim=1)
        with patch.object(
            ContextBlocks,
            "compute_visibility",
            side_effect=AssertionError("compute_visibility should not be called"),
        ):
            layout = AttentionLayout(
                query_positions=queries, key_positions=keys, context_blocks=blocks
            )
            self.assertIs(layout.visibility, blocks)
            self.assertIsNone(layout.mask_heads(8))
            mask_mod = layout.mask_mod(is_causal=True, sliding_window=4)
            mask = create_mask(
                mask_mod,
                2,
                layout.mask_heads(8),
                queries.shape[1],
                keys.shape[1],
                device=queries.device,
            )
            block_mask = create_block_mask(
                mask_mod,
                2,
                layout.mask_heads(8),
                queries.shape[1],
                keys.shape[1],
                device=queries.device,
            )
            self.assertEqual(block_mask.shape, (2, 1, queries.shape[1], keys.shape[1]))
        expected = dense_attention_mask(layout, is_causal=True, sliding_window=4)
        self.assertTrue(torch.equal(mask, expected))
        with self.assertRaisesRegex(ValueError, "batch"):
            layout.validate_dimensions(4, 8)

    def test_context_blocks_mask_mod_matches_dense_across_broadcast_and_constraints(
        self,
    ):
        cases = (
            dict(
                block_size=3,
                context_lengths=torch.tensor([[2, 5], [0, 3]]),
                context_valid=torch.tensor([[True, False, True, True, True, False]]),
                active_blocks=torch.tensor([[True, True], [True, False]]),
                query_batch=2,
            ),
            dict(
                block_size=2,
                context_lengths=torch.tensor([[1, 4]]),
                context_valid=torch.tensor([[True, True, False, True, True]]),
                active_blocks=torch.tensor([[True, False]]),
                query_batch=2,
            ),
            dict(
                block_size=2,
                context_lengths=torch.tensor([[0, 0]]),
                context_valid=torch.ones(1, 0, dtype=torch.bool),
                active_blocks=torch.tensor([[True, True]]),
                query_batch=1,
            ),
        )
        for index, case in enumerate(cases):
            blocks = ContextBlocks(
                block_size=case["block_size"],
                context_lengths=case["context_lengths"],
                context_valid=case["context_valid"],
                active_blocks=case["active_blocks"],
            )
            batch = case["query_batch"]
            num_context = blocks.num_context_rows
            num_queries = blocks.num_queries
            queries = torch.arange(num_queries)[None] + torch.arange(batch)[:, None]
            keys = torch.cat(
                (torch.arange(num_context).expand(batch, -1), queries), dim=1
            )
            layout = AttentionLayout(
                query_positions=queries, key_positions=keys, context_blocks=blocks
            )
            for is_causal, sliding_window in ((False, None), (True, 3)):
                with self.subTest(
                    case=index,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                ):
                    mask_mod = layout.mask_mod(
                        is_causal=is_causal, sliding_window=sliding_window
                    )
                    from_mod = create_mask(
                        mask_mod,
                        batch,
                        layout.mask_heads(4),
                        num_queries,
                        keys.shape[1],
                        device=queries.device,
                    )
                    from_dense = dense_attention_mask(
                        layout,
                        is_causal=is_causal,
                        sliding_window=sliding_window,
                    ).expand(batch, -1, -1, -1)
                    self.assertTrue(torch.equal(from_mod, from_dense))
            mod = layout.mask_mod(is_causal=False, sliding_window=None)
            oob = mod(
                torch.tensor(0),
                torch.tensor(0),
                torch.tensor([0, num_queries, num_queries + 5]),
                torch.tensor([keys.shape[1], 0, keys.shape[1] + 5]),
            )
            self.assertEqual(oob.tolist(), [False, False, False])

    def test_dense_preparation_scope_shares_context_blocks_base_visibility(self):
        blocks = ContextBlocks(
            block_size=3,
            context_lengths=torch.tensor([[2, 5], [1, 4]]),
            context_valid=torch.ones(2, 6, dtype=torch.bool),
            active_blocks=torch.tensor([[True, True], [True, False]]),
        )
        queries = torch.tensor([[2, 3, 4, 5, 6, 7], [1, 2, 3, 4, 5, 6]])
        keys = torch.cat((torch.arange(6).expand(2, -1), queries), dim=1)
        layout = AttentionLayout(
            query_positions=queries, key_positions=keys, context_blocks=blocks
        )
        for backend in (EagerAttentionBackend(), SDPAAttentionBackend()):
            with (
                self.subTest(backend=type(backend).__name__),
                patch.object(
                    ContextBlocks,
                    "compute_visibility",
                    wraps=blocks.compute_visibility,
                ) as compute,
                AttentionPreparationScope() as scope,
            ):
                sliding = scope.prepare(
                    backend,
                    layout,
                    batch_size=2,
                    num_heads=4,
                    is_causal=True,
                    sliding_window=3,
                )
                full = scope.prepare(
                    backend,
                    layout,
                    batch_size=2,
                    num_heads=4,
                    is_causal=False,
                    sliding_window=None,
                )
            self.assertEqual(compute.call_count, 1)
            self.assertIs(sliding.layout, layout)
            self.assertIs(full.layout, layout)
            for prepared, is_causal, sliding_window in (
                (sliding, True, 3),
                (full, False, None),
            ):
                ref_mask, ref_active = backend.prepare(
                    layout,
                    batch_size=2,
                    num_heads=4,
                    is_causal=is_causal,
                    sliding_window=sliding_window,
                )
                self.assertTrue(
                    torch.equal(prepared.mask.visibility, ref_mask.visibility)
                )
                self.assertTrue(
                    torch.equal(prepared.mask.empty_rows, ref_mask.empty_rows)
                )
                self.assertTrue(torch.equal(prepared.active_queries, ref_active))
