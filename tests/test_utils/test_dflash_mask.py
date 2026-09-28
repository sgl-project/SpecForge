import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash_mask import (
    build_block_mask,
    build_dense_mask,
)
from specforge.utils import get_device_type


def _reference_dflash_mask(
    anchor_positions,
    block_keep_mask,
    S,
    block_size,
    device,
    sliding_window=None,
    is_causal=None,
):
    """Independent reference using absolute positions in each serving block."""
    B, N = anchor_positions.shape
    mask = torch.zeros(
        B, 1, N * block_size, S + N * block_size, dtype=torch.bool, device=device
    )
    causal = (sliding_window is not None) if is_causal is None else is_causal
    for b in range(B):
        for block in range(N):
            if not block_keep_mask[b, block].item():
                continue
            anchor = anchor_positions[b, block].item()
            # Serving sees only the committed context prefix and this draft block.
            keys = [(pos, pos) for pos in range(anchor)]
            keys += [
                (S + block * block_size + offset, anchor + offset)
                for offset in range(block_size)
            ]
            for offset in range(block_size):
                query_pos = anchor + offset
                for column, key_pos in keys:
                    if causal and key_pos > query_pos:
                        continue
                    if (
                        sliding_window is not None
                        and abs(key_pos - query_pos) >= sliding_window
                    ):
                        continue
                    mask[b, 0, block * block_size + offset, column] = True
    return mask


class _RecordingDraftModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(
            layer_types=["sliding_attention", "full_attention"],
            sliding_window=8,
        )
        self.sliding_window = 8
        self.attention_mask = None

    def forward(self, noise_embedding, attention_mask, **kwargs):
        self.attention_mask = attention_mask
        return noise_embedding


class TestDFlashMask(unittest.TestCase):

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA FlexAttention required")
    def test_flex_output_and_gradients_match_sdpa(self):
        from torch.nn.attention.flex_attention import flex_attention

        attention = torch.compile(flex_attention)
        device = torch.device("cuda")
        torch.manual_seed(17)
        for causal in (None, False, True):
            for window in (None, 1, 3):
                with self.subTest(is_causal=causal, window=window):
                    args = dict(
                        anchor_positions=torch.tensor([[2, 6]], device=device),
                        block_keep_mask=torch.tensor([[True, False]], device=device),
                        S=8,
                        block_size=4,
                        device=device,
                        sliding_window=window,
                        is_causal=causal,
                    )
                    dense_mask = build_dense_mask(**args)
                    block_mask = build_block_mask(**args)
                    q = torch.randn(1, 2, 8, 32, device=device, requires_grad=True)
                    k = torch.randn(1, 2, 16, 32, device=device, requires_grad=True)
                    v = torch.randn_like(k, requires_grad=True)
                    output = attention(q, k, v, block_mask=block_mask)
                    expected = torch.nn.functional.scaled_dot_product_attention(
                        q, k, v, attn_mask=dense_mask
                    )
                    torch.testing.assert_close(output, expected, atol=1e-4, rtol=1e-4)
                    grad = torch.randn_like(output)
                    actual_grads = torch.autograd.grad(output, (q, k, v), grad)
                    expected_grads = torch.autograd.grad(expected, (q, k, v), grad)
                    for actual, reference in zip(actual_grads, expected_grads):
                        torch.testing.assert_close(
                            actual, reference, atol=1e-4, rtol=1e-4
                        )

    def test_causality_and_windows_against_absolute_position_reference(self):
        device = torch.device("cpu")
        anchors = torch.tensor([[0, 4, 7], [2, 6, 0]])
        keep = torch.tensor([[True, False, True], [True, True, False]])
        for causal in (None, False, True):
            for window in (None, 1, 3, 8):
                with self.subTest(is_causal=causal, window=window):
                    args = dict(
                        anchor_positions=anchors,
                        block_keep_mask=keep,
                        S=8,
                        block_size=5,
                        device=device,
                        sliding_window=window,
                        is_causal=causal,
                    )
                    expected = _reference_dflash_mask(**args)
                    dense = build_dense_mask(**args)
                    torch.testing.assert_close(dense, expected)
                    block_mask = build_block_mask(**args)
                    q = torch.arange(15).unsqueeze(1)
                    kv = torch.arange(23).unsqueeze(0)
                    for batch in range(2):
                        actual = block_mask.mask_mod(batch, 0, q, kv)
                        torch.testing.assert_close(actual, expected[batch, 0])

    def test_bidirectional_window_counts_draft_tokens(self):
        mask = build_dense_mask(
            torch.tensor([[6]]),
            torch.tensor([[True]]),
            6,
            4,
            torch.device("cpu"),
            sliding_window=3,
            is_causal=False,
        )
        expected = ([4, 5, 6, 7, 8], [5, 6, 7, 8, 9], [6, 7, 8, 9], [7, 8, 9])
        for row, columns in enumerate(expected):
            self.assertEqual(mask[0, 0, row].nonzero().flatten().tolist(), columns)

    def test_checkpoint_causality_reaches_both_layer_masks(self):
        from pathlib import Path
        from transformers import Qwen3Config

        checkpoint = Qwen3Config.from_json_file(
            str(
                Path(__file__).resolve().parents[2] / "configs/qwen3.8-27b-dflash2.json"
            )
        )
        self.assertIs(checkpoint.is_causal, False)
        for model_class in (OnlineDFlashModel, OnlineDominoModel, OnlineDSparkModel):
            for causal in (None, False, True):
                with self.subTest(model=model_class.__name__, is_causal=causal):
                    draft = _RecordingDraftModel()
                    if causal is not None:
                        draft.config.is_causal = (
                            checkpoint.is_causal if not causal else True
                        )
                    model = model_class(
                        draft_model=draft,
                        target_lm_head=nn.Identity(),
                        target_embed_tokens=nn.Embedding(32, 8),
                        mask_token_id=31,
                        block_size=4,
                        attention_backend="sdpa",
                    )
                    anchors, keep = torch.tensor([[6]]), torch.tensor([[True]])
                    with mock.patch.object(
                        model, "_sample_anchor_positions", return_value=(anchors, keep)
                    ):
                        model._forward_draft_blocks(
                            input_ids=torch.arange(12).unsqueeze(0),
                            hidden_states=torch.randn(1, 12, 8),
                            loss_mask=torch.ones(1, 12),
                        )
                    for name, window in (
                        ("full_attention", None),
                        ("sliding_attention", 8),
                    ):
                        expected = _reference_dflash_mask(
                            anchors,
                            keep,
                            12,
                            4,
                            torch.device("cpu"),
                            sliding_window=window,
                            is_causal=causal,
                        )
                        torch.testing.assert_close(draft.attention_mask[name], expected)

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device(get_device_type())

    def _compare_masks(
        self,
        anchor_positions,
        block_keep_mask,
        S,
        block_size,
        sliding_window=None,
    ):
        """Compare build_dense_mask against element-level reference (ground truth)."""
        anchor_positions = anchor_positions.to(self.device)
        block_keep_mask = block_keep_mask.to(self.device)

        sdpa_mask = build_dense_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
            sliding_window=sliding_window,
        )

        ref_mask = _reference_dflash_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
            sliding_window=sliding_window,
        )

        self.assertEqual(
            sdpa_mask.shape,
            ref_mask.shape,
            f"Shape mismatch: sdpa {sdpa_mask.shape} vs ref {ref_mask.shape}",
        )
        self.assertTrue(
            torch.equal(sdpa_mask, ref_mask),
            f"Mask mismatch with S={S}, block_size={block_size}, "
            f"sliding_window={sliding_window}, anchors={anchor_positions.tolist()}, "
            f"keep={block_keep_mask.tolist()}\n"
            f"Diff positions: {(sdpa_mask != ref_mask).nonzero(as_tuple=False).tolist()}",
        )

    def _compare_block_mask_consistency(
        self,
        anchor_positions,
        block_keep_mask,
        S,
        block_size,
        sliding_window=None,
    ):
        """Verify build_block_mask block-level mask is consistent with reference."""
        anchor_positions = anchor_positions.to(self.device)
        block_keep_mask = block_keep_mask.to(self.device)

        block_mask = build_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
            sliding_window=sliding_window,
        )

        ref_mask = _reference_dflash_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
            sliding_window=sliding_window,
        )

        dense_blocks = block_mask.to_dense()  # (B, H, Q_blocks, KV_blocks)
        BM_BLOCK = 128
        B, N = anchor_positions.shape
        Q_LEN = N * block_size
        KV_LEN = S + N * block_size
        n_q_blocks = (Q_LEN + BM_BLOCK - 1) // BM_BLOCK
        n_kv_blocks = (KV_LEN + BM_BLOCK - 1) // BM_BLOCK

        ref_int = ref_mask.squeeze(1).int()  # (B, Q_LEN, KV_LEN)
        for b in range(B):
            for qi in range(n_q_blocks):
                for ki in range(n_kv_blocks):
                    q_start = qi * BM_BLOCK
                    q_end = min(q_start + BM_BLOCK, Q_LEN)
                    k_start = ki * BM_BLOCK
                    k_end = min(k_start + BM_BLOCK, KV_LEN)
                    has_nonzero = ref_int[b, q_start:q_end, k_start:k_end].any().item()
                    block_val = dense_blocks[b, 0, qi, ki].item()
                    self.assertEqual(
                        block_val,
                        int(has_nonzero),
                        f"Block ({qi},{ki}) for batch {b} has incorrect occupancy",
                    )

    def test_basic_single_batch_single_block(self):
        """Single batch, single draft block."""
        anchor_positions = torch.tensor([[64]])
        block_keep_mask = torch.tensor([[True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_basic_single_batch_multi_block(self):
        """Single batch, multiple draft blocks."""
        anchor_positions = torch.tensor([[32, 64, 96]])
        block_keep_mask = torch.tensor([[True, True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_multi_batch(self):
        """Multiple batches with different anchors."""
        anchor_positions = torch.tensor([[16, 48, 80], [32, 64, 100]])
        block_keep_mask = torch.tensor([[True, True, True], [True, True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_invalid_blocks(self):
        """Some blocks are masked out (block_keep_mask=False)."""
        anchor_positions = torch.tensor([[20, 50, 80, 110]])
        block_keep_mask = torch.tensor([[True, False, True, False]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_all_blocks_invalid(self):
        """All blocks invalid — mask should be all zeros."""
        anchor_positions = torch.tensor([[30, 60]])
        block_keep_mask = torch.tensor([[False, False]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_anchor_at_zero(self):
        """Anchor at position 0 — no context tokens visible."""
        anchor_positions = torch.tensor([[0, 64]])
        block_keep_mask = torch.tensor([[True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_anchor_at_boundary(self):
        """Anchor exactly at S — all context tokens visible."""
        anchor_positions = torch.tensor([[128]])
        block_keep_mask = torch.tensor([[True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=4)

    def test_large_block_size(self):
        """Larger draft block size."""
        anchor_positions = torch.tensor([[50, 150]])
        block_keep_mask = torch.tensor([[True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=256, block_size=16)

    def test_block_size_1(self):
        """Minimal block_size=1."""
        anchor_positions = torch.tensor([[10, 30, 50]])
        block_keep_mask = torch.tensor([[True, True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=64, block_size=1)

    def test_sliding_window_moves_with_query_offset(self):
        """The context window advances while the draft block stays causal."""
        anchor_positions = torch.tensor([[6]])
        block_keep_mask = torch.tensor([[True]])
        mask = build_dense_mask(
            anchor_positions=anchor_positions.to(self.device),
            block_keep_mask=block_keep_mask.to(self.device),
            S=12,
            block_size=4,
            device=self.device,
            sliding_window=4,
        )
        expected_visible_keys = (
            [3, 4, 5, 12],
            [4, 5, 12, 13],
            [5, 12, 13, 14],
            [12, 13, 14, 15],
        )
        for query_offset, expected in enumerate(expected_visible_keys):
            with self.subTest(query_offset=query_offset):
                actual = mask[0, 0, query_offset].nonzero().flatten().tolist()
                self.assertEqual(actual, expected)

    def test_sliding_window_one_has_only_self_attention(self):
        """A one-token window excludes both context and other draft tokens."""
        anchor_positions = torch.tensor([[4, 9]])
        block_keep_mask = torch.tensor([[True, True]])
        self._compare_masks(
            anchor_positions,
            block_keep_mask,
            S=12,
            block_size=3,
            sliding_window=1,
        )

    def test_sliding_window_block_mask_consistency(self):
        anchor_positions = torch.tensor([[12, 24]])
        block_keep_mask = torch.tensor([[True, True]])
        self._compare_block_mask_consistency(
            anchor_positions,
            block_keep_mask,
            S=32,
            block_size=4,
            sliding_window=8,
        )

    def test_invalid_sliding_window(self):
        anchor_positions = torch.tensor([[12]], device=self.device)
        block_keep_mask = torch.tensor([[True]], device=self.device)
        for factory in (build_dense_mask, build_block_mask):
            with self.subTest(factory=factory.__name__):
                with self.assertRaisesRegex(ValueError, "sliding_window must be > 0"):
                    factory(
                        anchor_positions=anchor_positions,
                        block_keep_mask=block_keep_mask,
                        S=16,
                        block_size=4,
                        device=self.device,
                        sliding_window=0,
                    )

    def test_all_dflash_families_build_mixed_layer_masks(self):
        anchors = torch.tensor([[12]], device=self.device)
        keep = torch.tensor([[True]], device=self.device)
        for model_class in (OnlineDFlashModel, OnlineDominoModel, OnlineDSparkModel):
            with self.subTest(model_class=model_class.__name__):
                draft_model = _RecordingDraftModel().to(self.device)
                model = model_class(
                    draft_model=draft_model,
                    target_lm_head=nn.Identity(),
                    target_embed_tokens=nn.Embedding(32, 8).to(self.device),
                    mask_token_id=31,
                    block_size=4,
                    attention_backend="sdpa",
                )
                with mock.patch.object(
                    model,
                    "_sample_anchor_positions",
                    return_value=(anchors, keep),
                ):
                    model._forward_draft_blocks(
                        input_ids=torch.arange(16, device=self.device).unsqueeze(0),
                        hidden_states=torch.randn(1, 16, 8, device=self.device),
                        loss_mask=torch.ones(1, 16, device=self.device),
                    )

                masks = draft_model.attention_mask
                self.assertEqual(set(masks), {"full_attention", "sliding_attention"})
                self.assertTrue(masks["full_attention"][0, 0, 0, 0].item())
                self.assertFalse(masks["sliding_attention"][0, 0, 0, 0].item())

    def test_mixed_validity_multi_batch(self):
        """Multi-batch with mixed block validity patterns."""
        anchor_positions = torch.tensor([[10, 40, 70, 100], [20, 50, 80, 110]])
        block_keep_mask = torch.tensor(
            [[True, False, True, True], [False, True, False, True]]
        )
        self._compare_masks(anchor_positions, block_keep_mask, S=128, block_size=8)

    def test_various_context_lengths(self):
        """Sweep over various context lengths."""
        for S in [64, 128, 256, 512]:
            with self.subTest(S=S):
                anchor_positions = torch.tensor([[S // 4, S // 2, 3 * S // 4]])
                block_keep_mask = torch.tensor([[True, True, True]])
                self._compare_masks(
                    anchor_positions, block_keep_mask, S=S, block_size=4
                )

    def test_various_block_sizes(self):
        """Sweep over various draft block sizes."""
        for block_size in [1, 2, 4, 8, 16]:
            with self.subTest(block_size=block_size):
                anchor_positions = torch.tensor([[32, 80]])
                block_keep_mask = torch.tensor([[True, True]])
                self._compare_masks(
                    anchor_positions, block_keep_mask, S=128, block_size=block_size
                )

    def test_many_blocks(self):
        """Large number of draft blocks."""
        N = 32
        anchors = torch.arange(10, 10 + N * 4, 4).unsqueeze(0)
        keep = torch.ones(1, N, dtype=torch.bool)
        keep[0, ::3] = False
        self._compare_masks(anchors, keep, S=256, block_size=4)

    def test_consecutive_anchors(self):
        """Anchors placed consecutively."""
        anchor_positions = torch.tensor([[0, 1, 2, 3]])
        block_keep_mask = torch.tensor([[True, True, True, True]])
        self._compare_masks(anchor_positions, block_keep_mask, S=64, block_size=4)

    def test_random_stress(self):
        """Randomized stress test with multiple random configurations."""
        rng = torch.Generator().manual_seed(123)
        for trial in range(5):
            with self.subTest(trial=trial):
                B = torch.randint(1, 4, (1,), generator=rng).item()
                N = torch.randint(1, 8, (1,), generator=rng).item()
                S = 64 * torch.randint(1, 5, (1,), generator=rng).item()
                block_size = [1, 2, 4, 8][
                    torch.randint(0, 4, (1,), generator=rng).item()
                ]

                anchor_positions = torch.stack(
                    [
                        torch.randperm(S, generator=rng)[:N].sort().values
                        for _ in range(B)
                    ]
                )
                block_keep_mask = torch.rand(B, N, generator=rng) > 0.3

                self._compare_masks(
                    anchor_positions, block_keep_mask, S=S, block_size=block_size
                )

    def test_block_mask_consistency(self):
        """Verify BlockMask block-level mask is consistent with element-level reference."""
        anchor_positions = torch.tensor([[32, 64, 96]])
        block_keep_mask = torch.tensor([[True, True, True]])
        self._compare_block_mask_consistency(
            anchor_positions, block_keep_mask, S=128, block_size=4
        )

    def test_block_mask_consistency_mixed(self):
        """Verify BlockMask consistency with mixed validity."""
        anchor_positions = torch.tensor([[10, 40, 70, 100], [20, 50, 80, 110]])
        block_keep_mask = torch.tensor(
            [[True, False, True, True], [False, True, False, True]]
        )
        self._compare_block_mask_consistency(
            anchor_positions, block_keep_mask, S=128, block_size=8
        )

    def test_sliding_block_mask_matches_element_reference(self):
        """Flex and dense masks implement the same sliding-layer rule."""
        anchors = torch.tensor([[6, 12]], device=self.device)
        keep = torch.tensor([[True, False]], device=self.device)
        mask_args = {
            "anchor_positions": anchors,
            "block_keep_mask": keep,
            "S": 16,
            "block_size": 4,
            "device": self.device,
            "sliding_window": 5,
        }
        dense_mask = build_dense_mask(**mask_args)
        block_mask = build_block_mask(**mask_args)
        q_idx = torch.arange(8, device=self.device).unsqueeze(1)
        kv_idx = torch.arange(24, device=self.device).unsqueeze(0)
        flex_mask = block_mask.mask_mod(0, 0, q_idx, kv_idx)
        self.assertTrue(torch.equal(flex_mask, dense_mask[0, 0]))


if __name__ == "__main__":
    unittest.main(verbosity=2)
