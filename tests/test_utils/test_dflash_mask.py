import unittest

import torch

from specforge.core.dflash import create_dflash_block_mask, create_dflash_sdpa_mask


def _reference_dflash_mask(anchor_positions, block_keep_mask, S, block_size, device):
    """Element-level reference mask mirroring the mask_mod inside create_dflash_block_mask.

    This uses plain Python loops so correctness is obvious by inspection.
    """
    B, N = anchor_positions.shape
    Q_LEN = N * block_size
    KV_LEN = S + N * block_size

    mask = torch.zeros(B, 1, Q_LEN, KV_LEN, dtype=torch.bool, device=device)
    for b in range(B):
        for q_idx in range(Q_LEN):
            q_block_id = q_idx // block_size
            anchor_pos = anchor_positions[b, q_block_id].item()
            is_valid = block_keep_mask[b, q_block_id].item()
            if not is_valid:
                continue
            for kv_idx in range(KV_LEN):
                is_context = kv_idx < S
                ctx_visible = is_context and (kv_idx < anchor_pos)

                is_draft = kv_idx >= S
                kv_block_id = (kv_idx - S) // block_size
                draft_visible = is_draft and (q_block_id == kv_block_id)

                if ctx_visible or draft_visible:
                    mask[b, 0, q_idx, kv_idx] = True
    return mask


class TestDFlashMask(unittest.TestCase):

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device("cuda")

    def _compare_masks(self, anchor_positions, block_keep_mask, S, block_size):
        """Compare create_dflash_sdpa_mask against element-level reference (ground truth)."""
        anchor_positions = anchor_positions.to(self.device)
        block_keep_mask = block_keep_mask.to(self.device)

        sdpa_mask = create_dflash_sdpa_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
        )

        ref_mask = _reference_dflash_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
        )

        self.assertEqual(
            sdpa_mask.shape,
            ref_mask.shape,
            f"Shape mismatch: sdpa {sdpa_mask.shape} vs ref {ref_mask.shape}",
        )
        self.assertTrue(
            torch.equal(sdpa_mask, ref_mask),
            f"Mask mismatch with S={S}, block_size={block_size}, "
            f"anchors={anchor_positions.tolist()}, keep={block_keep_mask.tolist()}\n"
            f"Diff positions: {(sdpa_mask != ref_mask).nonzero(as_tuple=False).tolist()}",
        )

    def _compare_block_mask_consistency(
        self, anchor_positions, block_keep_mask, S, block_size
    ):
        """Verify create_dflash_block_mask block-level mask is consistent with reference."""
        anchor_positions = anchor_positions.to(self.device)
        block_keep_mask = block_keep_mask.to(self.device)

        block_mask = create_dflash_block_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
        )

        ref_mask = _reference_dflash_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            S=S,
            block_size=block_size,
            device=self.device,
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
                    if has_nonzero:
                        self.assertEqual(
                            block_val,
                            1,
                            f"Block ({qi},{ki}) for batch {b} should be 1 but got 0",
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
