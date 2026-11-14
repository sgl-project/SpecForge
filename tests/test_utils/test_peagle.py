"""P-EAGLE COD sampling preserves target validity and document boundaries."""

import unittest

import torch

from specforge.algorithms.objectives.protocols.peagle import (
    create_peagle_mask_mod,
    generate_cod_sample_indices,
)


class PEagleInputsTest(unittest.TestCase):
    def test_cod_sampling_uses_valid_targets_for_parallel_depths(self):
        torch.manual_seed(0)
        loss_mask = torch.tensor([[0, 1, 1, 1, 0, 1]])

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=loss_mask.shape[1],
            loss_mask=loss_mask,
            num_depths=4,
            down_sample_ratio=1.0,
            down_sample_ratio_min=1.0,
        )

        self.assertEqual(anchor_pos[: loss_mask.shape[1]].tolist(), list(range(6)))
        self.assertEqual(depth[: loss_mask.shape[1]].tolist(), [0] * 6)

        sampled_target_pos = anchor_pos + depth
        parallel_depth_mask = depth > 0
        self.assertTrue(torch.all(sampled_target_pos[parallel_depth_mask] >= 0))
        self.assertTrue(torch.all(sampled_target_pos[parallel_depth_mask] < 6))
        self.assertTrue(
            torch.all(loss_mask[0, sampled_target_pos[parallel_depth_mask]] == 1)
        )

    def test_cod_sampling_never_emits_negative_anchors(self):
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 8, dtype=torch.long)

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=loss_mask.shape[1],
            loss_mask=loss_mask,
            num_depths=4,
            down_sample_ratio=1.0,
            down_sample_ratio_min=1.0,
        )

        sampled_target_pos = anchor_pos + depth
        self.assertTrue(torch.all(anchor_pos >= 0))
        self.assertTrue(torch.all(sampled_target_pos < loss_mask.shape[1]))
        self.assertTrue(torch.all(loss_mask[0, sampled_target_pos] == 1))

    def test_cod_sampling_does_not_cross_packed_document_boundaries(self):
        torch.manual_seed(0)
        loss_mask = torch.ones(1, 6, dtype=torch.long)
        lengths = torch.tensor([3, 3], dtype=torch.long)

        anchor_pos, depth = generate_cod_sample_indices(
            seq_length=loss_mask.shape[1],
            loss_mask=loss_mask,
            lengths=lengths,
            num_depths=3,
            down_sample_ratio=1.0,
            down_sample_ratio_min=1.0,
        )

        target_pos = anchor_pos + depth
        document_ids = torch.tensor([0, 0, 0, 1, 1, 1])
        parallel = depth > 0
        self.assertTrue(
            torch.all(
                document_ids[anchor_pos[parallel]] == document_ids[target_pos[parallel]]
            )
        )

    def test_peagle_mask_respects_documents_depth_order_and_padding(self):
        anchor_pos = torch.tensor([0, 1, 1, 2, 4, 4, 5])
        depth = torch.tensor([0, 0, 1, 0, 0, 1, 0])
        lengths = torch.tensor([3, 2])
        mask_mod = create_peagle_mask_mod(
            anchor_pos=anchor_pos,
            depth=depth,
            lengths=lengths,
            total_seq_len=6,
        )

        def allowed(q_idx, kv_idx):
            return bool(
                mask_mod(
                    None,
                    None,
                    torch.tensor(q_idx),
                    torch.tensor(kv_idx),
                ).item()
            )

        self.assertTrue(allowed(2, 1))  # same rollout, depth 1 attends depth 0
        self.assertTrue(allowed(2, 0))  # depth 1 also attends causal depth-0 context
        self.assertFalse(allowed(1, 2))  # depth 0 cannot attend a future depth
        self.assertFalse(allowed(4, 3))  # different packed documents
        self.assertFalse(allowed(6, 6))  # padding anchor position
