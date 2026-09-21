import unittest

import torch

from specforge.modeling.target.target_head import TargetHead


class TargetHeadPreprocessTest(unittest.TestCase):
    def test_shifts_loss_mask_with_inputs_and_target(self):
        input_ids = torch.tensor([[10, 11, 12, 13]])
        target = torch.tensor([[[20], [21], [22], [23]]])
        loss_mask = torch.tensor([[0, 0, 1, 1]])

        shifted_ids, shifted_target, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_ids.tolist(), [[11, 12, 13, 0]])
        self.assertEqual(shifted_target.squeeze(-1).tolist(), [[21, 22, 23, 0]])
        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0, 1, 1, 0]])

    def test_shifts_batched_disjoint_and_boundary_masks(self):
        input_ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        target = torch.arange(20).reshape(2, 5, 2)
        loss_mask = torch.tensor(
            [[True, False, True, True, False], [False, False, False, False, True]]
        )

        _, _, shifted_mask = TargetHead.preprocess(
            None,
            input_ids,
            target,
            loss_mask,
        )

        self.assertEqual(shifted_mask.dtype, torch.bool)
        self.assertEqual(
            shifted_mask.squeeze(-1).tolist(),
            [
                [False, True, True, False, False],
                [False, False, False, True, False],
            ],
        )

    def test_single_token_sequence_has_no_trainable_eagle_row(self):
        _, _, shifted_mask = TargetHead.preprocess(
            None,
            torch.tensor([[7]]),
            torch.tensor([[[11, 12]]]),
            torch.tensor([[1]]),
        )

        self.assertEqual(shifted_mask.squeeze(-1).tolist(), [[0]])


if __name__ == "__main__":
    unittest.main()
