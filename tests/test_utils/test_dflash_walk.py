"""Strict accepted prefixes and sparse anchor walks have independent references."""

import unittest

import torch

from specforge.algorithms.metrics import blocks as family_model


class DFlashWalkTest(unittest.TestCase):
    def test_sparse_padded_and_empty_walks(self):
        cases = (
            ([0, 1, 2, 3, 4, 5], [2, 0, 5, 1, 0, 3], [1] * 6, (9, 3)),
            ([0, 1, 2, 10], [3, 0, 0, 0], [1] * 4, (5, 2)),
            ([1, 3, 8, 0], [0, 4, 0, 8], [1, 0, 1, 0], (2, 2)),
            ([0, 0], [9, 9], [0, 0], (0, 0)),
        )
        for positions, counts, valid, expected in cases:
            with self.subTest(positions=positions, counts=counts, valid=valid):
                result = family_model.compute_walk_accepted_length_terms(
                    torch.tensor([counts], dtype=torch.float32),
                    torch.tensor([positions]),
                    torch.tensor([valid], dtype=torch.bool),
                )
                self.assertEqual(tuple(value.item() for value in result), expected)

    def test_prefix_stops_at_masked_holes_and_scatters_global_indices(self):
        targets = torch.tensor([[[1, 2, 3], [4, 5, 6]], [[1, 2, 3], [4, 5, 6]]])
        valid = torch.tensor(
            [[[1, 0, 1], [0, 1, 1]], [[1, 1, 1], [0, 0, 0]]], dtype=torch.bool
        )
        result = family_model._scatter_accepted_prefix(
            targets, targets, valid, torch.tensor([[1, 3]]), 5
        )
        self.assertEqual(result.tolist(), [[0, 1, 0, 0, 0], [0, 3, 0, 0, 0]])


if __name__ == "__main__":
    unittest.main()
