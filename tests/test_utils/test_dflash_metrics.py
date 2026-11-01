"""CPU checks for prefix survival, censoring, and first-failure attribution."""

import unittest

import torch

from specforge.algorithms.common.dflash_metrics import hard_label_prefix_counts
from specforge.core.chunking import checkpointed_chunk_reduce


class PrefixMetricsTest(unittest.TestCase):
    @staticmethod
    def _inputs():
        def events(rows):
            return torch.tensor([rows], dtype=torch.bool)

        # Four blocks: a late error, a covered selector error, a coverage miss,
        # and an immediate selector error followed by later marginal hits.
        unary = events([[1, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
        covered = events([[1, 1, 1, 1], [1, 1, 1, 1], [1, 0, 1, 1], [1, 1, 0, 1]])
        selector = events([[1, 1, 1, 0], [1, 0, 1, 1], [1, 0, 0, 1], [0, 1, 0, 1]])
        return unary, covered, selector, torch.ones_like(unary)

    def test_distinct_prefix_populations_and_failure_decomposition(self):
        counts = hard_label_prefix_counts(*self._inputs())
        torch.testing.assert_close(
            counts.reached,
            torch.tensor([[0.0, 4, 2, 1, 1], [0.0, 4, 4, 3, 2], [0.0, 4, 3, 1, 1]]),
        )
        torch.testing.assert_close(
            counts.accepted,
            torch.tensor([[0.0, 2, 1, 1, 0], [0.0, 4, 3, 2, 2], [0.0, 3, 1, 1, 0]]),
        )
        torch.testing.assert_close(
            counts.selector_covered, torch.tensor([0.0, 4, 2, 1, 1])
        )
        torch.testing.assert_close(
            counts.selector_unary_correct, torch.tensor([0.0, 2, 2, 1, 0])
        )
        # At position 2, the three reached selector prefixes split equally into
        # an accepted token, a missing candidate, and a covered ranking error.
        self.assertEqual(counts.accepted[2, 2], 1)
        self.assertEqual((counts.reached[2] - counts.selector_covered)[2], 1)
        self.assertEqual((counts.selector_covered - counts.accepted[2])[2], 1)
        # Unary / oracle / selector lengths, each including one anchor.
        torch.testing.assert_close(
            1 + counts.accepted.sum(-1) / 4, torch.tensor([2.0, 3.75, 2.25])
        )
        survival = counts.accepted / 4
        self.assertTrue(torch.all((survival >= 0) & (survival <= 1)))
        self.assertTrue(torch.all(survival[:, 2:] <= survival[:, 1:-1]))

    def test_masked_tails_and_internal_gaps_are_not_model_failures(self):
        supervised = torch.tensor(
            [[[1, 1, 0, 0], [1, 0, 1, 1], [0, 0, 0, 0]]], dtype=torch.bool
        )
        hit = torch.ones_like(supervised)
        counts = hard_label_prefix_counts(hit, hit, hit, supervised)
        expected = torch.tensor([0.0, 2, 1, 0, 0])
        torch.testing.assert_close(counts.eligible, expected)
        torch.testing.assert_close(counts.reached, expected.expand(3, -1))
        torch.testing.assert_close(counts.accepted, counts.reached)

    def test_plain_dflash_and_anchor_only_blocks(self):
        unary, covered, _, supervised = self._inputs()
        counts = hard_label_prefix_counts(unary, covered, None, supervised)
        torch.testing.assert_close(counts.reached[2], torch.zeros(5))
        torch.testing.assert_close(counts.accepted[2], torch.zeros(5))
        torch.testing.assert_close(counts.selector_covered, torch.zeros(5))
        empty = unary[..., :0]
        counts = hard_label_prefix_counts(empty, empty, None, empty)
        torch.testing.assert_close(counts.reached, torch.zeros(3, 1))
        torch.testing.assert_close(counts.eligible, torch.zeros(1))

    def test_counts_are_invariant_to_metric_chunk_size(self):
        inputs = self._inputs()
        expected = hard_label_prefix_counts(*inputs)
        for chunk_size in (0, 1, 2, 3):
            with self.subTest(chunk_size=chunk_size):
                actual = checkpointed_chunk_reduce(
                    hard_label_prefix_counts, *inputs, chunk_size=chunk_size, dim=1
                )
                for value, reference in zip(actual, expected):
                    torch.testing.assert_close(value, reference)
