# coding=utf-8
"""Tests for reusable memory-bounded objective reductions."""

import unittest

import torch

from specforge.core.chunking import checkpointed_chunk_reduce


class CheckpointedChunkReduceTest(unittest.TestCase):
    def test_sums_additive_terms_and_preserves_gradients(self):
        values = torch.arange(12, dtype=torch.double).reshape(2, 6)
        values.requires_grad_()
        weights = torch.linspace(0.5, 1.5, 12, dtype=torch.double).reshape(2, 6)

        def terms(value_chunk, weight_chunk, optional):
            self.assertIsNone(optional)
            return (
                (value_chunk * weight_chunk).sum(),
                (value_chunk.square() * weight_chunk).sum(),
            )

        linear, quadratic = checkpointed_chunk_reduce(
            terms,
            values,
            weights,
            None,
            chunk_size=2,
            dim=1,
        )

        torch.testing.assert_close(linear, (values * weights).sum())
        torch.testing.assert_close(quadratic, (values.square() * weights).sum())
        quadratic.backward()
        torch.testing.assert_close(values.grad, 2 * values.detach() * weights)

    def test_zero_chunk_size_runs_one_full_slice(self):
        calls = []
        values = torch.arange(5)

        def terms(chunk):
            calls.append(tuple(chunk.shape))
            return (chunk.sum(),)

        (total,) = checkpointed_chunk_reduce(
            terms,
            values,
            chunk_size=0,
        )

        self.assertEqual(calls, [(5,)])
        self.assertEqual(total.item(), 10)

    def test_rejects_misaligned_inputs(self):
        with self.assertRaisesRegex(ValueError, "must be aligned"):
            checkpointed_chunk_reduce(
                lambda left, right: (left.sum() + right.sum(),),
                torch.zeros(2, 3),
                torch.zeros(2, 4),
                chunk_size=1,
                dim=1,
            )


if __name__ == "__main__":
    unittest.main()
