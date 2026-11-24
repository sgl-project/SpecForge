"""Chunk boundaries do not change prediction-set reductions or gradients."""

import unittest

import torch

from specforge.algorithms.objectives.execution import execute_batch
from specforge.algorithms.objectives.losses import PositionWeighting, TokenLoss
from specforge.algorithms.objectives.predictions import PredictionBatch, Predictions
from specforge.algorithms.objectives.projection import Projection


class ChunkedReductionTest(unittest.TestCase):
    def test_chunking_preserves_prediction_means(self):
        for reduction in ("sum_means", "tokens"):
            for fixed_denominator in (False, True):

                def run(chunk_size):
                    torch.manual_seed(31)
                    head = torch.nn.Linear(4, 7).double()
                    sets = []
                    for mask in ([1, 0.5, 0, 1, 0], [1, 1, 0.5], [0, 0]):
                        hidden = torch.randn(
                            len(mask), 1, 4, dtype=torch.float64, requires_grad=True
                        )
                        sets.append(
                            Predictions(
                                hidden=hidden,
                                projection=Projection(lambda x: {"logits": head(x)}),
                                mask=torch.tensor(mask, dtype=torch.float64)[:, None],
                                positions=torch.zeros(len(mask), 1, dtype=torch.long),
                                labels=torch.randint(7, (len(mask), 1)),
                                denominator=(
                                    torch.tensor(7.0) if fixed_denominator else None
                                ),
                            )
                        )
                    result = execute_batch(
                        PredictionBatch(
                            sets, reduction=reduction, chunk_size=chunk_size
                        ),
                        TokenLoss(kind="ce"),
                        PositionWeighting(),
                        training=True,
                    )
                    gradients = torch.autograd.grad(
                        result.loss, [p.hidden for p in sets] + list(head.parameters())
                    )
                    return result, gradients

                expected, expected_gradients = run(0)
                for chunk_size in (1, 2, 4, 8):
                    with self.subTest(
                        reduction=reduction, fixed=fixed_denominator, chunk=chunk_size
                    ):
                        actual, gradients = run(chunk_size)
                        torch.testing.assert_close(actual.loss, expected.loss)
                        torch.testing.assert_close(
                            actual.denominator, expected.denominator
                        )
                        torch.testing.assert_close(actual.ratios, expected.ratios)
                        torch.testing.assert_close(gradients, expected_gradients)
