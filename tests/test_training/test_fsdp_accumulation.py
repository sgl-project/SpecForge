"""FULL_SHARD reduce-scatters every micro-step; SHARD_GRAD_OP still defers via no_sync."""

import contextlib
import unittest
from unittest import mock

import torch

from specforge.training.backend import FSDPTrainingBackend, ParallelConfig


class FSDPAccumulationTest(unittest.TestCase):
    def test_full_shard_never_enters_no_sync(self):
        for strategy, deferrals in (("FULL_SHARD", 0), ("SHARD_GRAD_OP", 1)):
            with self.subTest(strategy=strategy):
                backend = FSDPTrainingBackend(
                    ParallelConfig(sharding_strategy=strategy)
                )
                backend._wrapped, backend._wrapper_kind = True, "fsdp"
                backend.module = mock.Mock(
                    no_sync=mock.Mock(return_value=contextlib.nullcontext())
                )
                for is_boundary in (False, True):
                    weight = torch.ones((), requires_grad=True)
                    backend.backward(weight * 2, is_boundary=is_boundary)
                    self.assertEqual(weight.grad.item(), 2.0)
                self.assertEqual(backend.module.no_sync.call_count, deferrals)


if __name__ == "__main__":
    unittest.main()
