"""The expert-parallel gradient norm must count replicated parameters once.

Every rank in the EP group holds the same attention/mHC/router/shared-expert
gradients and a disjoint slice of the routed experts.  Summing every square
over the group counts the replicas ep_size times and inflates the norm by up
to sqrt(ep_size) -- 2.83x at EP=8 -- which clips every step that much harder
and makes the configured learning rate meaningless.
"""

import math
import unittest
from unittest import mock

import torch
import torch.nn as nn

from specforge.optimizer import BF16Optimizer, _rank_local_parameter_ids


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.replicated = nn.Linear(4, 4, bias=False)
        self.local_expert = nn.Linear(4, 4, bias=False)
        self.local_expert._specforge_rank_local_parameters = True


def _fake_distributed(group_size):
    """Simulate `group_size` ranks that all hold identical tensors."""

    def all_reduce(tensor, op=None, group=None):
        tensor.mul_(group_size)

    module = mock.MagicMock()
    module.is_available.return_value = True
    module.is_initialized.return_value = True
    module.get_world_size.return_value = group_size
    module.all_reduce.side_effect = all_reduce
    module.ReduceOp.SUM = "sum"
    return module


class ExpertParallelGradNormTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.model = _Model()
        self.optimizer = BF16Optimizer(self.model, lr=1e-3, max_grad_norm=1e9)
        for parameter in self.model.parameters():
            parameter.grad = torch.randn_like(parameter)
        self.replicated_sq = (
            self.model.replicated.weight.grad.float().square().sum().item()
        )
        self.local_sq = (
            self.model.local_expert.weight.grad.float().square().sum().item()
        )

    def test_marker_finds_only_the_rank_local_module(self):
        ids = _rank_local_parameter_ids(self.model)
        self.assertEqual(ids, frozenset({id(self.model.local_expert.weight)}))

    def test_partitioned_norm_counts_replicas_once(self):
        group_size = 4
        self.optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=True, partition_replicated=True
        )
        with mock.patch("specforge.optimizer.dist", _fake_distributed(group_size)):
            total_norm, _ = self.optimizer._grad_norm_and_clip_coefficient()

        expected = math.sqrt(self.replicated_sq + group_size * self.local_sq)
        self.assertAlmostEqual(float(total_norm), expected, places=4)

    def test_unpartitioned_norm_is_the_inflated_one_it_replaces(self):
        group_size = 4
        self.optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=True, partition_replicated=False
        )
        with mock.patch("specforge.optimizer.dist", _fake_distributed(group_size)):
            total_norm, _ = self.optimizer._grad_norm_and_clip_coefficient()

        inflated = math.sqrt(group_size * (self.replicated_sq + self.local_sq))
        self.assertAlmostEqual(float(total_norm), inflated, places=4)
        # The whole point: the old path was strictly larger.
        exact = math.sqrt(self.replicated_sq + group_size * self.local_sq)
        self.assertGreater(inflated, exact)

    def test_partition_without_distributed_still_sums_both_halves(self):
        self.optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=True, partition_replicated=True
        )
        module = mock.MagicMock()
        module.is_available.return_value = False
        with mock.patch("specforge.optimizer.dist", module):
            total_norm, _ = self.optimizer._grad_norm_and_clip_coefficient()

        expected = math.sqrt(self.replicated_sq + self.local_sq)
        self.assertAlmostEqual(float(total_norm), expected, places=4)

    def test_unmarked_model_keeps_the_plain_sum(self):
        plain = nn.Linear(4, 4, bias=False)
        plain.weight.grad = torch.randn_like(plain.weight)
        optimizer = BF16Optimizer(plain, lr=1e-3, max_grad_norm=1e9)
        optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=True, partition_replicated=True
        )
        # Nothing is marked, so every parameter is treated as a rank-local
        # slice -- exactly the behaviour of a fully sharded optimizer.
        with mock.patch("specforge.optimizer.dist", _fake_distributed(4)):
            total_norm, _ = optimizer._grad_norm_and_clip_coefficient()
        expected = math.sqrt(4 * plain.weight.grad.float().square().sum().item())
        self.assertAlmostEqual(float(total_norm), expected, places=4)


if __name__ == "__main__":
    unittest.main()
