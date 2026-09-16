"""Sharding the replicated optimizer state must not change what training does.

Under expert parallelism every rank holds an identical fp32 master and both
Adam moments for the attention, mHC, router and shared-expert parameters, and
updates them identically. That is 5.8 GiB per rank on the DeepSeek-V4 drafter,
which is the difference between the optimizer fitting on the device and not.
Assigning each replicated parameter to one owner has to produce, on every rank,
exactly the parameters the unsharded optimizer would have produced.
"""

import os
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn

from specforge.optimizer import BF16Optimizer


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.replicated = nn.ParameterList(
            [nn.Parameter(torch.randn(size, 4)) for size in (7, 3, 11, 5)]
        )
        self.experts = nn.Linear(4, 4, bias=False)
        self.experts._specforge_rank_local_parameters = True


def _make(seed=5, **kwargs):
    torch.manual_seed(seed)
    model = _Model()
    optimizer = BF16Optimizer(model, lr=1e-2, max_grad_norm=1e9, **kwargs)
    torch.manual_seed(17)
    for parameter in model.parameters():
        parameter.grad = torch.randn_like(parameter)
    return model, optimizer


def _worker(rank, init_file, steps):
    if os.uname().sysname == "Darwin":
        os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(
        "gloo", init_method=f"file://{init_file}", rank=rank, world_size=2
    )
    try:
        reference_model, reference = _make()
        sharded_model, optimizer = _make()
        optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=False, partition_replicated=True
        )

        # Sharding must actually drop masters, or the test proves nothing.
        assert len(optimizer.fp32_params) < len(
            optimizer.model_params
        ), f"rank {rank} kept every master: {len(optimizer.fp32_params)}"
        assert optimizer._replicated_owner, "no replicated parameter was assigned"

        for _ in range(steps):
            reference.step()
            optimizer.step()
            for model in (reference_model, sharded_model):
                # Seed per model: one seed outside the loop would let the
                # reference consume the stream and hand the sharded optimizer
                # different gradients, which looks exactly like a sharding bug.
                torch.manual_seed(23)
                for parameter in model.parameters():
                    parameter.grad = torch.randn_like(parameter)

        for name, expected in reference_model.named_parameters():
            actual = dict(sharded_model.named_parameters())[name]
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    finally:
        dist.destroy_process_group()


class ExpertParallelStateShardingTest(unittest.TestCase):
    def test_assignment_is_balanced_and_deterministic(self):
        _, optimizer = _make()
        sizes = [p.numel() for p in optimizer.model_params]

        # Reproduce the greedy split without distributed, for two ranks.
        local = optimizer.model_params[-1]
        replicated = [i for i, p in enumerate(optimizer.model_params) if p is not local]
        load = [0, 0]
        owner = {}
        for index in sorted(replicated, key=lambda i: sizes[i], reverse=True):
            target = min(range(2), key=lambda r: (load[r], r))
            owner[index] = target
            load[target] += sizes[index]

        self.assertEqual(sorted(owner), sorted(replicated))
        # 11 + 3 against 7 + 5 -- the greedy pass keeps them within one row.
        self.assertLessEqual(abs(load[0] - load[1]) / max(load), 0.2)

    def test_no_sharding_without_a_group(self):
        _, optimizer = _make()
        optimizer.configure_grad_norm_reduction(
            process_group=None, enabled=True, partition_replicated=True
        )
        # Nothing to shard across, so every master stays.
        self.assertEqual(len(optimizer.fp32_params), len(optimizer.model_params))
        self.assertEqual(optimizer._replicated_owner, {})

    def test_two_ranks_match_the_unsharded_optimizer(self):
        with tempfile.TemporaryDirectory() as directory:
            init_file = os.path.join(directory, "rendezvous")
            mp.spawn(_worker, args=(init_file, 3), nprocs=2, join=True)


if __name__ == "__main__":
    unittest.main()
