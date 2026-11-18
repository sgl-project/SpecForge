import tempfile
import unittest
from copy import deepcopy
from dataclasses import replace
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from specforge.training.strategies.base import StepContext, StepOutput


class _Strategy:
    replay_statistics = ()

    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        return StepOutput(
            loss=self.model(batch.tensors["input"]).square().mean(), metrics={}
        )


class _Optimizer(torch.optim.SGD):
    def step(self):
        super().step()
        self.zero_grad(set_to_none=True)


class _ReplayStrategy(_Strategy):
    replay_statistics = ("count",)

    def prepass(self, batch, ctx):
        return replace(
            self.forward_loss(batch, ctx), statistics={"count": torch.ones(1)}
        )


def _count_allreduce(state, bucket):
    state["calls"] += 1
    buffer = bucket.buffer()
    dist.all_reduce(buffer)
    buffer.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


def _input(rank, micro_step):
    return torch.arange(1, 9, dtype=torch.float32).reshape(2, 4) / (
        rank + micro_step + 2
    )


def _worker(rank, directory, replay):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{directory}/rendezvous",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        torch.manual_seed(77)
        model = nn.Linear(4, 2, bias=False)
        reference = deepcopy(model)
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2, sharding_strategy="NO_SHARD"),
            optimizer_factory=lambda target: _Optimizer(target.parameters(), lr=0.01),
        )
        wrapped = backend.prepare_model(model)
        state = {"calls": 0}
        wrapped.register_comm_hook(state, _count_allreduce)
        strategy = _ReplayStrategy(wrapped) if replay else _Strategy(wrapped)
        runner = DataParallelRunner(strategy, backend)
        ctx = StepContext(accumulation_steps=2)
        reference_optimizer = _Optimizer(reference.parameters(), lr=0.01)
        check = unittest.TestCase()
        for window in range(2):
            batches = [
                TrainBatch(
                    sample_ids=[str(micro_step)],
                    strategy="dflash",
                    tensors={"input": _input(rank, micro_step)},
                )
                for micro_step in (2 * window, 2 * window + 1)
            ]
            if replay:
                run_window(runner, iter(batches), ctx)
            else:
                check.assertIsNone(
                    runner.forward_backward(batches[0], ctx, is_boundary=False)
                )
                # Inside the window nothing is synchronized.
                check.assertEqual(state["calls"], window)
                check.assertTrue(wrapped.require_backward_grad_sync)
                runner.apply_step(
                    runner.forward_backward(batches[1], ctx, is_boundary=True)
                )
            # One gradient synchronization per window, replayed or not.
            check.assertEqual(state["calls"], window + 1)
            check.assertTrue(wrapped.require_backward_grad_sync)
            # One effective batch spans both ranks and both microsteps.
            inputs = torch.cat(
                [
                    _input(r, step)
                    for r in range(2)
                    for step in (2 * window, 2 * window + 1)
                ]
            )
            reference(inputs).square().mean().backward()
            reference_optimizer.step()
            torch.testing.assert_close(
                model.weight, reference.weight, rtol=0, atol=1e-6
            )
    finally:
        dist.destroy_process_group()


class DDPAccumulationTest(unittest.TestCase):
    def test_reduces_only_at_optimizer_boundaries(self):
        self._check_accumulation(False)

    def test_replay_reduces_only_at_optimizer_boundaries(self):
        self._check_accumulation(True)

    def _check_accumulation(self, replay):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(
                _worker, args=(directory, replay), nprocs=2, join=True
            )


if __name__ == "__main__":
    unittest.main()
