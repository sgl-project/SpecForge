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
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput


class _Strategy(DraftTrainStrategy):
    name = "ddp"
    required_features = {"input"}

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
    def forward_loss(self, batch, ctx=None):
        with torch.no_grad():
            output = super().forward_loss(batch, ctx)

        def replay_loss(window_ratios):
            return _Strategy.forward_loss(self, batch, ctx)

        return replace(output, replay_loss=replay_loss)


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
        core = TrainerCore(strategy, backend, accumulation_steps=2)
        reference_optimizer = _Optimizer(reference.parameters(), lr=0.01)
        check = unittest.TestCase()
        for micro_step in range(4):
            batch = TrainBatch(
                sample_ids=[str(micro_step)],
                strategy="dflash",
                tensors={"input": _input(rank, micro_step)},
            )
            core.train_step(batch)
            check.assertEqual(state["calls"], (micro_step + 1) // 2)
            check.assertTrue(wrapped.require_backward_grad_sync)
            if micro_step % 2 == 1:
                # One effective batch spans both ranks and both microsteps.
                inputs = torch.cat(
                    [
                        _input(r, step)
                        for r in range(2)
                        for step in (micro_step - 1, micro_step)
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
