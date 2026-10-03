import json
import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as multiprocessing
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepOutput


class _Strategy:
    def __init__(self, model):
        self.model = model

    def forward_loss(self, batch, ctx=None):
        return StepOutput(
            loss=self.model(batch.tensors["input"]).square().mean(), metrics={}
        )


class _Optimizer(torch.optim.SGD):
    def step(self):
        super().step()
        self.zero_grad(set_to_none=True)


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


def _worker(rank, directory):
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
        reference = nn.Linear(4, 2, bias=False)
        reference.load_state_dict(model.state_dict())
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2, sharding_strategy="NO_SHARD"),
            optimizer_factory=lambda target: _Optimizer(target.parameters(), lr=0.01),
        )
        wrapped = backend.prepare_model(model)
        state = {"calls": 0}
        wrapped.register_comm_hook(state, _count_allreduce)
        core = TrainerCore(_Strategy(wrapped), backend, accumulation_steps=2)
        reference_optimizer = _Optimizer(reference.parameters(), lr=0.01)
        counts, differences = [], []
        for micro_step in range(4):
            batch = TrainBatch(
                sample_ids=[str(micro_step)],
                strategy="dflash",
                tensors={"input": _input(rank, micro_step)},
            )
            core.train_step(batch)
            counts.append(state["calls"])
            if micro_step % 2 == 1:
                for reference_rank in range(2):
                    for reference_micro in (micro_step - 1, micro_step):
                        loss = (
                            reference(_input(reference_rank, reference_micro))
                            .square()
                            .mean()
                            / 4
                        )
                        loss.backward()
                reference_optimizer.step()
                differences.append(
                    float((model.weight - reference.weight).abs().max().detach())
                )
        Path(directory, f"rank{rank}.json").write_text(
            json.dumps(
                {
                    "collective_counts": counts,
                    "weight_differences": differences,
                    "sync_restored": wrapped.require_backward_grad_sync,
                }
            )
        )
    finally:
        dist.destroy_process_group()


class DDPAccumulationTest(unittest.TestCase):
    def test_reduces_only_at_optimizer_boundaries(self):
        with tempfile.TemporaryDirectory() as directory:
            multiprocessing.spawn(_worker, args=(directory,), nprocs=2, join=True)
            for rank in range(2):
                result = json.loads(Path(directory, f"rank{rank}.json").read_text())
                with self.subTest(rank=rank):
                    self.assertEqual(result["collective_counts"], [0, 1, 1, 2])
                    self.assertLess(max(result["weight_differences"]), 1e-6)
                    self.assertTrue(result["sync_restored"])


if __name__ == "__main__":
    unittest.main()
