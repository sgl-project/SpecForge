"""Two-rank CPU correctness and collective-count test for DDP accumulation."""

import os
import tempfile
import unittest
from datetime import timedelta
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput


class Strategy(DraftTrainStrategy):
    name = "dflash"
    required_features = {"x", "y"}

    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        loss = (self.model(batch.tensors["x"]) - batch.tensors["y"]).square().mean()
        return StepOutput(loss=loss, metrics={})

    def checkpoint_state_filter(self, state_dict):
        return state_dict


def worker(rank, rendezvous, accumulation_steps):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + rendezvous,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    torch.manual_seed(812)
    model = torch.nn.Linear(3, 2, bias=False)
    reference = torch.nn.Linear(3, 2, bias=False)
    reference.load_state_dict(model.state_dict())
    backend = FSDPTrainingBackend(
        ParallelConfig(world_size=2, sharding_strategy="NO_SHARD"),
        optimizer_factory=lambda m: torch.optim.SGD(m.parameters(), lr=0.1),
    )
    wrapped = backend.prepare_model(model)
    calls = []

    def count_reduction(state, bucket):
        calls.append(1)
        tensor = bucket.buffer()
        dist.all_reduce(tensor)
        tensor /= 2
        result = torch.futures.Future()
        result.set_result(tensor)
        return result

    wrapped.register_comm_hook(None, count_reduction)
    core = TrainerCore(
        Strategy(wrapped), backend, accumulation_steps=accumulation_steps
    )
    generator = torch.Generator().manual_seed(321)
    xs = torch.randn(2, 3, accumulation_steps, 2, 3, generator=generator)
    ys = torch.randn(2, 3, accumulation_steps, 2, 2, generator=generator)
    reference_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    for step in range(3):
        for micro in range(accumulation_steps):
            result = core.train_step(
                TrainBatch(
                    sample_ids=[f"{rank}-{step}-{micro}"],
                    strategy="dflash",
                    tensors={"x": xs[rank, step, micro], "y": ys[rank, step, micro]},
                )
            )
            boundary = micro == accumulation_steps - 1
            assert len(calls) == step + int(boundary), (step, micro, len(calls))
            assert result.optimizer_stepped == boundary
        reference_optimizer.zero_grad()
        (
            reference(xs[:, step].flatten(0, 2)) - ys[:, step].flatten(0, 2)
        ).square().mean().backward()
        reference_optimizer.step()
        torch.testing.assert_close(model.weight, reference.weight, rtol=1e-6, atol=1e-7)
        backend.optimizer.zero_grad(set_to_none=True)
    dist.destroy_process_group()


@unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "requires Gloo")
class TestDDPAccumulation(unittest.TestCase):
    def test_accumulated_updates_match_global_batch_and_reduce_once(self):
        for accumulation_steps, bucket_cap in ((1, None), (4, None), (4, "8")):
            with self.subTest(
                accumulation_steps=accumulation_steps, bucket_cap=bucket_cap
            ):
                with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ):
                    os.environ.pop("SPECFORGE_DDP_BUCKET_CAP_MB", None)
                    if bucket_cap is not None:
                        os.environ["SPECFORGE_DDP_BUCKET_CAP_MB"] = bucket_cap
                    mp.spawn(
                        worker,
                        args=(os.path.join(directory, "gloo"), accumulation_steps),
                        nprocs=2,
                        join=True,
                    )


if __name__ == "__main__":
    unittest.main()
