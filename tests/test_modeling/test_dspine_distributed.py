"""Uneven DSpine supervision must match a globally normalized optimizer step."""

import unittest
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.runner import DataParallelRunner
from specforge.training.strategies.base import StepContext
from tests.test_modeling.dspine_fixtures import training_inputs, training_model
from tests.test_runtime._fixtures import build_strategy, train_window


def fixed_anchors(valid, num_anchors, available=None):
    batch, device = valid.shape[0], valid.device
    return torch.ones(batch, 1, dtype=torch.long, device=device), torch.ones(
        batch, 1, dtype=torch.bool, device=device
    )


def uneven_batch():
    inputs = training_inputs()
    inputs["loss_mask"][0] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    inputs["loss_mask"][1] = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return inputs


def per_microbatch_objective(model, batch, width, ctx):
    """Normalize refinement locally, with one detached alignment CE per window."""
    size = batch["input_ids"].shape[0]
    parts = [
        {key: value[start : start + width] for key, value in batch.items()}
        for start in range(0, size, width)
    ]
    step = dict(step=ctx.global_step, total_steps=ctx.total_steps)
    with torch.no_grad():
        pairs = [model(part, **step).ratios["dspine/backbone_ce"] for part in parts]
        window_ce = sum(n for n, _ in pairs) / sum(d for _, d in pairs)
    return (
        sum(model(part, alignment_ce=window_ce, **step).numerator for part in parts)
        / size
    )


def combined_batch(accumulation_steps):
    batches = [uneven_batch() for _ in range(accumulation_steps)]
    return {name: torch.cat([batch[name] for batch in batches]) for name in batches[0]}


def count_gradient_allreduce(state, bucket):
    state["calls"] += 1
    buffer = bucket.buffer()
    dist.all_reduce(buffer)
    buffer.div_(dist.get_world_size())
    future = torch.futures.Future()
    future.set_result(buffer)
    return future


class CountingRunner(DataParallelRunner):
    """Notes the gradient all-reduces issued by the end of each micro-batch."""

    def __init__(self, strategy, backend, communication):
        super().__init__(strategy, backend)
        self.communication = communication
        self.communication_counts = []

    def forward_backward(self, batch, ctx, *, is_boundary):
        pending = super().forward_backward(batch, ctx, is_boundary=is_boundary)
        self.communication_counts.append(self.communication["calls"])
        return pending


def distributed_step(rank, directory, accumulation_steps):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2
    )
    try:
        model = training_model(chunk_size=1)
        anchor_patch = patch(
            "specforge.algorithms.objectives.inputs.sample_anchors", fixed_anchors
        )
        anchor_patch.start()
        inputs = combined_batch(accumulation_steps)
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="NO_SHARD"))
        wrapped = backend.prepare_model(model)
        communication = {"calls": 0}
        wrapped.register_comm_hook(communication, count_gradient_allreduce)
        backend.set_optimizer(torch.optim.SGD(model.draft_model.parameters(), lr=0.0))
        strategy = build_strategy(wrapped)
        runner = CountingRunner(strategy, backend, communication)
        train_window(
            runner,
            [
                TrainBatch(
                    [str(index)],
                    "dspine",
                    {
                        name: tensor[index : index + 1]
                        for name, tensor in inputs.items()
                    },
                )
                for index in range(rank, 2 * accumulation_steps, 2)
            ],
            StepContext(global_step=300, total_steps=600),
        )
        torch.save(
            {
                "gradients": {
                    name: parameter.grad
                    for name, parameter in model.draft_model.named_parameters()
                },
                "communication_counts": runner.communication_counts,
                "sync_restored": wrapped.require_backward_grad_sync,
            },
            Path(directory) / f"gradients-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


def fsdp_mixed_precision_step(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=1
    )
    try:
        model = training_model(chunk_size=1, dtype=torch.bfloat16)
        wrapped = FSDP(
            model,
            device_id=torch.device("cpu"),
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, buffer_dtype=torch.float32
            ),
            ignored_modules=[
                model.target_head,
                model.draft_model.embed_tokens,
                model.draft_model.lm_head,
            ],
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={type(model.draft_model.backbone.layers[0])},
            ),
        )
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(wrapped, wrap=False)
        backend.set_optimizer(torch.optim.SGD(model.draft_model.parameters(), lr=0.0))
        strategy = build_strategy(wrapped)
        runner = DataParallelRunner(strategy, backend)
        inputs = training_inputs()
        for name in ("hidden_states", "target_last_hidden_states"):
            inputs[name] = inputs[name].to(torch.bfloat16)
        batches = [
            TrainBatch(
                [str(index)],
                "dspine",
                {name: tensor[index : index + 1] for name, tensor in inputs.items()},
            )
            for index in range(2)
        ]
        for step in (100, 300):
            wrapped.zero_grad(set_to_none=True)
            metrics = train_window(
                runner,
                batches,
                StepContext(global_step=step, total_steps=600),
            )
            assert torch.isfinite(torch.tensor(metrics.loss))
            assert model.draft_model.transfer_codes.dtype == torch.float32
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.draft_model.parameters()
                if parameter.requires_grad
            )
    finally:
        dist.destroy_process_group()


class DSpineDistributedTest(unittest.TestCase):

    @unittest.skipIf(not dist.is_gloo_available(), "CPU distributed test requires Gloo")
    def test_uneven_rank_losses_match_one_concatenated_batch(self):
        for accumulation_steps in [1, 2]:
            with self.subTest(accumulation_steps=accumulation_steps):
                with TemporaryDirectory() as directory:
                    tmp_path = Path(directory)
                    model = training_model(chunk_size=1)
                    self.enterContext(
                        patch(
                            "specforge.algorithms.objectives.inputs.sample_anchors",
                            fixed_anchors,
                        )
                    )
                    loss = per_microbatch_objective(
                        model,
                        combined_batch(accumulation_steps),
                        1,
                        StepContext(global_step=300, total_steps=600),
                    )
                    loss.backward()
                    expected = {
                        name: parameter.grad
                        for (name, parameter) in model.draft_model.named_parameters()
                        if parameter.requires_grad
                    }
                    mp.spawn(
                        distributed_step,
                        args=(str(tmp_path), accumulation_steps),
                        nprocs=2,
                        join=True,
                    )
                    for rank in range(2):
                        actual = torch.load(
                            tmp_path / f"gradients-{rank}.pt", weights_only=True
                        )
                        self.assertEqual(
                            actual["communication_counts"],
                            [0] * (accumulation_steps - 1) + [1],
                        )
                        self.assertTrue(actual["sync_restored"])
                        for name, gradient in expected.items():
                            self.assertTrue(
                                gradient is not None
                                and actual["gradients"][name] is not None,
                                name,
                            )
                            torch.testing.assert_close(
                                actual["gradients"][name],
                                gradient,
                                atol=2e-05,
                                rtol=0.0002,
                            )

    @unittest.skipIf(not dist.is_gloo_available(), "CPU FSDP test requires Gloo")
    def test_fsdp_bf16_training_with_fp32_buffers_and_window_replay(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            mp.spawn(
                fsdp_mixed_precision_step, args=(str(tmp_path),), nprocs=1, join=True
            )


if __name__ == "__main__":
    unittest.main()
