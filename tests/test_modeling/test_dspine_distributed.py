"""Uneven DSpine supervision must match a globally normalized optimizer step."""

from functools import partial
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import training_inputs, training_model


def fixed_anchors(sequence_length, loss_mask, device, max_valid_anchors=None):
    batch = loss_mask.shape[0]
    return torch.ones(batch, 1, dtype=torch.long, device=device), torch.ones(
        batch, 1, dtype=torch.bool, device=device
    )


def uneven_batch():
    inputs = training_inputs()
    inputs["loss_mask"][0] = torch.tensor([0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    inputs["loss_mask"][1] = torch.tensor([0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    return inputs


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


def distributed_step(rank, directory, accumulation_steps):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=2
    )
    try:
        model = training_model(chunk_size=1)
        model._sample_anchor_positions = fixed_anchors
        inputs = combined_batch(accumulation_steps)
        backend = FSDPTrainingBackend(ParallelConfig(sharding_strategy="NO_SHARD"))
        wrapped = backend.prepare_model(model)
        communication = {"calls": 0}
        wrapped.register_comm_hook(communication, count_gradient_allreduce)
        communication_counts = []
        backend.set_optimizer(torch.optim.SGD(model.draft_model.parameters(), lr=0.0))
        core = TrainerCore(
            DSpineTrainStrategy(wrapped), backend, accumulation_steps=accumulation_steps
        )
        for microstep in range(accumulation_steps):
            index = 2 * microstep + rank
            core.train_step(
                TrainBatch(
                    [str(index)],
                    "dspine",
                    {
                        name: tensor[index : index + 1]
                        for name, tensor in inputs.items()
                    },
                ),
                StepContext(global_step=300, total_steps=600),
            )
            communication_counts.append(communication["calls"])
        torch.save(
            {
                "gradients": {
                    name: parameter.grad
                    for name, parameter in model.draft_model.named_parameters()
                },
                "communication_counts": communication_counts,
                "sync_restored": wrapped.require_backward_grad_sync,
            },
            Path(directory) / f"gradients-{rank}.pt",
        )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_gloo_available(), reason="CPU distributed test requires Gloo"
)
@pytest.mark.parametrize("accumulation_steps", [1, 2, 4])
def test_uneven_rank_losses_match_one_concatenated_batch(tmp_path, accumulation_steps):
    model = training_model(chunk_size=1)
    model._sample_anchor_positions = fixed_anchors
    loss, _, _ = model(
        **combined_batch(accumulation_steps), global_step=300, total_steps=600
    )
    loss.backward()
    expected = {
        name: parameter.grad for name, parameter in model.draft_model.named_parameters()
    }
    mp.spawn(
        distributed_step, args=(str(tmp_path), accumulation_steps), nprocs=2, join=True
    )
    for rank in range(2):
        actual = torch.load(tmp_path / f"gradients-{rank}.pt", weights_only=True)
        assert actual["communication_counts"] == [0] * (accumulation_steps - 1) + [1]
        assert actual["sync_restored"]
        for name, gradient in expected.items():
            assert gradient is not None and actual["gradients"][name] is not None, name
            torch.testing.assert_close(
                actual["gradients"][name], gradient, atol=2e-5, rtol=2e-4
            )


def fsdp_mixed_precision_step(rank, directory):
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"file://{directory}/rendezvous", rank=rank, world_size=1
    )
    try:
        model = training_model(chunk_size=1).to(torch.bfloat16)
        wrapped = FSDP(
            model,
            device_id=torch.device("cpu"),
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, buffer_dtype=torch.float32
            ),
            ignored_modules=[model.lm_head, model.embed_tokens],
            auto_wrap_policy=partial(
                transformer_auto_wrap_policy,
                transformer_layer_cls={type(model.draft_model.layers[0])},
            ),
        )
        backend = FSDPTrainingBackend(ParallelConfig())
        backend.prepare_model(wrapped, wrap=False)
        backend.set_optimizer(torch.optim.SGD(model.draft_model.parameters(), lr=0.0))
        core = TrainerCore(DSpineTrainStrategy(wrapped), backend, accumulation_steps=2)
        inputs = training_inputs()
        for name in ("hidden_states", "target_last_hidden_states"):
            inputs[name] = inputs[name].to(torch.bfloat16)
        for step in (100, 300):
            wrapped.zero_grad(set_to_none=True)
            for index in range(2):
                result = core.train_step(
                    TrainBatch(
                        [str(index)],
                        "dspine",
                        {
                            name: tensor[index : index + 1]
                            for name, tensor in inputs.items()
                        },
                    ),
                    StepContext(global_step=step, total_steps=600),
                )
            assert result.optimizer_stepped
            assert torch.isfinite(torch.tensor(result.loss))
            assert model.draft_model.transfer_codes.dtype == torch.float32
            assert all(
                parameter.grad is not None and torch.isfinite(parameter.grad).all()
                for parameter in model.draft_model.parameters()
            )
    finally:
        dist.destroy_process_group()


@pytest.mark.skipif(not dist.is_gloo_available(), reason="CPU FSDP test requires Gloo")
def test_fsdp_bf16_training_with_fp32_buffers_and_window_replay(tmp_path):
    mp.spawn(fsdp_mixed_precision_step, args=(str(tmp_path),), nprocs=1, join=True)
