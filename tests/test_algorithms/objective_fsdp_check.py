"""Run with torchrun --nproc-per-node=2 -m tests.test_algorithms.objective_fsdp_check."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.base import ObjectiveResult, microbatch_mean
from specforge.modeling import build_draft_model
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from tests.test_algorithms.test_objectives import dflash2_config, training_batch


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)
        config = dflash2_config()
        model = build_draft_model(config, device=device)
        runtime = build_objective_model(model, options={"num_anchors": 2})
        batch = training_batch(config, device)
        torch.manual_seed(19)
        with torch.no_grad():
            reference = runtime(batch, step=0, total_steps=10).loss
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2, param_dtype=torch.float32)
        )
        wrapped = backend.prepare_model(runtime, optimizer_target=model)
        torch.manual_seed(19)
        result = wrapped(batch, step=0, total_steps=10)
        torch.testing.assert_close(result.loss, reference)
        result.loss.backward()
        gradients = [p.grad for p in model.candidate_selector.parameters() if p.numel()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)
        # DSpark uses a global denominator per microbatch, with DDP averaging.
        numerator = torch.tensor(2.0, device=device, requires_grad=True)
        result = microbatch_mean(
            ObjectiveResult(numerator, numerator.new_tensor(dist.get_rank() + 1)),
            global_denominator=True,
        )
        result.loss.backward()
        torch.testing.assert_close(numerator.grad, numerator.new_tensor(2 / 3))
        if dist.get_rank() == 0:
            print(
                "Two-rank FSDP selector backward and global microbatch normalization passed"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
