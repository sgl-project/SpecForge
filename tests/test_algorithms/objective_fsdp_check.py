"""Run with torchrun --nproc-per-node=2 -m tests.test_algorithms.objective_fsdp_check."""

import os
from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.modeling import build_draft_model
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import StepContext
from specforge.training.program import program_spec
from specforge.training.runner import DataParallelRunner
from tests.test_algorithms.test_objectives import dflash2_config, training_batch
from tests.test_runtime._fixtures import FakeProgram, fake_spec, reference_runtime


def main():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("nccl", timeout=timedelta(seconds=90))
    try:
        device = torch.device("cuda", torch.cuda.current_device())
        torch.manual_seed(7)
        config = dflash2_config(bound=True)
        model = build_draft_model(config, device=device)
        tables = [model.get_submodule(name) for name in config.target_bindings]
        for table in tables:
            table.load_target_weight(
                torch.randn(config.vocab_size, config.hidden_size, device=device)
            )
        runtime = build_objective_model(
            model, options={"num_anchors": 2}, frozen_modules=tables
        )
        batch = training_batch(config, device)
        torch.manual_seed(19)
        with torch.no_grad():
            reference = runtime(batch, step=0, total_steps=10).loss
        backend = FSDPTrainingBackend(
            ParallelConfig(world_size=2),
            precision=reference_runtime(torch.float32).precision,
        )
        whole = [p.numel() for table in tables for p in table.parameters()]
        wrapped = backend.prepare_model(
            runtime,
            optimizer_target=model,
            frozen_modules=runtime.frozen_modules,
            blocks=program_spec(runtime).partition.blocks,
        )
        # The bound tables stay whole on each rank; outside a forward a sharded
        # parameter would hold its shard.
        assert [p.numel() for table in tables for p in table.parameters()] == whole
        torch.manual_seed(19)
        result = wrapped(batch, step=0, total_steps=10)
        torch.testing.assert_close(result.loss, reference)
        result.loss.backward()
        gradients = [p.grad for p in model.candidate_selector.parameters() if p.numel()]
        assert all(g is not None and torch.isfinite(g).all() for g in gradients)

        # The microbatch reduction: the runner pools a term's denominator over
        # the two ranks, which hold 1 and 2, between forward and backward.
        class Terms(FakeProgram):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(2.0))

            def forward(self, tensors, **context):
                denominator = self.weight.new_tensor(dist.get_rank() + 1.0)
                return ObjectiveResult(
                    self.weight.detach(),
                    denominator.new_ones(()),
                    terms=((1.0, self.weight, denominator),),
                )

        program = Terms().to(device)
        replicated = FSDPTrainingBackend(
            ParallelConfig(world_size=2, sharding_strategy="NO_SHARD")
        )
        runner = DataParallelRunner(
            replicated.prepare_model(program),
            replicated,
            fake_spec(reduction="microbatch"),
        )
        runner.forward_backward(
            TrainBatch(["0"], "check", {}), StepContext(), is_boundary=True
        )
        # d/dw of 2 * w / 3 on each rank, averaged over the ranks.
        torch.testing.assert_close(
            program.weight.grad, program.weight.new_tensor(2 / 3)
        )
        if dist.get_rank() == 0:
            print("Two-rank FSDP selector backward and microbatch reduction passed")
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
