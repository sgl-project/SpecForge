# coding=utf-8
"""NO_SHARD (DDP) gradient accumulation: no reduction on non-boundary micro-steps.

DDP decides at forward time whether the backward reduces, so ``no_sync()``
around ``backward()`` alone still synchronizes every micro-step.  The backend's
``micro_step()`` context covers the forward; ``TrainerCore`` uses it.  2 GPUs.
"""

import json
import os
import tempfile
import unittest

import torch

WORLD, BS, ACC, STEPS = 2, 2, 2, 3
CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0


def _worker(rank, world_size, workdir, results_dir):
    import torch.distributed as dist

    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29584")

    from specforge.algorithms.eagle3.model import OnlineEagle3Model
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.target.target_head import TargetHead
    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDP1TrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import Eagle3TrainStrategy, StepContext

    draft_config = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
    head = TargetHead.from_pretrained(
        os.path.join(workdir, "target"), lm_head_key="lm_head.weight"
    )
    loader = fx.build_offline_eagle3_loader(
        os.path.join(workdir, "features"), batch_size=BS, run_id="ddp-acc",
        ttt_length=3, max_len=512, ref_slice=slice(rank * BS, (rank + 1) * BS),
    )
    batch = next(iter(loader))  # DISTINCT per rank

    def fresh():
        torch.manual_seed(0)
        dm = AutoDraftModel.from_config(
            draft_config, attention_backend="flex_attention", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(os.path.join(workdir, "vm.pt"))
        dm.freeze_embedding()
        return OnlineEagle3Model(draft_model=dm, length=3, attention_backend="flex_attention").cuda()

    def make_backend():
        pc = ParallelConfig.from_distributed()
        pc.sharding_strategy = "NO_SHARD"
        return FSDP1TrainingBackend(
            pc, optimizer_factory=lambda m: BF16Optimizer(
                m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=100
            ),
        )

    def grads_equal_across_ranks(param):
        g = param.grad.detach().float() if param.grad is not None else torch.zeros_like(param, dtype=torch.float32)
        gathered = [torch.empty_like(g) for _ in range(world_size)]
        dist.all_gather(gathered, g)
        return bool(torch.equal(gathered[0], gathered[1]))

    results = {}
    # path A: TrainerCore accumulation (micro_step context + backward)
    backend = make_backend()
    model = fresh()
    wrapped = backend.prepare_model(model, optimizer_target=model.draft_model)
    core = TrainerCore(Eagle3TrainStrategy(wrapped, target_head=head), backend, accumulation_steps=ACC)
    param = next(p for p in wrapped.module.draft_model.parameters() if p.requires_grad)
    equal = []
    for step in range(STEPS):
        ctx = StepContext(global_step=step, total_steps=STEPS)
        for _ in range(ACC):
            core.train_step(batch, ctx)
            torch.cuda.synchronize()
            equal.append(grads_equal_across_ranks(param))
    results["grads_equal_across_ranks_after_micro"] = equal
    full_a = backend.state_dict()["model"]

    # path B: reduce on every micro-step, same math up to reduction order
    backend_b = make_backend()
    model_b = fresh()
    wrapped_b = backend_b.prepare_model(model_b, optimizer_target=model_b.draft_model)
    strategy_b = Eagle3TrainStrategy(wrapped_b, target_head=head)
    for step in range(STEPS):
        ctx = StepContext(global_step=step, total_steps=STEPS)
        for _ in range(ACC):
            out = strategy_b.forward_loss(batch, ctx)
            backend_b.backward(out.loss / ACC, is_boundary=True)
        backend_b.step()
    full_b = backend_b.state_dict()["model"]
    if rank == 0:
        results["weight_diff"] = max(
            float((full_a[k].float() - full_b[k].float()).abs().max()) for k in full_a
        )

    # misuse is loud: a non-boundary backward whose forward ran outside micro_step()
    out = strategy_b.forward_loss(batch, StepContext(global_step=0, total_steps=1))
    try:
        backend_b.backward(out.loss, is_boundary=False)
        results["misuse_raises"] = False
    except RuntimeError:
        results["misuse_raises"] = True

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(results, f)
    dist.barrier()
    dist.destroy_process_group()


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestDDPAccumulation(unittest.TestCase):
    def test_no_reduction_before_the_boundary(self):
        import torch.multiprocessing as mp

        from tests.test_runtime import _fixtures as fx

        workdir = tempfile.mkdtemp(prefix="ddp_acc_")
        fx.write_draft_config(os.path.join(workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(workdir, "target"))
        fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        fx.write_offline_files(os.path.join(workdir, "features"), n=4)
        results_dir = os.path.join(workdir, "results")
        os.makedirs(results_dir)
        mp.spawn(_worker, args=(WORLD, workdir, results_dir), nprocs=WORLD, join=True)
        r0 = json.load(open(os.path.join(results_dir, "rank0.json")))
        # distinct data per rank: local grads differ until the boundary reduces them
        self.assertEqual(r0["grads_equal_across_ranks_after_micro"], [False, True] * STEPS, r0)
        self.assertLess(r0["weight_diff"], 1e-2, r0)
        self.assertTrue(r0["misuse_raises"], r0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
