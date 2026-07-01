# coding=utf-8
"""Phase D gate: grad-accumulation no_sync() is math-equivalent to reducing every step.

Spawns a 2-rank FSDP (pure data-parallel) group and, on every rank, runs one
optimizer step with accumulation_steps=2 two ways over the SAME micro-batch:
- ``TrainerCore`` (the new path): no_sync() on the non-boundary micro-step, so the
  gradient is reduced once, on the boundary backward;
- a bare path that forces the reduction on every micro-step (is_boundary=True).

The resulting draft weights must match: no_sync only defers *when* the (linear)
reduction happens, not its result. Using the same micro-batch twice keeps the
comparison exact (scaling the summed gradient by a power of two commutes with
rounding), so the check is bit-tight rather than merely close.

GPU-only; needs >=2 visible GPUs. Uses flex_attention (no flash-attn v2 needed).
Run on the H200 box via rcli.
"""

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
BS_FIXTURE = 2


def _worker(rank, world_size, workdir, results_dir):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(
        rank, world_size, tp_size=1, sp_ulysses_size=1, sp_ring_size=1, port="29581"
    )

    from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.modeling.target import TargetHead
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.strategies.base import Eagle3TrainStrategy
    from specforge.training.controller import TrainerCore

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    TTT, BS, ACC = 3, 2, 2
    draft_config = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
    target_dir = os.path.join(workdir, "target")
    vocab_path = os.path.join(workdir, "vm.pt")
    feat_dir = os.path.join(workdir, "features")

    def build_model():
        dm = AutoEagle3DraftModel.from_config(
            draft_config, attention_backend="flex_attention", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(vocab_path)
        dm.freeze_embedding()
        return OnlineEagle3Model(
            draft_model=dm, length=TTT, attention_backend="flex_attention"
        ).cuda()

    model_a = build_model()
    model_b = build_model()
    model_b.draft_model.load_state_dict(model_a.draft_model.state_dict())  # identical
    head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")

    files = sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir))
    ds = OfflineEagle3Dataset(files, max_len=512, ttt_length=TTT)
    data = DataCollatorWithPadding()([ds[j] for j in range(BS)])
    batch = TrainBatch(
        sample_ids=["a", "b"],
        strategy="eagle3",
        tensors=dict(data),
        metadata={"target_repr": "hidden_state", "ttt_length": TTT},
    )

    def opt_factory(m):
        return BF16Optimizer(
            m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
        )

    pc = ParallelConfig.from_distributed(tp_size=1, sp_ulysses_size=1, sp_ring_size=1)

    # Path A: no_sync via TrainerCore accumulation (reduce once, on the boundary).
    backend_a = FSDPTrainingBackend(pc, optimizer_factory=opt_factory)
    backend_a.prepare_model(model_a, wrap=True, optimizer_target=model_a.draft_model)
    # Count the deferrals: every NON-boundary micro-step must enter no_sync()
    # (gradient reduction deferred), so the boundary backward is the single
    # reduction per optimizer step — the roadmap's "one all-reduce per step".
    no_sync_entries = {"n": 0}
    _orig_no_sync = backend_a.module.no_sync

    def _counting_no_sync(*args, **kwargs):
        no_sync_entries["n"] += 1
        return _orig_no_sync(*args, **kwargs)

    backend_a.module.no_sync = _counting_no_sync
    strat_a = Eagle3TrainStrategy(backend_a.module, target_head=head)
    core_a = TrainerCore(strat_a, backend_a, accumulation_steps=ACC)
    for _ in range(ACC):
        rep_a = core_a.train_step(batch)

    # Path B: force the reduction on every micro-step (is_boundary=True), same math.
    backend_b = FSDPTrainingBackend(pc, optimizer_factory=opt_factory)
    backend_b.prepare_model(model_b, wrap=True, optimizer_target=model_b.draft_model)
    strat_b = Eagle3TrainStrategy(backend_b.module, target_head=head)
    for _ in range(ACC):
        out = strat_b.forward_loss(batch)
        backend_b.backward(out.loss / ACC, is_boundary=True)
    gn_b = backend_b.step()

    sd_a = strat_a.checkpoint_state_filter(backend_a.state_dict()["model"])
    sd_b = strat_b.checkpoint_state_filter(backend_b.state_dict()["model"])
    max_diff = max(float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a)
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(
            {
                "rank": rank,
                "max_weight_diff": max_diff,
                "gn_a": rep_a.grad_norm,
                "gn_b": float(gn_b.item()),
                "no_sync_entries": no_sync_entries["n"],
                "acc": ACC,
            },
            f,
        )


@unittest.skipUnless(CUDA and NGPU >= WORLD, f"requires >={WORLD} GPUs")
class TestNoSyncEquiv(unittest.TestCase):
    def test_no_sync_matches_per_step_reduction(self):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix="no_sync_equiv_")
        from tests.test_runtime import _fixtures as fx

        fx.write_draft_config(os.path.join(workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(workdir, "target"))
        fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        fx.write_offline_files(os.path.join(workdir, "features"), n=BS_FIXTURE, seq=64)

        results_dir = os.path.join(workdir, "results")
        os.makedirs(results_dir, exist_ok=True)
        mp.spawn(_worker, args=(WORLD, workdir, results_dir), nprocs=WORLD, join=True)

        for r in range(WORLD):
            with open(os.path.join(results_dir, f"rank{r}.json")) as f:
                res = json.load(f)
            # no_sync only defers the (linear) reduction — same weights out.
            self.assertLess(
                res["max_weight_diff"],
                1e-4,
                msg=f"rank {r}: no_sync weights diverged by {res['max_weight_diff']}",
            )
            # every non-boundary micro-step deferred its reduction: exactly one
            # gradient reduction (the boundary backward) per optimizer step.
            self.assertEqual(
                res["no_sync_entries"],
                res["acc"] - 1,
                msg=f"rank {r}: expected {res['acc'] - 1} no_sync deferrals, "
                f"got {res['no_sync_entries']}",
            )
            self.assertAlmostEqual(res["gn_a"], res["gn_b"], places=2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
