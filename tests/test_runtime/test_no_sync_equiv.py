# coding=utf-8
"""Gate: grad-accumulation no_sync() is math-equivalent to reducing every micro-step.

2-rank FSDP DP group with DISTINCT per-rank micro-batches, so a skipped or wrong
cross-rank reduction shows up as a weight diff between path A (TrainerCore,
no_sync off the boundary) and path B (reduce on every micro-step). Reusing one
batch per rank keeps the check exact: power-of-two loss scaling commutes with
rounding. GPU-only (>=2 GPUs, flex_attention); run on the H200 box via rcli."""

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 2
BS = 2  # per-rank micro-batch
BS_FIXTURE = WORLD * BS  # distinct samples for every rank


def _worker(rank, world_size, workdir, results_dir):
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(rank, world_size, port="29581")

    from specforge.core.eagle3 import OnlineEagle3Model
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
    from specforge.modeling.target.target_head import TargetHead
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import Eagle3TrainStrategy

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    TTT, ACC = 3, 2
    draft_config = AutoDraftModelConfig.from_file(os.path.join(workdir, "draft.json"))
    target_dir = os.path.join(workdir, "target")
    vocab_path = os.path.join(workdir, "vm.pt")
    feat_dir = os.path.join(workdir, "features")

    def build_model():
        dm = AutoDraftModel.from_config(
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
    ds = OfflineEagle3Dataset(files, max_len=512)
    # DISTINCT shard per rank — with identical shards a broken (skipped/no-op)
    # cross-rank reduction would be invisible.
    data = DataCollatorWithPadding()([ds[j] for j in range(rank * BS, (rank + 1) * BS)])
    batch = TrainBatch(
        sample_ids=[f"r{rank}-{j}" for j in range(BS)],
        strategy="eagle3",
        tensors=dict(data),
        metadata={"target_repr": "hidden_state", "ttt_length": TTT},
    )

    def opt_factory(m):
        return BF16Optimizer(
            m, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
        )

    pc = ParallelConfig.from_distributed()

    # Path A: no_sync via TrainerCore accumulation (reduce once, on the boundary).
    backend_a = FSDPTrainingBackend(pc, optimizer_factory=opt_factory)
    backend_a.prepare_model(model_a, wrap=True, optimizer_target=model_a.draft_model)
    # Count deferrals: every NON-boundary micro-step must enter no_sync(), so
    # the boundary backward is the single reduction per optimizer step.
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

    # state_dict() is collective on every rank; the wrapped FSDP gather is
    # rank0-only (CPU tensors on rank0, {} elsewhere).
    full_a = backend_a.state_dict()["model"]
    full_b = backend_b.state_dict()["model"]
    max_diff = None
    if full_a:
        sd_a = strat_a.checkpoint_state_filter(full_a)
        sd_b = strat_b.checkpoint_state_filter(full_b)
        max_diff = max(
            float((sd_a[k].float() - sd_b[k].float()).abs().max()) for k in sd_a
        )
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

        checked_weights = 0
        for r in range(WORLD):
            with open(os.path.join(results_dir, f"rank{r}.json")) as f:
                res = json.load(f)
            # no_sync only defers the (linear) reduction — same weights out.
            if res["max_weight_diff"] is not None:
                checked_weights += 1
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
        self.assertGreaterEqual(
            checked_weights, 1, "no rank saw the gathered full weights"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
