# coding=utf-8
"""M6 gate: the trainer split is equivalent under tp>1 & sp>1 (>=4 ranks).

Spawns a 4-process group with TP=2 x SP(ulysses)=2 and, on every rank, runs ONE
offline EAGLE3 step through both the legacy path (``run_forward`` +
``run_backward_and_update``) and the new path (``Eagle3TrainStrategy`` +
``TrainerCore`` + ``FSDPTrainingBackend`` carrying the real 4-rank
``ParallelConfig``). Both consume the *identical* USP-sharded data for that rank,
so any divergence is the new path failing to preserve the math under real TP+SP —
the "scale-out claim met, not FSDP-only" gate.

GPU-only; needs >=4 visible GPUs AND flash-attn **v2** (the Ulysses/USP attention
path resolves flash-attn v2 varlen kernels in ``llama3_eagle`` — there is no SP
fallback). The cached ``sglang:dev`` image used for the other gates ships
flash-attn **v4**, whose API differs, so SpecForge's USP path is non-functional
there and this test skips. Run on a training image with flash-attn v2. Validated
up to the USP-engages point on 8xH200; full numerical assertion needs v2.
"""

import json
import os
import tempfile
import types
import unittest

import torch

CUDA = torch.cuda.is_available()
NGPU = torch.cuda.device_count() if CUDA else 0
WORLD = 4


def _has_usp_flash_attn() -> bool:
    # Authoritative check: the USP attention path resolves flash-attn *v2* varlen
    # kernels in llama3_eagle. Probe that exact symbol — it is None when flash-attn
    # is absent OR when an API-incompatible major version is installed (e.g. the
    # cached sglang:dev image ships flash-attn v4, whose API differs, so SpecForge's
    # USP path is non-functional there). A bare `import flash_attn` is NOT enough.
    try:
        from specforge.modeling.draft.llama3_eagle import _std_flash_unpad_input

        return _std_flash_unpad_input is not None
    except Exception:
        return False


def _worker(rank, world_size, workdir, results_dir):
    """One rank: build old+new with identical weights, one step, record result."""
    from tests.test_runtime import _fixtures as fx

    fx.init_rank_distributed(
        rank, world_size, tp_size=2, sp_ulysses_size=2, sp_ring_size=1, port="29573"
    )

    from scripts.train_eagle3 import run_backward_and_update, run_forward
    from specforge import AutoDraftModelConfig, AutoEagle3DraftModel, OnlineEagle3Model
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.modeling.target import TargetHead
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.contracts import TrainBatch
    from specforge.runtime.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.runtime.training.strategy import Eagle3TrainStrategy
    from specforge.runtime.training.trainer import TrainerCore

    torch.manual_seed(0)
    torch.use_deterministic_algorithms(True, warn_only=True)

    TTT, BS = 3, 2
    cfg = os.path.join(workdir, "draft.json")
    target_dir = os.path.join(workdir, "target")
    vocab_path = os.path.join(workdir, "vm.pt")
    feat_dir = os.path.join(workdir, "features")
    draft_config = AutoDraftModelConfig.from_file(cfg)

    def build_model():
        # attention_backend="usp" so the forward actually exercises Ulysses SP
        # (UspAdapter + SeqAllToAll cross-rank attention), matching production
        # when sp>1; flex_attention would silently run per-rank-local attention
        # and the use_usp_preprocess-sharded data would shape-mismatch.
        dm = AutoEagle3DraftModel.from_config(
            draft_config, attention_backend="usp", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(vocab_path)
        dm.freeze_embedding()
        return OnlineEagle3Model(
            draft_model=dm, length=TTT, attention_backend="usp"
        ).cuda()

    model_a = build_model()
    model_b = build_model()
    model_b.draft_model.load_state_dict(model_a.draft_model.state_dict())  # identical
    head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")

    # USP-sharded offline data: this rank gets its SP slice (the real sp>1 path).
    files = sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir))
    ds = OfflineEagle3Dataset(
        files, max_len=512, ttt_length=TTT, use_usp_preprocess=True
    )
    data = DataCollatorWithPadding()([ds[j] for j in range(BS)])

    args = types.SimpleNamespace(
        is_vlm=False,
        target_model_backend="hf",
        shard_target_output=False,
        draft_accumulation_steps=1,
    )

    # --- legacy path ---
    opt_a = BF16Optimizer(
        model_a.draft_model,
        lr=1e-4,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=10,
    )
    plosses_a, _, _, _, _, _, _ = run_forward(
        args, model_a, data, head, is_online=False
    )
    loss_old = sum(0.8**i * plosses_a[i] for i in range(len(plosses_a))).item()
    gn_old = run_backward_and_update(args, plosses_a, opt_a, global_step=1)

    # --- new path under the real 4-rank ParallelConfig ---
    opt_b = BF16Optimizer(
        model_b.draft_model,
        lr=1e-4,
        max_grad_norm=0.5,
        warmup_ratio=0.0,
        total_steps=10,
    )
    pc = ParallelConfig.from_distributed(tp_size=2, sp_ulysses_size=2, sp_ring_size=1)
    backend = FSDPTrainingBackend(pc)
    backend.prepare_model(model_b, wrap=False, optimizer_target=model_b.draft_model)
    backend.set_optimizer(opt_b)
    strategy = Eagle3TrainStrategy(model_b, target_head=head)
    core = TrainerCore(strategy, backend, accumulation_steps=1)
    batch = TrainBatch(
        sample_ids=["a", "b"],
        strategy="eagle3",
        tensors=dict(data),
        metadata={"target_repr": "hidden_state", "ttt_length": TTT},
    )
    rep = core.train_step(batch)

    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as f:
        json.dump(
            {
                "rank": rank,
                "world": pc.world_size,
                "tp": pc.tp_size,
                "sp": pc.sp_size,
                "loss_old": loss_old,
                "loss_new": rep.loss,
                "gn_old": float(gn_old.item()),
                "gn_new": rep.grad_norm,
            },
            f,
        )


@unittest.skipUnless(
    CUDA and NGPU >= WORLD and _has_usp_flash_attn(),
    f"requires >={WORLD} GPUs and flash-attn v2 (USP/Ulysses attention path)",
)
class TestEquiv4Rank(unittest.TestCase):
    def test_equiv_tp2_sp2(self):
        import torch.multiprocessing as mp

        workdir = tempfile.mkdtemp(prefix="equiv_4rank_")
        from tests.test_runtime import _fixtures as fx

        # build shared fixtures once (deterministic, seed-fixed)
        fx.write_draft_config(os.path.join(workdir, "draft.json"))
        fx.write_target_head_dir(os.path.join(workdir, "target"))
        fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        fx.write_offline_files(os.path.join(workdir, "features"), n=8, seq=64)

        results_dir = os.path.join(workdir, "results")
        os.makedirs(results_dir, exist_ok=True)

        mp.spawn(_worker, args=(WORLD, workdir, results_dir), nprocs=WORLD, join=True)

        results = []
        for r in range(WORLD):
            with open(os.path.join(results_dir, f"rank{r}.json")) as f:
                results.append(json.load(f))
        self.assertEqual(len(results), WORLD)

        for res in results:
            self.assertEqual(res["world"], WORLD)
            self.assertEqual(res["tp"], 2)
            self.assertEqual(res["sp"], 2)
            # per-rank: new path reproduces legacy loss on the same SP slice
            self.assertAlmostEqual(
                res["loss_old"],
                res["loss_new"],
                places=3,
                msg=f"rank {res['rank']} loss: old={res['loss_old']} new={res['loss_new']}",
            )
            # grad-norm reduction matches (both reduce across the world group)
            self.assertAlmostEqual(
                res["gn_old"],
                res["gn_new"],
                places=2,
                msg=f"rank {res['rank']} grad_norm: old={res['gn_old']} new={res['gn_new']}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
