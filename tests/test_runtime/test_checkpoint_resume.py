# coding=utf-8
"""Phase D gate: checkpoint save -> resume restores weights, and continues training.

Two checks (single rank, GPU):
- ``test_checkpoint_resume`` — persisted draft weights + global step round-trip
  exactly into a fresh model, and the on-disk layout splits the rank0 shared
  payload from the per-rank optimizer/RNG file.
- ``test_resume_loss_curve_continuity`` — after save->resume (draft weights +
  optimizer + LR scheduler + RNG + **data position** restored), the loss curve
  continues to match an uninterrupted run over a sequence of DISTINCT batches —
  so the resumed run demonstrably trains on the batches the interrupted run had
  not reached, not on the epoch prefix again. Continuity is asserted to a
  tolerance, not bit-exact: BF16Optimizer reconstructs its fp32 master from the
  persisted bf16 weights on resume, exactly as the legacy trainer does, so the
  master loses the low mantissa bits the uninterrupted run kept.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


def _make_batches(feat_dir, bs, ttt, count):
    """``count`` DISTINCT collated eagle3 batches (batch i = samples [i*bs, (i+1)*bs))
    so a resumed run only matches the reference if it resumes at the right batch."""
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.runtime.contracts import TrainBatch

    files = sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir))
    ds = OfflineEagle3Dataset(files, max_len=512)
    collate = DataCollatorWithPadding()
    batches = []
    for i in range(count):
        data = collate([ds[j] for j in range(i * bs, (i + 1) * bs)])
        batches.append(
            TrainBatch(
                sample_ids=[str(j) for j in range(i * bs, (i + 1) * bs)],
                strategy="eagle3",
                tensors=dict(data),
                metadata={"target_repr": "hidden_state", "ttt_length": ttt},
            )
        )
    return batches


@unittest.skipUnless(CUDA, "checkpoint resume requires CUDA")
class TestCheckpointResume(unittest.TestCase):
    def test_checkpoint_resume(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29563")

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.data.preprocessing import OfflineEagle3Dataset
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.modeling.target import TargetHead
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.runtime.training.backend import (
            FSDPTrainingBackend,
            ParallelConfig,
        )
        from specforge.runtime.training.strategy import Eagle3TrainStrategy
        from specforge.runtime.training.trainer import TrainerController, TrainerCore

        TTT, BS, N = 3, 2, 6
        workdir = tempfile.mkdtemp(prefix="ckpt_resume_")
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(workdir, "target"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)
        out_dir = os.path.join(workdir, "out")

        draft_config = AutoDraftModelConfig.from_file(cfg)

        def build_model():
            dm = AutoEagle3DraftModel.from_config(
                draft_config,
                attention_backend="flex_attention",
                torch_dtype=torch.bfloat16,
            ).cuda()
            dm.load_vocab_mapping(vocab_path)
            dm.freeze_embedding()
            return OnlineEagle3Model(
                dm, length=TTT, attention_backend="flex_attention"
            ).cuda()

        head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
        ds = OfflineEagle3Dataset(
            sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir)), max_len=512
        )
        collate = DataCollatorWithPadding()

        def make_batches():
            out = []
            for s in range(0, N, BS):
                data = collate([ds[j] for j in range(s, s + BS)])
                out.append(
                    TrainBatch(
                        sample_ids=[str(j) for j in range(s, s + BS)],
                        strategy="eagle3",
                        tensors=dict(data),
                        metadata={"target_repr": "hidden_state", "ttt_length": TTT},
                    )
                )
            return out

        model = build_model()
        opt = BF16Optimizer(
            model.draft_model,
            lr=1e-3,
            max_grad_norm=0.5,
            warmup_ratio=0.0,
            total_steps=10,
        )
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)  # register module (no FSDP at 1 rank)
        backend.set_optimizer(opt)
        strategy = Eagle3TrainStrategy(model, target_head=head)
        core = TrainerCore(strategy, backend)
        ctrl = TrainerController(
            core, run_id="r", output_dir=out_dir, max_steps=3, num_epochs=2
        )
        step = ctrl.fit(make_batches())
        self.assertEqual(step, 3)
        wv = ctrl.save_checkpoint(step)

        # reload into a fresh model and compare persisted (non-embedding) weights
        from specforge.training.checkpoint import CheckpointManager

        ckpt_dir = wv.checkpoint_uri[len("file://") :]
        shared = torch.load(
            os.path.join(ckpt_dir, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(shared["global_step"], 3)
        self.assertEqual(shared["strategy"], "eagle3")
        self.assertEqual(shared["world_size"], 1)
        # rank-local state lives in per-rank files, NOT the shared payload
        self.assertNotIn("optimizer_state_dict", shared)
        self.assertTrue(
            os.path.exists(os.path.join(ckpt_dir, "training_state_rank0.pt"))
        )
        # the one reader merges this rank's optimizer/RNG back in
        ckpt = CheckpointManager.read_resume_state(wv.checkpoint_uri)
        self.assertIn("optimizer_state_dict", ckpt)
        self.assertIn("rng_state", ckpt)

        fresh = build_model()
        missing, unexpected = fresh.draft_model.load_state_dict(
            ckpt["draft_state_dict"], strict=False
        )
        self.assertEqual(unexpected, [])  # all persisted keys belong to the draft
        # every persisted weight must now match the trained model bit-for-bit
        trained = strategy.checkpoint_state_filter(backend.state_dict()["model"])
        fresh_sd = fresh.draft_model.state_dict()
        for k, v in trained.items():
            self.assertTrue(
                torch.equal(v.cpu(), fresh_sd[k].cpu()), msg=f"weight {k} mismatch"
            )

    def test_resume_loss_curve_continuity(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29564")

        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.training.backend import (
            FSDPTrainingBackend,
            ParallelConfig,
        )
        from specforge.runtime.training.strategy import Eagle3TrainStrategy
        from specforge.runtime.training.trainer import TrainerController, TrainerCore

        TTT, BS, TOTAL, CUT = 3, 2, 6, 3
        workdir = tempfile.mkdtemp(prefix="ckpt_continuity_")
        # TOTAL distinct batches: resuming on the wrong data cannot pass.
        feat_dir = fx.write_offline_files(
            os.path.join(workdir, "features"), n=BS * TOTAL
        )
        batches = _make_batches(feat_dir, BS, TTT, TOTAL)

        def build_run(run_id, out_dir):
            torch.manual_seed(0)
            torch.use_deterministic_algorithms(True, warn_only=True)
            model, head = fx.build_eagle3(workdir, ttt=TTT)
            backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
            backend.prepare_model(model, wrap=False)
            backend.set_optimizer(
                BF16Optimizer(
                    model.draft_model,
                    lr=1e-3,
                    max_grad_norm=0.5,
                    warmup_ratio=0.0,
                    total_steps=100,
                )
            )
            core = TrainerCore(Eagle3TrainStrategy(model, target_head=head), backend)
            return model, head, backend, core

        # Reference: one uninterrupted run of TOTAL steps.
        losses_ref = []
        _, _, _, core_r = build_run("ref", os.path.join(workdir, "ref"))
        ctrl_r = TrainerController(
            core_r,
            run_id="ref",
            output_dir=os.path.join(workdir, "ref"),
            max_steps=TOTAL,
            log_interval=1,
            logger=lambda m, s: losses_ref.append(m["loss"]),
        )
        torch.manual_seed(0)
        ctrl_r.fit(batches)
        self.assertEqual(len(losses_ref), TOTAL)

        # Interrupted phase 1: CUT steps, then save.
        out_a = os.path.join(workdir, "rez")
        losses_a = []
        model_a, _, backend_a, core_a = build_run("rez", out_a)
        ctrl_a = TrainerController(
            core_a,
            run_id="rez",
            output_dir=out_a,
            max_steps=CUT,
            log_interval=1,
            logger=lambda m, s: losses_a.append(m["loss"]),
        )
        torch.manual_seed(0)
        ctrl_a.fit(batches)
        ck = ctrl_a.save_checkpoint(CUT)
        self.assertEqual(ctrl_a.epoch_batch, CUT)  # data position at the cut
        # phase 1 mirrors the reference exactly for the first CUT steps
        for i in range(CUT):
            self.assertAlmostEqual(losses_a[i], losses_ref[i], places=4)
        # trained (post-CUT) draft weights, to verify resume restores them exactly
        w_trained = {
            k: v.detach().float().cpu().clone()
            for k, v in model_a.draft_model.state_dict().items()
            if "embed" not in k.lower()
        }

        # Phase 2: fresh model, restore weights + optimizer + scheduler + RNG +
        # data position — through the one checkpoint reader.
        from specforge.training.checkpoint import CheckpointManager

        state = CheckpointManager.read_resume_state(ck.checkpoint_uri)
        self.assertEqual(state["epoch_batch"], CUT)
        # position is also persisted batch-size-independently, in samples
        self.assertEqual(state["epoch_samples"], CUT * BS)
        # Same seed as the reference: the draft's FROZEN embedding is set at
        # construction and is not in the checkpoint (it is loaded from the target,
        # not trained), so it must be reconstructed identically — exactly as a real
        # resume rebuilds it from the same target. The training RNG, by contrast, is
        # restored from the checkpoint below, not from this seed.
        torch.manual_seed(0)
        model_b, head_b = fx.build_eagle3(workdir, ttt=TTT)
        model_b.draft_model.load_state_dict(state["draft_state_dict"], strict=False)
        backend_b = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend_b.prepare_model(model_b, wrap=False)
        backend_b.set_optimizer(
            BF16Optimizer(
                model_b.draft_model,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=100,
            )
        )
        backend_b.load_state_dict(
            {"optimizer": state["optimizer_state_dict"], "rng": state["rng_state"]}
        )
        core_b = TrainerCore(
            Eagle3TrainStrategy(model_b, target_head=head_b), backend_b
        )

        # Resume must restore the trained draft weights bit-for-bit.
        w_resumed = {
            k: v.detach().float().cpu()
            for k, v in model_b.draft_model.state_dict().items()
            if "embed" not in k.lower()
        }
        max_wdiff = max(
            float((w_trained[k] - w_resumed[k]).abs().max()) for k in w_trained
        )
        self.assertLess(max_wdiff, 1e-6, "resume did not restore trained draft weights")

        losses_b = []
        ctrl_b = TrainerController(
            core_b,
            run_id="rez",
            output_dir=out_a,
            max_steps=TOTAL,
            start_step=CUT,
            start_batch=state["epoch_batch"],  # reposition the data stream
            log_interval=1,
            logger=lambda m, s: losses_b.append(m["loss"]),
        )
        # No reseed here — the RNG was restored from the checkpoint. fit skips
        # the first CUT batches of the interrupted epoch and trains the rest.
        ctrl_b.fit(batches)
        self.assertEqual(len(losses_b), TOTAL - CUT)

        # The resumed curve continues the reference curve. Not bit-exact: the
        # BF16Optimizer fp32 master is rebuilt from the persisted bf16 weights, so a
        # sub-1% drift accrues over the continuation (the weights themselves are
        # restored bit-for-bit, asserted above). delta comfortably separates this
        # from a real resume bug (a lost optimizer/weight state diverges by ~1+).
        for i in range(TOTAL - CUT):
            self.assertAlmostEqual(
                losses_b[i],
                losses_ref[CUT + i],
                delta=0.05,
                msg=f"resume diverged at step {CUT + i + 1}: "
                f"{losses_b[i]} vs {losses_ref[CUT + i]}",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
