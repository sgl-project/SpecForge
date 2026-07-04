# coding=utf-8
"""Checkpoint save/resume gates.

CPU: the production ``Trainer(resume_from=...)`` entrypoint (wrap=False), its
fail-fast resume validation, and fit() re-entry at/after max_steps. GPU (run on
the H200 box via rcli): on-disk layout + weight round-trip, and loss-curve
continuity after resume (weights/optimizer/scheduler/RNG/data position).
"""

import contextlib
import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

CUDA = torch.cuda.is_available()


def _write_feature_files(d, n, seq=8):
    """``n`` offline eagle3-schema files with DISTINCT input_ids per sample, so
    which batches a run trained on is observable in the loss/weights."""
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        torch.save(
            {
                "input_ids": torch.arange(seq) + i * seq,
                "loss_mask": torch.ones(seq, dtype=torch.long),
                "hidden_state": torch.randn(1, seq, 4),
                "aux_hidden_state": torch.randn(1, seq, 12),
            },
            os.path.join(d, f"{i:04d}.ckpt"),
        )
    return d


def _x_transform(raw):
    return {"x": raw["input_ids"].float()[None, :]}


def _x_collate(features):
    return {"x": torch.cat([f["x"] for f in features], dim=0)}


def _fake_seam():
    """CPU fakes over the real seam ABCs (imported lazily: specforge is heavy)."""
    import torch.nn as nn

    from specforge.runtime.training.backend import TrainingBackend
    from specforge.runtime.training.strategy import DraftTrainStrategy, StepOutput

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

    class Composite(nn.Module):
        # mirrors OnlineEagle3Model's shape: the checkpointed unit is .draft_model
        def __init__(self):
            super().__init__()
            self.draft_model = Draft()

    class Strategy(DraftTrainStrategy):
        name = "eagle3"  # matches the offline refs' strategy tag
        required_features = {"x"}

        def __init__(self, model, seen=None):
            self.model = model
            self.seen = [] if seen is None else seen

        def trainable_module(self):
            return self.model

        def forward_loss(self, batch, ctx=None):
            self.validate_batch(batch)
            self.seen.extend(batch.sample_ids)
            w = self.model.draft_model.w
            loss = ((w - batch.tensors["x"].float().mean()) ** 2).sum()
            return StepOutput(loss=loss, metrics={})

        def checkpoint_state_filter(self, state_dict):
            pre = "draft_model."
            return {
                k[len(pre) :]: v for k, v in state_dict.items() if k.startswith(pre)
            }

    class Backend(TrainingBackend):
        name = "fake"

        def __init__(self, model):
            self.model = model
            self.steps = 0

        def prepare_model(self, model):
            return model

        def backward(self, loss, *, is_boundary=True):
            loss.backward()

        def step(self):
            self.steps += 1
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.data -= 0.1 * p.grad
                        p.grad = None
            return torch.tensor(1.0)

        def state_dict(self):
            return {"model": self.model.state_dict(), "optimizer": None, "rng": {}}

        def load_state_dict(self, state):
            pass

    return Composite, Strategy, Backend


@contextlib.contextmanager
def _no_fsdp_wrap():
    """Force the production backend to register without FSDP (CPU, world 1)."""
    from specforge.runtime.training.backend import FSDPTrainingBackend

    orig = FSDPTrainingBackend.prepare_model

    def unwrapped(self, model, *, wrap=True, optimizer_target=None):
        return orig(self, model, wrap=False, optimizer_target=optimizer_target)

    with mock.patch.object(FSDPTrainingBackend, "prepare_model", unwrapped):
        yield


class TestTrainerResumeEntrypoint(unittest.TestCase):
    """Production resume seam on CPU: ``Trainer(resume_from=...)`` restores
    weights, fp32 masters, scheduler position and data position, and fail-fasts
    on checkpoint/run mismatches."""

    _created_pg = False

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        # BF16Optimizer.load_state_dict logs via rank0, which needs a group.
        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(
                os.path.join(tempfile.mkdtemp(prefix="ckpt_pg_"), "store"), 1
            )
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            cls._created_pg = True

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls._created_pg:
            dist.destroy_process_group()

    def _make_trainer(
        self,
        out_dir,
        *,
        feat_dir,
        max_steps,
        resume_from=None,
        run_id="rz",
        seen=None,
        model=None,
    ):
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore
        from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
        from specforge.training.trainer import Trainer

        Composite, Strategy, _ = _fake_seam()
        seen = [] if seen is None else seen
        model = Composite() if model is None else model
        refs = OfflineManifestReader(feat_dir, run_id="data").read()
        spec = SimpleNamespace(
            name="eagle3",
            make_strategy=lambda wrapped, *, target_head: Strategy(wrapped, seen),
        )
        with _no_fsdp_wrap():
            trainer = Trainer(
                spec=spec,
                controller=DataFlowController(run_id),
                store=LocalFeatureStore("st"),
                ref_source={"refs": refs},
                model=model,
                target_head=None,
                optimizer_factory=lambda m: BF16Optimizer(
                    m, lr=0.05, max_grad_norm=1.0, warmup_ratio=0.0, total_steps=100
                ),
                run_id=run_id,
                output_dir=out_dir,
                batch_size=2,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=max_steps,
                save_interval=0,
                eval_interval=0,
                tp_size=1,
                sp_ulysses_size=1,
                sp_ring_size=1,
                logger=None,
                log_interval=50,
                collate_fn=_x_collate,
                per_sample_transform=_x_transform,
                resume_from=resume_from,
            )
        return trainer, model, seen

    def test_resume_continues_where_the_run_stopped(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_cpu_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=8)

        # Reference: one uninterrupted run over all 4 batches.
        t_ref, model_ref, seen_ref = self._make_trainer(
            os.path.join(workdir, "ref"), feat_dir=feat_dir, max_steps=4, run_id="ref"
        )
        self.assertEqual(t_ref.fit(), 4)
        w_ref = model_ref.draft_model.w.detach().clone()

        # Interrupted phase 1: 2 steps, save mid-epoch.
        out = os.path.join(workdir, "out")
        t1, model1, seen1 = self._make_trainer(out, feat_dir=feat_dir, max_steps=2)
        self.assertEqual(t1.fit(), 2)
        self.assertEqual(t1.controller._epoch_batch, 2)
        ck = t1.save_checkpoint()
        self.assertEqual(ck.global_step, 2)
        w_cut = model1.draft_model.w.detach().clone()
        masters_cut = [t.detach().cpu().clone() for t in t1.backend.optimizer.fp32_params]

        # Phase 2: resume through the production entrypoint (file:// URI).
        t2, model2, seen2 = self._make_trainer(
            out, feat_dir=feat_dir, max_steps=4, resume_from=ck.checkpoint_uri
        )
        self.assertTrue(torch.equal(model2.draft_model.w.detach(), w_cut))
        # exact fp32 masters restored, not re-cloned from the trained weights
        for restored, saved in zip(t2.backend.optimizer.fp32_params, masters_cut):
            self.assertTrue(torch.equal(restored.detach().cpu(), saved))
        # scheduler position survived — a reset would read 0 (warmup 0: the
        # cosine after_scheduler carries the position, not the outer wrapper)
        self.assertEqual(t2.backend.optimizer.scheduler.after_scheduler.last_epoch, 2)
        ctrl = t2.controller
        self.assertEqual(
            (ctrl.global_step, ctrl._epoch_batch, ctrl._epoch_samples), (2, 2, 4)
        )

        self.assertEqual(t2.fit(), 4)
        # only the unseen tail was trained, in the reference order
        self.assertEqual(
            seen2,
            ["data:00000004", "data:00000005", "data:00000006", "data:00000007"],
        )
        self.assertEqual(seen1 + seen2, seen_ref)
        # exact state restore => the continuation reproduces the reference run
        self.assertLess(
            abs(float(model2.draft_model.w.detach() - w_ref)),
            1e-9,
            "resumed run diverged from the uninterrupted reference",
        )

        # Resume at global_step == max_steps: fit() trains nothing.
        t3, model3, seen3 = self._make_trainer(
            os.path.join(workdir, "noop"),
            feat_dir=feat_dir,
            max_steps=2,
            resume_from=ck.checkpoint_uri,
        )
        self.assertEqual(t3.fit(), 2)
        self.assertEqual(seen3, [])
        self.assertTrue(torch.equal(model3.draft_model.w.detach(), w_cut))

    def test_resume_validation_fails_fast(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_bad_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=8)

        def write_ckpt(name, **overrides):
            d = os.path.join(workdir, name)
            os.makedirs(d)
            payload = {
                "draft_state_dict": {"w": torch.zeros(1)},
                "global_step": 2,
                "epoch": 0,
                "epoch_batch": 2,
                "epoch_samples": 4,
                "strategy": "eagle3",
                "run_id": "rz",
                "world_size": 1,
            }
            payload.update(overrides)
            torch.save(payload, os.path.join(d, "training_state.pt"))
            torch.save(
                {"optimizer": None, "rng": {}},
                os.path.join(d, "training_state_rank0.pt"),
            )
            return d

        cases = [
            ("strategy", {"strategy": "dflash"}, "was written by strategy"),
            ("dataset", {"dataset_size": 999}, "dataset_size=999 but this run has"),
            (
                "accum",
                {"accumulation_steps": 4},
                "accumulation_steps=4 but this run has",
            ),
            (
                "weights",
                {"draft_state_dict": {"bogus.weight": torch.zeros(1)}},
                "do not match this model",
            ),
        ]
        for name, overrides, anchor in cases:
            with self.subTest(name):
                with self.assertRaisesRegex(ValueError, anchor):
                    self._make_trainer(
                        os.path.join(workdir, f"out_{name}"),
                        feat_dir=feat_dir,
                        max_steps=4,
                        resume_from=write_ckpt(name, **overrides),
                    )


class TestFitReentry(unittest.TestCase):
    """fit() at/after max_steps and re-entry after a mid-epoch return (CPU fakes)."""

    def _controller(self, seen, max_steps, out_dir, **kw):
        from specforge.runtime.training.trainer import TrainerController, TrainerCore

        Composite, Strategy, Backend = _fake_seam()
        model = Composite()
        backend = Backend(model)
        core = TrainerCore(Strategy(model, seen), backend, accumulation_steps=1)
        ctrl = TrainerController(
            core, run_id="r", output_dir=out_dir, max_steps=max_steps, num_epochs=1, **kw
        )
        return ctrl, backend

    @staticmethod
    def _batches(n):
        from specforge.runtime.contracts import TrainBatch

        return [
            TrainBatch(
                sample_ids=[f"b{i}"],
                strategy="eagle3",
                tensors={"x": torch.full((2,), float(i))},
                metadata={},
            )
            for i in range(n)
        ]

    def test_resume_at_max_steps_returns_without_consuming(self):
        class Explode:
            def __iter__(self):
                raise AssertionError("fit() consumed data at global_step >= max_steps")

        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, backend = self._controller(seen, 3, d, start_step=3)
            self.assertEqual(ctrl.fit(Explode()), 3)
            ctrl.global_step = 5  # past the cap must return too
            self.assertEqual(ctrl.fit(Explode()), 5)
            self.assertEqual(seen, [])
            self.assertEqual(backend.steps, 0)

    def test_refit_after_midepoch_return_does_not_retrain_prefix(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, backend = self._controller(seen, 2, d)
            batches = self._batches(5)
            self.assertEqual(ctrl.fit(batches), 2)
            self.assertEqual(seen, ["b0", "b1"])
            ctrl.max_steps = 4
            self.assertEqual(ctrl.fit(batches), 4)
            # the live epoch position drives the skip: b0/b1 are not re-trained
            self.assertEqual(seen, ["b0", "b1", "b2", "b3"])
            self.assertEqual(backend.steps, 4)


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
        backend.prepare_model(model, wrap=False)  # no FSDP at 1 rank
        backend.set_optimizer(opt)
        strategy = Eagle3TrainStrategy(model, target_head=head)
        core = TrainerCore(strategy, backend)
        ctrl = TrainerController(
            core, run_id="r", output_dir=out_dir, max_steps=3, num_epochs=2
        )
        step = ctrl.fit(make_batches())
        self.assertEqual(step, 3)
        ck = ctrl.save_checkpoint(step)

        from specforge.training.checkpoint import CheckpointManager

        # run-scoped on-disk names
        ckpt_dir = ck.checkpoint_uri[len("file://") :]
        self.assertEqual(os.path.basename(ckpt_dir), "r-step3")
        self.assertTrue(os.path.islink(os.path.join(out_dir, "r-latest")))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "latest")))
        mgr = CheckpointManager(out_dir, "r")
        self.assertEqual(
            os.path.realpath(mgr.latest_dir()), os.path.realpath(ckpt_dir)
        )

        shared = torch.load(
            os.path.join(ckpt_dir, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(shared["global_step"], 3)
        self.assertEqual(shared["strategy"], "eagle3")
        self.assertEqual(shared["world_size"], 1)
        self.assertEqual(shared["epoch_batch"], 3)
        self.assertEqual(shared["epoch_samples"], N)
        # rank-local state lives in per-rank files, NOT the shared payload
        self.assertNotIn("optimizer_state_dict", shared)
        self.assertTrue(
            os.path.exists(os.path.join(ckpt_dir, "training_state_rank0.pt"))
        )

        # the one reader passes this rank's file through untouched as 'backend'
        ckpt = CheckpointManager.read_resume_state(ck.checkpoint_uri)
        self.assertNotIn("optimizer_state_dict", ckpt)
        self.assertNotIn("rng_state", ckpt)
        self.assertEqual(set(ckpt["backend"]), {"optimizer", "rng"})
        opt_state = ckpt["backend"]["optimizer"]
        trainable = [p for p in model.draft_model.parameters() if p.requires_grad]
        self.assertEqual(len(opt_state["fp32_params"]), len(trainable))
        for t in opt_state["fp32_params"]:
            self.assertEqual(t.dtype, torch.float32)
            self.assertEqual(t.device.type, "cpu")
        rng = ckpt["backend"]["rng"]
        self.assertEqual(set(rng), {"torch", "cuda"})
        # single bound-device CUDA state, not the legacy per-device list
        self.assertIsInstance(rng["cuda"], torch.Tensor)

        # Filter contract, checked against the LIVE trained module (not by
        # re-applying the filter): frozen-embed keys and nothing else dropped.
        trained_sd = {
            k: v.detach().cpu() for k, v in model.draft_model.state_dict().items()
        }
        embed_keys = {k for k in trained_sd if "embed" in k.lower()}
        self.assertTrue(embed_keys)  # the fixture freezes a real embedding
        persisted = ckpt["draft_state_dict"]
        self.assertEqual(set(persisted), set(trained_sd) - embed_keys)

        fresh = build_model()
        missing, unexpected = fresh.draft_model.load_state_dict(
            persisted, strict=False
        )
        self.assertEqual(unexpected, [])
        self.assertEqual(set(missing), embed_keys)
        fresh_sd = fresh.draft_model.state_dict()
        for k, v in persisted.items():
            self.assertTrue(
                torch.equal(v.cpu(), trained_sd[k]), msg=f"persisted {k} mismatch"
            )
            self.assertTrue(
                torch.equal(fresh_sd[k].cpu(), trained_sd[k]),
                msg=f"reloaded {k} mismatch",
            )

    def test_resume_loss_curve_continuity(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29564")

        from specforge.data.preprocessing import OfflineEagle3Dataset
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
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
        files = sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir))
        ds = OfflineEagle3Dataset(files, max_len=512)
        collate = DataCollatorWithPadding()
        batches = []
        for i in range(TOTAL):
            data = collate([ds[j] for j in range(i * BS, (i + 1) * BS)])
            batches.append(
                TrainBatch(
                    sample_ids=[str(j) for j in range(i * BS, (i + 1) * BS)],
                    strategy="eagle3",
                    tensors=dict(data),
                    metadata={"target_repr": "hidden_state", "ttt_length": TTT},
                )
            )

        def build_run():
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
            return model, backend, core

        # Reference: one uninterrupted run of TOTAL steps (loss + lr per step).
        losses_ref, lrs_ref = [], []
        _, backend_r, core_r = build_run()
        ctrl_r = TrainerController(
            core_r,
            run_id="ref",
            output_dir=os.path.join(workdir, "ref"),
            max_steps=TOTAL,
            log_interval=1,
            logger=lambda m, s: (
                losses_ref.append(m["loss"]),
                lrs_ref.append(backend_r.optimizer.get_learning_rate()),
            ),
        )
        torch.manual_seed(0)
        ctrl_r.fit(batches)
        self.assertEqual(len(losses_ref), TOTAL)

        # Interrupted phase 1: CUT steps, then save.
        out_a = os.path.join(workdir, "rez")
        losses_a = []
        model_a, backend_a, core_a = build_run()
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
        self.assertEqual(ctrl_a._epoch_batch, CUT)  # data position at the cut
        # phase 1 mirrors the reference exactly for the first CUT steps
        for i in range(CUT):
            self.assertAlmostEqual(losses_a[i], losses_ref[i], places=4)
        w_trained = {
            k: v.detach().float().cpu().clone()
            for k, v in model_a.draft_model.state_dict().items()
            if "embed" not in k.lower()
        }

        # Phase 2: fresh model, restore through the one checkpoint reader.
        from specforge.training.checkpoint import CheckpointManager

        state = CheckpointManager.read_resume_state(ck.checkpoint_uri)
        self.assertEqual(state["epoch_batch"], CUT)
        # position is also persisted batch-size-independently, in samples
        self.assertEqual(state["epoch_samples"], CUT * BS)
        self.assertNotIn("optimizer_state_dict", state)
        self.assertNotIn("rng_state", state)
        self.assertEqual(set(state["backend"]), {"optimizer", "rng"})
        # Same seed as the reference: the FROZEN embedding is not in the
        # checkpoint and must be rebuilt identically (as a real resume rebuilds
        # it from the same target); training RNG is restored from the checkpoint.
        torch.manual_seed(0)
        model_b, backend_b, core_b = build_run()
        model_b.draft_model.load_state_dict(state["draft_state_dict"], strict=False)
        backend_b.load_state_dict(state["backend"])

        # fp32 masters restored exactly (not re-quantized from bf16)
        for mb, ma in zip(
            backend_b.optimizer.fp32_params, backend_a.optimizer.fp32_params
        ):
            self.assertTrue(torch.equal(mb.detach().cpu(), ma.detach().cpu()))
        # scheduler position restored — a reset would read 0 (warmup 0: the
        # cosine after_scheduler carries the position, not the outer wrapper)
        self.assertEqual(
            backend_b.optimizer.scheduler.after_scheduler.last_epoch, CUT
        )
        self.assertAlmostEqual(
            backend_b.optimizer.get_learning_rate(), lrs_ref[CUT - 1], places=12
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
            start_samples=state["epoch_samples"],
            log_interval=1,
            logger=lambda m, s: losses_b.append(m["loss"]),
        )
        # No reseed: the RNG came from the checkpoint. fit skips the first CUT
        # batches of the interrupted epoch and trains the rest.
        ctrl_b.fit(batches)
        self.assertEqual(len(losses_b), TOTAL - CUT)

        # Weights, fp32 masters, optimizer, scheduler, RNG and data position are
        # all restored exactly, so only kernel-level nondeterminism separates the
        # resumed curve from the uninterrupted reference.
        for i in range(TOTAL - CUT):
            self.assertAlmostEqual(
                losses_b[i],
                losses_ref[CUT + i],
                delta=1e-3,
                msg=f"resume diverged at step {CUT + i + 1}: "
                f"{losses_b[i]} vs {losses_ref[CUT + i]}",
            )
        # both runs end at the same schedule position and lr
        self.assertEqual(
            backend_b.optimizer.scheduler.after_scheduler.last_epoch,
            backend_r.optimizer.scheduler.after_scheduler.last_epoch,
        )
        self.assertAlmostEqual(
            backend_b.optimizer.get_learning_rate(),
            backend_r.optimizer.get_learning_rate(),
            places=12,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
