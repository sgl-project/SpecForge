"""End-to-end CPU cooldown branch through the production ``Trainer(resume_from=...)``.

Reuses the CPU resume seam of ``test_checkpoint_resume`` (offline features,
fake strategy, production FSDP backend unwrapped, real ``BF16Optimizer`` and
``CheckpointManager``). Scaled schedule: total 100, warmup 2, D = 3, T = 6.
"""

from __future__ import annotations

import os
import tempfile
import unittest

import torch

from specforge.lr_scheduler import constant_linear_decay_lr
from tests.test_runtime import test_checkpoint_resume as harness

LR, TOTAL, RATIO, W, D, T = 0.05, 100, 0.02, 2, 3, 6


def _trainer(
    out, features, *, max_steps, scheduler, resume_from=None, d=D, t=T, lrs=None
):
    from specforge.optimizer import BF16Optimizer
    from specforge.runtime.control_plane import DataFlowController
    from specforge.runtime.data_plane.feature_store import LocalFeatureStore
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
    from specforge.training.trainer import Trainer

    lrs = [] if lrs is None else lrs

    class Recording(BF16Optimizer):
        def step(self, **kwargs):
            lrs.append(self.get_learning_rate())  # the LR this update runs with
            return super().step(**kwargs)

    decay = (
        dict(lr_decay_start_update=d, lr_decay_end_update=t)
        if scheduler == "constant_linear_decay"
        else {}
    )
    Composite, Strategy, _ = harness._fake_seam()
    model = Composite()
    refs = OfflineManifestReader(features, run_id="data").read()
    with harness._no_fsdp_wrap():
        trainer = Trainer(
            algorithm_name="eagle3",
            make_step_strategy=lambda wrapped, *, target_head: Strategy(wrapped),
            controller=DataFlowController("rz"),
            store=LocalFeatureStore("st"),
            ref_source={"refs": refs},
            model=model,
            target_head=None,
            optimizer_factory=lambda m: Recording(
                m,
                lr=LR,
                max_grad_norm=1.0,
                warmup_ratio=RATIO,
                total_steps=TOTAL,
                lr_scheduler=scheduler,
                **decay,
            ),
            run_id="rz",
            output_dir=str(out),
            batch_size=2,
            accumulation_steps=1,
            num_epochs=1,
            max_steps=max_steps,
            total_steps=TOTAL,
            save_interval=0,
            logger=None,
            log_interval=50,
            collate_fn=harness._x_collate,
            per_sample_transform=harness._x_transform,
            resume_from=resume_from,
        )
    return trainer, model, lrs


def _latest(out):
    return os.path.realpath(os.path.join(out, "rz-latest"))


def _expected(first, last):
    return [constant_linear_decay_lr(u, LR, W, D, T) for u in range(first, last + 1)]


class TestCooldownBranchResume(unittest.TestCase):
    created = False

    @classmethod
    def setUpClass(cls):
        import torch.distributed as dist

        cls._tmp = tempfile.TemporaryDirectory()
        if dist.is_available() and not dist.is_initialized():
            store = dist.FileStore(os.path.join(cls._tmp.name, "store"), 1)
            dist.init_process_group("gloo", store=store, rank=0, world_size=1)
            cls.created = True
        cls.features = harness._write_feature_files(
            os.path.join(cls._tmp.name, "features"), n=2 * T
        )

    @classmethod
    def tearDownClass(cls):
        import torch.distributed as dist

        if cls.created:
            dist.destroy_process_group()
            cls.created = False
        cls._tmp.cleanup()

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _out(self, name):
        return os.path.join(self.tmp.name, name)

    def test_constant_checkpoint_branches_and_matches_a_straight_cooldown_run(self):
        from specforge.training.checkpoint import CheckpointManager

        ref, ref_model, ref_lrs = _trainer(
            self._out("ref"),
            self.features,
            max_steps=T,
            scheduler="constant_linear_decay",
        )
        self.assertEqual(ref.fit(), T)
        self.assertEqual(ref_lrs, _expected(1, T))

        constant, _, constant_lrs = _trainer(
            self._out("constant"), self.features, max_steps=D, scheduler="constant"
        )
        self.assertEqual(constant.fit(), D)
        self.assertEqual(constant_lrs, ref_lrs[:D])

        branch = {
            "from_type": "constant",
            "branch_update": D,
            "decay_start_update": D,
            "decay_end_update": T,
        }
        cooldown, _, cooldown_lrs = _trainer(
            self._out("cooldown"),
            self.features,
            max_steps=T - 1,
            scheduler="constant_linear_decay",
            resume_from=_latest(self._out("constant")),
        )
        self.assertEqual(cooldown.backend.optimizer.lr_schedule_branch, branch)
        self.assertEqual(cooldown.fit(), T - 1)
        self.assertEqual(cooldown_lrs, _expected(D + 1, T - 1))
        saved = CheckpointManager.read_resume_state(_latest(self._out("cooldown")))
        self.assertEqual(saved["backend"]["optimizer"]["lr_schedule_branch"], branch)
        self.assertEqual(
            saved["backend"]["optimizer"]["lr_decay_window"],
            {"warmup_updates": W, "decay_start_update": D, "decay_end_update": T},
        )

        # A later resume keeps the branch and ends on the straight run's weights.
        final, final_model, final_lrs = _trainer(
            self._out("final"),
            self.features,
            max_steps=T,
            scheduler="constant_linear_decay",
            resume_from=_latest(self._out("cooldown")),
        )
        self.assertEqual(final.backend.optimizer.lr_schedule_branch, branch)
        self.assertEqual(final.fit(), T)
        self.assertEqual(final_lrs, _expected(T, T))
        self.assertTrue(
            torch.equal(
                final_model.draft_model.w.detach(), ref_model.draft_model.w.detach()
            )
        )

        # ... and refuses a changed decay window.
        with self.assertRaisesRegex(ValueError, "identical warmup/D/T"):
            _trainer(
                self._out("changed"),
                self.features,
                max_steps=T + 1,
                t=T + 1,
                scheduler="constant_linear_decay",
                resume_from=_latest(self._out("cooldown")),
            )

    def test_branch_after_the_decay_start_is_refused(self):
        constant, _, _ = _trainer(
            self._out("constant"), self.features, max_steps=D + 1, scheduler="constant"
        )
        self.assertEqual(constant.fit(), D + 1)
        with self.assertRaisesRegex(ValueError, f"from update {D + 1}: it is past"):
            _trainer(
                self._out("cooldown"),
                self.features,
                max_steps=T,
                scheduler="constant_linear_decay",
                resume_from=_latest(self._out("constant")),
            )

    def test_constant_resume_of_a_constant_checkpoint_is_unchanged(self):
        ref, ref_model, ref_lrs = _trainer(
            self._out("ref"), self.features, max_steps=T, scheduler="constant"
        )
        self.assertEqual(ref.fit(), T)
        first, _, _ = _trainer(
            self._out("first"), self.features, max_steps=D, scheduler="constant"
        )
        self.assertEqual(first.fit(), D)
        resumed, model, lrs = _trainer(
            self._out("resumed"),
            self.features,
            max_steps=T,
            scheduler="constant",
            resume_from=_latest(self._out("first")),
        )
        self.assertIsNone(resumed.backend.optimizer.lr_schedule_branch)
        self.assertEqual(resumed.fit(), T)
        self.assertEqual(lrs, ref_lrs[D:])
        self.assertEqual(lrs, [LR] * (T - D))
        self.assertTrue(
            torch.equal(model.draft_model.w.detach(), ref_model.draft_model.w.detach())
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
