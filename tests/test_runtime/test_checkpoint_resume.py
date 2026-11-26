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
    """A CPU program and a backend for it (imported lazily: specforge is heavy)."""
    import torch.nn as nn

    from specforge.algorithms.objectives.base import ObjectiveResult
    from specforge.training.backend import TrainingBackend
    from tests.test_runtime._fixtures import FakeProgram

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.w = nn.Parameter(torch.zeros(1))

    class Composite(FakeProgram):
        # mirrors ObjectiveModel's shape: the checkpointed unit is .draft_model
        def __init__(self, draft=None):
            super().__init__()
            self.draft_model = Draft() if draft is None else draft

        def forward(
            self, tensors, *, step, total_steps, collect_detailed_metrics, **replay
        ):
            w = self.draft_model.w
            loss = ((w - tensors["x"].float().mean()) ** 2).sum()
            return ObjectiveResult(loss, torch.ones(()))

        def draft_state(self, state):
            pre = "draft_model."
            return {k[len(pre) :]: v for k, v in state.items() if k.startswith(pre)}

        def kernel_record(self):
            selection = getattr(self, "loss_implementations", None)
            return {} if selection is None else dict(selection.names)

    class Backend(TrainingBackend):
        process_group = None
        name = "fake"

        def __init__(self, model):
            self.model = model
            self.steps = 0

        def prepare_model(self, model):
            return model

        def backward(self, loss, *, is_boundary=True):
            loss.backward()

        def scale_gradients(self, factor):
            with torch.no_grad():
                for p in self.model.parameters():
                    if p.grad is not None:
                        p.grad.mul_(factor)

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

    return Composite, Backend


def _spec():
    """What the fake program declares; named after the offline refs' strategy tag."""
    from tests.test_runtime._fixtures import fake_spec

    return fake_spec("x", name="eagle3")


def _recording_runner(seen):
    """A runner class that appends to ``seen`` the sample ids of each batch it forwards."""
    from specforge.training.runner import DataParallelRunner

    class Runner(DataParallelRunner):
        def _forward(self, batch, ctx, **replay):
            seen.extend(batch.sample_ids)
            return super()._forward(batch, ctx, **replay)

    return Runner


@contextlib.contextmanager
def _no_fsdp_wrap():
    """Force the production backend to register without FSDP (CPU, world 1)."""
    from specforge.training.backend import FSDPTrainingBackend

    orig = FSDPTrainingBackend.prepare_model

    def unwrapped(self, model, *, wrap=True, **arguments):
        return orig(self, model, wrap=False, **arguments)

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
        checkpoint_extra=None,
        strategy_kwargs=None,
        spec_name="eagle3",
        max_grad_norm=1.0,
        total_steps=100,
    ):
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore
        from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
        from specforge.training.trainer import Trainer

        Composite, _ = _fake_seam()
        seen = [] if seen is None else seen
        model = Composite() if model is None else model
        refs = OfflineManifestReader(feat_dir, run_id="data").read()
        with (
            _no_fsdp_wrap(),
            mock.patch(
                "specforge.training.trainer.DataParallelRunner", _recording_runner(seen)
            ),
        ):
            trainer = Trainer(
                algorithm_name=spec_name,
                controller=DataFlowController(run_id),
                store=LocalFeatureStore("st"),
                ref_source={"refs": refs},
                model=model,
                program_spec=_spec(),
                optimizer_factory=lambda m: BF16Optimizer(
                    m,
                    lr=0.05,
                    max_grad_norm=max_grad_norm,
                    warmup_ratio=0.0,
                    total_steps=(total_steps if total_steps is not None else max_steps),
                ),
                run_id=run_id,
                output_dir=out_dir,
                batch_size=2,
                accumulation_steps=1,
                num_epochs=1,
                max_steps=max_steps,
                total_steps=total_steps,
                save_interval=0,
                logger=None,
                log_interval=50,
                collate_fn=_x_collate,
                per_sample_transform=_x_transform,
                resume_from=resume_from,
                checkpoint_extra=checkpoint_extra,
                strategy_kwargs=strategy_kwargs,
            )
        return trainer, model, seen

    def test_resolved_loss_implementations_are_saved_and_restored(self):
        from specforge.algorithms.objectives.implementations.registry import (
            ExecutionSelection,
        )
        from specforge.training.checkpoint import CheckpointManager

        Composite, _ = _fake_seam()
        with tempfile.TemporaryDirectory() as workdir:
            features = _write_feature_files(os.path.join(workdir, "features"), n=4)
            out = os.path.join(workdir, "out")
            model = Composite()
            model.loss_implementations = ExecutionSelection()
            from dataclasses import asdict

            from specforge.algorithms.objectives.protocols.domino import DominoObjective

            model.objective = DominoObjective()
            from specforge.algorithms.objectives.execution_options import LossExecution

            contract = {
                "canonical_objective_options": asdict(model.objective),
                "canonical_loss_implementations": model.loss_implementations.names,
                "canonical_loss_execution": asdict(LossExecution(fused=True)),
            }
            trainer, _, _ = self._make_trainer(
                out,
                feat_dir=features,
                max_steps=1,
                model=model,
                checkpoint_extra=contract,
            )
            # Selection happens after Trainer construction, on the first real batch.
            model.loss_implementations.names["0:0"] = "token_ce"
            trainer.fit()
            checkpoint = os.path.realpath(os.path.join(out, "rz-latest"))
            state = CheckpointManager.read_resume_state(checkpoint)
            self.assertEqual(
                state["canonical_loss_implementations"], {"0:0": "token_ce"}
            )
            # Even if checkpoint_extra holds a separate stale mapping, saving
            # writes the exact CheckpointState.kernels snapshot agreed across ranks.
            trainer._controller.checkpoint_extra["canonical_loss_implementations"] = {
                "0:0": "stale_kernel"
            }
            trainer._controller.save_checkpoint(step=2)
            state_from_snapshot = CheckpointManager.read_resume_state(
                os.path.realpath(os.path.join(out, "rz-latest"))
            )
            self.assertEqual(
                state_from_snapshot["canonical_loss_implementations"],
                {"0:0": "token_ce"},
            )
            model = Composite()
            model.loss_implementations = ExecutionSelection()
            model.objective = DominoObjective()
            contract["canonical_objective_options"] = asdict(model.objective)
            contract["canonical_loss_implementations"] = (
                model.loss_implementations.names
            )
            with mock.patch.object(
                model.loss_implementations,
                "restore",
                wraps=model.loss_implementations.restore,
            ) as restore:
                self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=2,
                    model=model,
                    checkpoint_extra=contract,
                    resume_from=checkpoint,
                )
            restore.assert_called_once_with({"0:0": "token_ce"})
            # Missing or unknown saved options must not be filled from current defaults.
            saved_options = state["canonical_objective_options"]
            for options in (
                {
                    key: value
                    for key, value in saved_options.items()
                    if key != "lambda_base_start"
                },
                {**saved_options, "removed_option": 1},
            ):
                state["canonical_objective_options"] = options
                torch.save(state, os.path.join(checkpoint, "training_state.pt"))
                with (
                    self.subTest(options=options),
                    self.assertRaisesRegex(ValueError, "canonical_objective_options"),
                ):
                    self._make_trainer(
                        out,
                        feat_dir=features,
                        max_steps=2,
                        model=model,
                        checkpoint_extra=contract,
                        resume_from=checkpoint,
                    )
            state["canonical_objective_options"] = saved_options
            # Old fused checkpoints cannot prove which implementation ran.
            state.pop("canonical_loss_implementations")
            torch.save(state, os.path.join(checkpoint, "training_state.pt"))
            with self.assertRaisesRegex(ValueError, "does not record resolved"):
                self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=2,
                    model=model,
                    checkpoint_extra=contract,
                    resume_from=checkpoint,
                )

    def test_checkpoint_requires_recorded_loss_settings(self):
        from dataclasses import asdict

        from specforge.algorithms.objectives.execution_options import LossExecution

        with tempfile.TemporaryDirectory() as workdir:
            features = _write_feature_files(os.path.join(workdir, "features"), n=4)
            out = os.path.join(workdir, "out")
            trainer, _, _ = self._make_trainer(out, feat_dir=features, max_steps=1)
            trainer.fit()
            checkpoint = os.path.realpath(os.path.join(out, "rz-latest"))
            with self.assertRaisesRegex(ValueError, "canonical_loss_execution"):
                self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=2,
                    resume_from=checkpoint,
                    checkpoint_extra={
                        "canonical_loss_execution": asdict(LossExecution())
                    },
                )

    def test_resume_rejects_old_loss_metadata(self):
        with tempfile.TemporaryDirectory() as workdir:
            features = _write_feature_files(os.path.join(workdir, "features"), n=4)
            out = os.path.join(workdir, "out")
            trainer, _, _ = self._make_trainer(out, feat_dir=features, max_steps=1)
            trainer.fit()
            checkpoint = os.path.realpath(os.path.join(out, "rz-latest"))
            state_path = os.path.join(checkpoint, "training_state.pt")
            state = torch.load(state_path, weights_only=False)
            state["canonical_loss"] = {"kind": "ce"}
            torch.save(state, state_path)
            with self.assertRaisesRegex(ValueError, "convert_canonical_checkpoint"):
                self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=2,
                    resume_from=checkpoint,
                    checkpoint_extra={"canonical_loss": {"terms": {}}},
                )

    def test_unset_fused_follows_the_checkpoint(self):
        """A run that leaves fused unset continues with what its checkpoint recorded."""
        import types
        from dataclasses import asdict

        from specforge.algorithms.objectives.base import ObjectiveModel
        from specforge.algorithms.objectives.execution_options import LossExecution
        from specforge.algorithms.objectives.implementations.registry import (
            ExecutionSelection,
        )
        from specforge.training.checkpoint import CheckpointManager

        Composite, _ = _fake_seam()

        def model_with(execution):
            model = Composite()
            model.loss_implementations = ExecutionSelection()
            model.execution = execution
            model.adopt_recorded_execution = types.MethodType(
                ObjectiveModel.adopt_recorded_execution, model
            )
            return model

        def contract(model):
            return {
                "canonical_loss_implementations": model.loss_implementations.names,
                "canonical_loss_execution": asdict(model.execution),
            }

        for recorded in (False, True):
            with self.subTest(recorded=recorded), tempfile.TemporaryDirectory() as work:
                features = _write_feature_files(os.path.join(work, "features"), n=4)
                out = os.path.join(work, "out")
                writer = model_with(LossExecution(fused=recorded))
                trainer, _, _ = self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=1,
                    model=writer,
                    checkpoint_extra=contract(writer),
                )
                if recorded:
                    writer.loss_implementations.names["0:0"] = "token_ce"
                trainer.fit()
                checkpoint = os.path.realpath(os.path.join(out, "rz-latest"))

                unset = model_with(LossExecution())
                trainer, _, _ = self._make_trainer(
                    out,
                    feat_dir=features,
                    max_steps=2,
                    model=unset,
                    checkpoint_extra=contract(unset),
                    resume_from=checkpoint,
                )
                self.assertIs(unset.execution.fused, recorded)
                if recorded:
                    unset.loss_implementations.names["0:0"] = "token_ce"
                trainer.fit()
                state = CheckpointManager.read_resume_state(
                    os.path.realpath(os.path.join(out, "rz-latest"))
                )
                self.assertEqual(state["global_step"], 2)
                self.assertIs(state["canonical_loss_execution"]["fused"], recorded)

                # An explicit setting is this run's own request and must match.
                explicit = model_with(LossExecution(fused=not recorded))
                with self.assertRaisesRegex(ValueError, "canonical_loss_execution"):
                    self._make_trainer(
                        out,
                        feat_dir=features,
                        max_steps=3,
                        model=explicit,
                        checkpoint_extra=contract(explicit),
                        resume_from=checkpoint,
                    )
                # So must every other execution option.
                chunked = model_with(LossExecution(chunk_size=1))
                with self.assertRaisesRegex(ValueError, "canonical_loss_execution"):
                    self._make_trainer(
                        out,
                        feat_dir=features,
                        max_steps=3,
                        model=chunked,
                        checkpoint_extra=contract(chunked),
                        resume_from=checkpoint,
                    )

    def test_resume_continues_where_the_run_stopped(self):
        from specforge.training.checkpoint import CheckpointManager

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
        self.assertEqual(t1._controller._epoch_batch, 2)
        self.assertEqual(t1.last_checkpoint_step, 2)
        # Pin the immutable step-2 directory. ``rz-latest`` advances to step 4
        # when phase 2 completes, so retaining that mutable symlink would make
        # the later no-op check resume a different checkpoint than ``w_cut``.
        checkpoint_uri = f"file://{os.path.realpath(os.path.join(out, 'rz-latest'))}"
        checkpoint_state = CheckpointManager.read_resume_state(checkpoint_uri)
        self.assertEqual(checkpoint_state["effective_total_steps"], 100)
        w_cut = model1.draft_model.w.detach().clone()
        masters_cut = [
            t.detach().cpu().clone() for t in t1.backend.optimizer.fp32_params
        ]

        # Phase 2: resume through the production entrypoint (file:// URI).
        t2, model2, seen2 = self._make_trainer(
            out, feat_dir=feat_dir, max_steps=4, resume_from=checkpoint_uri
        )
        self.assertTrue(torch.equal(model2.draft_model.w.detach(), w_cut))
        # exact fp32 masters restored, not re-cloned from the trained weights
        for restored, saved in zip(t2.backend.optimizer.fp32_params, masters_cut):
            self.assertTrue(torch.equal(restored.detach().cpu(), saved))
        # scheduler position survived — a reset would read 0 (warmup 0: the
        # cosine after_scheduler carries the position, not the outer wrapper)
        self.assertEqual(t2.backend.optimizer.scheduler.after_scheduler.last_epoch, 2)
        ctrl = t2._controller
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
            resume_from=checkpoint_uri,
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
                "effective_total_steps": 100,
            }
            payload.update(overrides)
            torch.save(payload, os.path.join(d, "training_state.pt"))
            torch.save(
                {"optimizer": {}, "rng": {}},
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
            ("batch", {"batch_size": 1}, "batch_size=1 but this run has"),
            ("epochs", {"num_epochs": 2}, "num_epochs=2 but this run has"),
            ("tp", {"tp_size": 2}, "tp_size=2 but this run has"),
            (
                "ulysses",
                {"sp_ulysses_size": 2},
                "sp_ulysses_size=2 but this run has",
            ),
            (
                "ring",
                {"sp_ring_size": 2},
                "sp_ring_size=2 but this run has",
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

        peagle_semantics = (
            ("peagle_num_draft_layers", 2, 4),
            ("peagle_norm_before_residual", False, True),
            ("peagle_num_depths", 5, 6),
            ("peagle_down_sample_ratio", 0.7, 0.8),
            ("peagle_down_sample_ratio_min", 0.3, 0.2),
            ("peagle_mask_token_id", 0, 31),
        )
        for key, saved, current in peagle_semantics:
            with self.subTest(key):
                with self.assertRaisesRegex(
                    ValueError,
                    f"{key}={saved} but this run has {key}={current}",
                ):
                    self._make_trainer(
                        os.path.join(workdir, f"out_{key}"),
                        feat_dir=feat_dir,
                        max_steps=4,
                        resume_from=write_ckpt(key, **{key: saved}),
                        checkpoint_extra={key: current},
                    )

        required_peagle_contract = {
            key: current for key, _saved, current in peagle_semantics
        }
        with self.assertRaisesRegex(
            ValueError, "does not record required algorithm resume semantic"
        ):
            self._make_trainer(
                os.path.join(workdir, "out_peagle_missing_contract"),
                feat_dir=feat_dir,
                max_steps=4,
                resume_from=write_ckpt(
                    "peagle_missing_contract",
                    strategy="peagle",
                ),
                checkpoint_extra=required_peagle_contract,
                spec_name="peagle",
            )

    def test_resume_rejects_a_changed_implicit_schedule_horizon(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_horizon_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=8)
        source_dir = os.path.join(workdir, "source")
        source, _, _ = self._make_trainer(
            source_dir,
            feat_dir=feat_dir,
            max_steps=1,
            total_steps=None,
        )
        self.assertEqual(source.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(source_dir, "rz-latest"))

        with self.assertRaisesRegex(
            ValueError,
            "effective_total_steps=1 but this run has effective_total_steps=2",
        ):
            self._make_trainer(
                os.path.join(workdir, "changed"),
                feat_dir=feat_dir,
                max_steps=2,
                total_steps=None,
                resume_from=checkpoint,
            )

    def test_resume_rejects_missing_schedule_horizon(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_legacy_horizon_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=8)
        source_dir = os.path.join(workdir, "source")
        source, _, _ = self._make_trainer(
            source_dir,
            feat_dir=feat_dir,
            max_steps=1,
            total_steps=6,
        )
        self.assertEqual(source.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(source_dir, "rz-latest"))
        state_path = os.path.join(checkpoint, "training_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        state.pop("effective_total_steps")
        torch.save(state, state_path)

        with self.assertRaisesRegex(
            ValueError, "does not record effective_total_steps"
        ):
            self._make_trainer(
                os.path.join(workdir, "same"),
                feat_dir=feat_dir,
                max_steps=2,
                total_steps=6,
                resume_from=checkpoint,
            )

    def test_resume_rejects_partial_weights_except_provider_owned_omissions(self):
        import torch.nn as nn

        from specforge.algorithms.common.providers import (
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
            StepRuntimeConfig,
            checkpoint_key_fingerprint,
        )

        Program, _ = _fake_seam()

        class Draft(nn.Module):
            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))
                self.bias = nn.Parameter(torch.ones(1))
                self.embed_tokens = nn.Embedding(4, 1)
                self.embed_tokens.requires_grad_(False)

        class Composite(Program):
            def __init__(self):
                super().__init__(Draft())

        workdir = tempfile.mkdtemp(prefix="trainer_resume_keys_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        allowed_missing = {"embed_tokens.weight"}

        bogus_model = Composite()
        bogus_runtime = StepRuntimeConfig(
            options={},
            resume_contract={
                OMITTED_STATE_FINGERPRINT_CONTRACT_KEY: "caller-supplied-placeholder"
            },
            allowed_missing_checkpoint_keys=allowed_missing,
        )
        with self.assertRaisesRegex(ValueError, "does not match the live draft model"):
            self._make_trainer(
                os.path.join(workdir, "bogus-fingerprint"),
                feat_dir=feat_dir,
                max_steps=1,
                model=bogus_model,
                strategy_kwargs=bogus_runtime,
            )

        def runtime(model):
            return StepRuntimeConfig(
                options={},
                resume_contract={
                    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY: (
                        checkpoint_key_fingerprint(
                            model.draft_model,
                            allowed_missing,
                        )
                    )
                },
                allowed_missing_checkpoint_keys=allowed_missing,
            )

        source_model = Composite()
        initial, _, _ = self._make_trainer(
            os.path.join(workdir, "source"),
            feat_dir=feat_dir,
            max_steps=1,
            model=source_model,
            strategy_kwargs=runtime(source_model),
        )
        self.assertEqual(initial.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(workdir, "source", "rz-latest"))
        state_path = os.path.join(checkpoint, "training_state.pt")
        state = torch.load(state_path, map_location="cpu", weights_only=False)

        # EAGLE3 deliberately rebuilds its frozen target-copied embedding.
        state["draft_state_dict"].pop("embed_tokens.weight")
        torch.save(state, state_path)
        resumed_model = Composite()
        with torch.no_grad():
            resumed_model.draft_model.embed_tokens.weight.copy_(
                source_model.draft_model.embed_tokens.weight
            )
        resumed, _, _ = self._make_trainer(
            os.path.join(workdir, "allowed"),
            feat_dir=feat_dir,
            max_steps=2,
            resume_from=checkpoint,
            model=resumed_model,
            strategy_kwargs=runtime(resumed_model),
        )
        self.assertEqual(resumed.global_step, 1)

        # The same policy must not hide an unrelated missing trainable weight.
        state["draft_state_dict"].pop("bias")
        torch.save(state, state_path)
        corrupt_model = Composite()
        with torch.no_grad():
            corrupt_model.draft_model.embed_tokens.weight.copy_(
                source_model.draft_model.embed_tokens.weight
            )
        with self.assertRaisesRegex(ValueError, r"missing=\['bias'\]"):
            self._make_trainer(
                os.path.join(workdir, "corrupt"),
                feat_dir=feat_dir,
                max_steps=2,
                resume_from=checkpoint,
                model=corrupt_model,
                strategy_kwargs=runtime(corrupt_model),
            )

    def test_bound_algorithm_resume_contract_is_saved_and_validated(self):
        from specforge.algorithms.common.providers import StepRuntimeConfig

        workdir = tempfile.mkdtemp(prefix="trainer_resume_contract_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        source_runtime = StepRuntimeConfig(
            options={},
            resume_contract={
                "eagle3_test_objective": 0.8,
                "eagle3_optional_setting": None,
            },
            allowed_missing_checkpoint_keys=frozenset(),
        )
        source, _, _ = self._make_trainer(
            os.path.join(workdir, "source"),
            feat_dir=feat_dir,
            max_steps=1,
            strategy_kwargs=source_runtime,
        )
        self.assertEqual(source.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(workdir, "source", "rz-latest"))
        state = torch.load(
            os.path.join(checkpoint, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(state["eagle3_test_objective"], 0.8)
        self.assertIn("eagle3_optional_setting", state)
        self.assertIsNone(state["eagle3_optional_setting"])

        resumed, _, _ = self._make_trainer(
            os.path.join(workdir, "resumed"),
            feat_dir=feat_dir,
            max_steps=2,
            resume_from=checkpoint,
            strategy_kwargs=source_runtime,
        )
        self.assertEqual(resumed._controller.global_step, 1)

        changed_runtime = StepRuntimeConfig(
            options={},
            resume_contract={
                "eagle3_test_objective": 0.9,
                "eagle3_optional_setting": None,
            },
            allowed_missing_checkpoint_keys=frozenset(),
        )
        with self.assertRaisesRegex(
            ValueError,
            "eagle3_test_objective=0.8 but this run has eagle3_test_objective=0.9",
        ):
            self._make_trainer(
                os.path.join(workdir, "changed"),
                feat_dir=feat_dir,
                max_steps=2,
                resume_from=checkpoint,
                strategy_kwargs=changed_runtime,
            )

        changed_optional_runtime = StepRuntimeConfig(
            options={},
            resume_contract={
                "eagle3_test_objective": 0.8,
                "eagle3_optional_setting": 512,
            },
            allowed_missing_checkpoint_keys=frozenset(),
        )
        with self.assertRaisesRegex(
            ValueError,
            "eagle3_optional_setting=None but this run has "
            "eagle3_optional_setting=512",
        ):
            self._make_trainer(
                os.path.join(workdir, "changed_optional"),
                feat_dir=feat_dir,
                max_steps=2,
                resume_from=checkpoint,
                strategy_kwargs=changed_optional_runtime,
            )

    def test_direct_builder_requires_bound_provenance_for_omitted_embedding(self):
        from types import SimpleNamespace

        import torch.nn as nn

        from specforge.algorithms.common.providers import (
            MODEL_PROVENANCE_CONTRACT_KEY,
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
            STEP_OPTIONS_CONTRACT_KEY,
            StepRuntimeConfig,
            checkpoint_key_fingerprint,
        )
        from specforge.algorithms.objectives.registry import TrainingRequirements
        from specforge.launch import _assemble_trainer
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore
        from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

        Program, _ = _fake_seam()

        class Draft(nn.Module):
            _no_split_modules = []

            def __init__(self):
                super().__init__()
                self.w = nn.Parameter(torch.zeros(1))
                self.embed_tokens = nn.Embedding(4, 1)
                self.embed_tokens.requires_grad_(False)

        class Composite(Program):
            # the builder reads the program's declarations from these two
            objective_name = "eagle3"
            registration = SimpleNamespace(
                inputs=("x",), training=TrainingRequirements()
            )

            def __init__(self):
                super().__init__(Draft())

            def draft_state(self, state):
                state = super().draft_state(state)
                return {
                    key: value for key, value in state.items() if "embed" not in key
                }

        algorithm = SimpleNamespace(name="eagle3")
        workdir = tempfile.mkdtemp(prefix="direct_builder_eagle_resume_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        refs = OfflineManifestReader(feat_dir, run_id="data").read()

        def runtime(model, *, options=None):
            allowed_missing = {"embed_tokens.weight"}
            return StepRuntimeConfig(
                options={"name": "eagle3"} if options is None else options,
                resume_contract={
                    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY: (
                        checkpoint_key_fingerprint(
                            model.draft_model,
                            allowed_missing,
                        )
                    ),
                    MODEL_PROVENANCE_CONTRACT_KEY: (
                        ("source_identity_format", "stat-v2"),
                        ("test_model_source", "fixture-v1"),
                    ),
                },
                allowed_missing_checkpoint_keys=allowed_missing,
            )

        def runtime_without_provenance(model):
            allowed_missing = {"embed_tokens.weight"}
            return StepRuntimeConfig(
                options={},
                resume_contract={
                    OMITTED_STATE_FINGERPRINT_CONTRACT_KEY: (
                        checkpoint_key_fingerprint(
                            model.draft_model,
                            allowed_missing,
                        )
                    )
                },
                allowed_missing_checkpoint_keys=allowed_missing,
            )

        def assemble(
            model,
            *,
            output_dir,
            max_steps,
            resume_from=None,
            strategy_runtime=None,
        ):
            with _no_fsdp_wrap():
                return _assemble_trainer(
                    algorithm=algorithm,
                    controller=DataFlowController("direct"),
                    store=LocalFeatureStore("direct-store"),
                    ref_source={"refs": refs},
                    model=model,
                    optimizer_factory=lambda module: BF16Optimizer(
                        module,
                        lr=0.05,
                        max_grad_norm=1.0,
                        warmup_ratio=0.0,
                        total_steps=100,
                    ),
                    run_id="direct",
                    output_dir=output_dir,
                    batch_size=2,
                    accumulation_steps=1,
                    num_epochs=1,
                    max_steps=max_steps,
                    total_steps=100,
                    save_interval=0,
                    logger=None,
                    log_interval=50,
                    collate_fn=_x_collate,
                    per_sample_transform=_x_transform,
                    resume_from=resume_from,
                    strategy_kwargs=strategy_runtime,
                )

        output_dir = os.path.join(workdir, "out")
        source_model = Composite()
        first = assemble(
            source_model,
            output_dir=output_dir,
            max_steps=1,
            strategy_runtime=runtime(source_model),
        )
        self.assertEqual(first.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(output_dir, "direct-latest"))
        state = torch.load(
            os.path.join(checkpoint, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertNotIn("embed_tokens.weight", state["draft_state_dict"])

        with self.assertRaisesRegex(ValueError, "provider-bound StepRuntimeConfig"):
            assemble(
                Composite(),
                output_dir=output_dir,
                max_steps=2,
                resume_from=checkpoint,
            )

        with self.assertRaisesRegex(ValueError, MODEL_PROVENANCE_CONTRACT_KEY):
            assemble(
                source_model,
                output_dir=output_dir,
                max_steps=2,
                resume_from=checkpoint,
                strategy_runtime=runtime_without_provenance(source_model),
            )

        unchanged_model = Composite()
        with torch.no_grad():
            unchanged_model.draft_model.embed_tokens.weight.copy_(
                source_model.draft_model.embed_tokens.weight
            )
        with self.assertRaisesRegex(ValueError, STEP_OPTIONS_CONTRACT_KEY):
            assemble(
                unchanged_model,
                output_dir=output_dir,
                max_steps=2,
                resume_from=checkpoint,
                # Other options than the checkpoint recorded, that describe
                # the same run.
                strategy_runtime=runtime(
                    unchanged_model,
                    options={"name": "eagle3", "objective": "eagle3"},
                ),
            )
        # An option is recorded for resume, so one that describes nothing of
        # the program is refused before anything trains.
        for options, message in (
            ({"objective_scale": 2.0}, r"\['objective_scale'\] configure nothing"),
            ({"name": "eagle3-other"}, "whose name is 'eagle3'"),
            ({"objective": "dflash"}, "whose objective is 'eagle3'"),
            (
                {"required_features": ("x", "teacher")},
                r"whose required_features is \{'x'\}",
            ),
        ):
            with (
                self.subTest(options=options),
                self.assertRaisesRegex(ValueError, message),
            ):
                assemble(
                    Composite(),
                    output_dir=os.path.join(workdir, "refused"),
                    max_steps=1,
                    strategy_runtime=StepRuntimeConfig(
                        options=options,
                        resume_contract={},
                        allowed_missing_checkpoint_keys=frozenset(),
                    ),
                )
        described = StepRuntimeConfig(
            options={"objective": "eagle3", "required_features": ("x",)},
            resume_contract={},
            allowed_missing_checkpoint_keys=frozenset(),
        )
        assemble(
            Composite(),
            output_dir=os.path.join(workdir, "described"),
            max_steps=1,
            strategy_runtime=described,
        )

        changed_model = Composite()
        with torch.no_grad():
            changed_model.draft_model.embed_tokens.weight.fill_(123.0)
        with self.assertRaisesRegex(
            ValueError,
            OMITTED_STATE_FINGERPRINT_CONTRACT_KEY,
        ):
            assemble(
                changed_model,
                output_dir=output_dir,
                max_steps=2,
                resume_from=checkpoint,
                strategy_runtime=runtime(changed_model),
            )

        resumed_model = Composite()
        with torch.no_grad():
            resumed_model.draft_model.embed_tokens.weight.copy_(
                source_model.draft_model.embed_tokens.weight
            )
        resumed = assemble(
            resumed_model,
            output_dir=output_dir,
            max_steps=2,
            resume_from=checkpoint,
            strategy_runtime=runtime(resumed_model),
        )
        self.assertEqual(resumed.global_step, 1)
        self.assertEqual(resumed.fit(), 2)

    def test_resume_rejects_gradient_clip_drift(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_grad_clip_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        source, _, _ = self._make_trainer(
            os.path.join(workdir, "source"),
            feat_dir=feat_dir,
            max_steps=1,
            max_grad_norm=1.0,
        )
        self.assertEqual(source.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(workdir, "source", "rz-latest"))

        with self.assertRaisesRegex(
            ValueError,
            "max_grad_norm=1.0 but this run has max_grad_norm=2.0",
        ):
            self._make_trainer(
                os.path.join(workdir, "changed"),
                feat_dir=feat_dir,
                max_steps=2,
                max_grad_norm=2.0,
                resume_from=checkpoint,
            )

    def test_rank_that_cannot_restore_its_state_tells_the_others(self):
        from specforge.training.runner import DataParallelRunner

        workdir = tempfile.mkdtemp(prefix="trainer_resume_agreed_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        source, _, _ = self._make_trainer(
            os.path.join(workdir, "source"), feat_dir=feat_dir, max_steps=1
        )
        self.assertEqual(source.fit(), 1)
        checkpoint = os.path.realpath(os.path.join(workdir, "source", "rz-latest"))
        original = DataParallelRunner.agree
        told = []

        def agree(runner, error, *, value=None):
            told.append(error)
            return original(runner, error, value=value)

        resume = dict(
            feat_dir=feat_dir, max_steps=2, max_grad_norm=2.0, resume_from=checkpoint
        )
        with (
            mock.patch.object(DataParallelRunner, "agree", agree),
            # This rank raises what it failed with, after the agreement.
            self.assertRaisesRegex(ValueError, "max_grad_norm=1.0 but this run has"),
        ):
            self._make_trainer(os.path.join(workdir, "changed"), **resume)
        self.assertEqual(len(told), 1)
        self.assertRegex(told[0], "^ValueError: .*max_grad_norm=1.0 but this run has")
        # A rank that restored hears of it and raises as well.
        told_by_peer = RuntimeError("rank 1: ValueError: wrong master shape")
        with (
            mock.patch.object(DataParallelRunner, "agree", side_effect=told_by_peer),
            self.assertRaisesRegex(RuntimeError, "rank 1: ValueError"),
        ):
            self._make_trainer(
                os.path.join(workdir, "peer"),
                feat_dir=feat_dir,
                max_steps=2,
                resume_from=checkpoint,
            )

    def test_completed_checkpoint_resume_is_a_checkpoint_noop(self):
        workdir = tempfile.mkdtemp(prefix="trainer_resume_complete_")
        feat_dir = _write_feature_files(os.path.join(workdir, "features"), n=4)
        out = os.path.join(workdir, "out")
        complete, model, _ = self._make_trainer(out, feat_dir=feat_dir, max_steps=None)
        self.assertEqual(complete.fit(), 2)
        self.assertEqual(complete._controller.epoch, 1)
        checkpoint_uri = f"file://{os.path.realpath(os.path.join(out, 'rz-latest'))}"
        saved_weight = model.draft_model.w.detach().clone()

        resumed, resumed_model, seen = self._make_trainer(
            os.path.join(workdir, "resumed"),
            feat_dir=feat_dir,
            max_steps=None,
            resume_from=checkpoint_uri,
        )
        with mock.patch.object(resumed, "save_checkpoint") as save:
            self.assertEqual(resumed.fit(), 2)
        save.assert_not_called()
        self.assertEqual(seen, [])
        self.assertTrue(torch.equal(resumed_model.draft_model.w, saved_weight))


class TestFitReentry(unittest.TestCase):
    """fit() at/after max_steps and re-entry after a mid-epoch return (CPU fakes)."""

    def _controller(self, seen, max_steps, out_dir, **kw):
        from specforge.training.controller import TrainerController

        num_epochs = kw.pop("num_epochs", 1)
        Composite, Backend = _fake_seam()
        model = Composite()
        backend = Backend(model)
        spec = _spec()
        ctrl = TrainerController(
            _recording_runner(seen)(model, backend, spec),
            strategy_name=spec.name,
            run_id="r",
            output_dir=out_dir,
            max_steps=max_steps,
            num_epochs=num_epochs,
            **kw,
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

    def test_completed_epoch_returns_without_consuming(self):
        class Explode:
            def __iter__(self):
                raise AssertionError("completed resume consumed its online queue")

        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, backend = self._controller(seen, None, d, start_step=3, start_epoch=1)
            self.assertEqual(ctrl.fit(Explode()), 3)
            self.assertEqual(seen, [])
            self.assertEqual(backend.steps, 0)

    def test_prepositioned_online_queue_is_not_skipped_twice(self):
        class PrepositionedQueue:
            def __init__(self, batches):
                self.batches = batches

            def __iter__(self):
                return iter(self.batches)

            def seek(self, _):
                raise AssertionError("prepositioned online queue was sought again")

        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, backend = self._controller(
                seen,
                3,
                d,
                start_step=1,
                start_batch=1,
                start_samples=1,
                data_prepositioned=True,
            )
            queue = PrepositionedQueue(self._batches(3)[1:])
            self.assertEqual(ctrl.fit(queue), 3)
            self.assertEqual(seen, ["b1", "b2"])
            self.assertEqual(backend.steps, 2)

    def test_offline_resume_sets_saved_epoch_before_seeking_samples(self):
        class EpochData:
            def __init__(self, batches):
                self.batches = batches
                self.current = []
                self.skip = 0
                self.calls = []

            def set_epoch(self, epoch):
                self.calls.append(("set_epoch", epoch))
                self.current = self.batches[epoch]

            def seek(self, batches):
                self.calls.append(("seek", batches))
                self.skip = batches

            def __iter__(self):
                skip, self.skip = self.skip, 0
                return iter(self.current[skip:])

        epoch0 = self._batches(2)
        epoch1 = self._batches(2)
        for index, batch in enumerate(epoch1):
            batch.sample_ids = [f"epoch1-{index}"]
        data = EpochData({0: epoch0, 1: epoch1})
        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, _ = self._controller(
                seen,
                2,
                d,
                num_epochs=2,
                start_step=1,
                start_epoch=1,
                start_batch=1,
                start_samples=1,
            )
            self.assertEqual(ctrl.fit(data), 2)
        self.assertEqual(data.calls[:2], [("set_epoch", 1), ("seek", 1)])
        self.assertEqual(seen, ["epoch1-1"])

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

    def test_flattened_online_epochs_checkpoint_as_one_midstream_epoch(self):
        with tempfile.TemporaryDirectory() as d:
            seen = []
            ctrl, _ = self._controller(seen, 2, d, num_epochs=1)
            # Assembly has already expanded and independently truncated all
            # logical prompt epochs; TrainerController intentionally sees one
            # deterministic queue epoch.
            self.assertEqual(ctrl.fit(self._batches(4)), 2)
            checkpoint = ctrl.save_checkpoint(2)
            state = torch.load(
                os.path.join(
                    checkpoint.checkpoint_uri.removeprefix("file://"),
                    "training_state.pt",
                ),
                map_location="cpu",
                weights_only=False,
            )
        self.assertEqual(state["epoch"], 0)
        self.assertEqual(state["epoch_batch"], 2)
        self.assertEqual(state["epoch_samples"], 2)
        self.assertEqual(state["global_step"], 2)


@unittest.skipUnless(CUDA, "checkpoint resume requires CUDA")
class TestCheckpointResume(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if torch.distributed.is_initialized():
            return

        from specforge.distributed import destroy_distributed
        from tests.test_runtime import _fixtures as fx

        env = mock.patch.dict(os.environ)
        env.start()
        cls.addClassCleanup(env.stop)
        cls.addClassCleanup(destroy_distributed)
        fx.build_single_rank_distributed(port="29563")

    def test_checkpoint_resume(self):
        torch.manual_seed(0)
        from specforge.optimizer import BF16Optimizer
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerController
        from tests.test_runtime import _fixtures as fx

        TTT, BS, N = 3, 2, 6
        workdir = tempfile.mkdtemp(prefix="ckpt_resume_")
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(workdir, "target"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)
        out_dir = os.path.join(workdir, "out")

        def build_model():
            return fx.load_eagle3(
                cfg, target_dir, vocab_path, ttt=TTT, backend="flex_attention"
            )

        loader = fx.build_offline_eagle3_loader(
            feat_dir,
            batch_size=BS,
            run_id="checkpoint-data",
            ttt_length=TTT,
            max_len=512,
        )

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
        runner = fx.build_runner(model, backend)
        ctrl = TrainerController(
            runner,
            strategy_name=runner.spec.name,
            run_id="r",
            output_dir=out_dir,
            max_steps=3,
            num_epochs=2,
        )
        step = ctrl.fit(loader)
        self.assertEqual(step, 3)
        ck = ctrl.save_checkpoint(step)

        from specforge.training.checkpoint import CheckpointManager

        # run-scoped on-disk names
        ckpt_dir = ck.checkpoint_uri[len("file://") :]
        self.assertEqual(os.path.basename(ckpt_dir), "r-step3")
        self.assertTrue(os.path.islink(os.path.join(out_dir, "r-latest")))
        self.assertFalse(os.path.exists(os.path.join(out_dir, "latest")))
        mgr = CheckpointManager(out_dir, "r")
        self.assertEqual(os.path.realpath(mgr.latest_dir()), os.path.realpath(ckpt_dir))

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
        self.assertEqual(set(rng), {"torch", "device_type", "cuda", "npu"})
        # single bound-device CUDA state, not the legacy per-device list
        self.assertEqual(rng["device_type"], "cuda")
        self.assertIsInstance(rng["cuda"], torch.Tensor)
        self.assertIsNone(rng["npu"])

        # Canonical checkpoints retain frozen embeddings as well as trainable
        # parameters, so reload does not depend on a separate target directory.
        trained_sd = {
            k: v.detach().cpu() for k, v in model.draft_model.state_dict().items()
        }
        persisted = ckpt["draft_state_dict"]
        self.assertEqual(set(persisted), set(trained_sd))

        fresh = build_model()
        missing, unexpected = fresh.draft_model.load_state_dict(persisted, strict=True)
        self.assertEqual(unexpected, [])
        self.assertEqual(missing, [])
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
        from specforge.optimizer import BF16Optimizer
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerController
        from tests.test_runtime import _fixtures as fx

        TTT, BS, TOTAL, CUT = 3, 2, 6, 3
        workdir = tempfile.mkdtemp(prefix="ckpt_continuity_")
        # TOTAL distinct batches: resuming on the wrong data cannot pass.
        feat_dir = fx.write_offline_files(
            os.path.join(workdir, "features"), n=BS * TOTAL
        )

        def build_loader():
            return fx.build_offline_eagle3_loader(
                feat_dir,
                batch_size=BS,
                run_id="continuity-data",
                ttt_length=TTT,
                max_len=512,
            )

        self.addCleanup(
            torch.set_deterministic_debug_mode, torch.get_deterministic_debug_mode()
        )
        torch.use_deterministic_algorithms(True, warn_only=True)

        def build_run():
            torch.manual_seed(0)
            model, _ = fx.build_eagle3(workdir, ttt=TTT)
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
            return model, backend, fx.build_runner(model, backend)

        # Reference: one uninterrupted run of TOTAL steps (loss + lr per step).
        losses_ref, lrs_ref = [], []
        _, backend_r, runner_r = build_run()
        ctrl_r = TrainerController(
            runner_r,
            strategy_name=runner_r.spec.name,
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
        ctrl_r.fit(build_loader())
        self.assertEqual(len(losses_ref), TOTAL)

        # Interrupted phase 1: CUT steps, then save.
        out_a = os.path.join(workdir, "rez")
        losses_a = []
        model_a, backend_a, runner_a = build_run()
        ctrl_a = TrainerController(
            runner_a,
            strategy_name=runner_a.spec.name,
            run_id="rez",
            output_dir=out_a,
            max_steps=CUT,
            log_interval=1,
            logger=lambda m, s: losses_a.append(m["loss"]),
        )
        torch.manual_seed(0)
        ctrl_a.fit(build_loader())
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
        model_b, backend_b, runner_b = build_run()
        model_b.draft_model.load_state_dict(state["draft_state_dict"], strict=False)
        backend_b.load_state_dict(state["backend"])

        # fp32 masters restored exactly (not re-quantized from bf16)
        for mb, ma in zip(
            backend_b.optimizer.fp32_params, backend_a.optimizer.fp32_params
        ):
            self.assertTrue(torch.equal(mb.detach().cpu(), ma.detach().cpu()))
        # scheduler position restored — a reset would read 0 (warmup 0: the
        # cosine after_scheduler carries the position, not the outer wrapper)
        self.assertEqual(backend_b.optimizer.scheduler.after_scheduler.last_epoch, CUT)
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
            runner_b,
            strategy_name=runner_b.spec.name,
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
        ctrl_b.fit(build_loader())
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


def _two_rank_fsdp_resume_worker(rank, workdir, use_cuda):
    """Compare an uninterrupted step against restarting from a sharded checkpoint."""
    import torch.distributed as dist
    import torch.nn as nn
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

    from specforge.algorithms.objectives.base import ObjectiveResult
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.checkpoint import CheckpointManager
    from specforge.training.controller import TrainerController
    from specforge.training.runner import DataParallelRunner
    from tests.test_runtime._fixtures import FakeProgram, fake_spec

    class Draft(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)
            self.out = nn.Linear(8, 4)

        def forward(self, x):
            return self.out(torch.relu(self.proj(x)))

    class Composite(FakeProgram):
        def __init__(self):
            super().__init__()
            self.draft_model = Draft()

        def forward(
            self, tensors, *, step, total_steps, collect_detailed_metrics, **replay
        ):
            x = tensors["x"]
            # Use RNG so rank-local RNG restoration is also verified.
            noise = torch.randn_like(x) * 0.01
            out = self.draft_model(x + noise)
            return ObjectiveResult((out * out).mean(), torch.ones((), device=x.device))

        def draft_state(self, state):
            prefix = "draft_model."
            return {
                k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)
            }

    if use_cuda:
        torch.cuda.set_device(rank)
        device = torch.device("cuda", rank)
        pg_backend = "nccl"
    else:
        torch.set_num_threads(1)
        device = torch.device("cpu")
        pg_backend = "gloo"

    dist.init_process_group(
        pg_backend,
        init_method=f"file://{workdir}/rendezvous",
        rank=rank,
        world_size=2,
    )
    try:
        orig_fsdp = FSDP

        def cpu_fsdp(module, *args, **kwargs):
            kwargs.setdefault("device_id", torch.device("cpu"))
            return orig_fsdp(module, *args, **kwargs)

        patch_ctx = (
            contextlib.nullcontext()
            if use_cuda
            else mock.patch("torch.distributed.fsdp.FullyShardedDataParallel", cpu_fsdp)
        )

        # Two deterministic rank-distinct batches: step 1 before save, step 2 after.
        step1_batch = TrainBatch(
            sample_ids=[f"s1-r{rank}"],
            strategy="sharded-resume",
            tensors={
                "x": torch.linspace(
                    0.1 * (rank + 1), 1.0 + 0.1 * rank, 16, device=device
                ).reshape(2, 8)
            },
        )
        step2_batch = TrainBatch(
            sample_ids=[f"s2-r{rank}"],
            strategy="sharded-resume",
            tensors={
                "x": torch.linspace(
                    -0.5 - 0.1 * rank, 0.7 + 0.2 * rank, 16, device=device
                ).reshape(2, 8)
            },
        )

        def build_run(out_dir, *, start_step=0, max_steps=2, draft_weights=None):
            torch.manual_seed(1234)
            model = Composite().to(device)
            if draft_weights is not None:
                model.draft_model.load_state_dict(draft_weights, strict=True)
            pc = ParallelConfig(
                world_size=2,
                sharding_strategy="SHARD_GRAD_OP",
                param_dtype=torch.float32,
                fsdp_process_group=dist.group.WORLD,
            )

            class _ZeroingAdamW(torch.optim.AdamW):
                def step(self, closure=None):
                    out = super().step(closure=closure)
                    self.zero_grad(set_to_none=True)
                    return out

            backend = FSDPTrainingBackend(
                pc,
                optimizer_factory=lambda target: _ZeroingAdamW(
                    target.parameters(), lr=1e-2
                ),
            )
            with patch_ctx:
                wrapped = backend.prepare_model(
                    model, optimizer_target=model.draft_model
                )
            runner = DataParallelRunner(
                wrapped, backend, fake_spec("x", name="sharded-resume")
            )
            ctrl = TrainerController(
                runner,
                strategy_name="sharded-resume",
                run_id="fsdp-resume",
                output_dir=out_dir,
                max_steps=max_steps,
                start_step=start_step,
                checkpoint_manager=CheckpointManager(out_dir, "fsdp-resume"),
            )
            return runner, backend, ctrl

        def snapshot_optimizer(opt):
            state = opt.state_dict()["state"]
            return {
                k: {
                    name: val.detach().cpu().clone() if torch.is_tensor(val) else val
                    for name, val in entry.items()
                }
                for k, entry in state.items()
            }

        # 1) Uninterrupted two-step reference.
        ref_runner, ref_backend, ref_ctrl = build_run(
            os.path.join(workdir, "ref"), max_steps=2
        )
        torch.manual_seed(100 + rank)
        ref_ctrl.fit([step1_batch, step2_batch])
        ref_state = ref_runner.checkpoint_state()
        ref_opt = snapshot_optimizer(ref_backend.optimizer)

        # 2) Interrupted run: step 1 -> save checkpoint -> restart -> step 2.
        ckpt_out = os.path.join(workdir, "ckpt")
        _, _, a_ctrl = build_run(ckpt_out, max_steps=1)
        torch.manual_seed(100 + rank)
        a_ctrl.fit([step1_batch])
        ck = a_ctrl.save_checkpoint(1)
        dist.barrier()

        ckpt_dir = ck.checkpoint_uri[len("file://") :]
        shared = torch.load(
            os.path.join(ckpt_dir, "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        assert "replicated_optimizer_state" not in shared
        assert os.path.exists(os.path.join(ckpt_dir, f"training_state_rank{rank}.pt"))

        resume = CheckpointManager.read_resume_state(ck.checkpoint_uri)
        assert resume["backend"]["optimizer"] is not None

        b_runner, b_backend, b_ctrl = build_run(
            os.path.join(workdir, "resumed"),
            start_step=1,
            max_steps=2,
            draft_weights=resume["draft_state_dict"],
        )
        b_runner.load_resume(resume["backend"])
        b_runner.agree(None)
        b_ctrl.fit([step2_batch])
        b_state = b_runner.checkpoint_state()
        b_opt = snapshot_optimizer(b_backend.optimizer)

        # Compare full gathered draft weights on rank 0.
        if rank == 0:
            assert set(b_state.draft) == set(ref_state.draft)
            for name, ref_tensor in ref_state.draft.items():
                if not torch.equal(b_state.draft[name].cpu(), ref_tensor.cpu()):
                    raise AssertionError(f"draft weight mismatch on {name}")

        # Compare each rank's sharded optimizer state tensors.
        assert set(b_opt) == set(ref_opt) and len(b_opt) > 0
        for param_id, ref_entry in ref_opt.items():
            b_entry = b_opt[param_id]
            assert set(b_entry) == set(ref_entry)
            for key, ref_val in ref_entry.items():
                if torch.is_tensor(ref_val):
                    if not torch.equal(b_entry[key], ref_val):
                        raise AssertionError(
                            f"rank {rank} optimizer shard mismatch on {param_id}.{key}"
                        )
                else:
                    assert b_entry[key] == ref_val
    finally:
        dist.destroy_process_group()


class TestTwoRankFSDPShardedResume(unittest.TestCase):
    def test_two_rank_fsdp_sharded_resume_matches_uninterrupted(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as workdir:
            mp.spawn(
                _two_rank_fsdp_resume_worker,
                args=(workdir, False),
                nprocs=2,
                join=True,
            )

    @unittest.skipUnless(
        CUDA and torch.cuda.device_count() >= 2,
        "two-GPU FSDP sharded resume requires >=2 CUDA devices",
    )
    def test_two_gpu_fsdp_sharded_resume_matches_uninterrupted(self):
        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as workdir:
            mp.spawn(
                _two_rank_fsdp_resume_worker,
                args=(workdir, True),
                nprocs=2,
                join=True,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
