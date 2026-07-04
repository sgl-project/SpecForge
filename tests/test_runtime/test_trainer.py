# coding=utf-8
"""TrainerCore grad-accum + TrainerController fit/checkpoint (CPU)."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch
from specforge.runtime.training.backend import TrainingBackend
from specforge.runtime.training.strategy import DraftTrainStrategy, StepOutput
from specforge.runtime.training.trainer import (
    Checkpoint,
    TrainerController,
    TrainerCore,
)


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))


class FakeStrategy(DraftTrainStrategy):
    name = "fake"
    required_features = {"x"}

    def __init__(self):
        self.model = TinyModel()
        self.last_ctx = None

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch: TrainBatch, ctx=None) -> StepOutput:
        self.validate_batch(batch)
        self.last_ctx = ctx  # capture what fit() threads in (StepContext regression)
        loss = (self.model.w * batch.tensors["x"].sum()).abs()
        return StepOutput(loss=loss, metrics={"accuracy": torch.tensor(0.5)})


class RecordingStrategy(FakeStrategy):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward_loss(self, batch, ctx=None):
        self.seen.append(batch.sample_ids[0])
        return super().forward_loss(batch, ctx)


class FakeBackend(TrainingBackend):
    name = "fake"

    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.backwards = 0
        self.boundaries = []  # is_boundary flag per backward (the no_sync gate)

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        self.backwards += 1
        self.boundaries.append(is_boundary)
        loss.backward()

    def step(self):
        self.steps += 1
        return torch.tensor(1.0)

    def state_dict(self):
        return {
            "model": {"draft_model.w": self.model.w.detach().clone()},
            "optimizer": None,
            "rng": {},
        }

    def load_state_dict(self, state):
        pass


def _batch():
    return TrainBatch(
        sample_ids=["s"], strategy="fake", tensors={"x": torch.ones(2)}, metadata={}
    )


class TestTrainerCore(unittest.TestCase):
    def test_accumulation_boundary(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=2)
        m0 = core.train_step(_batch())
        self.assertFalse(m0.optimizer_stepped)  # no optimizer step yet
        self.assertIsNone(m0.grad_norm)
        self.assertEqual(backend.steps, 0)
        m1 = core.train_step(_batch())
        self.assertTrue(m1.optimizer_stepped)  # step on the 2nd micro-batch
        self.assertIsNotNone(m1.grad_norm)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(backend.backwards, 2)
        # boundary known BEFORE backward so the backend can no_sync micro-steps
        self.assertEqual(backend.boundaries, [False, True])

    def test_metrics_carry_no_mode(self):
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        rep = core.train_step(_batch())
        self.assertNotIn("mode", rep.metrics)

    def test_validate_batch_missing_feature(self):
        strat = FakeStrategy()
        bad = TrainBatch(sample_ids=["s"], strategy="fake", tensors={}, metadata={})
        with self.assertRaises(ValueError):
            strat.forward_loss(bad)


class TestTrainerController(unittest.TestCase):
    def test_fit_and_checkpoint(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                max_steps=3,
                num_epochs=5,
            )
            data = [_batch() for _ in range(10)]
            step = ctrl.fit(data)
            self.assertEqual(step, 3)  # max_steps honored
            self.assertEqual(backend.steps, 3)
            ckpt = ctrl.save_checkpoint(step)
            self.assertIsInstance(ckpt, Checkpoint)
            self.assertTrue(ckpt.checkpoint_uri.startswith("file://"))
            self.assertEqual(ckpt.global_step, 3)


class TestBestTracking(unittest.TestCase):
    def test_best_fires_on_misaligned_eval_and_save_intervals(self):
        # eval_interval=1, save_interval=2: the intervals NEVER coincide on step 1,
        # yet the first eval (the run's best score) must create the best pointer —
        # a checkpoint is persisted on demand for it.
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                max_steps=3,
                num_epochs=1,
                eval_interval=1,
                save_interval=2,
            )
            ctrl.fit([_batch() for _ in range(5)], eval_data=[_batch()])
            # FakeStrategy's accuracy is constant 0.5, so step 1 is the best.
            self.assertTrue(os.path.isdir(os.path.join(d, "r-step1")))
            with open(os.path.join(d, "r.best_meta.json")) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["step"], 1)
            self.assertAlmostEqual(meta["score"], 0.5, places=6)


class TestEvalMetricsFlow(unittest.TestCase):
    def test_eval_metrics_reach_logger_and_last_metrics(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        logged = []
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                max_steps=2,
                num_epochs=1,
                eval_interval=1,
                log_interval=50,  # train metrics never hit the logger in 2 steps
                logger=lambda m, s: logged.append((dict(m), s)),
            )
            ctrl.fit([_batch() for _ in range(3)], eval_data=[_batch()])
        # eval metrics are logged at EVERY eval step, independent of log_interval
        self.assertEqual([s for _, s in logged], [1, 2])
        for m, _ in logged:
            self.assertIn("eval/avg_loss", m)
            self.assertAlmostEqual(m["eval/avg_acc"], 0.5, places=6)
        # merged next to the train keys
        self.assertIn("eval/avg_acc", ctrl.last_metrics)
        self.assertIn("loss", ctrl.last_metrics)
        self.assertNotIn("mode", ctrl.last_metrics)


def _named_batch(i):
    return TrainBatch(
        sample_ids=[f"b{i}"],
        strategy="fake",
        tensors={"x": torch.ones(2)},
        metadata={},
    )


class TestResumeDataPosition(unittest.TestCase):
    def test_start_batch_skips_consumed_prefix(self):
        # Resume mid-epoch: start_batch=k drops the first k batches of the FIRST
        # epoch only; later epochs iterate in full.
        strat = RecordingStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                num_epochs=2,
                start_batch=3,
                start_samples=3,
            )
            ctrl.fit([_named_batch(i) for i in range(5)])
        # epoch 0 resumes at b3; epoch 1 is complete
        self.assertEqual(strat.seen, ["b3", "b4", "b0", "b1", "b2", "b3", "b4"])
        self.assertEqual(ctrl._epoch_batch, 0)  # reset when the epoch completes

    def test_ctor_rejects_half_specified_position(self):
        # start_batch/start_samples describe ONE position; one without the other
        # is a corrupt resume.
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model))
        for kw in ({"start_batch": 3}, {"start_samples": 3}):
            with self.assertRaisesRegex(ValueError, "zero or nonzero together"):
                TrainerController(core, run_id="r", output_dir="/tmp-unused", **kw)

    def test_islice_skip_past_end_raises(self):
        # plain-iterable fallback: a silent empty epoch is banned.
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                num_epochs=1,
                start_batch=7,
                start_samples=7,
            )
            with self.assertRaisesRegex(
                ValueError, "skips past the end of the data"
            ):
                ctrl.fit([_batch() for _ in range(5)])

    def test_islice_skip_of_exactly_all_batches_is_allowed(self):
        # skip == available means the epoch was fully consumed pre-restart.
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                num_epochs=1,
                start_batch=5,
                start_samples=5,
            )
            step = ctrl.fit([_batch() for _ in range(5)])
        self.assertEqual(step, 0)
        self.assertEqual(backend.steps, 0)

    def test_reentry_after_max_steps_does_not_retrain_prefix(self):
        # a mid-epoch max_steps return keeps the live position; re-entering
        # fit() continues from it instead of re-training the prefix.
        strat = RecordingStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        data = [_named_batch(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core, run_id="r", output_dir=d, num_epochs=1, max_steps=2
            )
            self.assertEqual(ctrl.fit(data), 2)
            ctrl.max_steps = 4
            self.assertEqual(ctrl.fit(data), 4)
        self.assertEqual(strat.seen, ["b0", "b1", "b2", "b3"])


class TestStepContextThreading(unittest.TestCase):
    """Regression for the Domino schedule bug: fit() must thread a real schedule
    horizon into StepContext.total_steps (falling back to max_steps), not None —
    otherwise DominoTrainStrategy._lambda_base is pinned to 0 for the whole run and
    the base-loss warmup silently never happens."""

    def _last_ctx(self, **controller_kw):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core, run_id="r", output_dir=d, num_epochs=1, **controller_kw
            )
            ctrl.fit([_batch() for _ in range(3)])
        return strat.last_ctx

    def test_explicit_total_steps_is_threaded(self):
        ctx = self._last_ctx(total_steps=10, max_steps=3)
        self.assertEqual(ctx.total_steps, 10)  # explicit horizon wins over the cap

    def test_total_steps_falls_back_to_max_steps(self):
        # the common case: only max_steps set -> use it as the horizon so a
        # schedule-dependent loss is NOT silently disabled.
        ctx = self._last_ctx(max_steps=5)
        self.assertEqual(ctx.total_steps, 5)

    def test_total_steps_none_when_unbounded(self):
        ctx = self._last_ctx()  # neither total_steps nor max_steps set
        self.assertIsNone(ctx.total_steps)


if __name__ == "__main__":
    unittest.main(verbosity=2)
