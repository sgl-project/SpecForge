# coding=utf-8
"""TrainerCore grad-accum + TrainerController fit/checkpoint (CPU)."""

import json
import os
import tempfile
import unittest

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput
from specforge.training.controller import (
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


class FakeBackend(TrainingBackend):
    name = "fake"

    def __init__(self, model):
        self.model = model
        self.steps = 0
        self.backwards = 0

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        self.backwards += 1
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
            with open(os.path.join(d, "best_meta.json")) as fh:
                meta = json.load(fh)
            self.assertEqual(meta["step"], 1)
            self.assertAlmostEqual(meta["score"], 0.5, places=6)


class TestResumeDataPosition(unittest.TestCase):
    def test_start_batch_skips_consumed_prefix(self):
        # Resume mid-epoch: start_batch=k drops the first k batches of the FIRST
        # epoch only; later epochs iterate in full.
        class RecordingStrategy(FakeStrategy):
            def __init__(self):
                super().__init__()
                self.seen = []

            def forward_loss(self, batch, ctx=None):
                self.seen.append(batch.sample_ids[0])
                return super().forward_loss(batch, ctx)

        def batch(i):
            return TrainBatch(
                sample_ids=[f"b{i}"],
                strategy="fake",
                tensors={"x": torch.ones(2)},
                metadata={},
            )

        strat = RecordingStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                num_epochs=2,
                start_batch=3,
            )
            ctrl.fit([batch(i) for i in range(5)])
        # epoch 0 resumes at b3; epoch 1 is complete
        self.assertEqual(strat.seen, ["b3", "b4", "b0", "b1", "b2", "b3", "b4"])
        self.assertEqual(ctrl.epoch_batch, 5)


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
