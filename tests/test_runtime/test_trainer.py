# coding=utf-8
"""TrainerCore grad-accum + TrainerController fit/checkpoint (CPU)."""

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

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch: TrainBatch) -> StepOutput:
        self.validate_batch(batch)
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

    def backward(self, loss):
        self.backwards += 1
        loss.backward()

    def step(self):
        self.steps += 1
        return torch.tensor(1.0)

    def state_dict(self):
        return {"draft_model.w": self.model.w.detach().clone()}

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
