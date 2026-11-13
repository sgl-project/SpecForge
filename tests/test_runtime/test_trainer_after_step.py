"""TrainerCore runs the strategy's after_step at optimizer boundaries and logs its metrics."""

import unittest
from contextlib import nullcontext

import torch
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput


class _Backend:
    parallel_config = None

    def accumulation_context(self, *, is_boundary):
        return nullcontext()

    def backward(self, loss, *, is_boundary):
        loss.backward()

    def step(self, **kwargs):
        return None


class _Strategy(DraftTrainStrategy):
    name = "stub"
    required_features = set()

    def __init__(self):
        self.module = nn.Linear(2, 1)
        self.groups = []

    def trainable_module(self):
        return self.module

    def forward_loss(self, batch, ctx=None):
        return StepOutput(
            loss=self.module(batch.tensors["x"]).square().mean(), metrics={}
        )

    def after_step(self, *, process_group=None):
        self.groups.append(process_group)
        return {"moe/step_balance": torch.tensor(1.5)}


class _PlainStrategy(_Strategy):
    after_step = DraftTrainStrategy.after_step


class TrainerAfterStepTest(unittest.TestCase):
    def _batch(self):
        return TrainBatch(
            sample_ids=["a"], strategy="stub", tensors={"x": torch.ones(1, 2)}
        )

    def test_after_step_runs_once_per_optimizer_step_and_its_metrics_join(self):
        strategy = _Strategy()
        core = TrainerCore(strategy, _Backend(), accumulation_steps=2)
        first = core.train_step(self._batch())
        self.assertFalse(first.optimizer_stepped)
        self.assertNotIn("moe/step_balance", first.metrics)
        self.assertEqual(strategy.groups, [])
        second = core.train_step(self._batch())
        self.assertTrue(second.optimizer_stepped)
        self.assertEqual(second.metrics["moe/step_balance"], 1.5)
        self.assertEqual(strategy.groups, [None])
        # The default after_step reports nothing.
        core = TrainerCore(_PlainStrategy(), _Backend(), accumulation_steps=1)
        result = core.train_step(self._batch())
        self.assertTrue(result.optimizer_stepped)
        self.assertEqual(set(result.metrics), {"loss"})


if __name__ == "__main__":
    unittest.main()
