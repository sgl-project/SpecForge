"""The step frame's window order and what a runner refuses."""

import unittest
from dataclasses import replace

import torch
from torch import nn

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import IncompleteWindow, PendingStep, StepMetrics
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from specforge.training.strategies.base import StepContext, StepOutput


def batch(value):
    return TrainBatch(
        sample_ids=[str(value)],
        strategy="test",
        tensors={"x": torch.tensor([float(value)])},
    )


class RecordingRunner:
    """Records the calls the frame makes; pools by doubling, as two ranks would."""

    def __init__(self, replay_statistics=()):
        self.replay_statistics = replay_statistics
        self.calls = []

    def forward_backward(self, batch, ctx, *, is_boundary):
        self.calls.append(
            (
                "forward_backward",
                batch.sample_ids[0],
                is_boundary,
                {name: value.tolist() for name, value in ctx.replay_values.items()},
            )
        )
        return PendingStep("window") if is_boundary else None

    def apply_step(self, pending):
        self.calls.append(("apply_step", pending.take()))
        return StepMetrics({"loss": 0.0})

    def prepass(self, batch, ctx):
        self.calls.append(("prepass", batch.sample_ids[0], batch.tensors["x"].device))
        return {"count": batch.tensors["x"]}

    def reduce_sum(self, values):
        self.calls.append(("reduce_sum", values.tolist()))
        return values.mul_(2)


class StepMetricsTest(unittest.TestCase):
    def test_gradient_norm_is_the_optimizers_only(self):
        reported = StepMetrics({"loss": 1.0, "grad_norm": 42.0})
        self.assertIsNone(reported.grad_norm)
        measured = StepMetrics({"loss": 1.0, "grad_norm": 0.5}, has_grad_norm=True)
        self.assertEqual(measured.grad_norm, 0.5)

    def test_learning_rate_is_converted_on_access(self):
        self.assertIsNone(StepMetrics({}).learning_rate)
        rate = StepMetrics({}, learning_rate=torch.tensor(0.25)).learning_rate
        self.assertIsInstance(rate, float)
        self.assertEqual(rate, 0.25)


class RunWindowTest(unittest.TestCase):
    def test_hands_each_micro_batch_over_as_it_arrives(self):
        runner = RecordingRunner()

        def stream():
            for value in (1, 2, 3):
                # Nothing is collected first: the earlier batches already ran.
                self.assertEqual(len(runner.calls), value - 1)
                yield batch(value)

        run_window(runner, stream(), StepContext(accumulation_steps=3))
        self.assertEqual(
            runner.calls,
            [
                ("forward_backward", "1", False, {}),
                ("forward_backward", "2", False, {}),
                ("forward_backward", "3", True, {}),
                ("apply_step", "window"),
            ],
        )

    def test_replay_statistic_is_pooled_before_the_gradient_pass(self):
        runner = RecordingRunner(replay_statistics=("count",))
        first = batch(1)
        run_window(runner, [first, batch(2)], StepContext(accumulation_steps=2))
        cpu = torch.device("cpu")
        self.assertEqual(
            runner.calls,
            [
                ("prepass", "1", cpu),
                ("prepass", "2", cpu),
                ("reduce_sum", [3.0]),
                ("forward_backward", "1", False, {"count": [6.0]}),
                ("forward_backward", "2", True, {"count": [6.0]}),
                ("apply_step", "window"),
            ],
        )
        # Pooling is in place and must not reach the program's own tensor.
        self.assertEqual(first.tensors["x"].tolist(), [1.0])

    def test_short_window_raises_without_applying_a_step(self):
        for statistics in ((), ("count",)):
            with self.subTest(replay_statistics=statistics):
                runner = RecordingRunner(replay_statistics=statistics)
                with self.assertRaisesRegex(IncompleteWindow, "received 1 of 2"):
                    run_window(runner, [batch(1)], StepContext(accumulation_steps=2))
                self.assertNotIn("apply_step", [call[0] for call in runner.calls])

    def test_short_window_is_not_applied_whatever_the_runner_returned(self):
        class EagerRunner(RecordingRunner):
            def forward_backward(self, batch, ctx, *, is_boundary):
                super().forward_backward(batch, ctx, is_boundary=is_boundary)
                return PendingStep("early")

        runner = EagerRunner()
        with self.assertRaisesRegex(IncompleteWindow, "received 1 of 2"):
            run_window(runner, [batch(1)], StepContext(accumulation_steps=2))
        self.assertNotIn("apply_step", [call[0] for call in runner.calls])

    def test_boundary_without_a_step_is_refused(self):
        class ForgetfulRunner(RecordingRunner):
            def forward_backward(self, batch, ctx, *, is_boundary):
                super().forward_backward(batch, ctx, is_boundary=is_boundary)

        for statistics in ((), ("count",)):
            with self.subTest(replay_statistics=statistics):
                runner = ForgetfulRunner(replay_statistics=statistics)
                with self.assertRaisesRegex(RuntimeError, "no step for the window"):
                    run_window(runner, [batch(1)], StepContext())
                self.assertNotIn("apply_step", [call[0] for call in runner.calls])

    def test_window_longer_than_the_accumulation_is_refused(self):
        for statistics in ((), ("count",)):
            with self.subTest(replay_statistics=statistics):
                runner = RecordingRunner(replay_statistics=statistics)
                with self.assertRaisesRegex(ValueError, "at most 2 micro-batches"):
                    run_window(
                        runner,
                        [batch(1), batch(2), batch(3)],
                        StepContext(accumulation_steps=2),
                    )
                self.assertNotIn("apply_step", [call[0] for call in runner.calls])


class _Strategy:
    replay_statistics = ()

    def __init__(self):
        self.model = nn.Linear(1, 1, bias=False)

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        numerator = self.model(batch.tensors["x"]).sum()
        return StepOutput(
            loss=numerator, metrics={}, loss_terms=(numerator, torch.ones(()))
        )


class _ReplayStrategy(_Strategy):
    replay_statistics = ("count",)

    def prepass(self, batch, ctx):
        return replace(
            self.forward_loss(batch, ctx), statistics={"count": torch.ones(1)}
        )


def runner_over(strategy):
    backend = FSDPTrainingBackend(
        ParallelConfig(),
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    backend.prepare_model(strategy.model, wrap=False)
    return DataParallelRunner(strategy, backend)


class RunnerRefusalTest(unittest.TestCase):
    def test_step_is_applied_once(self):
        runner = runner_over(_Strategy())
        pending = runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        runner.apply_step(pending)
        with self.assertRaisesRegex(RuntimeError, "another window"):
            runner.apply_step(pending)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            pending.take()

    def test_next_window_waits_for_the_pending_step(self):
        runner = runner_over(_Strategy())
        runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        with self.assertRaisesRegex(RuntimeError, "has not been applied"):
            runner.forward_backward(batch(2), StepContext(), is_boundary=True)

    def test_boundary_has_to_close_the_window(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_Strategy())
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            runner.forward_backward(batch(1), ctx, is_boundary=True)
        runner.forward_backward(batch(1), ctx, is_boundary=False)
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            runner.forward_backward(batch(2), ctx, is_boundary=False)

    def test_window_keeps_the_length_it_started_with(self):
        runner = runner_over(_Strategy())
        runner.forward_backward(
            batch(1), StepContext(accumulation_steps=3), is_boundary=False
        )
        with self.assertRaisesRegex(RuntimeError, "accumulates 3 micro-batches"):
            runner.forward_backward(
                batch(2), StepContext(accumulation_steps=2), is_boundary=True
            )

    def test_replayed_window_keeps_the_length_of_its_first_prepass(self):
        runner = runner_over(_ReplayStrategy())
        runner.prepass(batch(1), StepContext(accumulation_steps=3))
        with self.assertRaisesRegex(RuntimeError, "accumulates 3 micro-batches"):
            runner.prepass(batch(2), StepContext(accumulation_steps=2))

    def test_malformed_metric_raises_before_the_update(self):
        class Strategy(_Strategy):
            def forward_loss(self, batch, ctx=None):
                output = super().forward_loss(batch, ctx)
                count = torch.ones(())
                return replace(
                    output,
                    ratio_metrics={"seen": (count, count)},
                    sum_metrics={"seen": count},
                )

        strategy = Strategy()
        runner = runner_over(strategy)
        before = strategy.model.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            run_window(runner, [batch(1), batch(2)], StepContext(accumulation_steps=2))
        # Found at the first micro-batch: no backward, no optimizer step.
        self.assertIsNone(strategy.model.weight.grad)
        torch.testing.assert_close(strategy.model.weight, before, rtol=0, atol=0)

    def test_failed_step_blocks_the_next_window(self):
        class Strategy(_Strategy):
            def forward_loss(self, batch, ctx=None):
                output = super().forward_loss(batch, ctx)
                return replace(
                    output, loss_terms=(output.loss_terms[0], torch.zeros(()))
                )

        runner = runner_over(Strategy())
        pending = runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runner.apply_step(pending)
        # The rejected window's gradients are still on the model.
        with self.assertRaisesRegex(RuntimeError, "has not been applied"):
            runner.forward_backward(batch(2), StepContext(), is_boundary=True)

    def test_failed_prepass_leaves_nothing_to_pair(self):
        class Strategy(_ReplayStrategy):
            fail = True

            def prepass(self, batch, ctx):
                if self.fail:
                    raise ValueError("missing feature")
                return super().prepass(batch, ctx)

        strategy = Strategy()
        runner = runner_over(strategy)
        with self.assertRaisesRegex(ValueError, "missing feature"):
            run_window(runner, [batch(1)], StepContext())
        strategy.fail = False
        self.assertIsNotNone(run_window(runner, [batch(1)], StepContext()).loss)

    def test_runner_left_inside_a_window_cannot_start_another(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_Strategy())
        with self.assertRaises(IncompleteWindow):
            run_window(runner, [batch(1)], ctx)
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            run_window(runner, [batch(2), batch(3)], ctx)

    def test_declared_replay_statistic_cannot_be_skipped(self):
        runner = runner_over(_ReplayStrategy())
        with self.assertRaisesRegex(RuntimeError, "were not pooled"):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)

    def test_replay_pairs_every_prepass_of_the_window(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_ReplayStrategy())
        with self.assertRaises(IncompleteWindow):
            run_window(runner, [batch(1)], ctx)
        # The abandoned window's prepass would shift every later pairing.
        with self.assertRaisesRegex(RuntimeError, "prepasses to pair"):
            run_window(runner, [batch(2), batch(3)], ctx)

    def test_replay_values_need_a_declaration(self):
        runner = runner_over(_Strategy())
        ctx = StepContext(replay_values={"count": torch.ones(1)})
        with self.assertRaisesRegex(RuntimeError, "declares no replay statistic"):
            runner.forward_backward(batch(1), ctx, is_boundary=True)


if __name__ == "__main__":
    unittest.main()
