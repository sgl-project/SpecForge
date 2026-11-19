"""The step frame's window order and what a runner refuses."""

import unittest
from dataclasses import replace
from unittest import mock

import torch
from torch import nn

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import (
    IncompleteWindow,
    PendingStep,
    RankFailure,
    StepContext,
    StepMetrics,
)
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from tests.test_runtime._fixtures import FakeProgram, fake_spec


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

    def test_step_before_the_boundary_is_refused(self):
        class EagerRunner(RecordingRunner):
            def forward_backward(self, batch, ctx, *, is_boundary):
                super().forward_backward(batch, ctx, is_boundary=is_boundary)
                return PendingStep("early")

        # A runner that closed after one micro-batch must not have its step
        # applied, whether or not the rest of the window arrives.
        for statistics in ((), ("count",)):
            for batches in ([batch(1)], [batch(1), batch(2)]):
                with self.subTest(replay_statistics=statistics, window=len(batches)):
                    runner = EagerRunner(replay_statistics=statistics)
                    with self.assertRaisesRegex(
                        RuntimeError, "received 1 of 2|a step for micro-batch 1 of 2"
                    ):
                        run_window(runner, batches, StepContext(accumulation_steps=2))
                    self.assertNotIn("apply_step", [call[0] for call in runner.calls])

    def test_boundary_without_a_step_is_refused(self):
        class ForgetfulRunner(RecordingRunner):
            def forward_backward(self, batch, ctx, *, is_boundary):
                super().forward_backward(batch, ctx, is_boundary=is_boundary)

        for statistics in ((), ("count",)):
            with self.subTest(replay_statistics=statistics):
                runner = ForgetfulRunner(replay_statistics=statistics)
                with self.assertRaisesRegex(RuntimeError, "no step for micro-batch 1"):
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


class _Program(FakeProgram):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1, bias=False)

    def forward(
        self, tensors, *, step, total_steps, collect_detailed_metrics, **replay
    ):
        return ObjectiveResult(self.linear(tensors["x"]).sum(), torch.ones(()))


class _ReplayProgram(_Program):
    """Declares the ``count`` statistic; a prepass reports one per micro-batch."""

    def forward(self, tensors, *, prepass=False, replay_values=None, **context):
        result = super().forward(tensors, **context)
        if prepass:
            result.statistics["count"] = torch.ones(1)
        return result


def runner_over(program, **declarations):
    backend = FSDPTrainingBackend(
        ParallelConfig(),
        optimizer_factory=lambda module: torch.optim.SGD(module.parameters(), lr=0.1),
    )
    backend.prepare_model(program, wrap=False)
    if isinstance(program, _ReplayProgram):
        declarations.setdefault("replay_statistics", ("count",))
    return DataParallelRunner(program, backend, fake_spec("x", **declarations))


class _EvaluatedProgram(_Program):
    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, tensors, *, collect_detailed_metrics, **context):
        self.seen.append(
            (self.training, torch.is_grad_enabled(), collect_detailed_metrics)
        )
        return super().forward(
            tensors, collect_detailed_metrics=collect_detailed_metrics, **context
        )


class RunnerModeTest(unittest.TestCase):
    def test_evaluation_runs_in_its_own_mode_and_restores_the_callers(self):
        for was_training in (True, False):
            for diagnostics in (False, True):
                with self.subTest(was_training=was_training, diagnostics=diagnostics):
                    program = _EvaluatedProgram()
                    program.train(was_training)
                    runner = runner_over(program, evaluation_diagnostics=diagnostics)
                    # What the program declares decides, not the caller's context.
                    ctx = StepContext(collect_detailed_metrics=not diagnostics)
                    with runner.evaluation():
                        output = runner.evaluate(batch(1), ctx)
                        # Evaluation mode holds between the batches of a pass.
                        self.assertFalse(program.training)
                        runner.evaluate(batch(2), ctx)
                    self.assertEqual(program.seen, [(False, False, diagnostics)] * 2)
                    self.assertFalse(output.loss.requires_grad)
                    self.assertEqual(program.training, was_training)

    def test_pass_without_batches_leaves_the_mode_as_it_was(self):
        program = _EvaluatedProgram()
        runner = runner_over(program)
        with runner.evaluation():
            self.assertFalse(program.training)
        self.assertTrue(program.training)

    def test_failed_evaluation_restores_the_mode(self):
        class Program(_EvaluatedProgram):
            def forward(self, tensors, **context):
                raise ValueError("bad batch")

        program = Program()
        runner = runner_over(program)
        with self.assertRaisesRegex(ValueError, "bad batch"):
            with runner.evaluation():
                runner.evaluate(batch(1), StepContext())
        self.assertTrue(program.training)

    def test_evaluate_belongs_to_a_pass(self):
        runner = runner_over(_EvaluatedProgram())
        with self.assertRaisesRegex(RuntimeError, "inside an evaluation"):
            runner.evaluate(batch(1), StepContext())
        with runner.evaluation():
            with self.assertRaisesRegex(RuntimeError, "inside an evaluation pass"):
                runner.forward_backward(batch(1), StepContext(), is_boundary=True)

    def test_training_calls_put_the_whole_model_in_training_mode(self):
        class Program(_EvaluatedProgram):
            """A training-mode module around a draft that was left in evaluation mode."""

            def __init__(self):
                super().__init__()
                self.draft = nn.Linear(1, 1)
                self.draft.eval()

        program = Program()
        self.assertTrue(program.training)
        runner = runner_over(program)
        run_window(runner, [batch(1)], StepContext())
        self.assertTrue(program.draft.training)
        # After an evaluation pass as well, on a rank that had no batch to restore it.
        with runner.evaluation():
            pass
        program.draft.eval()
        run_window(runner, [batch(2)], StepContext())
        self.assertTrue(program.draft.training)


class RunnerReductionTest(unittest.TestCase):
    def test_sum_and_maximum_are_taken_over_the_ranks(self):
        runner = runner_over(_Program())
        calls = []

        def all_reduce(values, *, op, group):
            calls.append((op, group))
            values.mul_(2)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=all_reduce),
        ):
            total = runner.reduce_sum(torch.tensor([1.0]))
            widest = runner.reduce_max(torch.tensor([3]))
        self.assertEqual(total.tolist(), [2.0])
        self.assertEqual(widest.tolist(), [6])
        self.assertEqual(
            calls,
            [
                (torch.distributed.ReduceOp.SUM, None),
                (torch.distributed.ReduceOp.MAX, None),
            ],
        )

    def test_one_rank_reduces_nothing(self):
        runner = runner_over(_Program())
        values = torch.tensor([3])
        with mock.patch("torch.distributed.all_reduce") as reduce:
            self.assertIs(runner.reduce_max(values), values)
        reduce.assert_not_called()


class _TermsProgram(_Program):
    """Two terms with denominators of their own, as the ``microbatch`` reduction takes them."""

    def forward(self, tensors, **context):
        value = self.linear(tensors["x"]).sum()
        result = ObjectiveResult(value.detach(), torch.ones(()))
        if self.training:
            result.terms = (
                (1.0, value, torch.tensor(2.0)),
                (0.5, value * value, torch.tensor(0.0)),
            )
        return result


class RunnerMicrobatchReductionTest(unittest.TestCase):
    def test_denominators_are_pooled_between_forward_and_backward(self):
        program = _TermsProgram()
        with torch.no_grad():
            program.linear.weight.fill_(3.0)
        runner = runner_over(program, reduction="microbatch")
        pooled = []

        def all_reduce(values, *, op, group):
            pooled.append((values.clone(), program.linear.weight.grad))
            if len(pooled) == 1:
                # The other rank holds denominators of 4 and 0.
                values.add_(4.0)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=all_reduce),
        ):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        # Each term's denominator is reduced before any gradient; the
        # window's own denominator follows the backward.
        self.assertEqual(len(pooled), 3)
        self.assertEqual([values.item() for values, _ in pooled[:2]], [2.0, 0.0])
        self.assertEqual([gradient for _, gradient in pooled[:2]], [None, None])
        self.assertIsNotNone(pooled[2][1])
        # d/dw of 2 * (w / 6 + 0.5 * w**2 / 1) at w = 3, x = 1: the world size
        # undoes the wrapper's average, a pooled zero divides by one.
        torch.testing.assert_close(
            program.linear.weight.grad, torch.tensor([[2 * (1 / 6 + 3.0)]])
        )

    def test_denominators_are_pooled_in_their_own_dtype(self):
        """A sum is rounded in the dtype it is taken in: 256 + 1 is 256 in bfloat16."""

        class Program(_Program):
            def forward(self, tensors, **context):
                value = self.linear(tensors["x"]).sum()
                result = ObjectiveResult(value.detach(), torch.ones(()))
                result.terms = (
                    (1.0, value, torch.tensor(256.0, dtype=torch.bfloat16)),
                    (1.0, value, torch.tensor(256.0)),
                )
                return result

        program = Program()
        with torch.no_grad():
            program.linear.weight.fill_(1.0)
        runner = runner_over(program, reduction="microbatch")
        pooled = []

        def all_reduce(values, *, op, group):
            if len(pooled) < 2:
                # The other rank holds one of each.
                values.add_(torch.ones_like(values))
            pooled.append(values.clone())

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=all_reduce),
        ):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        self.assertEqual(
            [(values.dtype, values.item()) for values in pooled[:2]],
            [(torch.bfloat16, 256.0), (torch.float32, 257.0)],
        )
        torch.testing.assert_close(
            program.linear.weight.grad, torch.tensor([[2 / 256 + 2 / 257]])
        )
        # The window counts its micro-batches in a dtype that can: a bfloat16
        # one stops at 256.
        _, counted = runner._loss_terms(program({"x": torch.ones(1)}))
        self.assertEqual(counted.dtype, torch.float32)

    def test_denominator_is_taken_as_the_program_gave_it(self):
        """Its shape is kept and it is never differentiated."""

        class Program(_Program):
            def forward(self, tensors, **context):
                weight = self.linear.weight.sum()
                result = ObjectiveResult(weight.detach(), torch.ones(()))
                # A one-element denominator, and one that carries a graph.
                result.terms = (
                    (1.0, weight, torch.tensor([4.0])),
                    (1.0, weight, weight * 1.0),
                )
                return result

        program = Program()
        with torch.no_grad():
            program.linear.weight.fill_(2.0)
        runner = runner_over(program, reduction="microbatch")
        runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        # d/dw of w / 4 + w / 2 with the second denominator held at 2.
        torch.testing.assert_close(program.linear.weight.grad, torch.tensor([[0.75]]))

    def test_one_rank_normalizes_by_its_own_denominators(self):
        program = _TermsProgram()
        with torch.no_grad():
            program.linear.weight.fill_(3.0)
        runner = runner_over(program, reduction="microbatch")
        with mock.patch("torch.distributed.all_reduce") as reduce:
            metrics = run_window(runner, [batch(1)], StepContext())
        reduce.assert_not_called()
        # 3 / 2 + 0.5 * 9 / 1, reported over a denominator of one.
        self.assertEqual(metrics.loss, 6.0)

    def test_window_averages_the_micro_batch_means(self):
        program = _TermsProgram()
        with torch.no_grad():
            program.linear.weight.fill_(1.0)
        runner = runner_over(program, reduction="microbatch")
        metrics = run_window(
            runner, [batch(2), batch(4)], StepContext(accumulation_steps=2)
        )
        # (2 / 2 + 0.5 * 4) and (4 / 2 + 0.5 * 16), as means and not pooled.
        self.assertEqual(metrics.loss, (3.0 + 10.0) / 2)

    def test_declaration_and_result_have_to_agree(self):
        runner = runner_over(_Program(), reduction="microbatch")
        with self.assertRaisesRegex(ValueError, "returned 0 loss terms"):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        runner = runner_over(_TermsProgram())
        with self.assertRaisesRegex(ValueError, "'window' reduction and returned 2"):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)

    def test_evaluation_returns_what_the_program_computed(self):
        program = _TermsProgram()
        runner = runner_over(program, reduction="microbatch")
        with (
            mock.patch.object(runner, "reduce_sum") as reduce,
            runner.evaluation(),
        ):
            result = runner.evaluate(batch(1), StepContext())
        # Not by a group of one either: the evaluator pools the pass.
        reduce.assert_not_called()
        self.assertEqual(result.terms, ())
        self.assertEqual(result.denominator.item(), 1.0)


class RunnerAgreementTest(unittest.TestCase):
    def gathered(self, others):
        """Two ranks: this one's outcome and the given one for the other."""
        return (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, mine, group: out.__setitem__(
                    slice(None), [mine, others]
                ),
            ),
        )

    def agree(self, others, error=None, value=None):
        runner = runner_over(_Program())
        patches = self.gathered(others)
        with patches[0], patches[1], patches[2], patches[3]:
            runner.agree(error, value=value)

    def test_equal_outcomes_pass(self):
        self.agree((None, {"0:0": "reference"}), value={"0:0": "reference"})

    def test_another_ranks_error_raises_here_too(self):
        with self.assertRaisesRegex(RuntimeError, "rank 1: disk full"):
            self.agree(("disk full", None))

    def test_own_error_is_reported_with_its_rank(self):
        with self.assertRaisesRegex(RuntimeError, "rank 0: disk full"):
            self.agree((None, None), error="disk full")

    def test_differing_values_raise(self):
        with self.assertRaisesRegex(RuntimeError, "ranks disagree.*token_ce"):
            self.agree((None, {"0:0": "token_ce"}), value={"0:0": "reference"})

    def test_one_process_gathers_nothing(self):
        runner = runner_over(_Program())
        with (
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch("torch.distributed.all_gather_object") as gather,
        ):
            runner.agree(None, value={"0:0": "reference"})
            with self.assertRaisesRegex(RuntimeError, "rank 0: disk full"):
                runner.agree("disk full")
        gather.assert_not_called()

    def test_group_of_one_agrees_with_itself(self):
        runner = runner_over(_Program())
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=1),
            mock.patch(
                "torch.distributed.all_gather_object",
                side_effect=lambda out, mine, group: out.__setitem__(0, mine),
            ) as gather,
        ):
            # One outcome has nothing to differ from, whatever ``==`` says of it.
            runner.agree(None, value=float("nan"))
            with self.assertRaisesRegex(RuntimeError, "rank 0: disk full"):
                runner.agree("disk full")
        self.assertEqual(gather.call_count, 2)

    def test_empty_error_is_an_error(self):
        with self.assertRaisesRegex(RuntimeError, "rank 1: $"):
            self.agree(("", None))


class RunnerCheckpointTest(unittest.TestCase):
    def test_state_is_the_backends_split_by_who_saves_it(self):
        class Program(_Program):
            def draft_state(self, state):
                return {name.upper(): tensor for name, tensor in state.items()}

            def kernel_record(self):
                return {"0:0": "reference"}

        runner = runner_over(Program())
        run_window(runner, [batch(1)], StepContext())
        state = runner.checkpoint_state()
        self.assertEqual(list(state.draft), ["LINEAR.WEIGHT"])
        # One process shards nothing: the optimizer state is this rank's own.
        self.assertIsNone(state.replicated)
        self.assertEqual(sorted(state.rank), ["optimizer", "rng"])
        self.assertIsNotNone(state.rank["optimizer"])
        self.assertEqual(state.kernels, {"0:0": "reference"})

    def test_resume_restores_the_optimizer_and_the_generator(self):
        runner = runner_over(_Program())
        # State a fresh optimizer does not have.
        runner.backend.optimizer.param_groups[0]["lr"] = 0.025
        saved = runner.checkpoint_state().rank
        expected = torch.rand(3)
        resumed = runner_over(_Program())
        self.assertEqual(resumed.backend.optimizer.param_groups[0]["lr"], 0.1)
        resumed.load_resume(saved)
        self.assertEqual(resumed.backend.optimizer.param_groups[0]["lr"], 0.025)
        torch.testing.assert_close(torch.rand(3), expected, rtol=0, atol=0)

    def test_only_the_first_rank_selects_the_draft_weights(self):
        """The others hold nothing of the model, or under sharding a part of it."""

        class Program(_Program):
            def draft_state(self, state):
                return {"w": state["linear.weight"]}

        runner = runner_over(Program())
        self.assertEqual(list(runner.checkpoint_state().draft), ["w"])
        partial = {**runner.backend.state_dict(), "model": {"frozen_table": 0}}
        for rank, selected in ((0, True), (1, False)):
            with (
                self.subTest(rank=rank),
                mock.patch("torch.distributed.is_available", return_value=True),
                mock.patch("torch.distributed.is_initialized", return_value=True),
                mock.patch("torch.distributed.get_rank", return_value=rank),
                mock.patch.object(runner.backend, "state_dict", lambda: partial),
            ):
                if selected:
                    # The writer's selection is not skipped, whatever it holds.
                    with self.assertRaises(RankFailure) as raised:
                        runner.checkpoint_state()
                    self.assertIsInstance(raised.exception.__cause__, KeyError)
                else:
                    self.assertEqual(runner.checkpoint_state().draft, {})

    def test_failure_after_the_gather_is_this_ranks_alone(self):
        runner = runner_over(_Program())
        optimizer = runner.backend.optimizer
        with mock.patch.object(
            optimizer, "state_dict", side_effect=MemoryError("masters")
        ):
            with self.assertRaisesRegex(RankFailure, "MemoryError: masters") as raised:
                runner.checkpoint_state()
        self.assertIsInstance(raised.exception.__cause__, MemoryError)
        # Inside the gather the ranks are apart already: nothing to agree on.
        with mock.patch.object(
            runner.backend, "_module_state_dict", side_effect=MemoryError("gather")
        ):
            with self.assertRaisesRegex(MemoryError, "gather"):
                runner.checkpoint_state()
        with mock.patch.object(
            optimizer, "load_state_dict", side_effect=ValueError("shape")
        ):
            with self.assertRaisesRegex(RankFailure, "ValueError: shape"):
                runner.load_resume({"optimizer": {}, "rng": None})


class RunnerRefusalTest(unittest.TestCase):
    def test_step_is_applied_once(self):
        runner = runner_over(_Program())
        pending = runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        runner.apply_step(pending)
        with self.assertRaisesRegex(RuntimeError, "another window"):
            runner.apply_step(pending)
        with self.assertRaisesRegex(RuntimeError, "only once"):
            pending.take()

    def test_next_window_waits_for_the_pending_step(self):
        runner = runner_over(_Program())
        runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        with self.assertRaisesRegex(RuntimeError, "has not been applied"):
            runner.forward_backward(batch(2), StepContext(), is_boundary=True)

    def test_boundary_has_to_close_the_window(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_Program())
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            runner.forward_backward(batch(1), ctx, is_boundary=True)
        runner.forward_backward(batch(1), ctx, is_boundary=False)
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            runner.forward_backward(batch(2), ctx, is_boundary=False)

    def test_window_keeps_the_length_it_started_with(self):
        runner = runner_over(_Program())
        runner.forward_backward(
            batch(1), StepContext(accumulation_steps=3), is_boundary=False
        )
        with self.assertRaisesRegex(RuntimeError, "accumulates 3 micro-batches"):
            runner.forward_backward(
                batch(2), StepContext(accumulation_steps=2), is_boundary=True
            )

    def test_replayed_window_keeps_the_length_of_its_first_prepass(self):
        runner = runner_over(_ReplayProgram())
        runner.prepass(batch(1), StepContext(accumulation_steps=3))
        with self.assertRaisesRegex(RuntimeError, "accumulates 3 micro-batches"):
            runner.prepass(batch(2), StepContext(accumulation_steps=2))

    def test_malformed_metric_raises_before_the_update(self):
        class Program(_Program):
            def forward(self, tensors, **context):
                result = super().forward(tensors, **context)
                count = torch.ones(())
                return replace(
                    result, ratios={"seen": (count, count)}, sums={"seen": count}
                )

        program = Program()
        runner = runner_over(program)
        before = program.linear.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            run_window(runner, [batch(1), batch(2)], StepContext(accumulation_steps=2))
        # Found at the first micro-batch: no backward, no optimizer step.
        self.assertIsNone(program.linear.weight.grad)
        torch.testing.assert_close(program.linear.weight, before, rtol=0, atol=0)

    def test_metric_name_reused_across_micro_batches_raises_before_the_update(self):
        class Program(_Program):
            calls = 0

            def forward(self, tensors, **context):
                result = super().forward(tensors, **context)
                count = torch.ones(())
                self.calls += 1
                if self.calls == 1:
                    return replace(result, ratios={"seen": (count, count)})
                return replace(result, sums={"seen": count})

        program = Program()
        runner = runner_over(program)
        before = program.linear.weight.detach().clone()
        with self.assertRaisesRegex(ValueError, "must be distinct"):
            run_window(runner, [batch(1), batch(2)], StepContext(accumulation_steps=2))
        torch.testing.assert_close(program.linear.weight, before, rtol=0, atol=0)

    def test_failed_backward_ends_the_runner(self):
        class Program(_Program):
            detach = True

            def forward(self, tensors, **context):
                result = super().forward(tensors, **context)
                if not self.detach:
                    return result
                return replace(result, numerator=result.numerator.detach())

        program = Program()
        runner = runner_over(program)
        with self.assertRaisesRegex(RuntimeError, "does not require grad"):
            run_window(runner, [batch(1)], StepContext())
        # Its denominator is already in the window's totals.
        program.detach = False
        with self.assertRaisesRegex(RuntimeError, "runner cannot continue"):
            run_window(runner, [batch(2)], StepContext())

    def test_failed_step_blocks_the_next_window(self):
        class Program(_Program):
            def forward(self, tensors, **context):
                return replace(
                    super().forward(tensors, **context), denominator=torch.zeros(())
                )

        runner = runner_over(Program())
        pending = runner.forward_backward(batch(1), StepContext(), is_boundary=True)
        with self.assertRaisesRegex(ValueError, "finite and positive"):
            runner.apply_step(pending)
        # The rejected window's gradients are still on the model.
        with self.assertRaisesRegex(RuntimeError, "has not been applied"):
            runner.forward_backward(batch(2), StepContext(), is_boundary=True)

    def test_failed_prepass_leaves_nothing_to_pair(self):
        class Program(_ReplayProgram):
            fail = True

            def forward(self, tensors, *, prepass=False, **context):
                if prepass and self.fail:
                    raise ValueError("missing feature")
                return super().forward(tensors, prepass=prepass, **context)

        program = Program()
        runner = runner_over(program)
        with self.assertRaisesRegex(ValueError, "missing feature"):
            run_window(runner, [batch(1)], StepContext())
        program.fail = False
        self.assertIsNotNone(run_window(runner, [batch(1)], StepContext()).loss)

    def test_runner_left_inside_a_window_cannot_start_another(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_Program())
        with self.assertRaises(IncompleteWindow):
            run_window(runner, [batch(1)], ctx)
        with self.assertRaisesRegex(RuntimeError, "does not close it"):
            run_window(runner, [batch(2), batch(3)], ctx)

    def test_declared_replay_statistic_cannot_be_skipped(self):
        runner = runner_over(_ReplayProgram())
        with self.assertRaisesRegex(RuntimeError, "were not pooled"):
            runner.forward_backward(batch(1), StepContext(), is_boundary=True)

    def test_replay_pairs_every_prepass_of_the_window(self):
        ctx = StepContext(accumulation_steps=2)
        runner = runner_over(_ReplayProgram())
        with self.assertRaises(IncompleteWindow):
            run_window(runner, [batch(1)], ctx)
        # The abandoned window's prepass would shift every later pairing.
        with self.assertRaisesRegex(RuntimeError, "prepasses to pair"):
            run_window(runner, [batch(2), batch(3)], ctx)

    def test_prepass_needs_a_declaration(self):
        runner = runner_over(_Program())
        with self.assertRaisesRegex(RuntimeError, "declares no replay statistic"):
            runner.prepass(batch(1), StepContext())
        # Nothing was kept for a replay: the next forward draws as it would.
        self.assertFalse(runner._prepass_rng)

    def test_replay_values_need_a_declaration(self):
        runner = runner_over(_Program())
        ctx = StepContext(replay_values={"count": torch.ones(1)})
        with self.assertRaisesRegex(RuntimeError, "declares no replay statistic"):
            runner.forward_backward(batch(1), ctx, is_boundary=True)


if __name__ == "__main__":
    unittest.main()
