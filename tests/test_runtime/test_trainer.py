# coding=utf-8
"""Step runner grad-accum + TrainerController fit/eval/checkpoint (CPU)."""

import gc
import inspect
import json
import os
import tempfile
import unittest
import weakref
from dataclasses import replace
from unittest import mock

import torch
import torch.nn as nn

from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.runtime.contracts import TrainBatch
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.training.backend import TrainingBackend
from specforge.training.checkpoint import CheckpointManager
from specforge.training.contracts import (
    IncompleteWindow,
    RankFailure,
    StepContext,
    StepMetrics,
    materialize_metrics,
)
from specforge.training.controller import Checkpoint, TrainerController
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner, _reduce_ratio_metrics
from tests.test_runtime._fixtures import FakeProgram, fake_spec, train_window


class TinyProgram(FakeProgram):
    """One weight: the loss is ``|w * sum(x)|`` and the accuracy one half."""

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))
        # The schedule keywords of every forward: what fit() threads in.
        self.contexts = []

    def forward(
        self, tensors, *, step, total_steps, collect_detailed_metrics, **replay
    ):
        self.contexts.append(
            {
                "step": step,
                "total_steps": total_steps,
                "collect_detailed_metrics": collect_detailed_metrics,
            }
        )
        loss = (self.w * tensors["x"].sum()).abs()
        one = torch.ones(())
        return ObjectiveResult(loss, one, ratios={"acc": (torch.tensor(0.5), one)})


class RecordingProgram(TinyProgram):
    """Records the ``index`` of every batch it runs."""

    def __init__(self):
        super().__init__()
        self.seen = []

    def forward(self, tensors, **context):
        self.seen.append(int(tensors["index"]))
        return super().forward(tensors, **context)


class WeightedProgram(TinyProgram):
    """The batch gives the loss numerator's coefficient and the counts."""

    def forward(self, tensors, **context):
        coefficient = tensors["x"].reshape(())
        denominator = tensors["denominator"].reshape(())
        correct = tensors["correct"].reshape(())
        numerator = self.w.reshape(()) * coefficient
        return ObjectiveResult(
            numerator,
            denominator,
            ratios={"acc": (correct, denominator)},
            sums={"correct_count": correct, "sample_count": denominator},
        )


class ReplayProgram(WeightedProgram):
    """Pools the accuracy counts over a window before its gradient pass."""

    def __init__(self):
        super().__init__()
        self.replayed_values = []

    def forward(self, tensors, *, prepass=False, replay_values=None, **context):
        result = super().forward(tensors, **context)
        if prepass:
            return replace(
                result, statistics={"acc": torch.stack(result.ratios["acc"])}
            )
        if replay_values:
            self.replayed_values.append(replay_values["acc"])
        return result


class FakeBackend(TrainingBackend):
    process_group = None
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

    def scale_gradients(self, factor):
        for parameter in self.model.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)

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


class RetryableCloseQueue(SampleRefQueue):
    loader_close_retryable = True


def _batch():
    return TrainBatch(
        sample_ids=["s"], strategy="fake", tensors={"x": torch.ones(2)}, metadata={}
    )


def _weighted_batch(coefficient, denominator, correct):
    return TrainBatch(
        sample_ids=["s"],
        strategy="fake",
        tensors={
            "x": torch.tensor(float(coefficient)),
            "denominator": torch.tensor(float(denominator)),
            "correct": torch.tensor(float(correct)),
        },
        metadata={},
    )


def _runner(program, backend, **declarations):
    """The step runner over ``program``, which requires ``x`` of a batch."""
    return DataParallelRunner(
        program, backend, fake_spec("x", name="fake", **declarations)
    )


def _controller(program, backend, **options):
    """A run loop that drives ``program`` and ``backend`` through the step runner."""
    runner = _runner(program, backend)
    return TrainerController(runner, strategy_name=runner.spec.name, **options)


class TestDataParallelRunner(unittest.TestCase):
    def test_window_order_follows_the_declared_replay_statistics(self):
        for replayed in (False, True):
            with self.subTest(replayed=replayed):
                program = ReplayProgram() if replayed else WeightedProgram()
                backend = FakeBackend(program)
                runner = _runner(
                    program,
                    backend,
                    replay_statistics=("acc",) if replayed else (),
                )
                ctx = StepContext(accumulation_steps=2)
                for window in range(3):
                    program.zero_grad(set_to_none=True)
                    progress_at_pull = []

                    def batches():
                        for batch in (
                            _weighted_batch(2, 1, 1),
                            _weighted_batch(30, 3, 1),
                        ):
                            progress_at_pull.append((backend.backwards, backend.steps))
                            yield batch

                    metrics = run_window(runner, batches(), ctx)
                    # A replayed window reads every micro-batch before its first
                    # backward; neither kind steps before its last micro-batch.
                    self.assertEqual(
                        progress_at_pull,
                        [
                            (2 * window, window),
                            (2 * window + int(not replayed), window),
                        ],
                    )
                    self.assertEqual(backend.backwards, 2 * (window + 1))
                    self.assertEqual(backend.steps, window + 1)
                    self.assertEqual(metrics.loss, 8.0)
                    self.assertEqual(metrics.materialize()["acc"], 0.5)
                    self.assertEqual(metrics.materialize()["sample_count"], 4.0)
                    torch.testing.assert_close(program.w.grad, torch.tensor([8.0]))
                self.assertEqual(backend.boundaries, [False, True] * 3)
                if replayed:
                    # Every replayed forward sees the counts of its whole window.
                    self.assertEqual(len(program.replayed_values), 6)
                    for values in program.replayed_values:
                        torch.testing.assert_close(values, torch.tensor([2.0, 4.0]))

    def test_accumulation_boundary(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        runner = _runner(program, backend)
        ctx = StepContext(accumulation_steps=2)
        # no optimizer step yet
        self.assertIsNone(runner.forward_backward(_batch(), ctx, is_boundary=False))
        self.assertEqual(backend.steps, 0)
        # step on the 2nd micro-batch
        pending = runner.forward_backward(_batch(), ctx, is_boundary=True)
        self.assertIsNotNone(pending)
        metrics = runner.apply_step(pending)
        self.assertIsNotNone(metrics.grad_norm)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(backend.backwards, 2)
        # boundary known BEFORE backward so the backend can no_sync micro-steps
        self.assertEqual(backend.boundaries, [False, True])

    def test_metrics_reduce_once_at_accumulation_boundary(self):
        program = TinyProgram()
        runner = _runner(program, FakeBackend(program))
        ctx = StepContext(accumulation_steps=2)

        def reduce_sum(tensor, **collective):
            tensor.mul_(2)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce", side_effect=reduce_sum
            ) as reduce,
        ):
            runner.forward_backward(_batch(), ctx, is_boundary=False)
            self.assertEqual(reduce.call_count, 0)
            runner.apply_step(runner.forward_backward(_batch(), ctx, is_boundary=True))

        # The loss denominator, then the pairs of the loss and the accuracy
        # in one collective.
        self.assertEqual(
            [call.args[0].numel() for call in reduce.call_args_list], [1, 4]
        )

    def test_metrics_carry_no_mode(self):
        program = TinyProgram()
        runner = _runner(program, FakeBackend(program))
        metrics = train_window(runner, [_batch()])
        self.assertNotIn("mode", metrics.materialize())

    def test_global_loss_normalization_matches_combined_batch(self):
        split_program = WeightedProgram()
        split_runner = _runner(split_program, FakeBackend(split_program))
        split_result = train_window(
            split_runner, [_weighted_batch(2, 1, 1), _weighted_batch(30, 3, 1)]
        )

        combined_program = WeightedProgram()
        combined_runner = _runner(combined_program, FakeBackend(combined_program))
        combined_result = train_window(combined_runner, [_weighted_batch(32, 4, 2)])

        torch.testing.assert_close(split_program.w.grad, combined_program.w.grad)
        self.assertEqual(split_program.w.grad.item(), 8.0)
        self.assertEqual(split_result.loss, combined_result.loss)
        self.assertEqual(split_result.materialize()["acc"], 0.5)
        self.assertEqual(combined_result.materialize()["acc"], 0.5)

    def test_global_loss_normalization_compensates_rank_averaging(self):
        program = WeightedProgram()
        backend = FakeBackend(program)
        backend.process_group = "dp"
        runner = _runner(program, backend)

        def add_remote_denominator(tensor, *, group, op=None):
            self.assertEqual(group, "dp")
            # The loss denominator is the scalar collective; the packed metrics
            # follow it.
            if tensor.dim() == 0:
                self.assertEqual(op, torch.distributed.ReduceOp.SUM)
                tensor.add_(7.0)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce",
                side_effect=add_remote_denominator,
            ),
        ):
            # This rank's gradient is 9 over a local denominator of 3.
            train_window(runner, [_weighted_batch(9, 3, 0)])

        torch.testing.assert_close(program.w.grad, torch.tensor([1.8]))

    def test_global_loss_normalization_rejects_invalid_denominator(self):
        for denominator in (0.0, -1.0, float("nan"), float("inf")):
            with self.subTest(denominator=denominator):
                program = WeightedProgram()
                backend = FakeBackend(program)
                runner = _runner(program, backend)
                with self.assertRaisesRegex(ValueError, "finite and positive"):
                    train_window(runner, [_weighted_batch(2, denominator, 0)])
                self.assertEqual(backend.steps, 0)
                torch.testing.assert_close(program.w.grad, torch.tensor([2.0]))

    def test_metrics_are_materialized_lazily_and_cached(self):
        program = TinyProgram()
        runner = _runner(program, FakeBackend(program))

        with mock.patch(
            "specforge.training.contracts.materialize_metrics",
            wraps=materialize_metrics,
        ) as materialize:
            metrics = train_window(runner, [_batch()])
            materialize.assert_not_called()

            self.assertEqual(metrics.loss, 2.0)
            materialize.assert_called_once()
            self.assertEqual(metrics.materialize()["acc"], 0.5)
            self.assertEqual(metrics.grad_norm, 1.0)
            materialize.assert_called_once()

    def test_ratio_metrics_are_reported_by_name_and_position(self):
        class RatioProgram(TinyProgram):
            def forward(self, tensors, **context):
                return replace(
                    super().forward(tensors, **context),
                    ratios={
                        "acc": (torch.tensor(2.0), torch.tensor(4.0)),
                        "ce_position": (
                            torch.tensor([1.0, 3.0]),
                            torch.tensor([2.0, 6.0]),
                        ),
                    },
                )

        program = RatioProgram()
        runner = _runner(program, FakeBackend(program))
        metrics = train_window(runner, [_batch()]).materialize()

        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["ce_position_0"], 0.5)
        self.assertEqual(metrics["ce_position_1"], 0.5)

    _RATIO_INPUTS = {
        "acc": (torch.tensor(2.0), torch.tensor(4.0)),
        "ce_position": (
            torch.tensor([1.0, 3.0]),
            torch.tensor([2.0, 6.0]),
        ),
    }

    def test_ratio_metrics_sum_numerators_and_denominators_before_dividing(self):
        remote = torch.tensor([8.0, 6.0, 3.0, 1.0, 2.0, 2.0])
        reduced_groups = []

        def all_reduce(packed, *, group):
            reduced_groups.append(group)
            packed.add_(remote)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch("torch.distributed.all_reduce", side_effect=all_reduce),
        ):
            metrics = _reduce_ratio_metrics(
                self._RATIO_INPUTS,
                device=torch.device("cpu"),
                process_group="dp",
                reduce=True,
            )

        self.assertEqual(reduced_groups, ["dp"])
        self.assertEqual(metrics["acc"], 1.0)
        self.assertEqual(metrics["ce_position_0"], 1.0)
        self.assertEqual(metrics["ce_position_1"], 0.5)

    def test_ratio_metrics_skip_all_reduce_when_world_size_is_one(self):
        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=1),
            mock.patch("torch.distributed.all_reduce") as all_reduce_mock,
        ):
            metrics = _reduce_ratio_metrics(
                self._RATIO_INPUTS,
                device=torch.device("cpu"),
                process_group="dp",
                reduce=True,
            )

        all_reduce_mock.assert_not_called()
        self.assertEqual(metrics["acc"], 0.5)
        self.assertEqual(metrics["ce_position_0"], 0.5)
        self.assertEqual(metrics["ce_position_1"], 0.5)

    def test_additive_telemetry_accumulates_and_resets_each_optimizer_window(self):
        program = WeightedProgram()
        runner = _runner(program, FakeBackend(program))
        second = train_window(
            runner, [_weighted_batch(2, 1, 1), _weighted_batch(30, 3, 1)]
        ).materialize()
        self.assertEqual(second["sample_count"], 4)
        self.assertEqual(second["correct_count"], 2)
        self.assertEqual(second["acc"], 0.5)
        fourth = train_window(
            runner, [_weighted_batch(1, 2, 0), _weighted_batch(1, 3, 1)]
        ).materialize()
        self.assertEqual(fourth["sample_count"], 5)
        self.assertEqual(fourth["correct_count"], 1)
        self.assertAlmostEqual(fourth["acc"], 0.2)

    def test_prefix_counts_and_ratios_share_one_dp_sum_with_unequal_populations(self):
        accepted = torch.tensor([2.0, 0.0], requires_grad=True)
        reached = torch.tensor([3.0, 0.0])
        remote = torch.tensor([1.0, 0.0, 7.0, 0.0, 1.0, 0.0, 7.0, 0.0])

        def all_reduce(packed, *, group):
            self.assertEqual(group, "dp")
            self.assertFalse(packed.requires_grad)
            packed.add_(remote)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce", side_effect=all_reduce
            ) as reduce,
        ):
            metrics = _reduce_ratio_metrics(
                {"acceptance": (accepted, reached)},
                sums={"accepted_count": accepted, "reached_count": reached},
                device=torch.device("cpu"),
                process_group="dp",
                reduce=True,
            )
        reduce.assert_called_once()
        self.assertAlmostEqual(float(metrics["acceptance_0"]), 0.3)
        self.assertEqual(metrics["accepted_count_0"], 3)
        self.assertEqual(metrics["reached_count_0"], 10)
        # Zero is the established empty-ratio convention; count zero explicitly
        # distinguishes this from an observed 0% acceptance rate.
        self.assertEqual(metrics["acceptance_1"], 0)
        self.assertEqual(metrics["reached_count_1"], 0)

    def test_additive_telemetry_without_ratios(self):
        metrics = _reduce_ratio_metrics(
            {},
            sums={"count": torch.tensor(7.0)},
            device=torch.device("cpu"),
            process_group=None,
            reduce=False,
        )
        self.assertEqual(metrics["count"], 7)

    def test_batch_missing_a_required_feature_is_refused(self):
        program = TinyProgram()
        runner = _runner(program, FakeBackend(program))
        bad = TrainBatch(sample_ids=["s"], strategy="fake", tensors={}, metadata={})
        with self.assertRaisesRegex(
            ValueError, r"fake batch missing required features \['x'\]"
        ):
            train_window(runner, [bad])


class TestTrainerController(unittest.TestCase):
    def test_failed_optimizer_step_does_not_advance_or_ack(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        backend.step = mock.Mock(
            side_effect=FloatingPointError("non-finite global grad norm")
        )
        acknowledged = []
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=1,
                num_epochs=1,
                ack_fn=lambda ids, step: acknowledged.append((list(ids), step)),
            )
            with self.assertRaisesRegex(
                FloatingPointError, "non-finite global grad norm"
            ):
                ctrl.fit([_batch()])

        self.assertEqual(ctrl.global_step, 0)
        self.assertEqual(acknowledged, [])

    def test_training_log_reports_pipeline_throughput_breakdown(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        logged = []
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "specforge.training.contracts.materialize_metrics",
                wraps=materialize_metrics,
            ) as materialize,
        ):
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=2,
                num_epochs=1,
                log_interval=2,
                data_parallel_size=4,
                logger=lambda metrics, step: logged.append((dict(metrics), step)),
            )
            self.assertEqual(ctrl.fit([_batch(), _batch()]), 2)

        materialize.assert_called_once()

        self.assertEqual(len(logged), 1)
        metrics, step = logged[0]
        self.assertEqual(step, 2)
        for name in (
            "perf/optimizer_steps_per_hour",
            "perf/optimizer_step_time_s",
            "perf/data_wait_time_s",
            "perf/train_compute_time_s",
            "perf/durable_ack_time_s",
            "perf/global_samples_per_second",
        ):
            self.assertIn(name, metrics)
            self.assertGreaterEqual(metrics[name], 0.0)
        self.assertGreater(metrics["perf/optimizer_steps_per_hour"], 0.0)
        self.assertGreater(metrics["perf/global_samples_per_second"], 0.0)
        # Both rates share one elapsed time: each step trains one local sample,
        # and every data-parallel rank trains as many.
        self.assertAlmostEqual(
            metrics["perf/global_samples_per_second"]
            * 3600.0
            / metrics["perf/optimizer_steps_per_hour"],
            4.0,
        )
        self.assertEqual(
            [context["collect_detailed_metrics"] for context in program.contexts],
            [False, True],
        )

    def test_loader_fetch_seconds_metric_is_per_sample(self):
        class PerfData(list):
            def perf_counters_snapshot(self, reset=False):
                self.reset_requested = reset
                return {
                    "wait_producer_s": 0.0,
                    "wait_fetch_s": 0.0,
                    "fetch_s": 8.0,
                    "fetch_bytes": 1 << 30,
                    "fetch_batches": 2.0,
                    "fetch_samples": 4.0,
                }

        program = TinyProgram()
        backend = FakeBackend(program)
        data = PerfData([_batch()])
        logged = []
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=1,
                num_epochs=1,
                log_interval=1,
                logger=lambda metrics, step: logged.append((dict(metrics), step)),
            )
            self.assertEqual(ctrl.fit(data), 1)

        self.assertTrue(data.reset_requested)
        self.assertEqual(logged[0][0]["perf/fetch_seconds_per_sample"], 2.0)

    def test_progress_bar_tracks_optimizer_steps_on_rank_zero(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        progress = mock.Mock()
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "specforge.training.controller.sys.stderr.isatty", return_value=True
            ),
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=False),
            mock.patch("tqdm.tqdm", return_value=progress) as build_progress,
        ):
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=3,
                num_epochs=1,
                start_step=1,
            )
            self.assertEqual(ctrl.fit([_batch(), _batch()]), 3)

        self.assertEqual(build_progress.call_args.kwargs["total"], 3)
        self.assertEqual(build_progress.call_args.kwargs["initial"], 1)
        self.assertEqual(progress.update.call_args_list, [mock.call(1), mock.call(1)])
        progress.close.assert_called_once_with()

    def test_progress_bar_is_disabled_off_rank_zero(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        with (
            tempfile.TemporaryDirectory() as d,
            mock.patch(
                "specforge.training.controller.sys.stderr.isatty", return_value=True
            ),
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_rank", return_value=1),
            mock.patch("tqdm.tqdm") as build_progress,
        ):
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=1,
                num_epochs=1,
            )
            self.assertIsNone(ctrl._make_progress_bar())

        build_progress.assert_not_called()

    def test_fit_and_checkpoint(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
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

    def test_checkpoint_is_written_only_once_the_ranks_agree(self):
        class Program(TinyProgram):
            def kernel_record(self):
                return {"0:0": "reference"}

        program = Program()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(program, FakeBackend(program), run_id="r", output_dir=d)
            with mock.patch.object(
                ctrl.runner, "agree", side_effect=RuntimeError("ranks disagree")
            ) as agree:
                with self.assertRaisesRegex(RuntimeError, "ranks disagree"):
                    ctrl.save_checkpoint(1)
            agree.assert_called_once_with(
                None, value=("loss kernels", {"0:0": "reference"})
            )
            self.assertEqual(os.listdir(d), [])

    def test_rank_that_cannot_collect_its_state_says_so_in_the_agreement(self):
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(program, FakeBackend(program), run_id="r", output_dir=d)
            lost = MemoryError("optimizer masters")

            def collect():
                try:
                    raise lost
                except MemoryError as error:
                    raise RankFailure("MemoryError: optimizer masters") from error

            with (
                mock.patch.object(ctrl.runner, "checkpoint_state", collect),
                mock.patch.object(
                    ctrl.runner, "agree", wraps=ctrl.runner.agree
                ) as agree,
                self.assertRaises(MemoryError) as raised,
            ):
                ctrl.save_checkpoint(1)
            # The other ranks are told, and raise, before this one raises what
            # it failed with.
            agree.assert_called_once_with("MemoryError: optimizer masters", value=None)
            self.assertIs(raised.exception, lost)
            self.assertEqual(os.listdir(d), [])

    def test_failure_inside_the_gather_is_not_agreed(self):
        """Its peers are inside the collective: the rank raises and the job ends."""
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(program, FakeBackend(program), run_id="r", output_dir=d)
            with (
                mock.patch.object(
                    ctrl.runner, "checkpoint_state", side_effect=MemoryError("gather")
                ),
                mock.patch.object(ctrl.runner, "agree") as agree,
                self.assertRaisesRegex(MemoryError, "gather"),
            ):
                ctrl.save_checkpoint(1)
            agree.assert_not_called()
            self.assertEqual(os.listdir(d), [])

    def test_rank_told_of_another_ranks_failure_writes_nothing(self):
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(program, FakeBackend(program), run_id="r", output_dir=d)
            told = RuntimeError("rank 1: MemoryError: optimizer masters")
            with (
                mock.patch.object(ctrl.runner, "agree", side_effect=told),
                self.assertRaisesRegex(RuntimeError, "rank 1: MemoryError"),
            ):
                ctrl.save_checkpoint(1)
            self.assertEqual(os.listdir(d), [])

    def test_max_steps_closes_prefetch_and_requeues_unyielded_refs(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            for index in range(4):
                torch.save(
                    {"x": torch.ones(2) * (index + 1)},
                    os.path.join(d, f"{index:03d}.ckpt"),
                )
            refs = OfflineManifestReader(
                d,
                run_id="run",
                strategy="fake",
                feature_keys=("x",),
                target_repr=None,
            ).read()
            refs_by_id = {ref.sample_id: ref for ref in refs}
            queue = RetryableCloseQueue()
            queue.put(refs)
            store = LocalFeatureStore("st")
            loader = FeatureDataLoader(
                store,
                queue,
                batch_size=1,
                strategy="fake",
                ack=False,
                num_workers=2,
            )

            def ack(ids, _step):
                queue.ack([refs_by_id[sample_id] for sample_id in ids])

            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=1,
                num_epochs=1,
                ack_fn=ack,
            )
            self.assertEqual(ctrl.fit(loader), 1)

        self.assertEqual(queue.in_flight(), 0)
        self.assertEqual(queue.depth(), 3)
        self.assertEqual(store.health()["active_leases"], 0)
        self.assertIsNone(loader._prefetch_state)

    def test_public_trainer_fit_has_no_eval_side_channel(self):
        from specforge.training.trainer import Trainer

        self.assertEqual(list(inspect.signature(Trainer.fit).parameters), ["self"])

    def test_natural_eos_rejects_incomplete_accumulation(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program, backend, accumulation_steps=2, run_id="r", output_dir=d
            )
            progress = mock.Mock()
            with (
                mock.patch.object(ctrl, "_make_progress_bar", return_value=progress),
                self.assertRaisesRegex(
                    IncompleteWindow,
                    "incomplete gradient accumulation: received 1 of 2",
                ),
            ):
                ctrl.fit([_batch() for _ in range(3)])

        self.assertEqual(ctrl.global_step, 1)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(backend.boundaries, [False, True, False])
        progress.close.assert_called_once_with()
        # The third batch was consumed and never committed: a retry finds the
        # stream exhausted and must not report the run as complete.
        with self.assertRaisesRegex(RuntimeError, "received 1 of 2"):
            ctrl.fit([])
        self.assertEqual(ctrl.global_step, 1)

    def test_window_that_raised_ends_the_run(self):
        class FailingProgram(TinyProgram):
            def forward(self, tensors, **context):
                if "bad" in tensors:
                    raise ValueError("bad batch")
                return super().forward(tensors, **context)

        program = FailingProgram()
        backend = FakeBackend(program)
        good = _batch()
        bad = replace(
            good, sample_ids=["bad"], tensors={**good.tensors, "bad": torch.ones(())}
        )
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program, backend, accumulation_steps=2, run_id="r", output_dir=d
            )
            with self.assertRaisesRegex(ValueError, "bad batch"):
                ctrl.fit([_batch(), _batch(), _batch(), bad])
            # The third batch's backward ran; skipping to the end of the data
            # on a retry would save a checkpoint that never trained on it.
            with self.assertRaisesRegex(RuntimeError, "never committed.*bad batch"):
                ctrl.fit([])
            # Saving would record a position past both uncommitted batches.
            with self.assertRaisesRegex(RuntimeError, "cannot be saved"):
                ctrl.save_checkpoint(ctrl.global_step)
            self.assertEqual(os.listdir(d), [])
        self.assertEqual(ctrl.global_step, 1)

    def test_failed_acknowledgement_ends_the_run(self):
        def ack(sample_ids, step):
            raise OSError("ledger unavailable")

        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                max_steps=1,
                ack_fn=ack,
            )
            with self.assertRaisesRegex(OSError, "ledger unavailable"):
                ctrl.fit([_batch()])
            # The update is applied and its samples are not durable: a retry
            # that stopped at max_steps would report the run complete.
            self.assertEqual(ctrl.global_step, 1)
            with self.assertRaisesRegex(RuntimeError, "ledger unavailable"):
                ctrl.fit([])

    def test_window_holds_one_micro_batch_at_a_time(self):
        """A batch may live on the training device; none outlives its turn."""
        for replay_statistics in ((), ("count",)):
            with self.subTest(replay_statistics=replay_statistics):
                alive = []
                streamed = []

                class Runner:
                    def forward_backward(self, batch, ctx, *, is_boundary):
                        gc.collect()
                        alive.append(sum(ref() is not None for ref in streamed))
                        return "pending" if is_boundary else None

                    def apply_step(self, pending):
                        return StepMetrics({"loss": 0.0})

                    def prepass(self, batch, ctx):
                        return {"count": torch.ones(1)}

                    def reduce_sum(self, values):
                        return values

                def stream():
                    for _ in range(6):
                        # Hold nothing while suspended, as a loader would not.
                        handed = [_batch()]
                        streamed.append(weakref.ref(handed[0]))
                        yield handed.pop()

                Runner.replay_statistics = replay_statistics
                with tempfile.TemporaryDirectory() as d:
                    ctrl = TrainerController(
                        Runner(),
                        strategy_name="fake",
                        accumulation_steps=3,
                        run_id="r",
                        output_dir=d,
                    )
                    ctrl.fit(stream())
                # Entering a forward, only the batch it is handed is alive; a
                # replayed forward is handed a host copy and none is.
                self.assertEqual(alive, [0 if replay_statistics else 1] * 6)


class TestBestTracking(unittest.TestCase):
    def test_best_checkpoint_does_not_require_periodic_saves(self):
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                max_steps=2,
                eval_interval=1,
                eval_data_factory=lambda: [_batch()],
                save_interval=0,
            )
            ctrl.fit([_batch(), _batch()])
            self.assertTrue(os.path.isdir(os.path.join(d, "r-step1")))
            with open(os.path.join(d, "r.best_meta.json")) as stream:
                self.assertEqual(json.load(stream)["step"], 1)

    def test_best_fires_on_misaligned_eval_and_save_intervals(self):
        # eval_interval=1, save_interval=2: the first eval is still checkpointed
        # on demand as the best even though it is not a periodic-save step.
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=3,
                num_epochs=1,
                eval_interval=1,
                eval_data_factory=lambda: [_batch()],
                save_interval=2,
            )
            ctrl.fit([_batch() for _ in range(5)])
            self.assertTrue(os.path.isdir(os.path.join(d, "r-step1")))
            with open(os.path.join(d, "r.best_meta.json")) as stream:
                meta = json.load(stream)
            self.assertEqual(meta["step"], 1)
            self.assertAlmostEqual(meta["score"], 0.5, places=6)


class TestEvalMetricsFlow(unittest.TestCase):
    def test_eval_metrics_reach_logger_and_last_metrics(self):
        program = TinyProgram()
        backend = FakeBackend(program)
        logged = []
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                max_steps=2,
                num_epochs=1,
                eval_interval=1,
                eval_data_factory=lambda: [_batch()],
                log_interval=50,
                logger=lambda metrics, step: logged.append((dict(metrics), step)),
            )
            ctrl.fit([_batch() for _ in range(3)])
        self.assertTrue(program.training)
        self.assertEqual([step for _, step in logged], [1, 2])
        for metrics, _ in logged:
            self.assertIn("eval/avg_loss", metrics)
            self.assertAlmostEqual(metrics["eval/avg_acc"], 0.5, places=6)
        self.assertIn("eval/avg_acc", ctrl.last_metrics)
        self.assertIn("loss", ctrl.last_metrics)

    def test_eval_does_not_request_the_detailed_metrics_it_discards(self):
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program, FakeBackend(program), run_id="r", output_dir=d, num_epochs=1
            )
            metrics = ctrl.evaluate([_batch()])
        self.assertIn("eval/avg_loss", metrics)
        self.assertFalse(program.contexts[-1]["collect_detailed_metrics"])

    def test_each_interval_gets_a_fresh_managed_eval_stream(self):
        events = []

        class ManagedEval:
            def __init__(self, pass_id):
                self.pass_id = pass_id

            def __enter__(self):
                events.append(("enter", self.pass_id))
                return self

            def __exit__(self, *_):
                events.append(("exit", self.pass_id))

            def __iter__(self):
                yield _batch()

        def factory():
            pass_id = sum(event[0] == "enter" for event in events) + 1
            return ManagedEval(pass_id)

        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                max_steps=2,
                eval_interval=1,
                eval_data_factory=factory,
            )
            ctrl.fit([_batch(), _batch()])
        self.assertEqual(
            events,
            [("enter", 1), ("exit", 1), ("enter", 2), ("exit", 2)],
        )


def _named_batch(i):
    return TrainBatch(
        sample_ids=[f"b{i}"],
        strategy="fake",
        tensors={"x": torch.ones(2), "index": torch.tensor(i)},
        metadata={},
    )


class TestResumeDataPosition(unittest.TestCase):
    def test_start_batch_skips_consumed_prefix(self):
        # Resume mid-epoch: start_batch=k drops the first k batches of the FIRST
        # epoch only; later epochs iterate in full.
        program = RecordingProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                num_epochs=2,
                start_batch=3,
                start_samples=3,
            )
            ctrl.fit([_named_batch(i) for i in range(5)])
        # epoch 0 resumes at b3; epoch 1 is complete
        self.assertEqual(program.seen, [3, 4, 0, 1, 2, 3, 4])
        self.assertEqual(ctrl._epoch_batch, 0)  # reset when the epoch completes

    def test_natural_exhaustion_checkpoint_records_completed_epochs(self):
        program = RecordingProgram()
        backend = FakeBackend(program)
        data = [_named_batch(i) for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(program, backend, run_id="r", output_dir=d, num_epochs=2)
            step = ctrl.fit(data)
            self.assertEqual(ctrl.epoch, 2)
            checkpoint = ctrl.save_checkpoint(step)
            state = CheckpointManager.read_resume_state(checkpoint.checkpoint_uri)

            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["epoch_batch"], 0)
            self.assertEqual(state["epoch_samples"], 0)

            resumed = RecordingProgram()
            resumed_ctrl = _controller(
                resumed,
                FakeBackend(resumed),
                run_id="r2",
                output_dir=d,
                num_epochs=2,
                start_step=state["global_step"],
                start_epoch=state["epoch"],
                start_batch=state["epoch_batch"],
                start_samples=state["epoch_samples"],
            )
            self.assertEqual(resumed_ctrl.fit(data), step)
            self.assertEqual(resumed.seen, [])

    def test_ctor_rejects_half_specified_position(self):
        # start_batch/start_samples describe ONE position; one without the other
        # is a corrupt resume.
        program = TinyProgram()
        backend = FakeBackend(program)
        for kw in ({"start_batch": 3}, {"start_samples": 3}):
            with self.assertRaisesRegex(ValueError, "zero or nonzero together"):
                _controller(
                    program, backend, run_id="r", output_dir="/tmp-unused", **kw
                )

    def test_islice_skip_past_end_raises(self):
        # plain-iterable fallback: a silent empty epoch is banned.
        program = TinyProgram()
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                num_epochs=1,
                start_batch=7,
                start_samples=7,
            )
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
                ctrl.fit([_batch() for _ in range(5)])

    def test_islice_skip_of_exactly_all_batches_is_allowed(self):
        # skip == available means the epoch was fully consumed pre-restart.
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
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
        program = RecordingProgram()
        data = [_named_batch(i) for i in range(5)]
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                FakeBackend(program),
                run_id="r",
                output_dir=d,
                num_epochs=1,
                max_steps=2,
            )
            self.assertEqual(ctrl.fit(data), 2)
            ctrl.max_steps = 4
            self.assertEqual(ctrl.fit(data), 4)
        self.assertEqual(program.seen, [0, 1, 2, 3])


class TestStepContextThreading(unittest.TestCase):
    """Regression for the Domino schedule bug: fit() must thread a real schedule
    horizon through StepContext.total_steps to the program (falling back to
    max_steps), not None: the Domino base-loss warmup is scheduled over it."""

    def _last_context(self, **controller_kw):
        program = TinyProgram()
        backend = FakeBackend(program)
        with tempfile.TemporaryDirectory() as d:
            ctrl = _controller(
                program,
                backend,
                run_id="r",
                output_dir=d,
                num_epochs=1,
                **controller_kw,
            )
            ctrl.fit([_batch() for _ in range(3)])
        return program.contexts[-1]

    def test_explicit_total_steps_is_threaded(self):
        context = self._last_context(total_steps=10, max_steps=3)
        # explicit horizon wins over the cap
        self.assertEqual(context["total_steps"], 10)

    def test_total_steps_falls_back_to_max_steps(self):
        # the common case: only max_steps set -> use it as the horizon so a
        # schedule-dependent loss is NOT silently disabled.
        context = self._last_context(max_steps=5)
        self.assertEqual(context["total_steps"], 5)

    def test_total_steps_none_when_unbounded(self):
        context = self._last_context()  # neither total_steps nor max_steps set
        self.assertIsNone(context["total_steps"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
