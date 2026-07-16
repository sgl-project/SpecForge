# coding=utf-8
"""TrainerCore grad-accum + TrainerController fit/eval/checkpoint (CPU)."""

import inspect
import json
import os
import tempfile
import unittest
from unittest import mock

import torch
import torch.nn as nn

from specforge.runtime.contracts import TrainBatch
from specforge.runtime.data_plane.feature_dataloader import FeatureDataLoader
from specforge.runtime.data_plane.feature_store import LocalFeatureStore
from specforge.runtime.data_plane.offline_reader import OfflineManifestReader
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue
from specforge.training.backend import TrainingBackend
from specforge.training.checkpoint import CheckpointManager
from specforge.training.controller import Checkpoint, TrainerController, TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput


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


class RetryableCloseQueue(SampleRefQueue):
    loader_close_retryable = True


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

    def test_metrics_reduce_once_at_accumulation_boundary(self):
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=2)

        def reduce_sum(tensor):
            tensor.mul_(2)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce", side_effect=reduce_sum
            ) as reduce,
        ):
            core.train_step(_batch())
            self.assertEqual(reduce.call_count, 0)
            core.train_step(_batch())

        self.assertEqual(reduce.call_count, 1)
        self.assertEqual(reduce.call_args.args[0].numel(), 2)

    def test_metrics_carry_no_mode(self):
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        rep = core.train_step(_batch())
        self.assertNotIn("mode", rep.metrics)

    def test_strategy_scalar_metrics_are_preserved(self):
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        result = core._result(
            StepOutput(
                loss=torch.tensor(2.0),
                metrics={
                    "loss": torch.tensor(99.0),
                    "accuracy": torch.tensor(0.5),
                    "ce_loss": torch.tensor(1.25),
                    "lambda_base": 0.75,
                    "non_scalar_debug": torch.tensor([1.0, 2.0]),
                },
            ),
            grad_norm=None,
            stepped=False,
        )

        self.assertEqual(result.metrics["loss"], 2.0)
        self.assertEqual(result.metrics["acc"], 0.5)
        self.assertEqual(result.metrics["ce_loss"], 1.25)
        self.assertEqual(result.metrics["lambda_base"], 0.75)
        self.assertNotIn("non_scalar_debug", result.metrics)

    @staticmethod
    def _eagle_output(
        *,
        corrects=(1.0, 3.0),
        acc_denoms=(2.0, 4.0),
        losses=(2.0, 4.0),
        loss_denoms=(2.0, 4.0),
        acceptance_rates=(0.25, 0.75),
    ):
        tensor_list = lambda values: [  # noqa: E731
            torch.tensor(value, dtype=torch.float32) for value in values
        ]
        return StepOutput(
            loss=torch.tensor(99.0),
            metrics={
                "plosses": tensor_list(losses),
                "acces": tensor_list(
                    [
                        correct / denominator
                        for correct, denominator in zip(corrects, acc_denoms)
                    ]
                ),
                "acceptance_rates": tensor_list(acceptance_rates),
                "acc_corrects": tensor_list(corrects),
                "acc_denoms": tensor_list(acc_denoms),
                "metric_losses": tensor_list(losses),
                "metric_loss_denoms": tensor_list(loss_denoms),
            },
        )

    def test_eagle_metrics_preserve_ttt_positions_and_count_weighting(self):
        strat = FakeStrategy()
        strat.ploss_decay = 0.5
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)

        result = core._result(self._eagle_output(), grad_norm=None, stepped=False)

        self.assertAlmostEqual(result.metrics["acc_0"], 0.5)
        self.assertAlmostEqual(result.metrics["acc_1"], 0.75)
        self.assertAlmostEqual(result.metrics["acc"], 4 / 6)
        self.assertAlmostEqual(result.metrics["ploss_0"], 2.0)
        self.assertAlmostEqual(result.metrics["ploss_1"], 4.0)
        self.assertAlmostEqual(result.metrics["loss"], 4.0)
        self.assertAlmostEqual(result.metrics["acceptance_rate_0"], 0.25)
        self.assertAlmostEqual(result.metrics["acceptance_rate_1"], 0.75)
        self.assertAlmostEqual(result.metrics["acceptance_rate"], 0.5)
        self.assertNotIn("acces", result.metrics)
        self.assertNotIn("acceptance_rates", result.metrics)
        self.assertNotIn("plosses", result.metrics)

    def test_eagle_metrics_sum_counts_across_data_parallel_ranks(self):
        strat = FakeStrategy()
        strat.ploss_decay = 0.8
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        # Remote rank rows use the same packed layout as controller.py:
        # correct, acc denom, loss numerator, loss denom, acceptance numerator,
        # acceptance denom.
        remote = torch.tensor(
            [
                [8.0, 1.0],
                [10.0, 2.0],
                [100.0, 16.0],
                [10.0, 2.0],
                [9.0, 1.0],
                [10.0, 2.0],
            ]
        )

        def all_reduce(value, *, op, group):
            self.assertIs(op, torch.distributed.ReduceOp.SUM)
            self.assertIsNone(group)
            value.add_(remote)

        with (
            mock.patch("torch.distributed.is_available", return_value=True),
            mock.patch("torch.distributed.is_initialized", return_value=True),
            mock.patch("torch.distributed.get_world_size", return_value=2),
            mock.patch(
                "torch.distributed.all_reduce", side_effect=all_reduce
            ) as reduce,
        ):
            result = core._result(self._eagle_output(), grad_norm=None, stepped=False)

        reduce.assert_called_once()
        self.assertAlmostEqual(result.metrics["acc_0"], 9 / 12, places=6)
        self.assertAlmostEqual(result.metrics["acc_1"], 4 / 6, places=6)
        self.assertAlmostEqual(result.metrics["acc"], 13 / 18, places=6)
        self.assertAlmostEqual(result.metrics["ploss_0"], 104 / 12, places=6)
        self.assertAlmostEqual(result.metrics["ploss_1"], 32 / 6, places=6)
        self.assertAlmostEqual(
            result.metrics["loss"], 104 / 12 + 0.8 * (32 / 6), places=5
        )
        self.assertAlmostEqual(result.metrics["acceptance_rate_0"], 9.5 / 12, places=6)
        self.assertAlmostEqual(result.metrics["acceptance_rate_1"], 4 / 6, places=6)

    def test_validate_batch_missing_feature(self):
        strat = FakeStrategy()
        bad = TrainBatch(sample_ids=["s"], strategy="fake", tensors={}, metadata={})
        with self.assertRaises(ValueError):
            strat.forward_loss(bad)


class TestTrainerController(unittest.TestCase):
    def test_progress_bar_tracks_optimizer_steps_on_rank_zero(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
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
            ctrl = TrainerController(
                core,
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
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
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
            ctrl = TrainerController(
                core,
                run_id="r",
                output_dir=d,
                max_steps=1,
                num_epochs=1,
            )
            self.assertIsNone(ctrl._make_progress_bar())

        build_progress.assert_not_called()

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

    def test_max_steps_closes_prefetch_and_requeues_unyielded_refs(self):
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
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

            ctrl = TrainerController(
                core,
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
        strat = FakeStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=2)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(core, run_id="r", output_dir=d)
            progress = mock.Mock()
            with (
                mock.patch.object(ctrl, "_make_progress_bar", return_value=progress),
                self.assertRaisesRegex(
                    RuntimeError, "incomplete gradient accumulation"
                ),
            ):
                ctrl.fit([_batch() for _ in range(3)])

        self.assertEqual(ctrl.global_step, 1)
        self.assertEqual(backend.steps, 1)
        self.assertEqual(core.accumulation_remainder, 1)
        self.assertEqual(backend.boundaries, [False, True, False])
        progress.close.assert_called_once_with()


class TestBestTracking(unittest.TestCase):
    def test_best_checkpoint_does_not_require_periodic_saves(self):
        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
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
                eval_data_factory=lambda: [_batch()],
                log_interval=50,
                logger=lambda metrics, step: logged.append((dict(metrics), step)),
            )
            ctrl.fit([_batch() for _ in range(3)])
        self.assertTrue(strat.model.training)
        self.assertEqual([step for _, step in logged], [1, 2])
        for metrics, _ in logged:
            self.assertIn("eval/avg_loss", metrics)
            self.assertAlmostEqual(metrics["eval/avg_acc"], 0.5, places=6)
        self.assertIn("eval/avg_acc", ctrl.last_metrics)
        self.assertIn("loss", ctrl.last_metrics)

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

        strat = FakeStrategy()
        core = TrainerCore(strat, FakeBackend(strat.model), accumulation_steps=1)
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(
                core,
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

    def test_natural_exhaustion_checkpoint_records_completed_epochs(self):
        strat = RecordingStrategy()
        backend = FakeBackend(strat.model)
        core = TrainerCore(strat, backend, accumulation_steps=1)
        data = [_named_batch(i) for i in range(2)]
        with tempfile.TemporaryDirectory() as d:
            ctrl = TrainerController(core, run_id="r", output_dir=d, num_epochs=2)
            step = ctrl.fit(data)
            self.assertEqual(ctrl.epoch, 2)
            checkpoint = ctrl.save_checkpoint(step)
            state = CheckpointManager.read_resume_state(checkpoint.checkpoint_uri)

            self.assertEqual(state["epoch"], 2)
            self.assertEqual(state["epoch_batch"], 0)
            self.assertEqual(state["epoch_samples"], 0)

            resumed = RecordingStrategy()
            resumed_core = TrainerCore(
                resumed, FakeBackend(resumed.model), accumulation_steps=1
            )
            resumed_ctrl = TrainerController(
                resumed_core,
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
            with self.assertRaisesRegex(ValueError, "skips past the end of the data"):
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
