# coding=utf-8
"""Asynchronous optimizer-boundary durable ack (CPU).

``TrainerController(async_ack=True)`` runs step N's ack on one background
thread while step N+1 computes. These tests pin the invariants that make that
safe: acks stay strictly ordered, every checkpoint/eval/fit return flushes the
pending ack first, an ack failure re-raises on the training thread, the DP ack
collectives run on their own Gloo group concurrently with training-thread
collectives, and a crash mid-ack resumes from the last (flushed) checkpoint.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import random
import sqlite3
import tempfile
import threading
import time
import traceback
import unittest
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch
import torch.distributed as dist
import torch.nn as nn

from specforge.runtime.contracts import FeatureSpec, SampleRef, TrainBatch
from specforge.runtime.control_plane.controller import DataFlowController
from specforge.runtime.control_plane.dp_ack import (
    DPAckController,
    new_durable_ack_process_group,
)
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    SQLiteMetadataStore,
)
from specforge.training.backend import TrainingBackend
from specforge.training.checkpoint import STATE_FILE, CheckpointManager
from specforge.training.controller import (
    TrainerController,
    TrainerCore,
    _AsyncAckRunner,
)
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

WORLD_SIZE = 2
ACK_THREAD = "specforge-durable-ack"
_GLOO_AVAILABLE = dist.is_available() and dist.is_gloo_available()


class _TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(1))


class _Strategy(DraftTrainStrategy):
    name = "fake"
    required_features = {"x"}

    def __init__(self, on_forward=None):
        self.model = _TinyModel()
        self.on_forward = on_forward

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch: TrainBatch, ctx=None) -> StepOutput:
        self.validate_batch(batch)
        if self.on_forward is not None and self.model.training:
            self.on_forward(batch, ctx)
        loss = (self.model.w * batch.tensors["x"].sum()).abs()
        return StepOutput(loss=loss, metrics={"accuracy": torch.tensor(0.5)})


class _Backend(TrainingBackend):
    name = "fake"

    def __init__(self, model, on_step=None):
        self.model = model
        self.on_step = on_step

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()

    def scale_gradients(self, factor):
        pass

    def step(self):
        if self.on_step is not None:
            self.on_step()
        return torch.tensor(1.0)

    def state_dict(self):
        return {
            "model": {"draft_model.w": self.model.w.detach().clone()},
            "optimizer": None,
            "rng": {},
        }

    def load_state_dict(self, state):
        pass


class _RecordingCheckpointManager(CheckpointManager):
    """Record the durable marker each checkpoint observes as it is written."""

    def __init__(self, output_dir, run_id, marker_step):
        super().__init__(output_dir, run_id)
        self.marker_step = marker_step
        self.saves = []

    def save(self, state, step, *, rank_state=None):
        self.saves.append((step, self.marker_step()))
        return super().save(state, step, rank_state=rank_state)


def _batch(*sample_ids: str) -> TrainBatch:
    return TrainBatch(
        sample_ids=list(sample_ids),
        strategy="fake",
        tensors={"x": torch.ones(2)},
        metadata={},
    )


def _ref(sample_id: str) -> SampleRef:
    return SampleRef(
        sample_id=sample_id,
        run_id="run0",
        source_task_id=f"task-{sample_id}",
        feature_store_uri=f"mooncake://run0/{sample_id}",
        feature_keys={"hidden_state": f"{sample_id}/hidden_state"},
        feature_specs={
            "hidden_state": FeatureSpec(
                name="hidden_state", shape=(2, 4), dtype="float32"
            )
        },
        strategy="eagle3",
        metadata={"target_repr": "hidden_state"},
    )


def _controller(output_dir, *, on_forward=None, on_step=None, **kwargs):
    strategy = _Strategy(on_forward)
    core = TrainerCore(strategy, _Backend(strategy.model, on_step))
    return TrainerController(core, run_id="run0", output_dir=output_dir, **kwargs)


class TestAsyncDurableAck(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="async_ack_")
        self.store = SQLiteMetadataStore(os.path.join(self.work, "ledger.sqlite"))
        self.dfc = DataFlowController("run0", metadata_store=self.store)

    def tearDown(self):
        self.store.close()

    def _marker_step(self):
        return self.store.durable_marker()["global_step"]

    def _durable_ack(self, *, delay_s=0.0, calls=None):
        def ack_fn(ids, step):
            time.sleep(delay_s)
            self.dfc.ack_train_refs("t0", ids, global_step=step, optimizer_durable=True)
            if calls is not None:
                calls.append((list(ids), step, threading.current_thread().name))

        return ack_fn

    def test_ack_overlaps_next_step_in_optimizer_step_order(self):
        # ctx.global_step is the last COMPLETED step while step N computes.
        computing = {step: threading.Event() for step in range(1, 5)}
        calls = []

        def ack_fn(ids, step):
            next_step = computing.get(step + 1)
            overlapped = next_step.wait(10) if next_step is not None else None
            calls.append((list(ids), step, threading.current_thread().name, overlapped))

        ctrl = _controller(
            self.work,
            on_forward=lambda _batch, ctx: computing[ctx.global_step + 1].set(),
            max_steps=4,
            ack_fn=ack_fn,
            async_ack=True,
        )
        self.assertEqual(ctrl.fit([_batch(f"s{i}") for i in range(1, 5)]), 4)

        # Step N's ack was still running when step N+1's forward began, yet the
        # acks remain one ordered stream on one background thread.
        self.assertEqual(
            calls,
            [
                (["s1"], 1, ACK_THREAD, True),
                (["s2"], 2, ACK_THREAD, True),
                (["s3"], 3, ACK_THREAD, True),
                (["s4"], 4, ACK_THREAD, None),
            ],
        )
        self.assertIsNone(ctrl._ack_runner)

    def test_default_ack_stays_synchronous_on_training_thread(self):
        calls = []
        ctrl = _controller(
            self.work, max_steps=2, ack_fn=self._durable_ack(calls=calls)
        )
        ctrl.fit([_batch("s1"), _batch("s2")])
        self.assertEqual(
            calls,
            [
                (["s1"], 1, threading.main_thread().name),
                (["s2"], 2, threading.main_thread().name),
            ],
        )

    def test_fit_returns_only_after_final_ack_is_durable(self):
        ctrl = _controller(
            self.work,
            max_steps=3,
            ack_fn=self._durable_ack(delay_s=0.2),
            async_ack=True,
        )
        ctrl.fit([_batch(f"s{i}") for i in range(1, 5)])
        marker = self.store.durable_marker()
        self.assertEqual(marker["global_step"], 3)
        self.assertEqual(marker["acked"], {"s1", "s2", "s3"})

    def test_checkpoint_is_never_written_ahead_of_its_ack(self):
        manager = _RecordingCheckpointManager(self.work, "run0", self._marker_step)
        ctrl = _controller(
            self.work,
            max_steps=4,
            save_interval=2,
            ack_fn=self._durable_ack(delay_s=0.2),
            async_ack=True,
            checkpoint_manager=manager,
        )
        ctrl.fit([_batch(f"s{i}") for i in range(1, 5)])
        self.assertEqual(manager.saves, [(2, 2), (4, 4)])

        # Direct saves outside fit keep the same invariant.
        ctrl.save_checkpoint(ctrl.global_step)
        self.assertEqual(manager.saves[-1], (4, 4))

    def test_eval_waits_for_the_pending_ack(self):
        seen_at_eval = []

        def eval_data():
            seen_at_eval.append(self._marker_step())
            return [_batch("eval")]

        ctrl = _controller(
            self.work,
            max_steps=4,
            eval_interval=2,
            eval_data_factory=eval_data,
            ack_fn=self._durable_ack(delay_s=0.2),
            async_ack=True,
        )
        ctrl.fit([_batch(f"s{i}") for i in range(1, 5)])
        self.assertEqual(seen_at_eval, [2, 4])

    def test_ack_failure_reraises_before_next_ack_or_checkpoint(self):
        error = OSError("ledger fsync failed")
        attempted = []

        def ack_fn(ids, step):
            attempted.append(step)
            if step == 2:
                raise error
            self.dfc.ack_train_refs("t0", ids, global_step=step, optimizer_durable=True)

        manager = _RecordingCheckpointManager(self.work, "run0", self._marker_step)
        ctrl = _controller(
            self.work,
            max_steps=6,
            save_interval=3,
            ack_fn=ack_fn,
            async_ack=True,
            checkpoint_manager=manager,
        )
        with self.assertRaises(OSError) as raised:
            ctrl.fit([_batch(f"s{i}") for i in range(1, 7)])

        # The failure surfaces on the training thread at the next boundary's
        # flush: step 3 neither starts its ack nor writes its checkpoint.
        self.assertIs(raised.exception, error)
        self.assertIn(
            "background durable ack of optimizer step 2",
            "\n".join(raised.exception.__notes__),
        )
        self.assertEqual(ctrl.global_step, 3)
        self.assertEqual(attempted, [1, 2])
        self.assertEqual(manager.saves, [])
        self.assertEqual(self._marker_step(), 1)
        self.assertIsNone(ctrl._ack_runner)

    def test_failure_on_the_last_step_is_raised_by_fit(self):
        def ack_fn(ids, step):
            raise RuntimeError(f"durable DP acknowledgement failed at {step}")

        ctrl = _controller(self.work, max_steps=1, ack_fn=ack_fn, async_ack=True)
        with self.assertRaisesRegex(RuntimeError, "acknowledgement failed at 1"):
            ctrl.fit([_batch("s1")])

    def test_training_failure_waits_for_inflight_ack_and_keeps_primary(self):
        ack_finished = threading.Event()

        def ack_fn(ids, step):
            time.sleep(0.2)
            ack_finished.set()
            raise ValueError("ack also failed")

        def on_forward(_batch, ctx):
            if ctx.global_step == 1:
                raise RuntimeError("forward failed on step 2")

        ctrl = _controller(
            self.work,
            on_forward=on_forward,
            max_steps=3,
            ack_fn=ack_fn,
            async_ack=True,
        )
        with self.assertRaisesRegex(RuntimeError, "forward failed on step 2") as raised:
            ctrl.fit([_batch(f"s{i}") for i in range(1, 4)])

        # Lifecycle cleanup after fit() must not race the in-flight ack.
        self.assertTrue(ack_finished.is_set())
        self.assertIn(
            "the background durable ack also failed: ValueError: ack also failed",
            "\n".join(raised.exception.__notes__),
        )

    def test_blocked_time_metric_excludes_overlapped_ack(self):
        def run(async_ack):
            logged = []
            ctrl = _controller(
                self.work,
                on_forward=lambda _batch, _ctx: time.sleep(0.1),
                max_steps=4,
                log_interval=4,
                logger=lambda metrics, _step: logged.append(dict(metrics)),
                ack_fn=lambda _ids, _step: time.sleep(0.05),
                async_ack=async_ack,
            )
            ctrl.fit([_batch(f"s{i}") for i in range(1, 5)])
            return logged[0]

        sync_metrics = run(False)
        self.assertGreaterEqual(sync_metrics["perf/durable_ack_time_s"], 0.045)
        self.assertNotIn("perf/durable_ack_background_time_s", sync_metrics)

        async_metrics = run(True)
        # Three acks completed inside the window (the fourth is still running
        # behind the logger); the training thread never waited on them.
        self.assertGreaterEqual(
            async_metrics["perf/durable_ack_background_time_s"], 0.03
        )
        self.assertLess(
            async_metrics["perf/durable_ack_time_s"],
            async_metrics["perf/durable_ack_background_time_s"],
        )

    def test_ack_group_is_not_created_without_a_process_group(self):
        self.assertIsNone(new_durable_ack_process_group())

    def test_interrupted_submit_cannot_strand_close(self):
        # SIGTERM unwinds the training thread as a BaseException. Even when it
        # lands inside submit(), the lifecycle close() must still return.
        acked = []
        runner = _AsyncAckRunner(lambda _ids, step: acked.append(step))
        notify_all = runner._cv.notify_all

        def interrupted_notify():
            runner._cv.notify_all = notify_all
            raise KeyboardInterrupt

        runner._cv.notify_all = interrupted_notify
        with self.assertRaises(KeyboardInterrupt):
            runner.submit(["s1"], 1)
        closer = threading.Thread(target=runner.close, daemon=True)
        closer.start()
        closer.join(5)
        self.assertFalse(closer.is_alive(), "close() waited on a never-woken ack")
        self.assertEqual(acked, [])


class _RetainingFeatureStore:
    retain_on_release = True

    def __init__(self) -> None:
        self.aborted = []

    def abort(self, sample_id, *, reason="aborted") -> None:
        self.aborted.append((sample_id, reason))


class _FakeDistributor:
    latest = None

    def __init__(self, *_args, **kwargs) -> None:
        self.kwargs = kwargs
        type(self).latest = self

    @staticmethod
    def inbox_path(inbox_dir: str, dp_rank: int) -> str:
        return os.path.join(inbox_dir, f"inbox-rank{dp_rank}.jsonl")

    def start(self):
        return self

    def stop(self) -> None:
        pass


def _build_consumer(work, *, resume_from=None, env=None, dist_patches=None):
    """Enter the real online-consumer builder below model/FSDP assembly.

    ``dist_patches`` replaces the default single-process ``torch.distributed``
    view (``is_initialized() -> False``) with a scripted one.
    """
    from contextlib import ExitStack

    from specforge.algorithms.builtin import builtin_algorithm_registry
    from specforge.launch import build_disagg_online_consumer
    from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel

    captured = {}
    channel = StreamingRefChannel(os.path.join(work, "refs.jsonl"))
    if resume_from is not None and channel.consumer_quantum() is None:
        channel.publish_consumer_quantum(1)
    features = _RetainingFeatureStore()
    if dist_patches is None:
        dist_patches = [
            mock.patch("torch.distributed.is_initialized", return_value=False)
        ]
    with ExitStack() as stack:
        stack.enter_context(mock.patch.dict(os.environ, env or {}))
        for patcher in dist_patches:
            stack.enter_context(patcher)
        stack.enter_context(
            mock.patch(
                "specforge.runtime.data_plane.ref_distributor.RefDistributor",
                _FakeDistributor,
            )
        )
        stack.enter_context(
            mock.patch(
                "specforge.launch._assemble_trainer",
                side_effect=lambda **kwargs: captured.update(kwargs)
                or SimpleNamespace(),
            )
        )
        build_disagg_online_consumer(
            algorithm=builtin_algorithm_registry().resolve("eagle3"),
            feature_store=features,
            channel=channel,
            draft_model=object(),
            optimizer_factory=object(),
            run_id="run0",
            output_dir=os.path.join(work, "output"),
            collate_fn=lambda features: {},
            metadata_db_path=os.path.join(work, "ledger.sqlite"),
            inbox_dir=os.path.join(work, "inboxes"),
            resume_from=resume_from,
        )
    return captured, _FakeDistributor.latest, features


class TestConsumerAsyncAckSwitch(unittest.TestCase):
    def _async_ack(self, env):
        with tempfile.TemporaryDirectory(prefix="async_ack_switch_") as work:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DISAGG_ASYNC_ACK", None)
                captured, _distributor, _features = _build_consumer(work, env=env)
            captured["controller"].store.close()
            return captured["async_ack"]

    def test_online_consumer_enables_async_ack_by_default(self):
        self.assertIs(self._async_ack({}), True)

    def test_env_disables_async_ack(self):
        for value in ("0", "false", "OFF"):
            with self.subTest(value=value):
                self.assertIs(self._async_ack({"DISAGG_ASYNC_ACK": value}), False)

    def _rank0_of_two(self, *, local, peer):
        """Rank 0 of a scripted 2-rank consumer; the peer reports ``peer``."""
        group = object()
        created = []

        def gather(object_list, obj, group=None):
            # The only setup gather is the preflight (error, async) exchange.
            object_list[:] = [obj, (None, peer)]

        def new_group():
            created.append(group)
            return group

        with tempfile.TemporaryDirectory(prefix="async_ack_2rank_switch_") as work:
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("DISAGG_ASYNC_ACK", None)
                captured, _distributor, _features = _build_consumer(
                    work,
                    env={"DISAGG_ASYNC_ACK": "1" if local else "0"},
                    dist_patches=[
                        mock.patch(
                            "torch.distributed.is_initialized", return_value=True
                        ),
                        mock.patch("torch.distributed.get_world_size", return_value=2),
                        mock.patch("torch.distributed.get_rank", return_value=0),
                        mock.patch(
                            "torch.distributed.all_gather_object", side_effect=gather
                        ),
                        mock.patch("torch.distributed.broadcast_object_list"),
                        mock.patch(
                            "specforge.runtime.control_plane.dp_ack."
                            "new_durable_ack_process_group",
                            side_effect=new_group,
                        ),
                    ],
                )
            controller = captured["controller"]
            controller.store.close()
            return captured["async_ack"], created, controller.process_group, group

    def test_async_ack_uses_the_dedicated_group_when_every_rank_enables_it(self):
        enabled, created, process_group, group = self._rank0_of_two(
            local=True, peer=True
        )
        self.assertIs(enabled, True)
        self.assertEqual(len(created), 1)
        self.assertIs(process_group, group)

    def test_any_rank_opting_out_keeps_every_rank_synchronous_without_gloo(self):
        # new_group is collective, so the decision must be rank-consistent; the
        # synchronous kill switch must not require Gloo connectivity either.
        for local, peer in ((False, True), (True, False), (False, False)):
            with self.subTest(local=local, peer=peer):
                enabled, created, process_group, _ = self._rank0_of_two(
                    local=local, peer=peer
                )
                self.assertIs(enabled, False)
                self.assertEqual(created, [])
                self.assertIsNone(process_group)

    def test_worker_honours_the_typed_switch_unless_env_is_explicit(self):
        from specforge.training.disaggregated import _consumer_async_ack

        def cfg(async_ack):
            return SimpleNamespace(
                deployment=SimpleNamespace(
                    disaggregated=(
                        None
                        if async_ack is None
                        else SimpleNamespace(async_ack=async_ack)
                    )
                )
            )

        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("DISAGG_ASYNC_ACK", None)
            self.assertIs(_consumer_async_ack(cfg(False)), False)
            self.assertIs(_consumer_async_ack(cfg(True)), True)
            self.assertIsNone(_consumer_async_ack(cfg(None)))
            os.environ["DISAGG_ASYNC_ACK"] = "1"
            # The builder parses the explicit environment value itself.
            self.assertIsNone(_consumer_async_ack(cfg(False)))


def _crash_mid_ack_worker(work: str) -> None:
    """Die inside step 3's background ack while step 4 is already computing."""
    store = SQLiteMetadataStore(os.path.join(work, "ledger.sqlite"))
    dfc = DataFlowController("run0", metadata_store=store)
    dfc.commit_samples("producer", [_ref(f"s{i}") for i in range(1, 5)])
    step4_computing = threading.Event()

    def on_forward(_batch, ctx):
        if ctx.global_step == 3:
            step4_computing.set()

    def ack_fn(ids, step):
        if step == 3:
            # Hard crash before the ledger commit, with the trainer one
            # optimizer step ahead of its durable marker.
            step4_computing.wait(10)
            os._exit(17)
        dfc.ack_train_refs("trainer", ids, global_step=step, optimizer_durable=True)

    ctrl = _controller(
        os.path.join(work, "output"),
        on_forward=on_forward,
        max_steps=4,
        save_interval=2,
        ack_fn=ack_fn,
        async_ack=True,
    )
    ctrl.fit([_batch(f"s{i}") for i in range(1, 5)])
    os._exit(0)


class TestCrashMidAsyncAck(unittest.TestCase):
    def test_hard_crash_mid_ack_resumes_from_the_flushed_checkpoint(self):
        with tempfile.TemporaryDirectory(prefix="async_ack_crash_") as work:
            process = multiprocessing.get_context("spawn").Process(
                target=_crash_mid_ack_worker, args=(work,)
            )
            process.start()
            process.join(60)
            if process.is_alive():
                process.terminate()
                process.join(5)
                self.fail("crash worker hung")
            self.assertEqual(process.exitcode, 17)

            output = os.path.join(work, "output")
            checkpoints = sorted(
                name for name in os.listdir(output) if name.startswith("run0-step")
            )
            # Step 2's checkpoint was written only after its ack; step 4 never
            # got past its boundary flush of the dying step-3 ack.
            self.assertEqual(checkpoints, ["run0-step2"])
            with sqlite3.connect(os.path.join(work, "ledger.sqlite")) as connection:
                marker = dict(connection.execute("SELECT k, v FROM marker").fetchall())
            self.assertEqual(json.loads(marker["global_step"]), 2)

            # The canonical consumer resume accepts the pair (marker == ckpt),
            # releases the durable prefix, and requeues the un-acked tail.
            checkpoint = os.path.join(output, "run0-step2")
            self.assertTrue(os.path.isfile(os.path.join(checkpoint, STATE_FILE)))
            captured, distributor, features = _build_consumer(
                work, resume_from=checkpoint
            )
            self.assertEqual(distributor.kwargs["skip_ids"], {"s1", "s2"})
            self.assertEqual(distributor.kwargs["requeued_ids"], {"s3", "s4"})
            self.assertEqual([item[0] for item in features.aborted], ["s1", "s2"])

            # Finish the run from the checkpoint over the replayed refs.
            controller = captured["controller"]
            replay = sorted(
                ref.sample_id for ref in controller.sample_queue.get(10, timeout_s=0)
            )
            self.assertEqual(replay, ["s3", "s4"])
            resumed = _controller(
                output,
                start_step=2,
                max_steps=4,
                ack_fn=lambda ids, step: controller.ack_train_refs(
                    "trainer", ids, global_step=step, optimizer_durable=True
                ),
                async_ack=True,
            )
            self.assertEqual(resumed.fit([_batch(sid) for sid in replay]), 4)
            marker = controller.store.durable_marker()
            self.assertEqual(marker["global_step"], 4)
            self.assertEqual(marker["acked"], {"s1", "s2", "s3", "s4"})
            controller.store.close()


def _write_result(results_dir: str, rank: int, payload: dict) -> None:
    with open(os.path.join(results_dir, f"rank{rank}.json"), "w") as handle:
        json.dump(payload, handle)


def _two_rank_async_ack_worker(
    rank: int,
    init_method: str,
    work: str,
    fail_cleanup_id: str | None,
    results_dir: str,
) -> None:
    """Async DP acks on the Gloo ack group vs. training-thread collectives."""
    payload = {}
    initialized = False
    store = None
    try:
        dist.init_process_group(
            "gloo",
            init_method=init_method,
            rank=rank,
            world_size=WORLD_SIZE,
            timeout=timedelta(seconds=30),
        )
        initialized = True
        ack_group = new_durable_ack_process_group()
        store = (
            SQLiteMetadataStore(os.path.join(work, "ledger.sqlite"))
            if rank == 0
            else InMemoryMetadataStore()
        )

        class Features(_RetainingFeatureStore):
            def abort(self, sample_id, *, reason="aborted"):
                super().abort(sample_id, reason=reason)
                if sample_id == fail_cleanup_id:
                    raise OSError(f"injected remove failure for {sample_id}")

        features = Features()
        controller = DPAckController(
            "run0",
            is_authority=rank == 0,
            metadata_store=store,
            feature_store=features,
            process_group=ack_group,
        )
        jitter = random.Random(rank)
        acked_steps = []

        def ack_fn(ids, step):
            # Desynchronize the two ranks' ack threads from their training
            # threads and from each other.
            time.sleep(jitter.uniform(0.0, 0.03))
            controller.ack_train_refs(
                f"trainer-{rank}", ids, global_step=step, optimizer_durable=True
            )
            acked_steps.append(step)

        def training_collective():
            # Stands in for the NCCL gradient/metric collectives the training
            # thread issues on the default group while an ack is in flight.
            value = torch.ones(1)
            dist.all_reduce(value)
            if value.item() != WORLD_SIZE:
                raise AssertionError(f"default-group all_reduce gave {value}")

        manager = _RecordingCheckpointManager(
            os.path.join(work, "output"),
            "run0",
            lambda: store.durable_marker()["global_step"],
        )
        ctrl = _controller(
            os.path.join(work, "output"),
            on_step=training_collective,
            max_steps=8,
            save_interval=3,
            ack_fn=ack_fn,
            async_ack=True,
            checkpoint_manager=manager,
        )
        error = None
        try:
            ctrl.fit([_batch(f"r{rank}-s{i}") for i in range(1, 9)])
        except RuntimeError as exc:
            error = str(exc)
        marker = store.durable_marker()
        payload = {
            "error": error,
            "global_step": ctrl.global_step,
            "acked_steps": acked_steps,
            "saves": manager.saves,
            "marker_step": marker["global_step"],
            "marker_acked": sorted(marker["acked"]),
            "aborted": [item[0] for item in features.aborted],
        }
        dist.barrier()
    except BaseException as exc:
        payload = {
            "worker_error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
        }
    finally:
        _write_result(results_dir, rank, payload)
        if store is not None and hasattr(store, "close"):
            store.close()
        if initialized:
            dist.destroy_process_group()


@unittest.skipUnless(_GLOO_AVAILABLE, "requires torch.distributed with Gloo")
class TestTwoRankAsyncAck(unittest.TestCase):
    def _spawn_two_ranks(self, *args, timeout_s: float = 90.0) -> list:
        context = multiprocessing.get_context("spawn")
        processes = [
            context.Process(target=_two_rank_async_ack_worker, args=(rank, *args))
            for rank in range(WORLD_SIZE)
        ]
        for process in processes:
            process.start()
        deadline = time.monotonic() + timeout_s
        for process in processes:
            process.join(max(0.0, deadline - time.monotonic()))
        stuck = [rank for rank, process in enumerate(processes) if process.is_alive()]
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(5)
        self.assertFalse(stuck, f"async ack deadlocked on ranks {stuck}")
        self.assertEqual([process.exitcode for process in processes], [0, 0])
        results = []
        for rank in range(WORLD_SIZE):
            with open(os.path.join(args[-1], f"rank{rank}.json")) as handle:
                result = json.load(handle)
            self.assertNotIn("worker_error", result, result.get("traceback"))
            results.append(result)
        return results

    @staticmethod
    def _init_method(work: str) -> str:
        return Path(os.path.join(work, "gloo-init")).resolve().as_uri()

    def test_ordered_acks_overlap_training_collectives(self):
        with tempfile.TemporaryDirectory(prefix="async_ack_2rank_") as work:
            results = self._spawn_two_ranks(self._init_method(work), work, None, work)

        for rank, result in enumerate(results):
            self.assertIsNone(result["error"])
            self.assertEqual(result["global_step"], 8)
            self.assertEqual(result["acked_steps"], list(range(1, 9)))
            self.assertEqual(result["aborted"], [f"r{rank}-s{i}" for i in range(1, 9)])
        # Only the authority records the union, and every checkpoint it wrote
        # saw its own step already durable.
        self.assertEqual(results[0]["marker_step"], 8)
        self.assertEqual(
            results[0]["marker_acked"],
            sorted(f"r{rank}-s{i}" for rank in range(2) for i in range(1, 9)),
        )
        self.assertEqual(results[0]["saves"], [[3, 3], [6, 6]])
        self.assertIsNone(results[1]["marker_step"])

    def test_rank_local_cleanup_failure_reaches_both_training_threads(self):
        with tempfile.TemporaryDirectory(prefix="async_ack_2rank_fail_") as work:
            results = self._spawn_two_ranks(
                self._init_method(work), work, "r1-s2", work
            )

        errors = [result["error"] for result in results]
        self.assertIsNotNone(errors[0])
        self.assertEqual(errors[0], errors[1])
        self.assertIn("rank-local feature cleanup failed", errors[0])
        self.assertIn("injected remove failure for r1-s2", errors[0])
        for result in results:
            # Both ranks stop at step 3's flush: no step-3 ack, no checkpoint.
            self.assertEqual(result["global_step"], 3)
            self.assertEqual(result["acked_steps"], [1])
            self.assertEqual(result["saves"], [])
        # The authority commit precedes cleanup, so step 2 is durable.
        self.assertEqual(results[0]["marker_step"], 2)


if __name__ == "__main__":
    unittest.main()
