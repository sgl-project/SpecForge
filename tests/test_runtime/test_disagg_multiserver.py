# coding=utf-8
"""Multi-server producer fan-out (no GPU, no server, no mooncake master).

The multi-server topology: ``build_disagg_online_producer(feature_source=[...])``
builds one RolloutWorker per SGLangServerCaptureAdapter (1 server : 1 adapter :
1 worker), all leasing DISJOINT prompts from the one controller and publishing
into the one channel, concurrently. A logical worker can continuously maintain
multiple capture calls without duplicating worker/controller state. Each stub
``post_fn`` stands in for one patched SGLang server writing into the shared fake
Mooncake backend — the same topology configured through ``specforge train``
producer and consumer roles.

Covers the failure matrix the single-server path never hits:
- disjoint + complete production across two live servers;
- one dead server: its leases fail retryable, the survivor re-leases and
  finishes the pool (no truncation, no hang);
- all servers dead: loud RuntimeError and a failure sentinel;
- a poisoned prompt (server rejects it every time): terminal failure after
  ``max_prompt_attempts`` instead of a partial-success EOF;
- continuous prompt ingestion: capture calls never drain at an ingest-chunk or
  epoch boundary, while the dispatched plan, stop/restart replay, backpressure
  and failure paths stay as before.
"""

import os
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.inference.adapters.server_capture import (
    ServerCaptureSchema,
    SGLangServerCaptureAdapter,
)
from specforge.launch import _epoch_online_prompts, build_disagg_online_producer
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.test_server_capture import (
    AUX_LAYERS,
    HIDDEN,
    _FakeMooncakeStore,
    _StubCaptureServer,
)

ALGORITHM = builtin_algorithm_registry().resolve("dflash")


def _SLEEP(_s):  # injected drive sleep: keep retry/backpressure spins bounded
    time.sleep(0.001)


def _prompts(n, seq=6):
    return [
        {
            "task_id": f"p{i}",
            "payload": {
                "input_ids": list(range(1, seq + 1 + i)),
                "loss_mask": [1] * (seq + i),
            },
        }
        for i in range(n)
    ]


def _adapter(store, post_fn, url="http://server:30000"):
    layout = ALGORITHM.providers.server_streaming_for("text").layout
    return SGLangServerCaptureAdapter(
        url,
        store,
        run_id="run0",
        algorithm=ALGORITHM.name,
        schema=ServerCaptureSchema(
            aux_feature=layout.aux_feature,
            last_hidden_feature=layout.last_hidden_feature,
            passthrough=layout.passthrough,
            attention_mask_feature=layout.attention_mask_feature,
        ),
        post_fn=post_fn,
    )


def _build(adapters, prompts, store, channel, **kw):
    channel.publish_consumer_quantum(kw.pop("consumer_quantum", 1))
    return build_disagg_online_producer(
        algorithm=ALGORITHM,
        feature_source=adapters,
        prompts=prompts,
        feature_store=store,
        channel=channel,
        run_id="run0",
        target_hidden_size=HIDDEN,
        target_repr=None,
        aux_hidden_state_layer_ids=AUX_LAYERS,
        sleep=_SLEEP,
        **kw,
    )


def _published_refs(path):
    return StreamingRefChannel(path).poll()


def _published_sample_ids(path):
    return [r.sample_id for r in _published_refs(path)]


class _FailingPublishChannel(StreamingRefChannel):
    def __init__(self, path, *, fail_after):
        super().__init__(path)
        self.fail_after = fail_after

    def publish(self, ref):
        if self.published >= self.fail_after:
            raise OSError(
                f"injected publish failure after {self.published} durable ref(s)"
            )
        super().publish(ref)


class _FsyncFailingPublishChannel(StreamingRefChannel):
    def publish(self, ref):
        with patch(
            "specforge.runtime.data_plane.streaming_ref_channel.os.fsync",
            side_effect=OSError("injected fsync failure"),
        ):
            super().publish(ref)


class _TrackingMooncakeFeatureStore(MooncakeFeatureStore):
    def __init__(self, *args, abort_failures=(), **kwargs):
        super().__init__(*args, **kwargs)
        self.abort_calls = []
        self.abort_failures = set(abort_failures)

    def abort(self, sample_id, *, reason="aborted"):
        self.abort_calls.append((sample_id, reason))
        if sample_id in self.abort_failures:
            raise RuntimeError(f"injected abort failure for {sample_id}")
        return super().abort(sample_id, reason=reason)


class TestMultiServerProducer(unittest.TestCase):
    def _workdir(self):
        return tempfile.mkdtemp(prefix="disagg_multisrv_")

    def test_single_worker_refills_capture_slots_continuously(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        call_lock = threading.Lock()
        server_lock = threading.Lock()
        first_call_started = threading.Event()
        release_first_call = threading.Event()
        refilled_while_first_active = threading.Event()
        calls = 0
        active_calls = 0
        max_active_calls = 0

        def controlled_post(url, json_body, timeout):
            nonlocal calls, active_calls, max_active_calls
            with call_lock:
                call_index = calls
                calls += 1
                active_calls += 1
                max_active_calls = max(max_active_calls, active_calls)
            try:
                if call_index == 0:
                    first_call_started.set()
                    if not release_first_call.wait(5):
                        raise TimeoutError("test did not release first capture call")
                elif call_index >= 2:
                    refilled_while_first_active.set()
                # The fake sink uses process-global torch RNG state; serialize
                # only that test implementation, not the request lifecycle.
                with server_lock:
                    return stub(url, json_body, timeout)
            finally:
                with call_lock:
                    active_calls -= 1

        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(
            [_adapter(store, controlled_post)],
            _prompts(6),
            store,
            channel,
            lease=1,
            producer_concurrency=2,
        )
        outcome = {}

        def run_producer():
            try:
                outcome["produced"] = drive()
            except BaseException as exc:
                outcome["error"] = exc

        thread = threading.Thread(target=run_producer, daemon=True)
        thread.start()
        try:
            self.assertTrue(first_call_started.wait(5))
            self.assertTrue(
                refilled_while_first_active.wait(5),
                "producer waited for the slow call instead of refilling its free slot",
            )
            self.assertEqual(workers[0].health()["in_flight"], 2)
        finally:
            release_first_call.set()
        thread.join(10)

        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome.get("produced"), 6)
        self.assertEqual(len(workers), 1)
        self.assertEqual(max_active_calls, 2)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].health()["committed"], 6)
        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), 6)
        self.assertEqual(len(set(ids)), 6)

    def test_two_servers_disjoint_and_complete(self):
        backend = _FakeMooncakeStore()
        stubs = [_StubCaptureServer(backend), _StubCaptureServer(backend)]
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        adapters = [
            _adapter(store, stubs[i], url=f"http://server{i}:3000{i}") for i in range(2)
        ]
        N = 12
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(adapters, _prompts(N), store, channel, lease=2)
        self.assertEqual(len(workers), 2)

        produced = drive()
        self.assertEqual(produced, N)
        self.assertTrue(channel.is_closed())
        # A disaggregated producer publishes refs directly to the channel. It
        # must not retain a second local training queue/ledger.
        self.assertIsNone(workers[0].controller.sample_queue)

        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), N)
        self.assertEqual(len(set(ids)), N)  # no duplicate publishes

        seen0, seen1 = set(stubs[0].expected), set(stubs[1].expected)
        self.assertEqual(seen0 | seen1, {f"run0:p{i}" for i in range(N)})
        self.assertEqual(seen0 & seen1, set())  # disjoint prompt slices
        # concurrency is real: both servers captured (lease=2 over 12 prompts
        # leaves plenty for the peer even in the worst interleaving)
        self.assertTrue(seen0 and seen1)

        # ref-level provenance mirrors the stub-level split (the post-hoc
        # audit trail a real multi-server run relies on)
        by_server = {}
        for r in _published_refs(channel.path):
            by_server.setdefault(r.metadata["server"], set()).add(r.sample_id)
        self.assertEqual(by_server.get("http://server0:30000"), seen0)
        self.assertEqual(by_server.get("http://server1:30001"), seen1)

    def test_watermark_must_cover_one_consumer_optimizer_window(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(4),
            store,
            channel,
            consumer_quantum=4,
            in_flight_high_watermark=2,
        )

        with self.assertRaisesRegex(ValueError, "optimizer-step quantum 4"):
            drive()
        self.assertEqual(channel.published, 0)
        self.assertIn("high watermark", channel.failure())

    def test_byte_watermark_resumes_after_durable_consumer_ack(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(3),
            store,
            channel,
            lease=1,
            resident_high_watermark_bytes=1,
            resident_low_watermark_bytes=0,
        )

        outcome = {}

        def run_producer():
            try:
                outcome["produced"] = drive()
            except BaseException as exc:  # expose a thread failure to the test
                outcome["error"] = exc

        thread = threading.Thread(target=run_producer, daemon=True)
        thread.start()
        reader = StreamingRefChannel(channel.path)
        consumed = 0
        observed_pauses = []
        deadline = time.monotonic() + 10
        while consumed < 3 and time.monotonic() < deadline:
            refs = reader.poll()
            if refs:
                pause_deadline = time.monotonic() + 2
                while time.monotonic() < pause_deadline:
                    snapshot = drive.flow_control.snapshot(
                        in_flight_refs=channel.in_flight_remote(),
                        resident_bytes=sum(ref.estimated_bytes for ref in refs),
                    )
                    if snapshot["paused"]:
                        break
                    time.sleep(0.001)
                observed_pause = snapshot["paused"]
                # RefDistributor forwards this source-channel acknowledgement
                # only after the inbox ack follows the durable optimizer marker.
                reader.mark_consumed(len(refs))
                consumed += len(refs)
                observed_pauses.append(observed_pause)
            else:
                time.sleep(0.001)
        thread.join(5)

        self.assertFalse(thread.is_alive(), "producer stayed byte-throttled")
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome.get("produced"), 3)
        self.assertEqual(consumed, 3)
        self.assertEqual(observed_pauses, [True, True, True])
        self.assertTrue(channel.is_closed())
        snapshot = drive.flow_control.snapshot(
            in_flight_refs=channel.in_flight_remote(), resident_bytes=0
        )
        self.assertGreaterEqual(snapshot["pause_transitions"], 1)
        self.assertGreaterEqual(snapshot["resume_transitions"], 1)

    def test_byte_watermark_cannot_block_the_first_optimizer_window(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(3),
            store,
            channel,
            consumer_quantum=3,
            lease=1,
            resident_high_watermark_bytes=1,
            resident_low_watermark_bytes=0,
        )

        outcome = {}

        def run_producer():
            try:
                outcome["produced"] = drive()
            except BaseException as exc:  # expose a thread failure to the test
                outcome["error"] = exc

        thread = threading.Thread(target=run_producer, daemon=True)
        thread.start()
        deadline = time.monotonic() + 2
        while channel.published < 3 and time.monotonic() < deadline:
            time.sleep(0.001)
        published_before_ack = channel.published

        # Keep cleanup bounded if this invariant regresses and the producer
        # pauses before publishing a complete window.
        reader = StreamingRefChannel(channel.path)
        cleanup_deadline = time.monotonic() + 2
        while thread.is_alive() and time.monotonic() < cleanup_deadline:
            refs = reader.poll()
            if refs:
                reader.mark_consumed(len(refs))
            time.sleep(0.001)
        thread.join(2)

        self.assertEqual(published_before_ack, 3)
        self.assertFalse(thread.is_alive())
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome.get("produced"), 3)

    def test_hard_byte_cap_aborts_unpublished_capture_and_fails_channel(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(1),
            store,
            channel,
            lease=1,
            feature_store_max_resident_bytes=1,
        )

        with self.assertRaisesRegex(MemoryError, "hard cap exceeded"):
            drive()

        self.assertEqual(channel.published, 0)
        self.assertEqual(_published_refs(channel.path), [])
        self.assertFalse(channel.is_closed())
        self.assertIn("MemoryError", channel.failure())
        self.assertEqual(backend._d, {})

    def test_publish_failure_before_first_ref_aborts_the_whole_captured_batch(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = _TrackingMooncakeFeatureStore(store=backend, store_id="run0")
        channel = _FailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl"), fail_after=0
        )
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(3),
            store,
            channel,
            lease=3,
            prompt_seed=5,
        )

        with self.assertRaisesRegex(OSError, "after 0 durable ref"):
            drive()

        self.assertEqual(channel.published, 0)
        self.assertEqual(_published_sample_ids(channel.path), [])
        self.assertEqual(
            store.abort_calls,
            [
                ("run0:p0", "producer-ref-publication-failed"),
                ("run0:p1", "producer-ref-publication-failed"),
                ("run0:p2", "producer-ref-publication-failed"),
            ],
        )
        self.assertEqual(backend._d, {})
        self.assertIn("OSError", channel.failure())

    def test_publish_failure_aborts_other_concurrent_capture_results(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = _TrackingMooncakeFeatureStore(store=backend, store_id="run0")
        channel = _FailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl"), fail_after=0
        )
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(4),
            store,
            channel,
            lease=1,
            producer_concurrency=2,
        )

        with self.assertRaisesRegex(OSError, "after 0 durable ref"):
            drive()

        self.assertEqual(channel.published, 0)
        self.assertEqual(len(store.abort_calls), 2)
        self.assertEqual(
            {reason for _sample_id, reason in store.abort_calls},
            {
                "producer-ref-publication-failed",
                "producer-driver-failed-before-publication",
            },
        )
        self.assertEqual(backend._d, {})

    def test_partial_publish_aborts_only_the_non_durable_suffix(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = _TrackingMooncakeFeatureStore(store=backend, store_id="run0")
        channel = _FailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl"), fail_after=1
        )
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(3),
            store,
            channel,
            lease=3,
            prompt_seed=5,
        )

        with self.assertRaisesRegex(OSError, "after 1 durable ref"):
            drive()

        self.assertEqual(_published_sample_ids(channel.path), ["run0:p0"])
        self.assertEqual(
            store.abort_calls,
            [
                ("run0:p1", "producer-ref-publication-failed"),
                ("run0:p2", "producer-ref-publication-failed"),
            ],
        )
        self.assertEqual(store.health()["resident_samples"], 1)
        self.assertTrue(backend._d)
        self.assertTrue(all("/run0:p0/" in key for key in backend._d))

    def test_fsync_failure_preserves_the_possibly_visible_ref(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = _TrackingMooncakeFeatureStore(store=backend, store_id="run0")
        channel = _FsyncFailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl")
        )
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(3),
            store,
            channel,
            lease=3,
            prompt_seed=5,
        )

        with self.assertRaisesRegex(OSError, "injected fsync failure"):
            drive()

        self.assertEqual(_published_sample_ids(channel.path), ["run0:p0"])
        self.assertEqual(
            store.abort_calls,
            [
                ("run0:p1", "producer-ref-publication-failed"),
                ("run0:p2", "producer-ref-publication-failed"),
            ],
        )
        self.assertEqual(store.health()["resident_samples"], 1)
        self.assertTrue(all("/run0:p0/" in key for key in backend._d))

    def test_publish_cleanup_failure_keeps_the_primary_error_as_the_cause(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = _TrackingMooncakeFeatureStore(
            store=backend,
            store_id="run0",
            abort_failures={"run0:p1"},
        )
        channel = _FailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl"), fail_after=0
        )
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(2),
            store,
            channel,
            lease=2,
            prompt_seed=5,
        )

        with self.assertRaisesRegex(
            RuntimeError,
            "reference publication failed.*cleanup.*run0:p1",
        ) as raised:
            drive()

        self.assertIsInstance(raised.exception.__cause__, OSError)
        self.assertEqual(
            [sample_id for sample_id, _reason in store.abort_calls],
            ["run0:p0", "run0:p1"],
        )
        self.assertIn("injected publish failure", str(raised.exception))
        self.assertIn("injected abort failure", str(raised.exception))
        self.assertIn("cleanup", channel.failure())

    def test_prompt_epochs_republish_with_unique_sample_ids(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        N, E = 3, 2
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=2,
            prompt_epochs=E,
        )

        produced = drive()
        self.assertEqual(produced, N * E)
        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), N * E)
        self.assertEqual(len(set(ids)), N * E)
        self.assertEqual(
            set(ids),
            {
                f"run0:epoch{epoch:04d}-prompt{idx:012d}"
                for epoch in range(E)
                for idx in range(N)
            },
        )
        self.assertTrue(channel.is_closed())

    def test_prompt_ingest_chunks_preserve_epoch_ids_and_release_payloads(self):
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        N, E = 5, 2
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=2,
            prompt_epochs=E,
            prompt_ingest_batch_size=2,
        )

        produced = drive()

        self.assertEqual(produced, N * E)
        self.assertEqual(workers[0].controller.status()["prompts"], 0)
        self.assertEqual(
            set(_published_sample_ids(channel.path)),
            {
                f"run0:epoch{epoch:04d}-prompt{idx:012d}"
                for epoch in range(E)
                for idx in range(N)
            },
        )

    def test_prompt_epoch_order_is_seeded_and_reconstruction_stable(self):
        prompts = _prompts(12)

        single_epoch = _epoch_online_prompts(prompts, 0, 1, seed=42)
        rebuilt_single_epoch = _epoch_online_prompts(prompts, 0, 1, seed=42)
        other_seed = _epoch_online_prompts(prompts, 0, 1, seed=43)
        epoch_zero = _epoch_online_prompts(prompts, 0, 3, seed=42)
        epoch_one = _epoch_online_prompts(prompts, 1, 3, seed=42)
        rebuilt_epoch_one = _epoch_online_prompts(prompts, 1, 3, seed=42)

        self.assertEqual(single_epoch, rebuilt_single_epoch)
        self.assertNotEqual(
            [item["task_id"] for item in single_epoch],
            [item["task_id"] for item in other_seed],
        )
        zero_order = [item["metadata"]["prompt_index"] for item in epoch_zero]
        one_order = [item["metadata"]["prompt_index"] for item in epoch_one]
        self.assertNotEqual(zero_order, one_order)
        self.assertEqual(epoch_one, rebuilt_epoch_one)
        self.assertEqual(
            {item["task_id"] for item in epoch_one},
            {f"epoch0001-prompt{idx:012d}" for idx in range(len(prompts))},
        )
        self.assertEqual(
            [item["task_id"] for item in epoch_one[5:]],
            [item["task_id"] for item in rebuilt_epoch_one[5:]],
        )

    def test_one_dead_server_survivor_completes_pool(self):
        backend = _FakeMooncakeStore()
        healthy = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")

        dead_leased = threading.Event()

        def dead_post(url, json_body, timeout):
            dead_leased.set()  # it held leases when it died
            raise ConnectionError("server 1 unreachable")

        def healthy_post(url, json_body, timeout):
            dead_leased.wait(5)  # let the dead server lease + fail first
            return healthy(url, json_body, timeout)

        adapters = [
            _adapter(store, healthy_post, url="http://server0:30000"),
            _adapter(store, dead_post, url="http://server1:30001"),
        ]
        N = 8
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(adapters, _prompts(N), store, channel, lease=2)

        produced = drive(max_rounds=10_000)
        # the dead worker's leases were failed retryable and re-leased by the
        # survivor: every prompt still becomes exactly one ref.
        self.assertEqual(produced, N)
        self.assertTrue(dead_leased.is_set())
        self.assertEqual(set(healthy.expected), {f"run0:p{i}" for i in range(N)})
        ids = _published_sample_ids(channel.path)
        self.assertEqual(sorted(ids), sorted(set(ids)))
        self.assertTrue(channel.is_closed())
        self.assertTrue(workers[1].health()["recent_failures"])

    def test_all_servers_dead_raises_loudly(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")

        def dead_post(url, json_body, timeout):
            raise ConnectionError("pool down")

        adapters = [
            _adapter(store, dead_post, url=f"http://server{i}:3000{i}")
            for i in range(2)
        ]
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            adapters, _prompts(4), store, channel, lease=2, max_worker_failures=2
        )
        with self.assertRaises(RuntimeError):
            drive(max_rounds=10_000)
        self.assertFalse(channel.is_closed())
        self.assertIn("RuntimeError", channel.failure())

    def test_poisoned_prompt_goes_terminal_not_infinite(self):
        # PR654 known rough edge: an all-failed round used to read as
        # pool-drained (silent truncation); with drained = pending==0 AND
        # leased==0 the loop instead retries — so a prompt the server rejects
        # every time must go terminal via max_prompt_attempts, not spin.
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend, error_sample_ids={"run0:p0"})
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        N = 5
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=2,
            max_prompt_attempts=3,
        )
        with self.assertRaisesRegex(RuntimeError, "terminally failed prompt"):
            drive(max_rounds=10_000)
        ids = _published_sample_ids(channel.path)
        self.assertEqual(set(ids), {f"run0:p{i}" for i in range(1, N)})
        self.assertFalse(channel.is_closed())
        self.assertIn("terminally failed prompt", channel.failure())

    def test_source_count_worker_count_conflict_raises(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        stub = _StubCaptureServer(backend)
        adapters = [_adapter(store, stub), _adapter(store, stub)]
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        with self.assertRaises(ValueError):
            _build(adapters, _prompts(2), store, channel, num_rollout_workers=3)

    def test_producer_concurrency_must_be_positive(self):
        backend = _FakeMooncakeStore()
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        stub = _StubCaptureServer(backend)
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))

        with self.assertRaisesRegex(ValueError, "producer_concurrency"):
            _build(
                [_adapter(store, stub)],
                _prompts(2),
                store,
                channel,
                producer_concurrency=0,
            )


def _plan_task_ids(prompts, epochs, seed):
    """The chunked producer's FIFO dispatch plan (epoch order, then shuffle)."""
    return [
        item["task_id"]
        for epoch in range(epochs)
        for item in _epoch_online_prompts(prompts, epoch, epochs, seed=seed)
    ]


def _record_leases(controller):
    """Record the controller's global lease order (serialized for the test)."""
    leased = []
    lock = threading.Lock()
    lease_prompt_tasks = controller.lease_prompt_tasks

    def recording_lease(worker_id, max_tasks):
        with lock:
            tasks = lease_prompt_tasks(worker_id, max_tasks)
            leased.extend(task.task_id for task in tasks)
        return tasks

    controller.lease_prompt_tasks = recording_lease
    return leased


class _OverlapCheckingServer:
    """Stub server proving capture calls overlap across the whole prompt plan.

    Every call except the one carrying the plan's final prompt holds until a
    successor call has started, so at least one capture call is always in
    flight. A producer that drains its capture slots at an ingest-chunk or
    epoch boundary leaves such a call without a successor; that is recorded
    as a gap after ``wait_s`` instead of hanging the test.
    """

    def __init__(self, stub, total_prompts, *, wait_s=2.0):
        self.stub = stub
        self.total_prompts = total_prompts
        self.wait_s = wait_s
        self.status_probe = None
        self.max_resident_prompts = 0
        self.dispatched = []
        self.gaps = []
        self._calls = 0
        self._cond = threading.Condition()
        self._stub_lock = threading.Lock()

    def __call__(self, url, json_body, timeout):
        with self._cond:
            index = self._calls
            self._calls += 1
            self.dispatched.extend(
                spec["sample_id"] for spec in json_body["spec_capture"]
            )
            if self.status_probe is not None:
                self.max_resident_prompts = max(
                    self.max_resident_prompts, self.status_probe()
                )
            self._cond.notify_all()
            final = len(self.dispatched) >= self.total_prompts
            if not final and not self._cond.wait_for(
                lambda: self._calls > index + 1, timeout=self.wait_s
            ):
                self.gaps.append(len(self.dispatched))
        # The fake sink uses process-global torch RNG state.
        with self._stub_lock:
            return self.stub(url, json_body, timeout)


class TestContinuousPromptIngest(unittest.TestCase):
    """The prompt feeder keeps every capture server busy across chunk and
    epoch boundaries without changing the dispatched plan."""

    def _workdir(self):
        return tempfile.mkdtemp(prefix="disagg_ingest_")

    def _refill_threshold(self, workers, concurrency, lease):
        from specforge.launch import _ONLINE_PROMPT_REFILL_ROUNDS

        return _ONLINE_PROMPT_REFILL_ROUNDS * workers * concurrency * lease

    def test_capture_calls_never_drain_between_ingest_chunks_or_epochs(self):
        N, E, chunk, lease = 11, 2, 3, 1
        for servers, concurrency in ((1, 2), (2, 2)):
            with self.subTest(servers=servers, concurrency=concurrency):
                backend = _FakeMooncakeStore()
                store = MooncakeFeatureStore(store=backend, store_id="run0")
                server = _OverlapCheckingServer(_StubCaptureServer(backend), N * E)
                adapters = [
                    _adapter(store, server, url=f"http://server{i}:3000{i}")
                    for i in range(servers)
                ]
                channel = StreamingRefChannel(
                    os.path.join(self._workdir(), "refs.jsonl")
                )
                workers, drive = _build(
                    adapters,
                    _prompts(N),
                    store,
                    channel,
                    lease=lease,
                    producer_concurrency=concurrency,
                    prompt_epochs=E,
                    prompt_seed=3,
                    prompt_ingest_batch_size=chunk,
                )
                controller = workers[0].controller
                server.status_probe = lambda: controller.status()["prompts"]

                produced = drive()

                self.assertEqual(produced, N * E)
                # Every call but the last overlapped a later call: the number
                # of in-flight captures never reached zero before the final
                # prompt, at chunk boundaries (every 3 prompts) and at the
                # epoch boundary alike.
                self.assertEqual(server.gaps, [])
                plan = _plan_task_ids(_prompts(N), E, seed=3)
                self.assertEqual(
                    sorted(server.dispatched),
                    sorted(f"run0:{task_id}" for task_id in plan),
                )
                # Continuous ingestion still bounds normalized residency.
                threshold = self._refill_threshold(servers, concurrency, lease)
                self.assertLessEqual(
                    server.max_resident_prompts,
                    threshold + chunk + servers * concurrency * lease,
                )
                self.assertEqual(controller.status()["prompts"], 0)
                self.assertTrue(channel.is_closed())

    def test_dispatch_order_matches_the_chunked_epoch_plan(self):
        N, seed = 10, 7
        cases = (
            # servers, concurrency, lease, chunk, epochs
            (1, 1, 1, 3, 3),
            (2, 2, 2, 3, 3),
            (3, 2, 1, 4, 2),
            (2, 1, 2, 4096, 3),
            (2, 2, 1, 3, 1),
        )
        for servers, concurrency, lease, chunk, epochs in cases:
            with self.subTest(
                servers=servers,
                concurrency=concurrency,
                lease=lease,
                chunk=chunk,
                epochs=epochs,
            ):
                backend = _FakeMooncakeStore()
                stub = _StubCaptureServer(backend)
                stub_lock = threading.Lock()

                def serialized_post(url, json_body, timeout):
                    with stub_lock:
                        return stub(url, json_body, timeout)

                store = MooncakeFeatureStore(store=backend, store_id="run0")
                adapters = [
                    _adapter(store, serialized_post, url=f"http://server{i}:3000{i}")
                    for i in range(servers)
                ]
                channel = StreamingRefChannel(
                    os.path.join(self._workdir(), "refs.jsonl")
                )
                workers, drive = _build(
                    adapters,
                    _prompts(N),
                    store,
                    channel,
                    lease=lease,
                    producer_concurrency=concurrency,
                    prompt_epochs=epochs,
                    prompt_seed=seed,
                    prompt_ingest_batch_size=chunk,
                )
                leased = _record_leases(workers[0].controller)

                produced = drive()

                plan = _plan_task_ids(_prompts(N), epochs, seed=seed)
                self.assertEqual(produced, len(plan))
                # Global lease order is the old chunk-by-chunk FIFO plan:
                # same task ids, same epoch identity, same per-epoch shuffle.
                self.assertEqual(leased, plan)
                for epoch in range(epochs):
                    epoch_plan = [
                        item["task_id"]
                        for item in _epoch_online_prompts(
                            _prompts(N), epoch, epochs, seed=seed
                        )
                    ]
                    epoch_ids = set(epoch_plan)
                    self.assertEqual(
                        [task_id for task_id in leased if task_id in epoch_ids],
                        epoch_plan,
                    )
                ids = _published_sample_ids(channel.path)
                self.assertEqual(
                    sorted(ids), sorted(f"run0:{task_id}" for task_id in plan)
                )
                self.assertEqual(len(ids), len(set(ids)))
                self.assertEqual(workers[0].controller.status()["prompts"], 0)
                self.assertTrue(channel.is_closed())

    def test_dead_server_failover_across_chunks_keeps_the_plan_multiset(self):
        backend = _FakeMooncakeStore()
        healthy = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        dead_leased = threading.Event()

        def dead_post(url, json_body, timeout):
            dead_leased.set()
            raise ConnectionError("server 1 unreachable")

        def healthy_post(url, json_body, timeout):
            dead_leased.wait(5)  # the dead server holds leases first
            return healthy(url, json_body, timeout)

        N, E = 10, 2
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        _workers, drive = _build(
            [
                _adapter(store, healthy_post, url="http://server0:30000"),
                _adapter(store, dead_post, url="http://server1:30001"),
            ],
            _prompts(N),
            store,
            channel,
            lease=2,
            prompt_epochs=E,
            prompt_seed=1,
            prompt_ingest_batch_size=3,
        )

        produced = drive(max_rounds=10_000)

        plan = _plan_task_ids(_prompts(N), E, seed=1)
        self.assertEqual(produced, N * E)
        self.assertTrue(dead_leased.is_set())
        ids = _published_sample_ids(channel.path)
        self.assertEqual(sorted(ids), sorted(f"run0:{task_id}" for task_id in plan))
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(channel.is_closed())

    def test_stop_mid_plan_then_restart_replays_the_identical_plan(self):
        N, E, chunk, seed, stop_after = 11, 2, 3, 5, 13
        plan = _plan_task_ids(_prompts(N), E, seed=seed)

        def attempt(should_stop=None):
            backend = _FakeMooncakeStore()
            store = MooncakeFeatureStore(store=backend, store_id="run0")
            channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
            workers, drive = _build(
                [_adapter(store, _StubCaptureServer(backend))],
                _prompts(N),
                store,
                channel,
                lease=1,
                prompt_epochs=E,
                prompt_seed=seed,
                prompt_ingest_batch_size=chunk,
            )
            leased = _record_leases(workers[0].controller)
            produced = drive(
                should_stop=None if should_stop is None else should_stop(channel)
            )
            return produced, leased, channel, workers[0].controller

        # A cooperative stop (consumer finished/stopped) past a chunk and an
        # epoch boundary: publication stops cleanly at a plan prefix and the
        # feeder stops ingesting instead of pulling in the rest of the plan.
        produced, leased, channel, controller = attempt(
            lambda ch: lambda: ch.published >= stop_after
        )
        self.assertEqual(produced, stop_after)
        first_ids = _published_sample_ids(channel.path)
        self.assertEqual(
            first_ids, [f"run0:{task_id}" for task_id in plan[:stop_after]]
        )
        self.assertEqual(leased, plan[:stop_after])
        self.assertTrue(channel.is_closed())
        self.assertIsNone(channel.failure())
        residual = controller.status()
        self.assertEqual(residual["prompts_leased"], 0)
        self.assertLess(residual["prompts"], self._refill_threshold(1, 1, 1) + chunk)
        self.assertLess(stop_after + residual["prompts"], len(plan))

        # A restarted producer rebuilds the same plan and sample ids, so the
        # resumed consumer's skip set (the prior attempt's durable ids)
        # removes exactly the already-trained prefix.
        produced, leased, channel, _controller = attempt()
        self.assertEqual(produced, len(plan))
        self.assertEqual(leased, plan)
        replay_ids = _published_sample_ids(channel.path)
        self.assertEqual(replay_ids, [f"run0:{task_id}" for task_id in plan])
        skip_ids = set(first_ids)
        self.assertEqual(
            [sample_id for sample_id in replay_ids if sample_id not in skip_ids],
            [f"run0:{task_id}" for task_id in plan[stop_after:]],
        )

    def test_ref_backpressure_also_bounds_prompt_ingestion(self):
        N, chunk, high = 30, 2, 3
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(
            [_adapter(store, stub)],
            _prompts(N),
            store,
            channel,
            lease=1,
            prompt_ingest_batch_size=chunk,
            in_flight_high_watermark=high,
            in_flight_low_watermark=high,
        )
        controller = workers[0].controller
        outcome = {}

        def run_producer():
            try:
                outcome["produced"] = drive()
            except BaseException as exc:  # expose a thread failure to the test
                outcome["error"] = exc

        thread = threading.Thread(target=run_producer, daemon=True)
        thread.start()
        reader = StreamingRefChannel(channel.path)
        consumed = 0
        max_resident = 0
        paused_resident = []
        deadline = time.monotonic() + 20
        while consumed < N and time.monotonic() < deadline:
            max_resident = max(max_resident, controller.status()["prompts"])
            in_flight = channel.in_flight_remote()
            if in_flight < high and consumed + in_flight < N:
                # A slow consumer: let the producer reach its ref watermark.
                time.sleep(0.001)
                continue
            if in_flight >= high:
                # Paused on the ref watermark: the feeder must not keep
                # normalizing the rest of the dataset in the meantime.
                time.sleep(0.02)
                paused_resident.append(controller.status()["prompts"])
            refs = reader.poll()
            if refs:
                reader.mark_consumed(len(refs))
                consumed += len(refs)
        thread.join(5)

        self.assertFalse(thread.is_alive(), "producer stayed ref-throttled")
        self.assertNotIn("error", outcome)
        self.assertEqual(outcome.get("produced"), N)
        self.assertTrue(paused_resident)
        # pending < threshold + one chunk, plus the single leased prompt.
        bound = self._refill_threshold(1, 1, 1) + chunk
        self.assertLessEqual(max(paused_resident), bound)
        self.assertLessEqual(max_resident, bound)
        self.assertTrue(channel.is_closed())

    def test_worker_fatal_error_stops_peer_workers_mid_plan(self):
        # Workers now outlive ingest chunks, so a fatal publish failure on one
        # worker must stop its peers instead of letting them drain the plan.
        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        stub_lock = threading.Lock()

        def slow_post(url, json_body, timeout):
            time.sleep(0.005)
            with stub_lock:
                return stub(url, json_body, timeout)

        class _OneShotFailingPublishChannel(StreamingRefChannel):
            # Only the failing worker sees an error; its peer's publishes
            # would keep succeeding if it were left running.
            failed = False

            def publish(self, ref):
                if self.published >= 3 and not self.failed:
                    self.failed = True
                    raise OSError("injected publish failure")
                super().publish(ref)

        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = _OneShotFailingPublishChannel(
            os.path.join(self._workdir(), "refs.jsonl")
        )
        N = 200
        workers, drive = _build(
            [
                _adapter(store, slow_post, url=f"http://server{i}:3000{i}")
                for i in range(2)
            ],
            _prompts(N),
            store,
            channel,
            lease=1,
            prompt_ingest_batch_size=4,
        )

        with self.assertRaisesRegex(OSError, "injected publish failure"):
            drive()

        status = workers[0].controller.status()
        self.assertEqual(status["prompts_leased"], 0)
        # Peers and the feeder stopped with the failed worker: most of the
        # plan was never leased or even ingested.
        self.assertLess(status["prompts"] + channel.published, N // 2)
        self.assertIn("injected publish failure", channel.failure())
        self.assertFalse(channel.is_closed())

    def test_prompt_normalization_failure_mid_plan_fails_the_channel(self):
        prompts = _prompts(12)

        class _BadRowPrompts:
            def __len__(self):
                return len(prompts)

            def __getitem__(self, index):
                if index == 9:
                    raise ValueError("processed dataset row 9 is corrupt")
                return prompts[index]

        backend = _FakeMooncakeStore()
        stub = _StubCaptureServer(backend)
        store = MooncakeFeatureStore(store=backend, store_id="run0")
        channel = StreamingRefChannel(os.path.join(self._workdir(), "refs.jsonl"))
        workers, drive = _build(
            [_adapter(store, stub)],
            _BadRowPrompts(),
            store,
            channel,
            lease=1,
            prompt_ingest_batch_size=2,
        )

        with self.assertRaisesRegex(ValueError, "row 9 is corrupt"):
            drive()

        status = workers[0].controller.status()
        self.assertEqual(status["prompts_leased"], 0)
        # Every capture that completed before the failure was published once.
        ids = _published_sample_ids(channel.path)
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(stub.expected))
        self.assertIn("row 9 is corrupt", channel.failure())
        self.assertFalse(channel.is_closed())


if __name__ == "__main__":
    unittest.main()
