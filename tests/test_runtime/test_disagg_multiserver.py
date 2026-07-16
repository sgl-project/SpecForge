# coding=utf-8
"""Multi-server producer fan-out (no GPU, no server, no mooncake master).

The multi-server topology: ``build_disagg_online_producer(feature_source=[...])``
builds one RolloutWorker per SGLangServerCaptureAdapter (1 server : 1 adapter :
1 worker), all leasing DISJOINT prompts from the one controller and publishing
into the one channel, concurrently. Each stub ``post_fn`` stands in for one
patched SGLang server writing into the shared fake Mooncake backend — the same
topology configured through ``specforge train`` producer and consumer roles.

Covers the failure matrix the single-server path never hits:
- disjoint + complete production across two live servers;
- one dead server: its leases fail retryable, the survivor re-leases and
  finishes the pool (no truncation, no hang);
- all servers dead: loud RuntimeError and a failure sentinel;
- a poisoned prompt (server rejects it every time): terminal failure after
  ``max_prompt_attempts`` instead of a partial-success EOF.
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


if __name__ == "__main__":
    unittest.main()
