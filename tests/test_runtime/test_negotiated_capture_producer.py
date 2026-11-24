"""Negotiated producer admission, lease settlement and exact-attempt cleanup."""

import copy
import os
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor

from specforge.inference.capture_protocol import CaptureProtocol
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from specforge.runtime.data_plane.streaming_ref_channel import StreamingRefChannel
from tests.test_runtime.capture_fixtures import ProtocolServer
from tests.test_runtime.test_disagg_multiserver import _adapter, _build, _prompts
from tests.test_runtime.test_server_capture import _FakeMooncakeStore


class BatchServer(ProtocolServer):
    def __init__(self, backend):
        super().__init__(backend)
        self.capability["begin_batch"] = True
        for item in self.capability["schema"]:
            item["dtype"] = "bfloat16"
        self.batches = []
        self.generates = []
        self.gate = None
        self.lock = threading.RLock()

    def __call__(self, url, json_body, timeout):
        body = copy.deepcopy(json_body)
        action = body.get("action", "generate")
        if action == "begin_batch":
            with self.lock:
                self.batches.append(body)
                replies = [
                    super(BatchServer, self).__call__(
                        url, dict(task, action="begin"), timeout
                    )
                    for task in body["tasks"]
                ]
            if self.gate:
                self.gate(action, body)
            return {"attempts": replies}
        if action == "generate":
            with self.lock:
                self.generates.append(body)
            if self.gate:
                self.gate(action, body)
        with self.lock:
            return super().__call__(url, body, timeout)


def wait_for(predicate, timeout=3):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.001)


class NegotiatedProducerTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.backend = _FakeMooncakeStore()
        self.store = MooncakeFeatureStore(store=self.backend, store_id="run0")
        self.server = BatchServer(self.backend)
        self.channel = StreamingRefChannel(os.path.join(self.directory.name, "refs"))

    def build(self, n=8, concurrency=4, servers=None, **kwargs):
        adapters = []
        for i, server in enumerate(servers or [self.server]):
            adapter = _adapter(self.store, server, url=f"http://server{i}")
            adapter._capture_protocol = CaptureProtocol(adapter.base_url, server, 1)
            adapters.append(adapter)
        lease = kwargs.pop("lease", 1)
        return _build(
            adapters,
            _prompts(n),
            self.store,
            self.channel,
            lease=lease,
            producer_concurrency=concurrency,
            **kwargs,
        )

    def drive_async(self, drive, **kwargs):
        pool = ThreadPoolExecutor(max_workers=1)
        future = pool.submit(drive, **kwargs)
        self.addCleanup(pool.shutdown)
        return future

    def test_returned_output_cannot_silently_exceed_reserved_schema(self):
        for item in self.server.capability["schema"]:
            item["dtype"] = "int8"  # Deliberately false server capability.
        workers, drive = self.build(n=2, concurrency=2, max_worker_failures=1)
        with self.assertRaisesRegex(RuntimeError, "dead workers"):
            drive()
        self.assertIn("exceeded negotiated byte reservation", self.channel.failure())
        self.assertEqual(self.channel.published, 0)
        self.assertFalse(self.backend._d)
        self.assertFalse(self.server.active)
        self.assertEqual(workers[0].health()["in_flight"], 0)

    def test_other_endpoint_registers_while_first_endpoint_is_blocked(self):
        entered, other_entered, release = (threading.Event() for _ in range(3))
        other = BatchServer(self.backend)

        def first_gate(action, body):
            if action == "begin_batch":
                entered.set()
                self.assertTrue(release.wait(3))

        def other_gate(action, body):
            if action == "begin_batch":
                self.assertTrue(entered.wait(2))
                other_entered.set()

        self.server.gate, other.gate = first_gate, other_gate
        _, drive = self.build(n=8, concurrency=2, servers=[self.server, other])
        future = self.drive_async(drive)
        try:
            self.assertTrue(other_entered.wait(2))
        finally:
            release.set()
        self.assertEqual(future.result(timeout=3), 8)

    def test_driver_failure_after_begin_retires_registered_prelease(self):
        workers, drive = self.build(n=4)

        def fail_before_generate():
            if self.server.active:
                raise RuntimeError("injected driver failure after begin")
            return False

        with self.assertRaisesRegex(RuntimeError, "driver failure after begin"):
            drive(should_stop=fail_before_generate)
        self.assertFalse(self.server.generates)
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)
        self.assertIsNotNone(self.channel.failure())

    def test_cancelled_registered_lease_releases_fence_and_requeues_prompt(self):
        workers, drive = self.build(n=4)
        source = workers[0].feature_source
        register = source.register_prepared_batches

        def cancel_before_generate(prepared):
            register(prepared)
            workers[0].stop()

        source.register_prepared_batches = cancel_before_generate
        with self.assertRaisesRegex(RuntimeError, "stopped or draining"):
            drive()
        self.assertFalse(self.server.generates)
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        status = workers[0].controller.status()
        self.assertEqual(status["prompts_leased"], 0)
        self.assertEqual(status["prompts_pending"], 4)

    def test_stale_prepared_cleanup_preserves_peer_replacement_attempt(self):
        workers, _drive = self.build(n=1, concurrency=2)
        source = workers[0].feature_source
        # Use real task leases and store fences.
        workers[0].start()
        workers[0].controller.ingest_prompts(_prompts(1))
        tasks = workers[0].lease_tasks(1)
        old = source.prepare_refs_batches([tasks], capture=workers[0].capture)
        source.register_prepared_batches(old)
        sid = old[0]["payloads"][0]["sample_id"]
        self.store.prepare_capture_retry(sid)  # A peer retires the old writer.
        new = source.prepare_refs_batches([tasks], capture=workers[0].capture)
        source.register_prepared_batches(new)
        fence = self.store._capture_fences[sid]
        source.discard_prepared_batches(old)
        self.assertIs(self.store._capture_fences[sid], fence)
        self.assertEqual(self.server.active[sid]["reservation_token"], fence.token)
        source.discard_prepared_batches(new)
        workers[0].fail_leased(tasks, "test-cleanup", unhealthy=False)
        self.assertFalse(self.server.active)

    def test_unknown_old_cleanup_never_recovers_replacement_pending_intent(self):
        workers, _drive = self.build(n=1, concurrency=2)
        source = workers[0].feature_source
        workers[0].start()
        workers[0].controller.ingest_prompts(_prompts(1))
        tasks = workers[0].lease_tasks(1)
        calls = []
        original = source._capture_protocol.post

        def lose_batch(url, json_body, timeout):
            calls.append(copy.deepcopy(json_body))
            reply = original(url, json_body, timeout)
            if json_body.get("action") == "begin_batch":
                raise TimeoutError("lost begin response")
            return reply

        source._capture_protocol.post = lose_batch
        old = source.prepare_refs_batches([tasks], capture=workers[0].capture)
        with self.assertRaises(TimeoutError):
            source.register_prepared_batches(old)
        sid = old[0]["payloads"][0]["sample_id"]
        self.store.prepare_capture_retry(sid)
        new = source.prepare_refs_batches([tasks], capture=workers[0].capture)
        with self.assertRaises(TimeoutError):
            source.register_prepared_batches(new)
        pending = self.store._capture_pending_begin[sid]
        calls_before = len(calls)
        source.discard_prepared_batches(old)
        self.assertEqual(len(calls), calls_before)
        self.assertIs(self.store._capture_pending_begin[sid], pending)
        self.assertEqual(self.server.active[sid]["attempt_id"], pending.attempt_id)
        source.discard_prepared_batches(new)
        workers[0].fail_leased(tasks, "test-cleanup", unhealthy=False)
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)

    def test_unknown_begin_reply_recovers_original_intents_before_retry(self):
        calls = []
        original = self.server.__call__
        lost = [False]

        def transport(url, json_body, timeout):
            calls.append(copy.deepcopy(json_body))
            reply = original(url, json_body, timeout)
            if json_body.get("action") == "begin_batch" and not lost[0]:
                lost[0] = True
                raise TimeoutError("lost begin reply")
            return reply

        workers, drive = self.build(n=4, servers=[transport])
        self.assertEqual(drive(), 4)
        first = next(body for body in calls if body.get("action") == "begin_batch")
        recovered = [body for body in calls if body.get("action") == "begin"]
        self.assertEqual(
            {b["attempt_id"] for b in recovered},
            {b["attempt_id"] for b in first["tasks"]},
        )
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)

    def test_http_409_refresh_never_retargets_original_begin_intents(self):
        import requests

        calls = []
        original = self.server.__call__
        rejected = [False]
        recovered = []

        def transport(url, json_body, timeout):
            calls.append(copy.deepcopy(json_body))
            if json_body.get("action") == "begin_batch" and not rejected[0]:
                # A response encoding failure may surface as HTTP 409 after
                # reservations were granted. It is not proof of rollback.
                original(url, json_body, timeout)
                rejected[0] = True
                self.server.capability["server_epoch"] = "refreshed-epoch"
                response = requests.Response()
                response.status_code = 409
                raise requests.HTTPError(
                    "ambiguous server rejection", response=response
                )
            if json_body.get("action") == "begin":
                recovered.append(copy.deepcopy(json_body))
                # Cleanup must recover the exact old reservation despite the
                # refreshed discovery; reject any new identity immediately.
                self.assertEqual(json_body["server_epoch"], "epoch")
            return original(url, json_body, timeout)

        workers, drive = self.build(n=4, servers=[transport], max_worker_failures=1)
        with self.assertRaisesRegex(RuntimeError, "dead workers"):
            drive()
        batch = next(body for body in calls if body.get("action") == "begin_batch")
        self.assertEqual(
            {body["attempt_id"] for body in recovered},
            {body["attempt_id"] for body in batch["tasks"]},
        )
        self.assertEqual(
            len([body for body in calls if body.get("action") == "begin_batch"]), 1
        )
        self.assertFalse(self.server.generates)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertFalse(self.server.active)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_oversize_lease_fails_before_registration(self):
        workers, drive = self.build(n=1, feature_store_max_resident_bytes=1)
        with self.assertRaisesRegex(MemoryError, "sample reservation"):
            drive()
        self.assertFalse(self.server.batches)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_persistent_capacity_refusal_times_out_and_settles_lease(self):
        attempts = []

        def refuse(url, json_body, timeout):
            if json_body.get("action") == "begin_batch":
                attempts.append(json_body["tasks"][0]["attempt_id"])
                return dict(
                    error="full",
                    error_code="capture_capacity_declined",
                    reservation_granted=False,
                )
            return self.server(url, json_body, timeout)

        workers, drive = self.build(n=1, servers=[refuse], peer_wait_timeout_s=0.05)
        with self.assertRaisesRegex(TimeoutError, "backpressure timed out"):
            drive()
        self.assertEqual(len(set(attempts)), 1)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertFalse(self.server.active)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_unreturned_outputs_consume_global_hard_cap_credit(self):
        # The canonical DFlash fixture requests aux only, plus two int64 passthrough tensors.
        aux = next(
            item for item in self.server.capability["schema"] if item["name"] == "aux"
        )
        per_token = aux["width"] * 2 + 16
        cap = per_token * 42  # Fits two longest prompts across both endpoints.
        other = BatchServer(self.backend)
        _, drive = self.build(
            n=16,
            concurrency=8,
            servers=[self.server, other],
            feature_store_max_resident_bytes=cap,
        )
        reader = StreamingRefChannel(self.channel.path)
        outstanding = 0
        peak = 0
        lock = threading.Lock()

        def gate(action, body):
            nonlocal outstanding, peak
            if action == "begin_batch":
                # Derive output lengths from request ids in this fixed fixture.
                size = sum(
                    (6 + int(task["sample_id"].split(":p")[1])) * per_token
                    for task in body["tasks"]
                )
                with lock:
                    outstanding += size
                    peak = max(peak, outstanding)
                self.assertLessEqual(peak, cap)

        self.server.gate = other.gate = gate
        future = self.drive_async(drive)
        while not future.done():
            refs = reader.poll()
            if refs:
                # Acknowledge the actual completed outputs. Not-yet-returned
                # writers remain in outstanding and continue occupying credit.
                for ref in refs:
                    self.store.abort(ref.sample_id, reason="test-consumer-completed")
                with lock:
                    outstanding -= sum(ref.estimated_bytes for ref in refs)
                reader.mark_consumed(len(refs))
            time.sleep(0.001)
        self.assertEqual(future.result(timeout=3), 16)
        self.assertLessEqual(peak, cap)

    def test_per_task_errors_retire_every_registered_attempt(self):
        self.server.stub.error_sample_ids = {f"run0:p{i}" for i in range(4)}
        workers, drive = self.build(
            n=4, max_prompt_attempts=1, feature_store_max_resident_bytes=100000
        )
        with self.assertRaisesRegex(RuntimeError, "terminally failed prompt"):
            drive()
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_fences)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertFalse(self.backend._d)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_second_adoption_failure_retires_every_unpublished_ref(self):
        workers, drive = self.build(n=2, lease=2, max_worker_failures=1)
        adopt = self.store.adopt
        adopted = []

        def fail_after_second_adoption(ref):
            adopt(ref)
            adopted.append(ref.sample_id)
            if len(adopted) == 2:
                raise RuntimeError("injected post-adoption failure")

        self.store.adopt = fail_after_second_adoption
        with self.assertRaisesRegex(RuntimeError, "injected post-adoption failure"):
            drive()
        self.assertEqual(len(adopted), 2)
        self.assertEqual(self.channel.published, 0)
        self.assertFalse(self.server.active)
        self.assertFalse(self.backend._d)
        self.assertFalse(self.store._capture_fences)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_partial_lease_errors_cleanup_only_unreturned_identities(self):
        self.server.stub.error_sample_ids = {"run0:p1", "run0:p3"}
        workers, drive = self.build(
            n=4,
            concurrency=2,
            lease=2,
            max_prompt_attempts=1,
            feature_store_max_resident_bytes=100000,
        )
        with self.assertRaisesRegex(RuntimeError, "terminally failed prompt"):
            drive()
        refs = self.channel.poll()
        self.assertEqual({ref.sample_id for ref in refs}, {"run0:p0", "run0:p2"})
        self.assertEqual(set(self.server.active), {"run0:p0", "run0:p2"})
        self.assertTrue(self.backend._d)
        self.assertEqual(set(self.store._capture_fences), {"run0:p0", "run0:p2"})
        # This fixture negotiates legacy ownership, so successful fences stay
        # live until consumer release; failed fences alone must be retired.
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def _stop_active_worker(self, method):
        entered, release = threading.Event(), threading.Event()

        def gate(action, body):
            if action == "generate":
                entered.set()
                self.assertTrue(release.wait(3))

        self.server.gate = gate
        workers, drive = self.build(n=8)
        future = self.drive_async(drive)
        try:
            self.assertTrue(entered.wait(2))
            getattr(workers[0], method)()
        finally:
            release.set()
        with self.assertRaisesRegex(
            RuntimeError, "stopped or draining with prompts remaining"
        ):
            future.result(timeout=3)
        self.assertEqual(self.channel.published, 1)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_pending"], 7)

    def test_stopped_worker_leaves_coordinator_without_spinning(self):
        self._stop_active_worker("stop")

    def test_draining_worker_leaves_coordinator_without_spinning(self):
        self._stop_active_worker("drain")

    def test_stopped_endpoint_releases_pending_plan_to_surviving_worker(self):
        entered, release = threading.Event(), threading.Event()
        other = BatchServer(self.backend)

        def first_gate(action, body):
            if action == "generate":
                entered.set()
                self.assertTrue(release.wait(3))

        def other_gate(action, body):
            if action == "generate":
                self.assertTrue(release.wait(3))

        self.server.gate, other.gate = first_gate, other_gate
        workers, drive = self.build(n=12, concurrency=2, servers=[self.server, other])
        future = self.drive_async(drive)
        try:
            self.assertTrue(entered.wait(2))
            workers[0].stop()
        finally:
            release.set()
        self.assertEqual(future.result(timeout=3), 12)
        self.assertEqual(workers[0].health()["state"], "stopped")
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_stop_after_registration_cleans_without_submitting_generate(self):
        stop = threading.Event()
        workers, drive = self.build(n=4, feature_store_max_resident_bytes=100000)
        source = workers[0].feature_source
        register = source.register_prepared_batches

        def stopping_register(prepared):
            register(prepared)
            stop.set()

        source.register_prepared_batches = stopping_register
        self.assertEqual(drive(should_stop=stop.is_set), 0)
        self.assertEqual([len(b["tasks"]) for b in self.server.batches], [1])
        self.assertFalse(self.server.generates)
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)
        self.assertTrue(self.channel.is_closed())

    def test_partial_preparation_is_loud_and_releases_every_lease(self):
        workers, drive = self.build(n=4, max_worker_failures=1)
        source = workers[0].feature_source
        prepare = source.prepare_refs_batches

        def short_prepare(groups, *, capture):
            return prepare(groups, capture=capture)[:-1]

        source.prepare_refs_batches = short_prepare
        with self.assertRaisesRegex(
            RuntimeError, "preparation returned 0 groups for one lease"
        ):
            drive()
        self.assertFalse(self.server.batches)
        self.assertFalse(self.server.generates)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_stop_releases_prepared_leases_waiting_for_credit(self):
        per_token = (
            sum(item["width"] * 2 for item in self.server.capability["schema"]) + 16
        )
        stop = threading.Event()
        workers, drive = self.build(
            n=6, feature_store_max_resident_bytes=per_token * 11
        )
        future = self.drive_async(drive, should_stop=stop.is_set)
        wait_for(lambda: self.channel.published >= 1)
        stop.set()
        self.assertGreaterEqual(future.result(timeout=3), 1)
        self.assertFalse(
            self.server.active.keys() - {ref.sample_id for ref in self.channel.poll()}
        )
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)


if __name__ == "__main__":
    unittest.main()
