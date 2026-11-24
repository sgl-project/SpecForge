"""Begin/recovery singleflight is scoped to original identities, never the store."""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from unittest import mock

from specforge.inference.capture_protocol import (
    CaptureCapacityDeclined,
    CaptureProtocol,
)
from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore
from tests.test_runtime.test_server_capture import _FakeMooncakeStore


class Transport:
    def __init__(self):
        self.calls = []
        self.gate = None
        self.lock = threading.Lock()
        self.lose_batch = False

    def __call__(self, url, *, json_body, timeout):
        body = dict(json_body)
        with self.lock:
            self.calls.append((url, body))
        if self.gate:
            self.gate(url, body)
        tasks = body["tasks"] if body["action"] == "begin_batch" else [body]
        replies = [
            {
                **task,
                "reservation_token": task["attempt_id"] + "-token",
                **(
                    {"object_key_layout": "attempt-v1"}
                    if task.get("publication_ownership") == "attempt-v1"
                    else {}
                ),
            }
            for task in tasks
        ]
        if body["action"] == "begin_batch":
            if self.lose_batch:
                self.lose_batch = False
                raise TimeoutError("lost batch reply after server reservation")
            return {"attempts": replies}
        return replies[0]


class ObservedMetadataLock:
    def __init__(self, wrapped, observed):
        self.wrapped, self.observed = wrapped, observed

    def __enter__(self):
        self.wrapped.acquire()
        return self

    def __exit__(self, *exc):
        self.wrapped.release()
        if threading.current_thread().name.startswith("recovery"):
            self.observed.set()


class BeginParallelTests(unittest.TestCase):
    def setUp(self):
        self.transport = Transport()
        self.store = MooncakeFeatureStore(store=_FakeMooncakeStore(), store_id="store")
        self.cap = dict(
            server_epoch="epoch", begin_batch=True, published_ownership="attempt-v1"
        )
        self.first = CaptureProtocol("http://endpoint-1", self.transport, 1)
        self.second = CaptureProtocol("http://endpoint-2", self.transport, 1)

    def request(self, sid, attempt=None):
        return (
            dict(
                store_id="store",
                sample_id=sid,
                gen=1,
                features={"aux": "hidden"},
                passthrough=[],
            ),
            [1, 2, 3],
            attempt or sid + "-attempt",
            ["hidden"],
        )

    def begin(self, protocol, sid):
        return self.store.begin_capture_attempts(
            protocol, self.cap, [self.request(sid)]
        )[0]

    def test_two_endpoints_enter_http_concurrently(self):
        barrier = threading.Barrier(2, timeout=2)
        self.transport.gate = lambda url, body: barrier.wait()
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.begin, self.first, "first")
            second = executor.submit(self.begin, self.second, "second")
            self.assertEqual(first.result(timeout=3)["attempt_id"], "first-attempt")
            self.assertEqual(second.result(timeout=3)["attempt_id"], "second-attempt")
        self.assertEqual(len(self.transport.calls), 2)
        self.assertFalse(self.store._capture_pending_begin)

    def test_fresh_prepare_does_not_wait_for_other_inflight_http(self):
        entered, release = threading.Event(), threading.Event()

        def gate(url, body):
            entered.set()
            self.assertTrue(release.wait(2))

        self.transport.gate = gate
        with ThreadPoolExecutor(max_workers=2) as executor:
            first = executor.submit(self.begin, self.first, "blocked")
            try:
                self.assertTrue(entered.wait(1))
                executor.submit(self.store.prepare_capture_retry, "unrelated").result(
                    timeout=0.5
                )
                self.assertFalse(first.done())
            finally:
                release.set()
            first.result(timeout=2)

    def test_same_sid_cross_endpoint_cannot_replace_pending_or_active_attempt(self):
        entered, release = threading.Event(), threading.Event()

        def gate(url, body):
            entered.set()
            self.assertTrue(release.wait(2))

        self.transport.gate = gate
        with ThreadPoolExecutor(max_workers=1) as executor:
            first = executor.submit(self.begin, self.first, "same")
            try:
                self.assertTrue(entered.wait(1))
                original = self.store._capture_pending_begin["same"]
                with self.assertRaisesRegex(RuntimeError, "pending"):
                    self.begin(self.second, "same")
                self.assertIs(self.store._capture_pending_begin["same"], original)
                self.assertEqual(len(self.transport.calls), 1)
            finally:
                release.set()
            first.result(timeout=2)
        with self.assertRaisesRegex(RuntimeError, "prior capture"):
            self.begin(self.second, "same")
        self.assertEqual(len(self.transport.calls), 1)

    def test_waiting_recovery_does_not_replay_completed_original_flight(self):
        entered, release, snapshot = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        def gate(url, body):
            entered.set()
            self.assertTrue(release.wait(2))

        self.transport.gate = gate
        self.store._lock = ObservedMetadataLock(self.store._lock, snapshot)
        with (
            ThreadPoolExecutor(max_workers=1) as original_executor,
            ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="recovery"
            ) as recovery_executor,
        ):
            original = original_executor.submit(self.begin, self.first, "same")
            try:
                self.assertTrue(entered.wait(1))
                recovery = recovery_executor.submit(
                    self.store._recover_capture_begin, "same"
                )
                self.assertTrue(snapshot.wait(1))
            finally:
                release.set()
            original.result(timeout=2)
            self.assertIsNone(recovery.result(timeout=2))
        self.assertEqual(len(self.transport.calls), 1)

    def test_waiting_recovery_never_retargets_replacement_intent(self):
        self.transport.lose_batch = True
        with self.assertRaises(TimeoutError):
            self.begin(self.first, "same")
        original = self.store._capture_pending_begin["same"]
        snapshot = threading.Event()
        self.store._lock = ObservedMetadataLock(self.store._lock, snapshot)
        replacement = replace(
            original,
            protocol=self.second,
            capability=dict(self.cap, server_epoch="new-epoch"),
            attempt_id="replacement",
            flight=threading.RLock(),
        )
        with original.flight:
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="recovery"
            ) as executor:
                recovery = executor.submit(self.store._recover_capture_begin, "same")
                self.assertTrue(snapshot.wait(1))
                with self.store._lock:
                    self.store._capture_pending_begin["same"] = replacement
                # The executor must not join while this thread owns its flight.
                original.flight.release()
                try:
                    self.assertIsNone(recovery.result(timeout=2))
                finally:
                    original.flight.acquire()
        self.assertIs(self.store._capture_pending_begin["same"], replacement)
        self.assertEqual(len(self.transport.calls), 1)

    def test_partial_track_failure_preserves_only_original_unfinished_intent(self):
        original_track = self.store.track_external_attempt
        failures = [True]

        def track(sid, **kwargs):
            if sid == "second" and failures[0]:
                failures[0] = False
                raise RuntimeError("metadata tracking failed")
            return original_track(sid, **kwargs)

        with mock.patch.object(self.store, "track_external_attempt", side_effect=track):
            with self.assertRaisesRegex(RuntimeError, "tracking failed"):
                self.store.begin_capture_attempts(
                    self.first,
                    self.cap,
                    [self.request("first"), self.request("second")],
                )
            self.assertNotIn("first", self.store._capture_pending_begin)
            intent = self.store._capture_pending_begin["second"]
            wire = self.store._recover_capture_begin("second")
        self.assertEqual(wire["reservation_token"], "second-attempt-token")
        self.assertEqual(intent.protocol.url, self.first.url)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(
            [body["action"] for _, body in self.transport.calls],
            ["begin_batch", "begin"],
        )

    def test_lost_batch_recovery_retains_original_endpoint_epoch_and_attempts(self):
        self.transport.lose_batch = True
        with self.assertRaises(TimeoutError):
            self.store.begin_capture_attempts(
                self.first, self.cap, [self.request("first"), self.request("second")]
            )
        intents = dict(self.store._capture_pending_begin)
        self.store._recover_capture_begin("second")
        self.store._recover_capture_begin("first")
        for url, body in self.transport.calls[1:]:
            self.assertEqual(url, self.first.url)
            self.assertEqual(body["server_epoch"], "epoch")
            self.assertEqual(body["attempt_id"], intents[body["sample_id"]].attempt_id)
        self.assertFalse(self.store._capture_pending_begin)

    def test_legacy_second_decline_keeps_first_fence_and_clears_unsent(self):
        self.cap.pop("begin_batch")

        def decline_second(url, body):
            if body["sample_id"] == "second":
                raise CaptureCapacityDeclined("full")

        self.transport.gate = decline_second
        with self.assertRaisesRegex(RuntimeError, "after 1 grants") as failure:
            self.store.begin_capture_attempts(
                self.first,
                self.cap,
                [self.request("first"), self.request("second"), self.request("third")],
            )
        self.assertNotIsInstance(failure.exception, CaptureCapacityDeclined)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(set(self.store._capture_fences), {"first"})
        self.assertEqual(
            [body["sample_id"] for _, body in self.transport.calls], ["first", "second"]
        )

    def test_legacy_first_lost_reply_keeps_only_sent_intent_for_recovery(self):
        self.cap.pop("begin_batch")
        self.transport.gate = mock.Mock(side_effect=TimeoutError("lost begin reply"))
        with self.assertRaises(TimeoutError):
            self.store.begin_capture_attempts(
                self.first, self.cap, [self.request("first"), self.request("second")]
            )
        self.assertEqual(set(self.store._capture_pending_begin), {"first"})
        self.transport.gate = None
        self.store._recover_capture_begin("first")
        self.assertEqual(self.transport.calls[0], self.transport.calls[1])
        self.assertEqual(set(self.store._capture_fences), {"first"})
        self.assertFalse(self.store._capture_pending_begin)

    def test_legacy_single_begin_fallback_reenters_batch_flight(self):
        self.cap.pop("begin_batch")
        with ThreadPoolExecutor(max_workers=1) as executor:
            replies = executor.submit(
                self.store.begin_capture_attempts,
                self.first,
                self.cap,
                [self.request("first"), self.request("second")],
            ).result(timeout=2)
        self.assertEqual(len(replies), 2)
        self.assertEqual(
            [body["action"] for _, body in self.transport.calls], ["begin", "begin"]
        )
        self.assertFalse(self.store._capture_pending_begin)


if __name__ == "__main__":
    unittest.main()
