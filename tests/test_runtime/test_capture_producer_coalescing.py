"""Throughput and failure contracts for overlapping negotiated capture calls."""

import threading
import unittest
from concurrent.futures import Future, ThreadPoolExecutor
from unittest import mock

from tests.test_runtime.test_negotiated_capture_producer import (
    NegotiatedProducerFixture,
    wait_for,
)
from tests.test_runtime.test_server_capture import _request_specs


class CoalescedProducerTests(NegotiatedProducerFixture, unittest.TestCase):
    def check_initial_registration(self, concurrency):
        workers, drive = self.build(n=concurrency, concurrency=concurrency)
        self.assertEqual(drive(), concurrency)
        self.assertEqual(
            [len(batch["tasks"]) for batch in self.server.batches], [concurrency]
        )
        self.assertEqual(
            [len(body["input_ids"]) for body in self.server.generates],
            [1] * concurrency,
        )
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)

    def test_four_slots_share_initial_registration(self):
        self.check_initial_registration(4)

    def test_twelve_slots_share_initial_registration(self):
        self.check_initial_registration(12)

    def test_done_generate_publishes_while_followup_begin_is_blocked(self):
        release_slow, begin_blocked, release_begin = (
            threading.Event() for _ in range(3)
        )
        slow = {}

        def gate(action, body):
            if action == "begin_batch" and len(self.server.batches) == 2:
                begin_blocked.set()
                self.assertTrue(release_begin.wait(3))
            if action == "generate":
                sid = _request_specs(body)[0]["sample_id"]
                if sid == slow["sid"]:
                    self.assertTrue(release_slow.wait(3))

        original = self.server.__call__

        # Capture the initial batch identity before any generate can execute.
        def transport(url, json_body, timeout):
            if json_body.get("action") == "begin_batch" and not slow:
                slow["sid"] = json_body["tasks"][0]["sample_id"]
            return original(url, json_body, timeout)

        self.server.gate = gate
        workers, drive = self.build(n=7, concurrency=3, servers=[transport])
        future = self.drive_async(drive)
        try:
            self.assertTrue(begin_blocked.wait(2))
            release_slow.set()
            wait_for(lambda: self.channel.published >= 3)
            self.assertFalse(future.done())
            self.assertFalse(release_begin.is_set())
        finally:
            release_slow.set()
            release_begin.set()
        self.assertEqual(future.result(timeout=3), 7)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertGreater(len(self.server.batches), 1)

    def test_late_idle_slots_coalesce_after_inflight_begin(self):
        second_begin = threading.Event()
        release_begin = threading.Event()
        release_initial = threading.Event()
        initial = []
        original = self.server.__call__

        def transport(url, json_body, timeout):
            if json_body.get("action") == "begin_batch" and not initial:
                initial.extend(task["sample_id"] for task in json_body["tasks"])
            return original(url, json_body, timeout)

        def gate(action, body):
            if action == "begin_batch" and len(self.server.batches) == 2:
                second_begin.set()
                self.assertTrue(release_begin.wait(3))
            if action == "generate":
                sid = _request_specs(body)[0]["sample_id"]
                if sid in initial[1:]:
                    self.assertTrue(release_initial.wait(3))

        self.server.gate = gate
        workers, drive = self.build(n=9, concurrency=4, servers=[transport])
        source = workers[0].feature_source
        prepared_late = set()
        prepare = source.prepare_refs_batches

        def observe_prepare(batches, *, capture):
            result = prepare(batches, capture=capture)
            if second_begin.is_set():
                prepared_late.update(
                    task.task_id for tasks in batches for task in tasks
                )
            return result

        source.prepare_refs_batches = observe_prepare
        future = self.drive_async(drive)
        try:
            self.assertTrue(second_begin.wait(2))
            release_initial.set()
            wait_for(lambda: len(prepared_late) == 3)
        finally:
            release_initial.set()
            release_begin.set()
        self.assertEqual(future.result(timeout=3), 9)
        self.assertEqual([len(b["tasks"]) for b in self.server.batches[:3]], [4, 1, 3])

    def test_generate_submit_failure_retires_registered_preleases(self):
        workers, drive = self.build(n=4)
        submit = ThreadPoolExecutor.submit

        def guarded_submit(executor, fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "run_leased":
                raise RuntimeError("injected generate submit failure")
            return submit(executor, fn, *args, **kwargs)

        with mock.patch.object(ThreadPoolExecutor, "submit", guarded_submit):
            with self.assertRaisesRegex(RuntimeError, "generate submit failure"):
                drive()
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)
        self.assertIsNotNone(self.channel.failure())

    def test_cancelled_generate_future_releases_prelease_and_fence(self):
        workers, drive = self.build(n=4, max_worker_failures=1)
        submit = ThreadPoolExecutor.submit

        def cancelling_submit(executor, fn, *args, **kwargs):
            if getattr(fn, "__name__", "") == "run_leased":
                future = Future()
                future.cancel()
                return future
            return submit(executor, fn, *args, **kwargs)

        with mock.patch.object(ThreadPoolExecutor, "submit", cancelling_submit):
            with self.assertRaisesRegex(RuntimeError, "dead workers"):
                drive()
        self.assertFalse(self.server.active)
        self.assertFalse(self.store._capture_pending_begin)
        self.assertEqual(workers[0].health()["in_flight"], 0)
        self.assertEqual(workers[0].controller.status()["prompts_leased"], 0)
