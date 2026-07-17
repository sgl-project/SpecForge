# coding=utf-8
"""Deterministic CPU tests for consumer-driven capture windows."""

from __future__ import annotations

import tempfile
import unittest

from specforge.runtime.contracts import FeatureSpec, SampleRef
from specforge.runtime.data_plane.windowed_capture import (
    CaptureFailedError,
    ConsumerFailedError,
    SQLiteWindowedCaptureRegistry,
    WindowedCaptureQueue,
    capture_contract_digest,
)


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


class _ReclaimStore:
    def __init__(self) -> None:
        self.reclaimed: list[tuple[str, int, str]] = []

    def reclaim(self, ref: SampleRef, *, reason: str = "consumed") -> None:
        self.reclaimed.append((ref.sample_id, int(ref.metadata["generation"]), reason))


class WindowedRegistryFixture(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.clock = _Clock()
        self.digest = capture_contract_digest(
            {
                "strategy": "dflash",
                "target_model_version": "qwen3-8b-fixture",
                "features": ["input_ids", "hidden_states", "loss_mask"],
            }
        )
        self.registry = self.make_registry(max_live_refs=8)

    def make_registry(
        self,
        *,
        path: str | None = None,
        max_live_refs: int,
        max_live_bytes: int | None = None,
        reservation_bytes: int | None = None,
    ) -> SQLiteWindowedCaptureRegistry:
        registry = SQLiteWindowedCaptureRegistry(
            path or f"{self.tempdir.name}/registry.db",
            max_live_refs=max_live_refs,
            max_live_bytes=max_live_bytes,
            capture_reservation_bytes=reservation_bytes,
            clock=self.clock,
            poll_s=0.001,
        )
        self.addCleanup(registry.close)
        return registry

    def initialize(
        self, *, samples: int = 8, consumers: tuple[str, ...] = ("a", "b", "c")
    ) -> None:
        self.registry.initialize_run(
            run_id="capture-run",
            contract_digest=self.digest,
            source_sample_ids=[f"source-{index}" for index in range(samples)],
            expected_consumers=consumers,
        )

    def register(
        self,
        consumer_id: str,
        *,
        lookbehind: int = 0,
        lookahead: int = 0,
        prefetch_depth: int = 0,
        max_outstanding: int = 1,
    ) -> None:
        self.registry.register_consumer(
            consumer_id,
            lookbehind=lookbehind,
            lookahead=lookahead,
            prefetch_depth=prefetch_depth,
            max_outstanding=max_outstanding,
        )

    @staticmethod
    def ref(request, *, estimated_bytes: int = 80) -> SampleRef:
        sample_id = f"capture-run:{request.key.source_sample_id}"
        return SampleRef(
            sample_id=sample_id,
            run_id="capture-run",
            source_task_id=request.key.source_sample_id,
            feature_store_uri=f"fixture://capture-run/{sample_id}",
            feature_keys={"input_ids": f"{sample_id}/input_ids"},
            feature_specs={
                "input_ids": FeatureSpec(name="input_ids", shape=(1, 4), dtype="int64")
            },
            strategy="dflash",
            estimated_bytes=estimated_bytes,
            metadata={"generation": request.generation, "target_repr": None},
        )

    def complete(self, request, *, estimated_bytes: int = 80) -> SampleRef:
        ref = self.ref(request, estimated_bytes=estimated_bytes)
        self.registry.mark_committing(request, ref)
        self.registry.complete_capture(request, ref)
        return ref


class TestWindowedCaptureIdentity(WindowedRegistryFixture):
    def test_contract_digest_is_canonical_and_contract_sensitive(self):
        left = capture_contract_digest({"layers": (1, 2), "features": {"a", "b"}})
        right = capture_contract_digest({"features": {"b", "a"}, "layers": [1, 2]})
        other = capture_contract_digest({"features": {"a"}, "layers": [1, 2]})

        self.assertEqual(left, right)
        self.assertNotEqual(left, other)

    def test_existing_run_rejects_source_or_capacity_identity_change(self):
        self.initialize(samples=2, consumers=("a",))
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            self.registry.initialize_run(
                run_id="capture-run",
                contract_digest=self.digest,
                source_sample_ids=["source-1", "source-0"],
                expected_consumers=("a",),
            )

        other = self.make_registry(
            path=f"{self.tempdir.name}/registry.db", max_live_refs=7
        )
        with self.assertRaisesRegex(RuntimeError, "identity mismatch"):
            other.initialize_run(
                run_id="capture-run",
                contract_digest=self.digest,
                source_sample_ids=["source-0", "source-1"],
                expected_consumers=("a",),
            )


class TestWindowedCaptureStateMachine(WindowedRegistryFixture):
    def test_three_misses_singleflight_and_share_one_generation(self):
        self.initialize(samples=2)
        for consumer_id in ("a", "b", "c"):
            self.register(consumer_id)
        tickets = [
            self.registry.request_acquire(consumer_id, 0)
            for consumer_id in ("a", "b", "c")
        ]

        requests = self.registry.claim_batch(8)

        self.assertEqual(len(requests), 1)
        self.assertEqual(requests[0].demand_consumers, ("a", "b", "c"))
        self.complete(requests[0])
        leases = [self.registry.wait_ready(ticket, timeout_s=1.0) for ticket in tickets]
        self.assertEqual({lease.generation for lease in leases}, {1})
        for lease in leases:
            self.registry.release_and_advance(lease.consumer_id, [lease])
        candidates = self.registry.begin_evictions()
        self.assertEqual([candidate.source_index for candidate in candidates], [0])

    def test_duplicate_request_is_rejected_without_duplicate_capture(self):
        self.initialize(samples=1, consumers=("a", "b"))
        self.register("a")
        self.register("b")
        first = self.registry.request_acquire("a", 0)
        with self.assertRaisesRegex(RuntimeError, "already has"):
            self.registry.request_acquire("a", 0)
        second = self.registry.request_acquire("b", 0)

        request = self.registry.claim_batch(2)[0]
        self.complete(request)
        self.assertEqual(
            self.registry.wait_ready(first, timeout_s=1.0).generation,
            self.registry.wait_ready(second, timeout_s=1.0).generation,
        )
        self.assertEqual(self.registry.snapshot()["capture_count"], 1)

    def test_heterogeneous_windows_replenish_independent_frontiers(self):
        self.registry.close()
        self.registry = self.make_registry(max_live_refs=5)
        self.initialize(samples=7, consumers=("fast", "slow"))
        self.register("fast", lookahead=3, prefetch_depth=4, max_outstanding=2)
        self.register("slow", lookahead=0, prefetch_depth=1)
        initial = self.registry.claim_batch(8)
        self.assertEqual([request.source_index for request in initial], [0, 1, 2, 3])
        for request in initial:
            self.complete(request)

        ticket = self.registry.request_acquire("fast", 0)
        lease = self.registry.wait_ready(ticket, timeout_s=1.0)
        self.assertTrue(lease.ready_at_request)
        self.registry.release_and_advance("fast", [lease])
        frontier = self.registry.claim_batch(8)

        self.assertEqual([request.source_index for request in frontier], [4])
        snapshot = self.registry.snapshot()
        self.assertEqual(snapshot["consumers"]["fast"]["cursor"], 1)
        self.assertEqual(snapshot["consumers"]["slow"]["cursor"], 0)

    def test_round_robin_demand_fairness(self):
        self.initialize(samples=6)
        for consumer_id in ("a", "b", "c"):
            self.register(consumer_id, lookahead=5, max_outstanding=3)
        for consumer_id, indices in (
            ("a", (0, 1, 2)),
            ("b", (3,)),
            ("c", (4,)),
        ):
            for source_index in indices:
                self.registry.request_acquire(consumer_id, source_index)

        first = self.registry.claim_batch(3)

        self.assertEqual({request.source_index for request in first}, {0, 3, 4})

    def test_retry_generation_is_unique_and_terminal_failure_wakes_waiter(self):
        self.initialize(samples=1, consumers=("a",))
        self.register("a")
        ticket = self.registry.request_acquire("a", 0)
        first = self.registry.claim_batch(1)[0]

        self.assertTrue(
            self.registry.fail_capture(
                first, "transient", retryable=True, max_retries=1
            )
        )
        second = self.registry.claim_batch(1)[0]
        self.assertEqual((first.generation, second.generation), (1, 2))
        self.assertFalse(
            self.registry.fail_capture(
                second, "still broken", retryable=True, max_retries=1
            )
        )
        with self.assertRaisesRegex(CaptureFailedError, "still broken"):
            self.registry.wait_ready(ticket, timeout_s=1.0)

    def test_terminal_failure_is_observed_by_all_waiters_before_recapture(self):
        self.initialize(samples=1)
        for consumer_id in ("a", "b", "c"):
            self.register(consumer_id)
        first = self.registry.request_acquire("a", 0)
        second = self.registry.request_acquire("b", 0)
        failed = self.registry.claim_batch(1)[0]
        self.assertFalse(
            self.registry.fail_capture(
                failed, "terminal generation", retryable=False, max_retries=0
            )
        )

        with self.assertRaisesRegex(CaptureFailedError, "terminal generation"):
            self.registry.wait_ready(first, timeout_s=1.0)
        late = self.registry.request_acquire("c", 0)
        self.assertEqual(self.registry.claim_batch(1), ())
        for ticket in (second, late):
            with self.assertRaisesRegex(CaptureFailedError, "terminal generation"):
                self.registry.wait_ready(ticket, timeout_s=1.0)

        retry = self.registry.request_acquire("a", 0)
        replacement = self.registry.claim_batch(1)[0]
        self.assertEqual(replacement.generation, failed.generation + 1)
        self.complete(replacement)
        self.assertEqual(
            self.registry.wait_ready(retry, timeout_s=1.0).generation,
            replacement.generation,
        )

    def test_failed_consumer_drops_only_its_waiter_and_window(self):
        self.initialize(samples=2, consumers=("failed", "healthy"))
        self.register("failed")
        self.register("healthy")
        failed_ticket = self.registry.request_acquire("failed", 0)
        healthy_ticket = self.registry.request_acquire("healthy", 0)
        request = self.registry.claim_batch(1)[0]

        self.registry.fail_consumer("failed", "trainer exited")
        self.complete(request)
        with self.assertRaisesRegex(ConsumerFailedError, "trainer exited"):
            self.registry.wait_ready(failed_ticket, timeout_s=1.0)
        healthy_lease = self.registry.wait_ready(healthy_ticket, timeout_s=1.0)
        self.assertEqual(healthy_lease.source_index, 0)
        self.registry.release_and_advance("healthy", [healthy_lease])
        self.assertEqual(self.registry.snapshot()["consumers"]["healthy"]["cursor"], 1)

    def test_read_lease_blocks_pressure_eviction_until_release(self):
        self.initialize(samples=1, consumers=("a",))
        self.register("a")
        ticket = self.registry.request_acquire("a", 0)
        request = self.registry.claim_batch(1)[0]
        self.complete(request)
        lease = self.registry.wait_ready(ticket, timeout_s=1.0)

        self.assertEqual(self.registry.begin_evictions(pressure=True), ())
        self.registry.release_and_advance("a", [lease])
        candidates = self.registry.begin_evictions(pressure=True)

        self.assertEqual([item.source_index for item in candidates], [0])

    def test_expired_consumer_releases_its_lease_without_stopping_peer(self):
        self.initialize(samples=1, consumers=("expired", "healthy"))
        self.register("expired")
        self.register("healthy")
        ticket = self.registry.request_acquire("expired", 0)
        request = self.registry.claim_batch(1)[0]
        self.complete(request)
        self.registry.wait_ready(ticket, timeout_s=1.0)
        self.registry.heartbeat("healthy")
        self.clock.now += 11.0
        self.registry.heartbeat("healthy")

        self.assertEqual(self.registry.expire_consumers(10.0), ("expired",))
        snapshot = self.registry.snapshot()

        self.assertEqual(snapshot["leases"], 0)
        self.assertNotEqual(snapshot["consumers"]["healthy"]["state"], "failed")

    def test_pressure_reclaim_unblocks_demand_without_evicting_hard_interest(self):
        self.registry.close()
        self.registry = self.make_registry(max_live_refs=2)
        self.initialize(samples=3, consumers=("fast",))
        self.register("fast", lookahead=2, prefetch_depth=2)
        for request in self.registry.claim_batch(2):
            self.complete(request)
        ticket = self.registry.request_acquire("fast", 2)
        self.assertEqual(self.registry.claim_batch(1), ())

        store = _ReclaimStore()
        self.assertEqual(self.registry.reclaim(store, limit=1, pressure=True), 1)
        demand = self.registry.claim_batch(1)

        self.assertEqual([request.source_index for request in demand], [2])
        self.assertNotIn(ticket.key.source_sample_id, store.reclaimed[0][0])
        self.assertLessEqual(self.registry.snapshot()["live_refs"], 2)

    def test_byte_reservations_are_hard_and_oversize_completion_is_rejected(self):
        self.registry.close()
        self.registry = self.make_registry(
            max_live_refs=4,
            max_live_bytes=200,
            reservation_bytes=100,
        )
        self.initialize(samples=3, consumers=("a",))
        self.register("a", lookahead=2, prefetch_depth=3)
        requests = self.registry.claim_batch(4)
        self.assertEqual(len(requests), 2)
        self.complete(requests[0], estimated_bytes=80)
        with self.assertRaisesRegex(ValueError, "exceeds reserved"):
            self.registry.mark_committing(
                requests[1], self.ref(requests[1], estimated_bytes=101)
            )
        self.registry.fail_capture(
            requests[1], "oversize", retryable=False, max_retries=0
        )
        snapshot = self.registry.snapshot()
        self.assertLessEqual(snapshot["peak_live_bytes"], 200)
        self.assertLessEqual(snapshot["peak_live_refs"], 2)


class TestWindowedCaptureResume(WindowedRegistryFixture):
    def test_owner_recovery_fences_old_generation_and_durable_cursor_rewinds(self):
        self.initialize(samples=2, consumers=("a",))
        self.register("a", lookahead=1, max_outstanding=2)
        ticket = self.registry.request_acquire("a", 0)
        interrupted = self.registry.claim_batch(1)[0]
        path = self.registry.path
        self.registry.close()

        resumed = self.make_registry(path=path, max_live_refs=8)
        self.registry = resumed
        with self.assertRaisesRegex(RuntimeError, "recover_inflight"):
            resumed.initialize_run(
                run_id="capture-run",
                contract_digest=self.digest,
                source_sample_ids=["source-0", "source-1"],
                expected_consumers=("a",),
            )
        resumed.initialize_run(
            run_id="capture-run",
            contract_digest=self.digest,
            source_sample_ids=["source-0", "source-1"],
            expected_consumers=("a",),
            recover_inflight=True,
        )
        replacement = resumed.claim_batch(1)[0]
        self.assertEqual((interrupted.generation, replacement.generation), (1, 2))
        self.complete(replacement)
        lease = resumed.wait_ready(ticket, timeout_s=1.0)
        resumed.release_and_advance("a", [lease])
        self.assertEqual(resumed.consumer_cursor("a"), 1)

        resumed.resume_consumer("a", durable_cursor=0)
        replay = resumed.request_acquire("a", 0)
        replay_lease = resumed.wait_ready(replay, timeout_s=1.0)
        self.assertTrue(replay_lease.ready_at_request)
        self.assertEqual(replay_lease.generation, 2)

    def test_committing_recovery_reclaims_payload_before_requeue(self):
        self.initialize(samples=1, consumers=("a",))
        self.register("a")
        ticket = self.registry.request_acquire("a", 0)
        interrupted = self.registry.claim_batch(1)[0]
        ref = self.ref(interrupted)
        self.registry.mark_committing(interrupted, ref)
        path = self.registry.path
        self.registry.close()

        resumed = self.make_registry(path=path, max_live_refs=8)
        self.registry = resumed
        with self.assertRaisesRegex(RuntimeError, "recovery_store"):
            resumed.initialize_run(
                run_id="capture-run",
                contract_digest=self.digest,
                source_sample_ids=["source-0"],
                expected_consumers=("a",),
                recover_inflight=True,
            )
        store = _ReclaimStore()
        resumed.initialize_run(
            run_id="capture-run",
            contract_digest=self.digest,
            source_sample_ids=["source-0"],
            expected_consumers=("a",),
            recover_inflight=True,
            recovery_store=store,
        )

        self.assertEqual(
            store.reclaimed, [(ref.sample_id, 1, "interrupted-capture-recovery")]
        )
        replacement = resumed.claim_batch(1)[0]
        self.assertEqual(replacement.generation, 2)
        self.complete(replacement)
        lease = resumed.wait_ready(ticket, timeout_s=1.0)
        self.assertEqual(lease.generation, 2)


class TestWindowedCaptureQueueAndSoak(WindowedRegistryFixture):
    def test_1p1c_queue_preserves_order_and_duplicate_ack_is_loud(self):
        self.registry.close()
        self.registry = self.make_registry(max_live_refs=3)
        self.initialize(samples=3, consumers=("trainer",))
        self.register("trainer", lookahead=2, prefetch_depth=3, max_outstanding=2)
        for request in self.registry.claim_batch(3):
            self.complete(request)
        queue = WindowedCaptureQueue(self.registry, "trainer", idle_timeout_s=1.0)

        first = queue.get(2)
        self.assertEqual(
            [ref.source_task_id for ref in first], ["source-0", "source-1"]
        )
        queue.ack(first)
        with self.assertRaisesRegex(RuntimeError, "not leased"):
            queue.ack(first)
        last = queue.get(1)
        queue.ack(last)
        self.assertEqual(queue.get(1), [])
        metrics = queue.metrics()
        self.assertEqual(metrics["refs"], 3)
        self.assertEqual(metrics["ready_at_request_refs"], 3)
        self.assertEqual(metrics["ready_at_request_ratio"], 1.0)
        self.assertEqual(metrics["next_fetch"], 3)
        self.assertEqual(metrics["in_flight"], 0)
        self.assertEqual(
            self.registry.snapshot()["consumers"]["trainer"]["state"],
            "completed",
        )

    def test_skewed_1p3c_soak_stays_within_slots_and_bytes(self):
        self.registry.close()
        self.registry = self.make_registry(
            max_live_refs=9,
            max_live_bytes=900,
            reservation_bytes=100,
        )
        total = 200
        consumers = ("fast", "medium", "slow")
        self.initialize(samples=total, consumers=consumers)
        self.register("fast", lookahead=4, prefetch_depth=5)
        self.register("medium", lookahead=2, prefetch_depth=3)
        self.register("slow", lookahead=0, prefetch_depth=1)
        store = _ReclaimStore()
        delivered = {consumer_id: [] for consumer_id in consumers}

        def consume_one(consumer_id: str) -> None:
            cursor = self.registry.consumer_cursor(consumer_id)
            if cursor >= total:
                return
            ticket = self.registry.request_acquire(consumer_id, cursor)
            requests = self.registry.claim_batch(16)
            if not requests:
                reclaimed = self.registry.reclaim(
                    store, limit=1, pressure=True, reason="capacity"
                )
                self.assertEqual(reclaimed, 1)
                requests = self.registry.claim_batch(16)
            for request in requests:
                self.complete(request)
            lease = self.registry.wait_ready(ticket, timeout_s=1.0)
            delivered[consumer_id].append(lease.source_index)
            self.registry.release_and_advance(consumer_id, [lease])
            self.registry.reclaim(store, limit=32)
            snapshot = self.registry.snapshot()
            self.assertLessEqual(snapshot["live_refs"], 9)
            self.assertLessEqual(snapshot["live_bytes"], 900)

        while any(self.registry.consumer_cursor(item) < total for item in consumers):
            for _ in range(5):
                consume_one("fast")
            for _ in range(2):
                consume_one("medium")
            consume_one("slow")

        for consumer_id in consumers:
            self.registry.complete_consumer(consumer_id)
        self.registry.reclaim(store, limit=32)
        snapshot = self.registry.snapshot()

        for consumer_id in consumers:
            self.assertEqual(delivered[consumer_id], list(range(total)))
        self.assertLessEqual(snapshot["peak_live_refs"], 9)
        self.assertLessEqual(snapshot["peak_live_bytes"], 900)
        self.assertEqual(snapshot["live_refs"], 0)
        self.assertEqual(self.registry.finalize_run(), "completed")


if __name__ == "__main__":
    unittest.main()
