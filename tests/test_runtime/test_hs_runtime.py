"""CPU tests of prefix-cache ownership and capture-controller lifecycle."""

import gc
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from unittest import mock

import torch
from specforge_sglang_capture.hs import (
    AdmissionBudgetError,
    CaptureCancelledError,
    CaptureCompletion,
    CaptureContractError,
    CaptureController,
    CapturedSpan,
    ExecutionIdentity,
    FeatureSchema,
    FeatureSpec,
    PrefixFeatureCache,
)


def schema():
    return FeatureSchema(
        (
            FeatureSpec("aux", 2, "float32", (1, 3)),
            FeatureSpec("last_hidden", 2, "float32", ()),
        )
    )


def identity(n=0, epoch=0):
    return ExecutionIdentity("server", f"request-{n}", f"attempt-{n}", epoch)


def features(n, start=0):
    rows = torch.arange(start * 2, (start + n) * 2, dtype=torch.float32).reshape(n, 2)
    return {"aux": rows.clone(), "last_hidden": (rows + 1000).clone()}


def populate(cache, tokens, values=None, namespace="ns"):
    values = features(len(tokens)) if values is None else values
    budget = schema().row_bytes * len(tokens)
    controller = CaptureController(cache, schema(), 2 * budget, budget, budget)
    execution = identity()
    lease = cache.lookup_and_pin(namespace, tokens, 0)
    reservation = controller.try_reserve_output(execution, len(tokens), 0, 0)
    plan = controller.freeze_plan(
        execution, namespace, tokens, len(tokens), 0, lease, reservation
    )
    try:
        span = CapturedSpan(execution, 0, len(tokens), values, None)
        controller.submit(plan, span).result(5).release()
    finally:
        controller.poll_completed()
        controller.poll_drained()
        controller.shutdown()


class GateEvent:
    def __init__(self):
        self.entered = threading.Event()
        self.finished = threading.Event()

    def synchronize(self):
        self.entered.set()
        if not self.finished.wait(5):
            raise RuntimeError("test producer event was not completed")


class CacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = PrefixFeatureCache(schema(), 4, 128)
        self.tokens = tuple(range(8))

    def test_schema_order_and_shape_are_semantic(self):
        a = FeatureSchema((FeatureSpec("aux", 2, "float32", (3, 1)),))
        b = FeatureSchema((FeatureSpec("aux", 2, "float32", (1, 3)),))
        self.assertNotEqual(a.signature, b.signature)
        self.assertEqual(schema().row_bytes, 16)
        for kwargs in [
            dict(name="unknown", width=2, dtype="float32", layer_ids=()),
            dict(name="aux", width=2, dtype="torch.float32", layer_ids=(1,)),
            dict(name="last_hidden", width=2, dtype="float32", layer_ids=(1,)),
        ]:
            with self.assertRaises(CaptureContractError):
                FeatureSpec(**kwargs)
        with self.assertRaises(CaptureContractError):
            schema().validate({"aux": torch.zeros(4, 2)}, 4)

    def test_keys_bind_entire_prefix_namespace_and_chunk_size(self):
        same_tail = (9, 10, 11, 12) + self.tokens[4:]
        keys = self.cache.block_keys("ns", self.tokens)
        self.assertEqual(len(keys[0].digest), 64)
        self.assertNotEqual(
            keys[1].digest, self.cache.block_keys("ns", same_tail)[1].digest
        )
        self.assertNotEqual(
            keys[0].digest, self.cache.block_keys("other", self.tokens)[0].digest
        )
        self.assertNotEqual(
            keys[0].digest,
            PrefixFeatureCache(schema(), 2, 128)
            .block_keys("ns", self.tokens)[0]
            .digest,
        )
        self.assertEqual(keys, self.cache.block_keys("ns", self.tokens))
        for tokens in [(True, 1, 2, 3), (-1, 1, 2, 3), (2**63, 1, 2, 3)]:
            with self.assertRaises(CaptureContractError):
                self.cache.block_keys("ns", tokens)

    def test_missing_first_block_stops_contiguous_coverage(self):
        # Retain only the later block to exercise a hole in the cache index.
        _, later = self.cache.block_keys("ns", self.tokens)
        self.assertTrue(self.cache._adopt_released_sample((later,), features(8)))
        lease = self.cache.lookup_and_pin("ns", self.tokens, 8)
        self.assertEqual(lease.coverage_end, 0)
        self.cache.release_lease(lease)

    def test_no_incomplete_tail_or_limit_chunk_reuse(self):
        populate(self.cache, self.tokens)
        for query, limit, expected in [
            (self.tokens, 8, 8),
            (self.tokens, 7, 4),
            (self.tokens[:7], 7, 4),
            (self.tokens[:3], 3, 0),
        ]:
            lease = self.cache.lookup_and_pin("ns", query, limit)
            self.assertEqual(lease.coverage_end, expected)
            self.cache.release_lease(lease)

    def test_missing_feature_adoption_retains_no_cache_blocks(self):
        keys = self.cache.block_keys("ns", self.tokens[:4])
        with self.assertRaises(CaptureContractError):
            self.cache._adopt_released_sample(keys, {"aux": torch.zeros(4, 2)})
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        lease = self.cache.lookup_and_pin("ns", self.tokens, 8)
        self.assertEqual(lease.coverage_end, 0)
        self.cache.release_lease(lease)

    def test_shrink_inside_chunk_consumes_handle_and_stale_release_is_safe(self):
        populate(self.cache, self.tokens)
        old = self.cache.lookup_and_pin("ns", self.tokens, 8)
        new = self.cache.shrink_lease(old, 3)
        self.cache.release_lease(old)
        self.assertEqual(self.cache.stats()["active_leases"], 1)
        self.assertEqual(self.cache.stats()["leased_blocks"], 1)
        out = features(8, 100)
        self.cache.copy_prefix_into(new, 3, out)
        self.assertTrue(torch.equal(out["aux"][:3], features(3)["aux"]))
        with self.assertRaises(CaptureContractError):
            self.cache.copy_prefix_into(old, 3, out)
        self.cache.release_lease(new)
        self.cache.release_lease(new)
        self.assertEqual(self.cache.stats()["active_leases"], 0)

    def test_pinned_blocks_cannot_be_evicted_and_cache_does_not_alias(self):
        values = features(8)
        populate(self.cache, self.tokens, values)
        values["aux"].fill_(-99)
        lease = self.cache.lookup_and_pin("ns", self.tokens, 8)
        different = (99, 98, 97, 96)
        populate(self.cache, different)
        blocked = self.cache.lookup_and_pin("ns", different, 4)
        self.assertEqual(blocked.coverage_end, 0)
        self.cache.release_lease(blocked)
        out = features(8)
        self.cache.copy_prefix_into(lease, 8, out)
        self.assertTrue(torch.equal(out["aux"], features(8)["aux"]))
        out["aux"].fill_(-999)
        self.cache.copy_prefix_into(lease, 8, out)
        self.assertTrue(torch.equal(out["aux"], features(8)["aux"]))
        self.cache.release_lease(lease)
        populate(self.cache, different)
        self.assertEqual(self.cache.stats()["cache_bytes"], 64)
        self.assertEqual(self.cache.stats()["evictions"], 2)

    def test_concurrent_adoption_retains_one_owner_for_duplicate_keys(self):
        keys = self.cache.block_keys("ns", self.tokens)
        with ThreadPoolExecutor(8) as pool:
            results = list(
                pool.map(
                    lambda _: self.cache._adopt_released_sample(keys, features(8)),
                    range(8),
                )
            )
        self.assertEqual(results.count(True), 1)
        self.assertEqual(self.cache.stats()["ready_blocks"], 2)
        self.assertEqual(self.cache.stats()["cache_owned_groups"], 1)
        self.assertEqual(self.cache.stats()["cache_bytes"], 128)

    def test_lookup_pin_is_atomic_against_eviction(self):
        cache = PrefixFeatureCache(schema(), 4, 64)
        populate(cache, self.tokens[:4])
        entered, resume = threading.Event(), threading.Event()
        complete = cache._complete

        def blocked(entry):
            entered.set()
            self.assertTrue(resume.wait(3))
            return complete(entry)

        with (
            mock.patch.object(cache, "_complete", blocked),
            ThreadPoolExecutor(2) as pool,
        ):
            lookup = pool.submit(cache.lookup_and_pin, "ns", self.tokens[:4], 4)
            self.assertTrue(entered.wait(3))
            key = cache.block_keys("ns", (8, 9, 10, 11))[0]
            eviction = pool.submit(cache._adopt_released_sample, (key,), features(4))
            resume.set()
            lease = lookup.result(3)
            self.assertEqual(lease.coverage_end, 4)
            self.assertFalse(eviction.result(3))
        cache.release_lease(lease)

    def test_release_during_cpu_reader_defers_reclaim(self):
        cache = PrefixFeatureCache(schema(), 4, 64)
        populate(cache, self.tokens[:4])
        lease = cache.lookup_and_pin("ns", self.tokens[:4], 4)
        dest = features(4)
        entered, resume = threading.Event(), threading.Event()
        real_copy = torch.Tensor.copy_

        def blocked(tensor, *args, **kwargs):
            entered.set()
            self.assertTrue(resume.wait(3))
            return real_copy(tensor, *args, **kwargs)

        with (
            mock.patch.object(torch.Tensor, "copy_", blocked),
            ThreadPoolExecutor(1) as pool,
        ):
            reader = pool.submit(cache.copy_prefix_into, lease, 4, dest)
            self.assertTrue(entered.wait(3))
            cache.release_lease(lease)
            keys = cache.block_keys("ns", (8, 9, 10, 11))
            self.assertFalse(cache._adopt_released_sample(keys, features(4)))
            resume.set()
            reader.result(3)
        self.assertTrue(cache._adopt_released_sample(keys, features(4)))
        self.assertEqual(cache.stats()["active_leases"], 0)

    def test_failed_adoption_releases_staged_views_and_preserves_empty_index(self):
        keys = self.cache.block_keys("ns", self.tokens)
        values = features(8)
        original = torch.Tensor.__getitem__
        staged = []
        retained_errors = []

        def slice_feature(tensor, index):
            view = original(tensor, index)
            if staged:
                error = RuntimeError("adoption view creation failed")
                error.view = view
                raise error
            staged.append(weakref.ref(view))
            return view

        with mock.patch.object(torch.Tensor, "__getitem__", slice_feature):
            try:
                self.cache._adopt_released_sample(keys, values)
            except RuntimeError as error:
                retained_errors.append(error)
        gc.collect()
        self.assertEqual(len(retained_errors), 1)
        self.assertEqual(len(staged), 1)
        self.assertIsNone(staged[0]())
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        self.assertEqual(self.cache.stats()["ready_blocks"], 0)

    def test_owner_group_charges_entire_storage_and_any_lease_blocks_eviction(self):
        cache = PrefixFeatureCache(schema(), 4, 160)
        tokens = tuple(range(9))
        populate(cache, tokens)
        self.assertEqual(cache.stats()["cache_bytes"], 144)
        self.assertEqual(cache.stats()["cache_logical_bytes"], 128)
        lease = cache.lookup_and_pin("ns", tokens, 4)
        different = (9, 8, 7, 6)
        populate(cache, different, namespace="other")
        self.assertEqual(cache.stats()["ready_blocks"], 2)
        self.assertEqual(cache.stats()["cache_bytes"], 144)
        out = features(4, 100)
        cache.copy_prefix_into(lease, 4, out)
        self.assertTrue(torch.equal(out["aux"], features(4)["aux"]))
        cache.release_lease(lease)
        populate(cache, different, namespace="other")
        self.assertEqual(cache.stats()["ready_blocks"], 1)
        self.assertEqual(cache.stats()["cache_owned_groups"], 1)
        self.assertEqual(cache.stats()["cache_bytes"], 64)
        self.assertEqual(cache.stats()["evictions"], 2)

    def test_contiguous_owner_prefix_uses_one_copy_per_feature(self):
        cache = PrefixFeatureCache(schema(), 4, 160)
        tokens = tuple(range(9))
        populate(cache, tokens)
        lease = cache.lookup_and_pin("ns", tokens, 8)
        out = features(8, 100)
        copies = []
        real_copy = torch.Tensor.copy_

        def copy(target, source, *args, **kwargs):
            copies.append(tuple(target.shape))
            return real_copy(target, source, *args, **kwargs)

        with mock.patch.object(torch.Tensor, "copy_", copy):
            cache.copy_prefix_into(lease, 8, out)
        cache.release_lease(lease)
        self.assertEqual(copies, [(8, 2), (8, 2)])
        self.assertTrue(torch.equal(out["aux"], features(8)["aux"]))

    def test_adoption_counts_hidden_backing_storage_and_rejects_oversize(self):
        values = {name: tensor[:4] for name, tensor in features(9).items()}
        keys = self.cache.block_keys("ns", self.tokens[:4])
        self.assertFalse(self.cache._adopt_released_sample(keys, values))
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        larger = PrefixFeatureCache(schema(), 4, 160)
        self.assertTrue(larger._adopt_released_sample(keys, values))
        self.assertEqual(larger.stats()["cache_bytes"], 144)
        self.assertEqual(larger.stats()["cache_logical_bytes"], 64)


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.cache = PrefixFeatureCache(schema(), 4, 1024)
        self.controller = CaptureController(self.cache, schema(), 512, 512, 512)
        self.samples = []

    def tearDown(self):
        for sample in self.samples:
            sample.release()
        self.controller.poll_completed()
        self.controller.poll_drained()
        self.controller.shutdown()

    def plan(self, n=8, L=0, execution=None, namespace="ns", passthrough=0):
        execution = identity() if execution is None else execution
        tokens = tuple(range(n))
        lease = self.cache.lookup_and_pin(namespace, tokens, n)
        self.assertGreaterEqual(lease.coverage_end, L)
        lease = self.cache.shrink_lease(lease, L)
        reservation = self.controller.try_reserve_output(execution, n, L, passthrough)
        self.assertIsNotNone(reservation)
        return self.controller.freeze_plan(
            execution, namespace, tokens, n, L, lease, reservation
        )

    def submit(self, plan, event=None):
        span = CapturedSpan(
            plan.execution, plan.L, plan.N, features(plan.N - plan.L, plan.L), event
        )
        return self.controller.submit(plan, span)

    def take(self, future):
        sample = future.result(5)
        self.samples.append(sample)
        return sample

    def test_full_sample_from_non_aligned_prefix_and_suffix(self):
        cold = self.take(self.submit(self.plan(n=9)))
        self.assertTrue(torch.equal(cold.features["aux"], features(9)["aux"]))
        self.assertEqual(self.cache.stats()["ready_blocks"], 0)
        cold.release()
        self.assertEqual(self.cache.stats()["ready_blocks"], 2)
        self.controller.poll_completed()
        self.controller.poll_drained()
        plan = self.plan(n=9, L=5, execution=identity(1))
        warm = self.take(self.submit(plan))
        for name in warm.features:
            self.assertTrue(torch.equal(warm.features[name], features(9)[name]))
        warm.features["aux"].fill_(-99)
        lease = self.cache.lookup_and_pin("ns", tuple(range(9)), 9)
        out = features(9)
        self.cache.copy_prefix_into(lease, 8, out)
        self.cache.release_lease(lease)
        self.assertTrue(torch.equal(out["aux"][:8], features(8)["aux"]))
        self.assertEqual(self.controller.stats()["active_leases"], 0)

    def test_output_charge_survives_completion_and_cancel_until_sink_release(self):
        self.controller = CaptureController(self.cache, schema(), 512, 128, 512)
        sample = self.take(self.submit(self.plan()))
        self.assertEqual(self.controller.stats()["output_bytes"], 128)
        self.assertEqual(self.controller.stats()["pinned_bytes"], 0)
        self.assertIsNone(self.controller.try_reserve_output(identity(1), 8, 0, 0))
        drain = self.controller.invalidate(
            identity(), "HTTP timeout after publication started"
        )
        self.assertFalse(drain.wait(0.01))
        self.assertEqual(sample.features["aux"].shape, (8, 2))
        sample.release()
        self.assertTrue(drain.wait(1))
        self.assertEqual(self.controller.stats()["output_bytes"], 0)
        with self.assertRaises(CaptureContractError):
            _ = sample.features

    def test_hard_budget_oversize_and_passthrough(self):
        for args in [(100, 0, 0), (8, 0, 500)]:
            with self.assertRaises(AdmissionBudgetError):
                self.controller.try_reserve_output(identity(), *args)
        self.assertEqual(self.controller.stats()["output_reservations"], 0)
        res = self.controller.try_reserve_output(identity(), 8, 0, 17)
        self.assertEqual(res.host_bytes, 145)
        self.controller.release_output(res)
        self.controller.release_output(res)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)

    def test_pinned_and_device_pressure_are_independent_hard_limits(self):
        for limits in [(128, 1024, 1024), (1024, 1024, 128)]:
            with self.subTest(limits=limits):
                controller = CaptureController(self.cache, schema(), *limits)
                reserved = controller.try_reserve_output(identity(30), 8, 0, 0)
                self.assertIsNone(controller.try_reserve_output(identity(31), 4, 0, 0))
                self.assertEqual(controller.stats()["pinned_bytes"], 128)
                self.assertEqual(controller.stats()["device_bytes"], 128)
                with self.assertRaises(AdmissionBudgetError):
                    controller.try_reserve_output(identity(32), 9, 0, 0)
                controller.release_output(reserved)
                self.assertEqual(controller.stats()["output_bytes"], 0)
                controller.shutdown()

    def test_admission_stats_count_retries_and_hard_oversize(self):
        for kind, limits in (
            ("pinned", (128, 1024, 1024)),
            ("output", (1024, 128, 1024)),
            ("device", (1024, 1024, 128)),
        ):
            with self.subTest(kind=kind):
                controller = CaptureController(self.cache, schema(), *limits)
                try:
                    held = controller.try_reserve_output(identity(70), 8, 0, 0)
                    for _ in range(2):
                        self.assertIsNone(
                            controller.try_reserve_output(identity(71), 4, 0, 0)
                        )
                    with self.assertRaises(AdmissionBudgetError):
                        controller.try_reserve_output(identity(72), 100, 0, 0)
                    stats = controller.stats()
                    self.assertEqual(stats["admission_attempts"], 4)
                    self.assertEqual(stats["admission_reserved_attempts"], 1)
                    self.assertEqual(stats["admission_budget_declined_attempts"], 2)
                    self.assertEqual(stats["admission_hard_oversize_attempts"], 1)
                    controller.release_output(held)
                finally:
                    controller.shutdown()

    def test_controller_pending_and_fragmentation_decline_admission(self):
        self.controller._max_pending = 1
        held = self.controller.try_reserve_output(identity(80), 1, 0, 0)
        self.assertIsNone(self.controller.try_reserve_output(identity(81), 1, 0, 0))
        self.controller.release_output(held)
        self.controller.poll_drained()
        self.assertEqual(
            self.controller.stats()["admission_budget_declined_attempts"], 1
        )
        self.controller._max_pending = 4096
        # Four 128-byte slots fill the 512-byte arena. Two separated free
        # slots have enough total bytes for 192, but no contiguous DMA slot.
        slots = [
            self.controller.try_reserve_output(identity(90 + i), 8, 0, 0)
            for i in range(4)
        ]
        self.controller.release_output(slots[0])
        self.controller.release_output(slots[2])
        self.controller.poll_drained()
        self.assertIsNone(self.controller.try_reserve_output(identity(94), 12, 0, 0))
        stats = self.controller.stats()
        self.assertEqual(stats["admission_budget_declined_attempts"], 2)
        self.assertEqual(stats["pinned_free_bytes"], 256)
        self.assertEqual(stats["pinned_free_slots"], 2)
        for slot in slots:
            self.controller.release_output(slot)

    def test_pinned_arena_is_one_retained_bounded_allocation_across_sizes(self):
        real_empty = torch.empty
        allocations = []

        def allocate(*args, **kwargs):
            self.assertTrue(kwargs.pop("pin_memory"))
            value = real_empty(*args, **kwargs)
            allocations.append(value)
            return value

        with mock.patch("torch.empty", allocate):
            for rows in (1, 8, 3, 16, 2):
                views = self.controller._staging_views(rows)
                for spec in schema().features:
                    value = views[spec.name]
                    self.assertEqual(tuple(value.shape), (rows, spec.width))
                    self.assertEqual(
                        value.untyped_storage().data_ptr(),
                        self.controller._pinned_arena.untyped_storage().data_ptr(),
                    )
        self.assertEqual(len(allocations), 1)
        self.assertEqual(allocations[0].numel(), 512)
        self.assertEqual(self.controller.stats()["pinned_arena_bytes"], 512)
        self.assertEqual(self.controller.stats()["pinned_bytes"], 0)

    def test_pinned_arena_capacity_is_conservative_and_alignment_is_charged(self):
        controller = CaptureController(self.cache, schema(), 96, 1024, 1024)
        self.assertEqual(controller.stats()["pinned_capacity_bytes"], 64)
        with self.assertRaises(AdmissionBudgetError):
            controller.try_reserve_output(identity(60), 5, 0, 0)
        controller.shutdown()
        mixed = FeatureSchema(
            (
                FeatureSpec("aux", 1, "bfloat16", (1,)),
                FeatureSpec("last_hidden", 1, "float32", ()),
            )
        )
        controller = CaptureController(
            PrefixFeatureCache(mixed, 1, 64), mixed, 16, 64, 64
        )
        reservation = controller.try_reserve_output(identity(61), 1, 0, 0)
        self.assertEqual(reservation.pinned_bytes, 8)
        self.assertEqual(reservation.device_bytes, 6)
        controller.release_output(reservation)
        controller.shutdown()

    def test_quarantined_arena_forbids_new_reservation_and_landing_views(self):
        self.controller._arena_poisoned = True
        with self.assertRaisesRegex(CaptureContractError, "quarantined"):
            self.controller.try_reserve_output(identity(62), 1, 0, 0)
        with self.assertRaisesRegex(CaptureContractError, "quarantined"):
            self.controller._staging_views(1)

    def test_admission_rollback_releases_frozen_lease_and_all_budgets(self):
        populate(self.cache, tuple(range(8)))
        plan = self.plan(L=4)
        self.controller.release_output(plan.reservation)
        self.cache.release_lease(plan.lease)
        for key in ("output_bytes", "pinned_bytes", "device_bytes", "active_leases"):
            self.assertEqual(self.controller.stats()[key], 0)
        with self.assertRaises(CaptureContractError):
            self.submit(plan)

    def test_cancel_waits_producer_event_and_future_cannot_cancel_cleanup(self):
        plan = self.plan()
        gate = GateEvent()
        future = self.submit(plan, gate)
        self.assertTrue(gate.entered.wait(3))
        self.assertFalse(future.cancel())
        drain = self.controller.invalidate(plan.execution, "retracted")
        self.assertFalse(drain.done())
        self.assertEqual(self.controller.stats()["device_bytes"], 128)
        gate.finished.set()
        with self.assertRaises(CaptureCancelledError):
            future.result(3)
        self.assertTrue(drain.wait(1))
        for key in (
            "output_bytes",
            "pinned_bytes",
            "device_bytes",
            "active_leases",
            "pending_copies",
        ):
            self.assertEqual(self.controller.stats()[key], 0)
        self.assertEqual(self.cache.stats()["ready_blocks"], 0)

    def test_old_epoch_completion_cannot_complete_new_execution(self):
        old = self.plan(execution=identity(0, 0))
        gate = GateEvent()
        old_future = self.submit(old, gate)
        self.assertTrue(gate.entered.wait(3))
        self.controller.invalidate(old.execution, "retract")
        new = self.plan(execution=identity(0, 1))
        new_future = self.submit(new)
        gate.finished.set()
        with self.assertRaises(CaptureCancelledError):
            old_future.result(3)
        self.take(new_future)
        completions = self.controller.poll_completed()
        self.assertEqual(
            {c.execution for c in completions}, {old.execution, new.execution}
        )
        self.assertIsInstance(
            next(c.error for c in completions if c.execution == old.execution),
            CaptureCancelledError,
        )
        self.assertIsNotNone(
            next(c.sample for c in completions if c.execution == new.execution)
        )

    def test_wrong_span_or_underlying_storage_is_rejected_before_submit(self):
        plan = self.plan()
        with self.assertRaises(CaptureContractError):
            self.controller.submit(
                plan, CapturedSpan(plan.execution, 1, 8, features(7), None)
            )
        too_large = features(8)
        too_large["aux"] = torch.arange(1000, dtype=torch.float32)[:16].reshape(8, 2)
        with self.assertRaises(CaptureContractError):
            self.controller.submit(
                plan, CapturedSpan(plan.execution, 0, 8, too_large, None)
            )
        self.assertEqual(self.controller.stats()["pending_copies"], 0)
        self.controller.release_output(plan.reservation)

    def test_worker_failure_drains_and_future_reports_original_error(self):
        plan = self.plan()
        with mock.patch.object(
            self.controller,
            "_copy_suffix",
            side_effect=RuntimeError("D2H submission failed"),
        ):
            future = self.submit(plan)
            with self.assertRaisesRegex(RuntimeError, "D2H submission failed"):
                future.result(3)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)
        self.assertEqual(self.controller.stats()["device_bytes"], 0)
        self.assertEqual(self.cache.stats()["active_leases"], 0)
        completions = self.controller.poll_completed()
        self.assertEqual(len(completions), 1)
        self.assertIsNotNone(completions[0].error)

    def test_cache_adoption_failure_keeps_complete_sample_delivery(self):
        sample = self.take(self.submit(self.plan()))
        self.assertTrue(torch.equal(sample.features["aux"], features(8)["aux"]))
        with mock.patch.object(
            self.cache,
            "_adopt_released_sample",
            side_effect=RuntimeError("cache metadata failed"),
        ):
            sample.release()
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        self.assertEqual(self.controller.stats()["fill_failures"], 1)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)

    def test_cache_capacity_declines_adoption_without_blocking_release(self):
        self.cache.capacity_bytes = 64
        sample = self.take(self.submit(self.plan()))
        sample.release()
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)

    def test_release_transfers_same_storage_without_allocating_or_copying(self):
        sample = self.take(self.submit(self.plan(n=9)))
        pointers = {
            name: tensor.untyped_storage().data_ptr()
            for name, tensor in sample.features.items()
        }
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        self.assertEqual(self.controller.stats()["output_bytes"], 144)
        original = self.cache._adopt_released_sample

        def adopt(keys, values):
            # The output charge remains live throughout cache admission.
            self.assertEqual(self.controller.stats()["output_bytes"], 144)
            return original(keys, values)

        with (
            mock.patch.object(self.cache, "_adopt_released_sample", adopt),
            mock.patch(
                "torch.empty", side_effect=AssertionError("release allocated a tensor")
            ),
            mock.patch.object(
                torch.Tensor,
                "copy_",
                side_effect=AssertionError("release copied a tensor"),
            ),
        ):
            sample.release()
        self.assertEqual(self.controller.stats()["output_bytes"], 0)
        self.assertEqual(self.cache.stats()["cache_bytes"], 144)
        self.assertEqual(self.cache.stats()["cache_logical_bytes"], 128)
        self.assertEqual(self.cache.stats()["cache_owned_groups"], 1)
        for entry in self.cache._entries.values():
            for name, tensor in entry.features.items():
                self.assertEqual(tensor.untyped_storage().data_ptr(), pointers[name])
        with self.assertRaises(CaptureContractError):
            _ = sample.features

    def test_cancelled_sample_does_not_transfer_output_to_cache(self):
        plan = self.plan()
        sample = self.take(self.submit(plan))
        drain = self.controller.invalidate(plan.execution, "retract after assembly")
        self.assertFalse(drain.done())
        sample.release()
        self.assertTrue(drain.wait(1))
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)

    def test_no_complete_chunk_does_not_retain_full_output(self):
        sample = self.take(self.submit(self.plan(n=3)))
        references = [weakref.ref(tensor) for tensor in sample.features.values()]
        sample.release()
        gc.collect()
        self.assertEqual(self.cache.stats()["cache_bytes"], 0)
        self.assertTrue(all(reference() is None for reference in references))

    def test_workers_assemble_independent_jobs_concurrently(self):
        gate = GateEvent()
        first = self.submit(self.plan(execution=identity(60)), gate)
        self.assertTrue(gate.entered.wait(3))
        second = self.submit(self.plan(execution=identity(61)))
        second_sample = self.take(second)
        self.assertFalse(first.done())
        self.assertTrue(torch.equal(second_sample.features["aux"], features(8)["aux"]))
        gate.finished.set()
        self.take(first)

    def test_reserved_slots_are_disjoint_and_coalesce_only_after_release(self):
        first = self.controller.try_reserve_output(identity(70), 8, 0, 0)
        second = self.controller.try_reserve_output(identity(71), 8, 0, 0)
        real_empty = torch.empty

        def allocate(*args, **kwargs):
            kwargs.pop("pin_memory", None)
            return real_empty(*args, **kwargs)

        with mock.patch("torch.empty", allocate):
            a = self.controller._staging_views(8, first)
            b = self.controller._staging_views(8, second)
            for value in a.values():
                value.fill_(10)
            for value in b.values():
                value.fill_(20)
            self.assertTrue(all(bool(torch.all(value == 10)) for value in a.values()))
            self.assertTrue(all(bool(torch.all(value == 20)) for value in b.values()))
            # No reader remains when a slot is explicitly released here.
            a.clear()
            self.controller.release_output(first)
            with self.assertRaises(CaptureContractError):
                self.controller._staging_views(8, first)
            third = self.controller.try_reserve_output(identity(72), 8, 0, 0)
            c = self.controller._staging_views(8, third)
            for value in c.values():
                value.fill_(30)
            self.assertTrue(all(bool(torch.all(value == 20)) for value in b.values()))
            b.clear()
            c.clear()
        self.controller.release_output(second)
        self.controller.release_output(third)
        self.assertEqual(self.controller._pinned_free, [(0, 512)])

    def test_corrupt_lease_fails_instead_of_zero_filling(self):
        populate(self.cache, tuple(range(8)))
        plan = self.plan(L=4)
        entry = self.cache._entries[self.cache.block_keys("ns", tuple(range(8)))[0]]
        entry.features.pop("last_hidden")
        future = self.submit(plan)
        with self.assertRaisesRegex(CaptureContractError, "missing or corrupt"):
            future.result(3)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)

    def test_shutdown_nonblocking_keeps_inflight_storage_until_actual_drain(self):
        plan = self.plan()
        gate = GateEvent()
        future = self.submit(plan, gate)
        self.assertTrue(gate.entered.wait(3))
        self.controller.shutdown(wait=False)
        self.assertEqual(self.controller.stats()["device_bytes"], 128)
        gate.finished.set()
        with self.assertRaises(CaptureCancelledError):
            future.result(3)
        self.controller.shutdown(wait=True)
        self.assertEqual(self.controller.stats()["output_bytes"], 0)

    def test_partial_copy_and_failed_wait_drain_both_owners_without_traceback_storage(
        self,
    ):
        # CPU tensors plus fully mocked CUDA entry points test scheduling/lifetime
        # logic deterministically. This is not CUDA numerical/device evidence.
        real_copy, real_empty = torch.Tensor.copy_, torch.empty
        for fail_wait in (False, True):
            with self.subTest(fail_wait=fail_wait):
                plan = self.plan(execution=identity(20 + int(fail_wait)))
                allocated, retained_errors = [], []
                copies = [0]

                class Producer:
                    syncs = 0

                    def synchronize(producer):
                        producer.syncs += 1

                producer = Producer()

                class Stream:
                    syncs = 0

                    def wait_event(stream, event):
                        if fail_wait:
                            raise RuntimeError("wait_event failed before dependency")

                    def synchronize(stream):
                        stream.syncs += 1

                stream = Stream()

                def allocate(*args, **kwargs):
                    kwargs.pop("pin_memory", None)
                    tensor = real_empty(*args, **kwargs)
                    allocated.append(weakref.ref(tensor))
                    return tensor

                def copy(tensor, *args, **kwargs):
                    copies[0] += 1
                    if copies[0] == 2:
                        raise RuntimeError(tensor)
                    return real_copy(tensor, *args, **kwargs)

                original = self.controller._copy_suffix

                def pretend_cuda(job):
                    job.is_cuda = True
                    return original(job)

                with (
                    mock.patch.object(self.controller, "_copy_suffix", pretend_cuda),
                    mock.patch("torch.cuda.device", return_value=nullcontext()),
                    mock.patch("torch.cuda.stream", return_value=nullcontext()),
                    mock.patch("torch.cuda.Stream", return_value=stream),
                    mock.patch("torch.empty", allocate),
                    mock.patch.object(torch.Tensor, "record_stream", return_value=None),
                    mock.patch.object(torch.Tensor, "copy_", copy),
                ):
                    future = self.submit(plan, producer)
                    try:
                        future.result(3)
                    except RuntimeError as error:
                        retained_errors.append(error)
                gc.collect()
                self.assertEqual(len(retained_errors), 1)
                self.assertEqual(producer.syncs, 1)
                self.assertEqual(stream.syncs, 1)
                # The single explicitly accounted arena intentionally survives
                # safe failures; no per-copy views/source tensors may survive.
                self.assertTrue(
                    all(
                        ref() is None or ref() is self.controller._pinned_arena
                        for ref in allocated
                    )
                )
                expected_arena = (
                    self.controller._pinned_arena.numel()
                    if self.controller._pinned_arena is not None
                    else 0
                )
                self.assertEqual(
                    self.controller.stats()["pinned_arena_bytes"], expected_arena
                )
                self.assertLessEqual(
                    expected_arena, self.controller.stats()["pinned_capacity_bytes"]
                )
                for key in (
                    "output_bytes",
                    "pinned_bytes",
                    "device_bytes",
                    "pending_copies",
                ):
                    self.assertEqual(self.controller.stats()[key], 0)
                self.controller.poll_completed()
                self.controller.poll_drained()

    def test_unproven_event_drain_quarantines_without_budget_reuse(self):
        cache = PrefixFeatureCache(schema(), 4, 128)
        controller = CaptureController(cache, schema(), 128, 128, 128)
        execution = identity(99)
        tokens = tuple(range(8))
        lease = cache.lookup_and_pin("ns", tokens, 8)
        reservation = controller.try_reserve_output(execution, 8, 0, 0)
        plan = controller.freeze_plan(execution, "ns", tokens, 8, 0, lease, reservation)

        class UnprovenEvent:
            def synchronize(self):
                raise RuntimeError("producer drain unavailable")

        future = controller.submit(
            plan, CapturedSpan(execution, 0, 8, features(8), UnprovenEvent())
        )
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            future.result(3)
        drain = controller.invalidate(execution, "cancel after failed event")
        self.assertFalse(drain.wait(0.01))
        controller.release_output(reservation)
        self.assertEqual(controller.stats()["device_bytes"], 128)
        self.assertEqual(controller.stats()["quarantined_executions"], 1)
        with self.assertRaisesRegex(RuntimeError, "not drained"):
            controller.shutdown(wait=True)

    def test_completion_requires_exactly_one_outcome(self):
        with self.assertRaises(CaptureContractError):
            CaptureCompletion(identity())
        with self.assertRaises(CaptureContractError):
            CaptureCompletion(identity(), sample=object(), error=RuntimeError())


if __name__ == "__main__":
    unittest.main()
