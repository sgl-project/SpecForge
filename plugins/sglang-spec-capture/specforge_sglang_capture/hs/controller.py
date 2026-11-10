"""Bounded capture ownership, asynchronous D2H, full assembly and safe drain.

B hands over compact independently owned suffix snapshots after admission.
Workers never touch a Req, execute a collective, or release sink-owned output.
"""

from __future__ import annotations

import os
import resource
import threading
import uuid
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor
from time import monotonic_ns, perf_counter
from dataclasses import dataclass, field

import torch

from .cache import PrefixFeatureCache, _token_tuple
from .protocol import (
    DTYPES, DTYPE_BYTES, AdmissionBudgetError, CaptureCancelledError, CaptureCompletion,
    CaptureContractError, CapturePlan, CapturedSpan, DrainHandle, ExecutionIdentity,
    ContentBlockKey, FeatureSchema, MaterializedSample, OutputReservation, PrefixLease,
    canonical_digest, checked_int, _normalized_error,
)


@dataclass
class _Reservation:
    handle: OutputReservation
    submitted: bool = False
    transient_released: bool = False
    release_requested: bool = False
    pinned_offset: int = 0


@dataclass
class _Execution:
    identity: ExecutionIdentity
    reservations: set[str] = field(default_factory=set)
    plan: CapturePlan | None = None
    cancelled: bool = False
    reason: str = ""
    jobs: int = 0
    sample: MaterializedSample | None = None
    owned_output: dict[str, torch.Tensor] = field(default_factory=dict)
    cache_keys: tuple[ContentBlockKey, ...] = ()
    fills: dict[str, object] = field(default_factory=dict)
    quarantined: list[object] = field(default_factory=list)
    completion_pending: bool = False
    drained_queued: bool = False
    drained_polled: bool = False
    drained: threading.Event = field(default_factory=threading.Event)


@dataclass
class _Job:
    plan: CapturePlan
    features: dict[str, torch.Tensor]
    ready_event: object | None
    is_cuda: bool
    state: _Execution
    future: Future
    diagnostics: dict[str, int | float] = field(default_factory=dict)


class _UnsafeDrain(RuntimeError):
    """The runtime must retain storage when CUDA/producer drain is unproven."""

    def __init__(self, error, retained):
        super().__init__(f"capture storage quarantined: drain could not be proven: {error}")
        self.retained = retained


class CaptureController:
    def __init__(self, cache: PrefixFeatureCache, schema: FeatureSchema,
                 pinned_budget_bytes: int = 256 * 1024**2,
                 output_budget_bytes: int = 512 * 1024**2,
                 device_budget_bytes: int = 256 * 1024**2, *,
                 max_workers: int = 4):
        if not isinstance(cache, PrefixFeatureCache) or schema != cache.schema:
            raise CaptureContractError("controller/cache schema mismatch")
        self.cache, self.schema = cache, schema
        self._diagnostics = os.environ.get("SPECFORGE_HS_DIAGNOSTICS", "0") == "1"
        self._diagnostic_every = max(1, int(os.environ.get("SPECFORGE_HS_DIAGNOSTICS_EVERY", "1")))
        self._diagnostic_limit = max(0, int(os.environ.get("SPECFORGE_HS_DIAGNOSTICS_MAX", "4096")))
        self._diagnostic_sequence = 0
        self._limits = {
            "pinned": checked_int(pinned_budget_bytes, "pinned_budget_bytes"),
            "output": checked_int(output_budget_bytes, "output_budget_bytes"),
            "device": checked_int(device_budget_bytes, "device_budget_bytes"),
        }
        # PyTorch rounds pinned allocations up to a power of two and retains
        # freed size buckets. One owned arena prevents cumulative idle buckets
        # from bypassing the configured hard bound. Default budgets are exact
        # powers of two; other budgets conservatively expose a smaller arena.
        self._pinned_configured_bytes = self._limits["pinned"]
        self._limits["pinned"] = (1 << (pinned_budget_bytes.bit_length() - 1)
                                  if pinned_budget_bytes else 0)
        self._pinned_arena = None
        self._slot_alignment = max(DTYPE_BYTES[f.dtype] for f in schema.features)
        self._pinned_free = [(0, self._limits["pinned"])]
        self._max_workers = checked_int(max_workers, "max_workers", 1)
        self._arena_poisoned = False
        self._current = dict.fromkeys(self._limits, 0)
        self._peaks = dict.fromkeys(self._limits, 0)
        self._lock = threading.RLock()
        self._reservations: dict[str, _Reservation] = {}
        self._executions: dict[ExecutionIdentity, _Execution] = {}
        self._completed: deque[CaptureCompletion] = deque()
        self._drained: deque[ExecutionIdentity] = deque()
        self._max_pending = max(1, min(4096, output_budget_bytes // schema.row_bytes))
        self._closed = False
        self._fill_failures = 0
        # Admission attempts include retries; these are not unique-request counts.
        # Reasons may overlap when more than one budget is currently exhausted.
        self._admission_stats = dict.fromkeys((
            "admission_attempts", "admission_reserved_attempts",
            "admission_budget_declined_attempts", "admission_hard_oversize_attempts",
            "admission_declined_pinned_attempts", "admission_declined_output_attempts",
            "admission_declined_device_attempts", "admission_declined_controller_max_pending_attempts",
            "admission_declined_fragmentation_attempts",
        ), 0)
        self._adoption_calls = 0
        self._adoption_unused_skips = 0
        self._adoption_total_ms = self._adoption_max_ms = 0.0
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers, thread_name_prefix="spec-feature-capture")

    def _staging_layout(self, rows):
        offset, layout = 0, {}
        for spec in self.schema.features:
            itemsize = DTYPE_BYTES[spec.dtype]
            offset = ((offset + itemsize - 1) // itemsize) * itemsize
            size = rows * spec.width * itemsize
            layout[spec.name] = (offset, size, spec)
            offset += size
        offset = ((offset + self._slot_alignment - 1) // self._slot_alignment) * self._slot_alignment
        return offset, layout

    def _staging_views(self, rows, reservation=None):
        required, layout = self._staging_layout(rows)
        if required > self._limits["pinned"]:
            raise AdmissionBudgetError("aligned suffix exceeds pinned arena capacity")
        with self._lock:
            if self._arena_poisoned:
                raise CaptureContractError("pinned arena quarantined after unproven DMA drain")
            offset = 0
            if reservation is not None:
                rec = self._reservation(reservation)
                if rec.transient_released or required > reservation.pinned_bytes:
                    raise CaptureContractError("staging slot is released or too small")
                offset = rec.pinned_offset
            elif self._current["pinned"]:
                raise CaptureContractError("unreserved staging view while slots are active")
            # Allocate once under the lock: workers must never race to create
            # two retained power-of-two pinned allocations.
            if self._pinned_arena is None:
                self._pinned_arena = torch.empty((self._limits["pinned"],), dtype=torch.uint8,
                                                device="cpu", pin_memory=True)
            return {name: self._pinned_arena[offset + start:offset + start + size]
                    .view(DTYPES[spec.dtype]).view(rows, spec.width)
                    for name, (start, size, spec) in layout.items()}

    def _allocate_slot_locked(self, amount):
        if not amount:
            return 0
        for i, (start, size) in enumerate(self._pinned_free):
            if size >= amount:
                if size == amount:
                    self._pinned_free.pop(i)
                else:
                    self._pinned_free[i] = (start + amount, size - amount)
                return start
        return None

    def _free_slot_locked(self, start, amount):
        if not amount:
            return
        self._pinned_free.append((start, amount))
        merged = []
        for offset, size in sorted(self._pinned_free):
            if merged and merged[-1][0] + merged[-1][1] == offset:
                old, length = merged[-1]
                merged[-1] = (old, length + size)
            else:
                merged.append((offset, size))
        self._pinned_free = merged

    def try_reserve_output(self, execution: ExecutionIdentity, N: int, L: int,
                           passthrough_bytes: int) -> OutputReservation | None:
        if not isinstance(execution, ExecutionIdentity):
            raise CaptureContractError("reservation requires execution identity")
        checked_int(N, "N", 1)
        checked_int(L, "L")
        checked_int(passthrough_bytes, "passthrough_bytes")
        if L > N:
            raise CaptureContractError("L exceeds N")
        charges = {"output": N * self.schema.row_bytes + passthrough_bytes,
                   "pinned": self._staging_layout(N - L)[0],
                   "device": (N - L) * self.schema.row_bytes}
        with self._lock:
            self._admission_stats["admission_attempts"] += 1
            for kind, amount in charges.items():
                if amount > self._limits[kind]:
                    self._admission_stats["admission_hard_oversize_attempts"] += 1
                    raise AdmissionBudgetError(f"request needs {amount} {kind} bytes; hard limit {self._limits[kind]}")
            if self._closed:
                raise CaptureContractError("capture controller is shut down")
            if self._arena_poisoned:
                raise CaptureContractError("capture staging quarantined; restart required")
            if execution in self._executions:
                raise CaptureContractError("execution identity already active; retry needs a new execution epoch")
            reasons = [kind for kind in charges
                       if self._current[kind] + charges[kind] > self._limits[kind]]
            if len(self._executions) >= self._max_pending:
                reasons.append("controller_max_pending")
            if reasons:
                self._admission_stats["admission_budget_declined_attempts"] += 1
                for reason in reasons:
                    self._admission_stats[f"admission_declined_{reason}_attempts"] += 1
                return None
            slot = self._allocate_slot_locked(charges["pinned"])
            if slot is None:
                self._admission_stats["admission_budget_declined_attempts"] += 1
                self._admission_stats["admission_declined_fragmentation_attempts"] += 1
                return None  # Fragmentation cannot authorize overlapping DMA slots.
            handle = OutputReservation(uuid.uuid4().hex, execution, charges["output"],
                                       charges["pinned"], charges["device"], N, L, passthrough_bytes)
            self._reservations[handle.reservation_id] = _Reservation(handle, pinned_offset=slot)
            state = _Execution(execution, {handle.reservation_id})
            self._executions[execution] = state
            self._admission_stats["admission_reserved_attempts"] += 1
            for kind, amount in charges.items():
                self._current[kind] += amount
                self._peaks[kind] = max(self._peaks[kind], self._current[kind])
            return handle

    def _reservation(self, handle: OutputReservation) -> _Reservation:
        rec = self._reservations.get(getattr(handle, "reservation_id", None))
        if rec is None or rec.handle is not handle:
            raise CaptureContractError("output reservation is foreign, stale or released")
        return rec

    def freeze_plan(self, execution: ExecutionIdentity, namespace: str, tokens: tuple[int, ...],
                    N: int, L: int, lease: PrefixLease,
                    reservation: OutputReservation) -> CapturePlan:
        _token_tuple(tokens)
        if len(tokens) != N:
            raise CaptureContractError("plan tokens do not match N")
        self.cache._validate_plan_lease(lease, L, namespace, tokens)
        digest = canonical_digest(["spec-plan-v1", execution.server_epoch, execution.request_id,
                  execution.attempt_id, execution.execution_epoch, namespace, self.schema.signature,
                  N, L, tokens])
        with self._lock:
            rec = self._reservation(reservation)
            state = self._executions.get(execution)
            if (reservation.execution != execution or reservation.N != N or reservation.L != L
                    or state is None or state.cancelled or rec.submitted):
                raise CaptureContractError("plan/reservation execution or range mismatch")
            if state.plan is not None:
                raise CaptureContractError("execution already has a frozen plan")
            plan = CapturePlan(execution, namespace, tokens, N, L, self.schema.signature,
                               lease, reservation, digest)
            state.plan = plan
            return plan

    def _release_transient_locked(self, rec: _Reservation) -> None:
        if rec.transient_released:
            return
        rec.transient_released = True
        self._free_slot_locked(rec.pinned_offset, rec.handle.pinned_bytes)
        self._current["pinned"] -= rec.handle.pinned_bytes
        self._current["device"] -= rec.handle.device_bytes

    def _release_host_locked(self, rec: _Reservation, state: _Execution) -> None:
        self._current["output"] -= rec.handle.host_bytes
        self._reservations.pop(rec.handle.reservation_id, None)
        state.reservations.discard(rec.handle.reservation_id)
        assert all(v >= 0 for v in self._current.values())

    def _maybe_drained_locked(self, state: _Execution) -> None:
        if state.jobs or state.reservations or state.fills or state.quarantined:
            return
        state.drained.set()
        if not state.drained_queued:
            self._drained.append(state.identity)
            state.drained_queued = True
        self._reap_locked(state)

    def _reap_locked(self, state: _Execution) -> None:
        if state.drained_polled and state.drained.is_set() and not state.completion_pending:
            if self._executions.get(state.identity) is state:
                self._executions.pop(state.identity)

    def release_output(self, reservation: OutputReservation) -> None:
        with self._lock:
            rec = self._reservations.get(getattr(reservation, "reservation_id", None))
            if rec is None or rec.handle is not reservation:
                return
            rec.release_requested = True
            if rec.submitted:
                # The worker/sink remains the owner. No timeout can revoke its read.
                return
            state = self._executions[reservation.execution]
            if state.plan is not None:
                self.cache.release_lease(state.plan.lease)
            self._release_transient_locked(rec)
            self._release_host_locked(rec, state)
            self._maybe_drained_locked(state)

    def _sample_released(self, reservation: OutputReservation) -> None:
        with self._lock:
            rec = self._reservations.get(reservation.reservation_id)
            if rec is None or rec.handle is not reservation:
                return
            state = self._executions[reservation.execution]
            state.sample = None
            owned, state.owned_output = state.owned_output, {}
            adoption_started = perf_counter()
            try:
                # MaterializedSample.release has already removed the sink's
                # tensor references. Readers may not access or modify them
                # after this ownership transfer. Keep the output charge until
                # the cache has charged every retained backing allocation.
                if owned and not state.cancelled and not rec.release_requested and not self._closed:
                    if state.cache_keys or not self.cache.skip_unused_cache_work:
                        self._adoption_calls += 1
                        self.cache._adopt_released_sample(state.cache_keys, owned)
                    else:
                        self._adoption_unused_skips += 1
            except Exception:
                self._fill_failures += 1
            finally:
                elapsed_ms = (perf_counter() - adoption_started) * 1000.0
                self._adoption_total_ms += elapsed_ms
                self._adoption_max_ms = max(self._adoption_max_ms, elapsed_ms)
                state.cache_keys = ()
                owned.clear()
                self._release_transient_locked(rec)
                self._release_host_locked(rec, state)
                self._maybe_drained_locked(state)

    def submit(self, plan: CapturePlan, span: CapturedSpan) -> Future[MaterializedSample]:
        if not isinstance(plan, CapturePlan) or not isinstance(span, CapturedSpan):
            raise CaptureContractError("submit requires a frozen plan and span")
        if span.execution != plan.execution or (span.start, span.end) != (plan.L, plan.N):
            raise CaptureContractError("captured span must be the plan's absolute [L,N)")
        if plan.schema_signature != self.schema.signature:
            raise CaptureContractError("plan schema mismatch")
        features = dict(span.features)
        self.schema.validate(features, plan.N - plan.L)
        devices = {tensor.device for tensor in features.values()}
        if len(devices) != 1 or next(iter(devices)).type not in ("cpu", "cuda"):
            raise CaptureContractError("all suffix features must share one CPU/CUDA device")
        is_cuda = next(iter(devices)).type == "cuda"
        if is_cuda and span.ready_event is None:
            raise CaptureContractError("CUDA snapshots require a producer ready event")
        if span.ready_event is not None and not callable(getattr(span.ready_event, "synchronize", None)):
            raise CaptureContractError("ready_event must support synchronize")
        # Count underlying storage, not tensor-view rows. B must hand over compact
        # owned snapshots; a tiny suffix view of a full padded batch is rejected.
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                    for t in features.values()}
        source_bytes = sum(storages.values())
        if source_bytes > plan.reservation.device_bytes:
            raise CaptureContractError("snapshot underlying storage exceeds the admission reservation")
        with self._lock:
            rec = self._reservation(plan.reservation)
            state = self._executions.get(plan.execution)
            if (state is None or state.plan is not plan or state.cancelled or rec.release_requested
                    or rec.submitted or self._closed):
                raise CaptureContractError("execution is stale, cancelled, submitted or closed")
            self.cache._validate_plan_lease(plan.lease, plan.L, plan.namespace, plan.tokens)
            rec.submitted = True
            state.jobs += 1
            future = Future()
            # Future.cancel() cannot cancel the cleanup worker and strand storage.
            future.set_running_or_notify_cancel()
            job = _Job(plan, features, span.ready_event, is_cuda, state, future)
            if self._diagnostics:
                sequence = self._diagnostic_sequence
                self._diagnostic_sequence += 1
                if sequence % self._diagnostic_every == 0 and sequence // self._diagnostic_every < self._diagnostic_limit:
                    job.diagnostics["p0_v1_materialize_submitted_ns"] = monotonic_ns()
            try:
                self._executor.submit(self._run_job, job)
            except BaseException:
                rec.submitted = False
                state.jobs -= 1
                raise
            return future

    def _cancelled(self, job: _Job) -> bool:
        with self._lock:
            rec = self._reservations.get(job.plan.reservation.reservation_id)
            return job.state.cancelled or rec is None or rec.release_requested

    def _copy_suffix(self, job: _Job) -> dict[str, torch.Tensor]:
        """Only this worker owns these sources/staging until the real copy drain."""
        if not job.is_cuda:
            if job.ready_event is not None:
                try:
                    ready_started = monotonic_ns() if job.diagnostics else 0
                    job.ready_event.synchronize()
                    if job.diagnostics:
                        job.diagnostics["p0_v1_producer_ready_wait_ms"] = (monotonic_ns() - ready_started) / 1e6
                except BaseException as exc:
                    raise _UnsafeDrain(exc, [job.ready_event]) from exc
            return {} if self._cancelled(job) else dict(job.features)
        staging = {}
        stream = None
        source = target = None
        device = next(iter(job.features.values())).device
        try:
            with torch.cuda.device(device):
                if self._cancelled(job):
                    job.ready_event.synchronize()
                    return {}
                with self._lock:
                    poisoned = self._arena_poisoned
                if poisoned:
                    # A previous failed DMA still owns this shared landing
                    # buffer. Drain this job's producer, but never touch arena.
                    job.ready_event.synchronize()
                    raise CaptureContractError("pinned arena quarantined after unproven DMA drain")
                stream = torch.cuda.Stream(device=device)
                with torch.cuda.stream(stream):
                    if job.diagnostics:
                        waiting = torch.cuda.Event(enable_timing=True)
                        ready = torch.cuda.Event(enable_timing=True)
                        copy_start = torch.cuda.Event(enable_timing=True)
                        waiting.record(stream)
                    stream.wait_event(job.ready_event)
                    if job.diagnostics:
                        ready.record(stream)
                        alloc_started = monotonic_ns()
                    staging = self._staging_views(job.plan.N - job.plan.L, job.plan.reservation)
                    if job.diagnostics:
                        job.diagnostics["p0_v1_staging_alloc_ms"] = (monotonic_ns() - alloc_started) / 1e6
                        copy_start.record(stream)
                    for name, source in job.features.items():
                        target = staging[name]
                        target.copy_(source, non_blocking=True)
                        source.record_stream(stream)
                    copied = torch.cuda.Event(enable_timing=True) if job.diagnostics else torch.cuda.Event()
                    copied.record(stream)
                copied.synchronize()
                if job.diagnostics:
                    # Read events only after the existing real copy drain. No
                    # timing-only synchronization is introduced on the hot path.
                    job.diagnostics["p0_v1_producer_ready_wait_ms"] = waiting.elapsed_time(ready)
                    job.diagnostics["p0_v1_d2h_copy_ms"] = copy_start.elapsed_time(copied)
                return staging
        except BaseException as exc:
            drain_errors = []
            # A failed wait_event may leave the copy stream independent of the
            # producer. Drain BOTH owners, even if either synchronize raises.
            if stream is not None:
                try:
                    stream.synchronize()
                except BaseException as drain_error:
                    drain_errors.append(drain_error)
            try:
                job.ready_event.synchronize()
            except BaseException as drain_error:
                drain_errors.append(drain_error)
            if drain_errors:
                raise _UnsafeDrain(drain_errors[0], [staging, stream, job.ready_event]) from exc
            source = target = None
            staging.clear()
            raise

    def _prepare_cache_adoption(self, job: _Job) -> tuple:
        # Hash/construct keys on the assembly worker, never in the sink release
        # callback (which may run on the scheduler thread).
        if self.cache.skip_unused_cache_work and (
                self.cache.capacity_bytes == 0 or job.plan.N < self.cache.chunk_tokens):
            return ()
        try:
            return self.cache.block_keys(job.plan.namespace, job.plan.tokens)
        except Exception:
            with self._lock:
                self._fill_failures += 1
            return ()

    def _run_job(self, job: _Job) -> None:
        output, staging = {}, {}
        sample = None
        completion = None
        unsafe = False
        try:
            if job.diagnostics:
                started_ns = monotonic_ns()
                job.diagnostics["p0_v1_materialize_started_ns"] = started_ns
                job.diagnostics["p0_v1_d2h_queue_ms"] = (started_ns - job.diagnostics["p0_v1_materialize_submitted_ns"]) / 1e6
                job.diagnostics["p0_v1_d2h_bytes"] = (job.plan.N - job.plan.L) * self.schema.row_bytes if job.is_cuda else 0
            copy_started = perf_counter()
            staging = self._copy_suffix(job)
            copied_at = perf_counter()
            job.features.clear()
            job.ready_event = None
            if self._cancelled(job):
                raise CaptureCancelledError(job.state.reason or "execution released")
            with torch.no_grad():
                if job.diagnostics:
                    alloc_started = monotonic_ns()
                    faults_started = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
                for spec in self.schema.features:
                    output[spec.name] = torch.empty((job.plan.N, spec.width), dtype=DTYPES[spec.dtype], device="cpu")
                if job.diagnostics:
                    prefix_started = monotonic_ns()
                    job.diagnostics["p0_v1_output_alloc_ms"] = (prefix_started - alloc_started) / 1e6
                    job.diagnostics["p0_v1_output_alloc_minor_faults"] = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - faults_started
                    faults_started = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
                try:
                    self.cache.copy_prefix_into(job.plan.lease, job.plan.L, output)
                finally:
                    self.cache.release_lease(job.plan.lease)
                if job.diagnostics:
                    suffix_started = monotonic_ns()
                    job.diagnostics["p0_v1_prefix_copy_ms"] = (suffix_started - prefix_started) / 1e6
                    job.diagnostics["p0_v1_prefix_copy_bytes"] = job.plan.L * self.schema.row_bytes
                    job.diagnostics["p0_v1_prefix_copy_minor_faults"] = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - faults_started
                    faults_started = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt
                for name in output:
                    output[name][job.plan.L:job.plan.N].copy_(staging[name])
                if job.diagnostics:
                    job.diagnostics["p0_v1_suffix_copy_ms"] = (monotonic_ns() - suffix_started) / 1e6
                    job.diagnostics["p0_v1_suffix_copy_bytes"] = (job.plan.N - job.plan.L) * self.schema.row_bytes
                    job.diagnostics["p0_v1_suffix_copy_minor_faults"] = resource.getrusage(resource.RUSAGE_THREAD).ru_minflt - faults_started
            staging.clear()
            with self._lock:
                self._release_transient_locked(self._reservation(job.plan.reservation))
            if self._cancelled(job):
                raise CaptureCancelledError(job.state.reason or "execution released")
            self.schema.validate(output, job.plan.N, "cpu")
            assembled_at = perf_counter()
            cache_keys = self._prepare_cache_adoption(job)
            filled_at = perf_counter()
            if job.diagnostics:
                job.diagnostics["p0_v1_key_prepare_ms"] = (filled_at - assembled_at) * 1000.0
                # Assembly and key preparation are complete. Sample/completion
                # bookkeeping and future delivery follow on this same worker.
                job.diagnostics["p0_v1_materialize_done_ns"] = monotonic_ns()
            sample = MaterializedSample(job.plan.execution, output, job.plan.N,
                        lambda: self._sample_released(job.plan.reservation), timings={
                            **job.diagnostics,
                            "d2h_wait_ms": (copied_at - copy_started) * 1000.0,
                            "assembly_ms": (assembled_at - copied_at) * 1000.0,
                            "cache_fill_ms": (filled_at - assembled_at) * 1000.0,
                        })
            with self._lock:
                job.state.owned_output = dict(output)
                job.state.cache_keys = cache_keys
                output.clear()
                if self._cancelled(job):
                    sample.release()
                    sample = None
                    raise CaptureCancelledError(job.state.reason or "execution released")
                job.state.sample = sample
            completion = CaptureCompletion(job.plan.execution, sample=sample)
        except _UnsafeDrain as exc:
            unsafe = True
            with self._lock:
                # A persistent arena must never be reused while any old DMA
                # may still access it. Also stop fresh admission immediately.
                self._arena_poisoned = True
                # Retain sources AND landing buffers. Keep every charge until a
                # real recovery mechanism proves drain; timeout is never success.
                job.state.quarantined.extend([job, exc.retained])
            completion = CaptureCompletion(job.plan.execution, error=RuntimeError(str(exc)))
        except BaseException as exc:
            completion = CaptureCompletion(job.plan.execution, error=_normalized_error(exc))
        finally:
            self.cache.release_lease(job.plan.lease)
            output.clear()
            if not unsafe:
                staging.clear()
                job.features.clear()
                job.ready_event = None
                with self._lock:
                    rec = self._reservations.get(job.plan.reservation.reservation_id)
                    if rec is not None:
                        self._release_transient_locked(rec)
                        if sample is None:
                            self._release_host_locked(rec, job.state)
            with self._lock:
                job.state.jobs -= 1
                job.state.completion_pending = True
                self._completed.append(completion)
                self._maybe_drained_locked(job.state)
            # Publish the result only after completion bookkeeping and resources.
            if completion.error is not None:
                job.future.set_exception(completion.error)
            else:
                job.future.set_result(completion.sample)

    def poll_completed(self) -> list[CaptureCompletion]:
        with self._lock:
            result = list(self._completed)
            self._completed.clear()
            for completion in result:
                state = self._executions.get(completion.execution)
                if state is not None:
                    state.completion_pending = False
                    self._reap_locked(state)
            return result

    def invalidate(self, execution: ExecutionIdentity, reason: str) -> DrainHandle:
        with self._lock:
            state = self._executions.get(execution)
            if state is None:
                event = threading.Event()
                event.set()
                return DrainHandle(event)
            state.cancelled, state.reason = True, str(reason)
            for fill in tuple(state.fills.values()):
                self.cache.abort_fill(fill)
            for token in tuple(state.reservations):
                rec = self._reservations.get(token)
                if rec is not None and not rec.submitted:
                    self.release_output(rec.handle)
            self._maybe_drained_locked(state)
            return DrainHandle(state.drained)

    def poll_drained(self) -> list[ExecutionIdentity]:
        with self._lock:
            result = list(self._drained)
            self._drained.clear()
            for execution in result:
                state = self._executions.get(execution)
                if state is not None:
                    state.drained_polled = True
                    self._reap_locked(state)
            return result

    def stats(self) -> dict[str, int | float]:
        result = self.cache.stats()
        with self._lock:
            result.update(self._admission_stats)
            result.update({f"{k}_bytes": v for k, v in self._current.items()})
            result.update({f"{k}_peak_bytes": v for k, v in self._peaks.items()})
            result.update(pending_executions=len(self._executions),
                          pending_copies=sum(s.jobs for s in self._executions.values()),
                          output_reservations=len(self._reservations),
                          quarantined_executions=sum(bool(s.quarantined) for s in self._executions.values()),
                          fill_failures=self._fill_failures,
                          capture_workers=self._max_workers,
                          cache_adoption_calls=self._adoption_calls,
                          cache_adoption_unused_skips=self._adoption_unused_skips,
                          cache_adoption_total_ms=self._adoption_total_ms,
                          cache_adoption_max_ms=self._adoption_max_ms,
                          pinned_free_bytes=sum(size for _, size in self._pinned_free),
                          pinned_free_slots=len(self._pinned_free))
            result.update(pinned_configured_bytes=self._pinned_configured_bytes,
                          pinned_capacity_bytes=self._limits["pinned"],
                          pinned_arena_bytes=(self._pinned_arena.numel() if self._pinned_arena is not None else 0),
                          pinned_arena_poisoned=int(self._arena_poisoned))
            return result

    def shutdown(self, wait: bool = True) -> None:
        with self._lock:
            self._closed = True
            executions = tuple(self._executions)
        for execution in executions:
            self.invalidate(execution, "controller shutdown")
        # Never cancel queued cleanup workers. They retain producer events/storage.
        self._executor.shutdown(wait=wait, cancel_futures=False)
        if wait:
            with self._lock:
                if any(not state.drained.is_set() for state in self._executions.values()):
                    raise RuntimeError("capture shutdown not drained: sink output or quarantined storage still owned")
