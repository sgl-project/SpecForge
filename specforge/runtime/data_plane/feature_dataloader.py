# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureDataLoader: ``SampleRef`` + ``FeatureStore`` -> ``TrainBatch``.

Leases refs (consume-once queue or re-iterable ref list), applies the injected
per-sample transform and collate, and yields ``TrainBatch``es — no model
knowledge. clone-on-fetch (default unless the store's ``get()`` already returns
fresh tensors) clones tensors out of the store and releases the handle
immediately, so prefetch can never race a release.
"""

from __future__ import annotations

from specforge.runtime.data_plane.pool_diagnostics import get_diagnostics
from specforge.runtime.data_plane.feature_store import abort_feature_ref

import json
import logging
import math
import os
import queue as queue_module
import threading
import time
import sys
import traceback
from collections import OrderedDict
from concurrent import futures
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterator, List, Optional

import torch

from specforge.runtime.contracts import SampleRef, TrainBatch
from specforge.runtime.data_plane.batch_materialization import (
    ConsumerSchedule, allocate_sequence_batch, copy_sample_into,
)
from specforge.runtime.data_plane.feature_store import FeatureStore
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue

PerSampleTransform = Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]]
CollateFn = Callable[[List[Dict[str, torch.Tensor]]], Dict[str, Any]]

_PREFETCH_POLL_S = 0.1
_PREFETCH_JOIN_TIMEOUT_S = float(
    os.environ.get("SPECFORGE_LOADER_JOIN_TIMEOUT_S", "5.0")
)
# The join interval emits diagnostics; it does not reset this absolute bound.
_PREFETCH_DRAIN_TIMEOUT_S = float(
    os.environ.get("SPECFORGE_LOADER_DRAIN_TIMEOUT_S", "60.0")
)
logger = logging.getLogger(__name__)


class _PrefetchStopped(Exception):
    """A loader-owned batch stopped between completed sample reads."""


@dataclass
class _PreparedBatch:
    batch: TrainBatch
    ready: Dict[torch.device, torch.cuda.Event]

    def consume(self) -> TrainBatch:
        # Futures synchronize Python threads, not CUDA work. Hand off both
        # execution ordering and allocator lifetime to the consumer's streams.
        for device, event in self.ready.items():
            stream = torch.cuda.current_stream(device)
            stream.wait_event(event)
            for tensor in self.batch.tensors.values():
                if tensor.device == device:
                    tensor.record_stream(stream)
        return self.batch


@dataclass
class _OutstandingRef:
    ref: SampleRef
    materialized: bool = False
    touched: bool = False
    failed: bool = False


class _QueuePrefetchState:
    """One online iterator's worker, buffer, and not-yet-yielded leases."""

    def __init__(self, depth: int) -> None:
        self.buffer: queue_module.Queue[Any] = queue_module.Queue(maxsize=depth)
        self.stop = threading.Event()
        self.thread: Optional[threading.Thread] = None
        self.outstanding: "OrderedDict[str, _OutstandingRef]" = OrderedDict()
        self.outstanding_lock = threading.Lock()
        self.fetches: Dict[Any, List[SampleRef]] = {}
        self.fetches_lock = threading.Lock()
        self.shutdown_lock = threading.Lock()
        self.shutdown_complete = False
        self.status = "ACTIVE"
        self.progress_lock = threading.Lock()
        self.tasks: Dict[int, Dict[str, Any]] = {}
        self.materialized_samples = 0
        self.shutdown_started = None
        self.shutdown_deadline = None
        self.shutdown_budget_s = None
        self.diagnostic_intervals = 0

    def task_progress(self, **fields) -> None:
        with self.progress_lock:
            self.tasks.setdefault(threading.get_ident(), {}).update(
                thread_name=threading.current_thread().name,
                last_progress_monotonic=time.monotonic(), **fields,
            )

    def sample_materialized(self, ref: SampleRef) -> None:
        # Only called after the source read/copy and handle release completed.
        with self.outstanding_lock:
            entry = self.outstanding.get(ref.sample_id)
            if entry is not None:
                entry.materialized = True
        with self.progress_lock:
            self.materialized_samples += 1
        self.task_progress(sample_id=ref.sample_id, stage="sample_released")

    def sample_started(self, ref: SampleRef) -> None:
        with self.outstanding_lock:
            self.outstanding[ref.sample_id].touched = True
        self.task_progress(sample_id=ref.sample_id, stage="materializing_sample")

    def snapshot(self, *, status=None, include_stacks=False) -> Dict[str, Any]:
        with self.progress_lock:
            tasks = {str(tid): dict(task) for tid, task in self.tasks.items()}
            result = dict(status=status or self.status, stop_requested=self.stop.is_set(),
                          materialized_samples=self.materialized_samples,
                          diagnostic_intervals=self.diagnostic_intervals,
                          elapsed_s=(time.monotonic() - self.shutdown_started
                                     if self.shutdown_started is not None else 0.0),
                          deadline_monotonic=self.shutdown_deadline,
                          tasks=tasks)
        result["outstanding"] = [
            {"sample_id": item.ref.sample_id, "materialized": item.materialized,
             "touched": item.touched, "failed": item.failed}
            for item in self.outstanding_entries()
        ]
        if include_stacks:
            frames = sys._current_frames()
            tids = {int(tid) for tid, task in tasks.items() if task.get("running")}
            if self.thread is not None and self.thread.is_alive():
                tids.add(self.thread.ident)
            result["thread_stacks"] = {
                str(tid): "".join(traceback.format_stack(frames[tid], limit=16))[-8192:]
                for tid in tids if tid in frames
            }
        return result

    def track_fetch(self, future: Any, refs: List[SampleRef]) -> None:
        with self.fetches_lock:
            self.fetches[future] = list(refs)

    def claim_for_yield(self, future: Any) -> Optional[List[SampleRef]]:
        """Transfer a completed fetch and its refs to the consumer."""
        with self.fetches_lock:
            if self.stop.is_set():
                return None
            refs = self.fetches.pop(future, None)
            if refs is None:
                return None
            with self.outstanding_lock:
                for ref in refs:
                    self.outstanding.pop(ref.sample_id, None)
            return refs

    def settle_unyielded_fetch(self, future: Any) -> Optional[List[SampleRef]]:
        """Stop tracking a fetch while leaving its refs for close settlement."""
        with self.fetches_lock:
            refs = self.fetches.pop(future, None)
            if refs is None:
                return None
            return refs

    def mark_fetch_failed(self, refs):
        with self.outstanding_lock:
            for ref in refs:
                entry = self.outstanding.get(ref.sample_id)
                if entry is not None and entry.touched:
                    entry.failed = True

    def live_fetches(self) -> Dict[Any, List[SampleRef]]:
        with self.fetches_lock:
            return dict(self.fetches)

    def track(self, refs: List[SampleRef]) -> None:
        with self.outstanding_lock:
            for ref in refs:
                self.outstanding[ref.sample_id] = _OutstandingRef(ref)

    def mark_yielded_or_failed(self, refs: List[SampleRef]) -> None:
        with self.outstanding_lock:
            for ref in refs:
                self.outstanding.pop(ref.sample_id, None)

    def outstanding_entries(self) -> List[_OutstandingRef]:
        with self.outstanding_lock:
            return list(self.outstanding.values())

    def outstanding_ids(self) -> List[str]:
        with self.outstanding_lock:
            return list(self.outstanding)


def _default_collate(features: List[Dict[str, torch.Tensor]]) -> Dict[str, Any]:
    """Trivial stack collate (used only when no collate_fn is injected)."""
    keys = features[0].keys()
    return {k: torch.stack([f[k] for f in features], dim=0) for k in keys}


class FeatureDataLoader:
    def __init__(
        self,
        store: FeatureStore,
        queue: Optional[SampleRefQueue] = None,
        *,
        refs: Optional[List[SampleRef]] = None,
        batch_size: int = 1,
        collate_fn: Optional[CollateFn] = None,
        per_sample_transform: Optional[PerSampleTransform] = None,
        device: "torch.device | str" = "cpu",
        clone_on_fetch: Optional[bool] = None,
        drop_last: bool = True,
        strategy: str = "eagle3",
        ack: bool = True,
        gc_interval_s: Optional[float] = 15.0,
        num_workers: int = 0,
        pin_memory: bool = False,
    ) -> None:
        if (queue is None) == (refs is None):
            raise ValueError(
                "provide exactly one of `queue` (stream) or `refs` (re-iterable)"
            )
        self.store = store
        self.queue = queue
        self._refs = list(refs) if refs is not None else None
        self.batch_size = batch_size
        self.collate_fn = collate_fn or _default_collate
        self.per_sample_transform = per_sample_transform
        self.device = device
        # By default clone unless the store's get() already returns fresh
        # caller-owned tensors (e.g. Mooncake), where the clone only copies.
        # CLONE_ON_FETCH=0 skips the defensive clone; =1 forces it.
        if clone_on_fetch is None:
            clone_on_fetch = not getattr(store, "get_returns_fresh_tensors", False)
        clone_env = os.environ.get("CLONE_ON_FETCH")
        if clone_env == "0":
            clone_on_fetch = False
        elif clone_env == "1":
            clone_on_fetch = True
        self.clone_on_fetch = bool(clone_on_fetch)
        self.drop_last = drop_last
        self.strategy = strategy
        self.ack = ack
        if num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        self.num_workers = int(num_workers)
        self.pin_memory = bool(pin_memory)
        self.direct_batch = os.environ.get("SPECFORGE_TRAINER_DIRECT_BATCH", "0") == "1"
        self.feature_demand = os.environ.get("SPECFORGE_TRAINER_FEATURE_DEMAND", "0") == "1"
        self._consumer_schedule = None
        self._max_batches = None
        self._batch_sequence = 0
        self._sequence_lock = threading.Lock()
        self._fetch_local = threading.local()
        self._last_shutdown_diagnostics = {"status": "NOT_STARTED"}
        for name, value in (("SPECFORGE_LOADER_JOIN_TIMEOUT_S", _PREFETCH_JOIN_TIMEOUT_S),
                            ("SPECFORGE_LOADER_DRAIN_TIMEOUT_S", _PREFETCH_DRAIN_TIMEOUT_S)):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self._seek_batches = 0
        # Remote stores defer a physical free while the get() read-lease is live
        # (Mooncake remove -> -706): release() parks it and gc() must retry.
        # Pump gc on a cadence LONGER than the lease TTL — the store gives up
        # after max_release_attempts, so rapid retries would leak the object.
        self.gc_interval_s = gc_interval_s if hasattr(store, "gc") else None
        self._last_gc = time.monotonic()
        self._lifecycle_lock = threading.Lock()
        self._prefetch_state: Optional[_QueuePrefetchState] = None
        self._queue_closed = False
        self._perf_lock = threading.Lock()
        self._perf = {
            "wait_producer_s": 0.0,
            "wait_fetch_s": 0.0,
            "fetch_s": 0.0,
            "fetch_bytes": 0.0,
            "fetch_batches": 0.0,
            "fetch_samples": 0.0,
            "direct_batch_batches": 0.0,
            "direct_batch_fallback_batches": 0.0,
            "feature_demand_trimmed_batches": 0.0,
            "feature_requested_bytes": 0.0,
            "feature_materialized_bytes": 0.0,
            "feature_h2d_bytes": 0.0,
            "last_hidden_requested_bytes": 0.0,
            "last_hidden_h2d_bytes": 0.0,
            "prefetch_submitted_batches": 0.0,
            "prefetch_materialized_samples": 0.0,
            "prefetch_stopped_batches": 0.0,
            "prefetch_shutdown_timeouts": 0.0,
        }

    def _perf_add(self, **deltas: float) -> None:
        with self._perf_lock:
            for name, delta in deltas.items():
                self._perf[name] += delta

    def perf_counters_snapshot(self, reset: bool = False) -> Dict[str, float]:
        """Return loader counters accumulated since the last reset."""
        with self._perf_lock:
            snapshot = dict(self._perf)
            if reset:
                for name in self._perf:
                    self._perf[name] = 0.0
        return snapshot

    def configure_consumer_schedule(self, *, global_step, accumulation_steps,
                                    accumulation_remainder, log_interval, has_logger,
                                    max_batches=None):
        # Called on the consuming thread before any iterator/prefetch is started.
        if max_batches is not None and (int(max_batches) != max_batches or max_batches < 0):
            raise ValueError("max_batches must be a nonnegative integer or None")
        with self._lifecycle_lock:
            if self._prefetch_state is not None:
                raise RuntimeError("cannot change consumer schedule while prefetch is active")
        self._consumer_schedule = ConsumerSchedule(
            int(global_step), max(1, int(accumulation_steps)),
            int(accumulation_remainder), max(1, int(log_interval)), bool(has_logger),
        )
        with self._sequence_lock:
            self._max_batches = None if max_batches is None else int(max_batches)
            self._batch_sequence = 0

    def _batch_limit_reached(self):
        with self._sequence_lock:
            return self._max_batches is not None and self._batch_sequence >= self._max_batches

    def _next_batch_index(self):
        with self._sequence_lock:
            index = self._batch_sequence
            self._batch_sequence += 1
            return index

    def _timed_make_batch(self, refs: List[SampleRef], batch_index=None) -> TrainBatch:
        started = time.perf_counter()
        diag = get_diagnostics()
        with diag.batch(batch_index), diag.component("loader").span("fetch_materialize"):
            batch = self._make_batch(refs, batch_index=batch_index)
        self._perf_add(
            fetch_s=time.perf_counter() - started,
            fetch_bytes=float(sum(int(ref.estimated_bytes or 0) for ref in refs)),
            fetch_batches=1.0,
            fetch_samples=float(len(refs)),
        )
        return batch

    def _prepare_batch(self, refs: List[SampleRef], batch_index=None) -> _PreparedBatch:
        batch = self._timed_make_batch(refs, batch_index=batch_index)
        ready = {}
        for tensor in batch.tensors.values():
            if tensor.is_cuda and tensor.device not in ready:
                event = torch.cuda.Event()
                event.record(torch.cuda.current_stream(tensor.device))
                ready[tensor.device] = event
        return _PreparedBatch(batch, ready)

    def _prepare_queue_batch(self, refs, batch_index, state):
        previous = getattr(self._fetch_local, "state", None)
        self._fetch_local.state = state
        state.task_progress(running=True, batch_index=batch_index,
                            sample_id=None, stage="prepare")
        try:
            self._check_fetch_stop()
            return self._prepare_batch(refs, batch_index)
        except _PrefetchStopped:
            self._perf_add(prefetch_stopped_batches=1.0)
            raise
        finally:
            state.task_progress(running=False, stage="finished")
            self._fetch_local.state = previous

    def _check_fetch_stop(self, ref=None):
        state = getattr(self._fetch_local, "state", None)
        if state is not None:
            if state.stop.is_set():
                raise _PrefetchStopped()
            if ref is not None:
                state.sample_started(ref)

    def _sample_materialized(self, ref):
        state = getattr(self._fetch_local, "state", None)
        if state is not None:
            state.sample_materialized(ref)
            self._perf_add(prefetch_materialized_samples=1.0)

    def _maybe_gc(self) -> None:
        if self.gc_interval_s is None:
            return
        now = time.monotonic()
        if now - self._last_gc >= self.gc_interval_s:
            self._last_gc = now
            self.store.gc()

    def _validate_refs(self, refs: List[SampleRef]) -> None:
        strategies = {ref.strategy for ref in refs}
        if strategies != {self.strategy}:
            raise ValueError(
                f"loader strategy={self.strategy!r} received refs with "
                f"strategies={sorted(strategies)}"
            )

        schema_versions = {ref.schema_version for ref in refs}
        if len(schema_versions) != 1:
            raise ValueError(
                f"mixed schema versions in batch: {sorted(schema_versions)}"
            )

        target_reprs = {ref.metadata.get("target_repr") for ref in refs}
        if len(target_reprs) != 1:
            raise ValueError(
                f"mixed target_repr values in batch: {sorted(map(repr, target_reprs))}"
            )

        spec_sets = [set(ref.feature_specs) for ref in refs if ref.feature_specs]
        if spec_sets and any(spec_set != spec_sets[0] for spec_set in spec_sets[1:]):
            raise ValueError(f"mixed feature spec names in batch: {spec_sets}")

        if not spec_sets:
            return
        first_specs = next(ref.feature_specs for ref in refs if ref.feature_specs)
        for ref in refs:
            if not ref.feature_specs:
                continue
            for name, spec in ref.feature_specs.items():
                expected = first_specs[name]
                if spec.dtype != expected.dtype or len(spec.shape) != len(
                    expected.shape
                ):
                    raise ValueError(
                        f"incompatible feature spec for sample {ref.sample_id}, "
                        f"feature {name!r}: {spec} vs {expected}"
                    )

    def _materialize(self, ref: SampleRef) -> Dict[str, torch.Tensor]:
        self._check_fetch_stop(ref)
        tensors, handle = self.store.get(ref, device=self.device)
        try:
            if self.clone_on_fetch:
                tensors = {k: v.clone() for k, v in tensors.items()}
        finally:
            # ``get`` has already registered a read lease. Clone/device or
            # tensor validation failures must not strand that handle.
            self.store.release(handle, reason="loaded")
            self._sample_materialized(ref)
        self._maybe_gc()
        if self.per_sample_transform is not None:
            tensors = self.per_sample_transform(tensors)
        return tensors

    def _make_batch(self, refs: List[SampleRef], batch_index=None) -> TrainBatch:
        self._validate_refs(refs)
        layout = getattr(self.collate_fn, "sequence_batch_layout", None)
        if self.direct_batch and layout is not None and self.per_sample_transform is None:
            return self._make_direct_batch(refs, layout, batch_index)
        if self.direct_batch:
            self._perf_add(direct_batch_fallback_batches=1.0)
        per_sample = [self._materialize(r) for r in refs]
        self._check_fetch_stop()
        batch_tensors = self.collate_fn(per_sample)
        non_tensors = [
            name
            for name, value in batch_tensors.items()
            if not isinstance(value, torch.Tensor)
        ]
        if non_tensors:
            raise TypeError(f"collate_fn returned non-tensors for {non_tensors}")
        if self.pin_memory:
            batch_tensors = {
                name: (
                    value.pin_memory()
                    if value.device.type == "cpu" and not value.is_pinned()
                    else value
                )
                for name, value in batch_tensors.items()
            }
        return TrainBatch(
            sample_ids=[r.sample_id for r in refs],
            strategy=self.strategy,
            tensors=batch_tensors,
            metadata={
                "target_repr": refs[0].metadata.get("target_repr"),
                "ttt_length": refs[0].metadata.get("ttt_length"),
            },
        )

    def _make_direct_batch(self, refs, layout, batch_index):
        self._check_fetch_stop()
        names = None
        if (self.feature_demand and self.strategy == "dflash"
                and self._consumer_schedule is not None and batch_index is not None
                and not self._consumer_schedule.detailed_metrics(batch_index)):
            names = set(refs[0].feature_specs) - {"target_last_hidden_states"}
            self._perf_add(feature_demand_trimmed_batches=1.0)
        tensors, destinations = allocate_sequence_batch(
            refs, layout, device=self.device, names=names,
        )
        for ref, dest in zip(refs, destinations):
            self._check_fetch_stop(ref)
            _, handle = copy_sample_into(self.store, ref, dest)
            # get_into certifies completion of every source copy, including H2D.
            self.store.release(handle, reason="loaded")
            self._sample_materialized(ref)
            requested = sum(value.numel() * value.element_size() for value in dest.values())
            h2d = sum(value.numel() * value.element_size() for value in dest.values() if value.is_cuda)
            last = dest.get("target_last_hidden_states")
            last_bytes = last.numel() * last.element_size() if last is not None else 0
            self._perf_add(feature_requested_bytes=requested,
                           feature_materialized_bytes=requested, feature_h2d_bytes=h2d,
                           last_hidden_requested_bytes=last_bytes,
                           last_hidden_h2d_bytes=last_bytes if last is not None and last.is_cuda else 0)
        self._check_fetch_stop()
        self._maybe_gc()
        self._perf_add(direct_batch_batches=1.0)
        return TrainBatch(
            sample_ids=[ref.sample_id for ref in refs], strategy=self.strategy,
            tensors=tensors, metadata={"target_repr": refs[0].metadata.get("target_repr"),
                                      "ttt_length": refs[0].metadata.get("ttt_length"),
                                      "direct_batch": True, "consumer_batch_index": batch_index},
        )

    def _settle_incomplete_queue_batch(self, refs: List[SampleRef]) -> None:
        """Terminally discard and clean a short ``drop_last`` queue batch."""
        reason = (
            "online stream ended with an incomplete batch: "
            f"received {len(refs)} refs but batch_size={self.batch_size}"
        )
        self.queue.fail(refs, reason=reason, retryable=False)
        cleanup_errors = []
        for ref in refs:
            try:
                # A fresh Mooncake consumer has no put-side bookkeeping for a
                # never-materialized server ref. Seed it before aborting so the
                # remote keys are actually removed; local/file stores need no
                # adoption hook.
                adopt = getattr(self.store, "adopt", None)
                if callable(adopt):
                    adopt(ref)
                abort_feature_ref(self.store, ref, reason=reason)
            except Exception as exc:  # settle the full tail, then fail loudly
                cleanup_errors.append(f"{ref.sample_id}: {exc}")
        if cleanup_errors:
            raise RuntimeError(f"{reason}; cleanup errors={cleanup_errors}")

    def __iter__(self) -> Iterator[TrainBatch]:
        if self._refs is not None:
            yield from self._iter_refs()
        else:
            yield from self._iter_queue()

    def _iter_refs(self) -> Iterator[TrainBatch]:
        # Acking (the durable marker) is the trainer's job here, not the loader's.
        skip, self._seek_batches = self._seek_batches, 0
        chunks = []
        for start in range(skip * self.batch_size, len(self._refs), self.batch_size):
            if self._max_batches is not None and len(chunks) >= self._max_batches:
                break
            chunk = self._refs[start : start + self.batch_size]
            if self.drop_last and len(chunk) < self.batch_size:
                break
            chunks.append(chunk)
        if self.num_workers > 0:
            yield from self._iter_refs_prefetch(chunks)
            return
        for chunk in chunks:
            yield self._make_batch(chunk, batch_index=self._next_batch_index())

    def _iter_refs_prefetch(
        self, chunks: List[List[SampleRef]]
    ) -> Iterator[TrainBatch]:
        """Materialize fixed offline batches concurrently while preserving order."""
        from collections import deque
        from concurrent.futures import ThreadPoolExecutor

        chunk_iter = iter(chunks)
        pending = deque()
        with ThreadPoolExecutor(
            max_workers=self.num_workers,
            thread_name_prefix="feature-loader",
        ) as executor:
            for _ in range(self.num_workers):
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    break
                pending.append(executor.submit(self._prepare_batch, chunk, self._next_batch_index()))
            while pending:
                yield pending.popleft().result().consume()
                try:
                    chunk = next(chunk_iter)
                except StopIteration:
                    continue
                pending.append(executor.submit(self._prepare_batch, chunk, self._next_batch_index()))

    def seek(self, num_batches: int) -> None:
        """Skip the first ``num_batches`` of the NEXT iteration (refs mode; one-shot).

        Raises on a queue stream (which has no resumable position) and when the
        skip exceeds the available batches (a silently empty epoch is banned).
        """
        if self._refs is None:
            raise ValueError(
                "seek() applies to refs mode only; a queue stream is consume-once"
            )
        skip = max(0, int(num_batches))
        if self.drop_last:
            available = len(self._refs) // self.batch_size
        else:
            available = -(-len(self._refs) // self.batch_size)
        if skip > available:
            raise ValueError(
                f"seek({num_batches}) skips past the end of the data: only "
                f"{available} batches available ({len(self._refs)} refs at "
                f"batch_size={self.batch_size}, drop_last={self.drop_last}); "
                f"the resume position does not fit this dataset"
            )
        self._seek_batches = skip

    def _iter_queue(self) -> Iterator[TrainBatch]:
        with self._lifecycle_lock:
            if self._queue_closed:
                return
        depth = self.num_workers or int(os.environ.get("LOADER_PREFETCH", "0"))
        if depth > 0:
            yield from self._iter_queue_prefetch(depth)
            return
        while not self._batch_limit_reached():
            wait_started = time.perf_counter()
            refs = self.queue.get(self.batch_size, timeout_s=0.0)
            self._perf_add(wait_producer_s=time.perf_counter() - wait_started)
            if not refs:
                return
            if self.drop_last and len(refs) < self.batch_size:
                self._settle_incomplete_queue_batch(refs)
                return
            try:
                fetch_started = time.perf_counter()
                batch = self._timed_make_batch(refs, batch_index=self._next_batch_index())
                self._perf_add(wait_fetch_s=time.perf_counter() - fetch_started)
            except Exception as exc:
                self.queue.fail(refs, reason=f"materialize:{exc}", retryable=False)
                raise
            yield batch
            if self.ack:
                self.queue.ack(refs)

    def _iter_queue_prefetch(self, depth: int) -> Iterator[TrainBatch]:
        state = _QueuePrefetchState(depth)
        eos = object()

        def put_interruptibly(item: Any) -> bool:
            while not state.stop.is_set():
                try:
                    state.buffer.put(item, timeout=_PREFETCH_POLL_S)
                    return True
                except queue_module.Full:
                    continue
            return False

        def get_refs_interruptibly() -> List[SampleRef]:
            get_interruptible = getattr(self.queue, "get_interruptible", None)
            if callable(get_interruptible):
                return get_interruptible(
                    self.batch_size,
                    stop_event=state.stop,
                )
            return self.queue.get(self.batch_size, timeout_s=_PREFETCH_POLL_S)

        # The buffer carries FUTURES, so up to `depth` batches materialize
        # concurrently while the worker goes back to leasing; the consuming
        # thread resolves them in submission order. Lease/ack semantics are
        # unchanged.
        pool = futures.ThreadPoolExecutor(
            max_workers=depth, thread_name_prefix="loader-fetch"
        )

        def _worker() -> None:
            try:
                while not state.stop.is_set():
                    if self._batch_limit_reached():
                        # EOF is not cancellation: every submitted legal batch
                        # must still be resolved and yielded in order.
                        put_interruptibly(eos)
                        return
                    refs = get_refs_interruptibly()
                    if not refs:
                        if not state.stop.is_set():
                            put_interruptibly(eos)
                        return
                    state.track(refs)
                    if state.stop.is_set():
                        return
                    if self.drop_last and len(refs) < self.batch_size:
                        try:
                            self._settle_incomplete_queue_batch(refs)
                        finally:
                            state.mark_yielded_or_failed(refs)
                        put_interruptibly(eos)
                        return
                    future = pool.submit(self._prepare_queue_batch, refs,
                                         self._next_batch_index(), state)
                    state.track_fetch(future, refs)
                    self._perf_add(prefetch_submitted_batches=1.0)
                    if not put_interruptibly((future, refs)):
                        return
            except BaseException as exc:  # loud failure, never a silent hang
                put_interruptibly(exc)
            finally:
                # EOF stops submissions, not consumption: futures queued before
                # eos must still complete and be yielded in order. Do not block
                # this worker on fetches either. Early-stop cancellation and
                # bounded waiting belong to _shutdown_prefetch, even if this
                # worker has already exited after enqueueing eos.
                pool.shutdown(wait=False, cancel_futures=False)

        worker = threading.Thread(
            target=_worker,
            name="loader-prefetch",
            # A broken backend must not make interpreter shutdown impossible.
            # close() still joins and reports a bounded, loud lifecycle error.
            daemon=True,
        )
        state.thread = worker
        with self._lifecycle_lock:
            if self._queue_closed:
                return
            if self._prefetch_state is not None:
                raise RuntimeError("FeatureDataLoader already has an active iterator")
            self._prefetch_state = state
            worker.start()

        try:
            while not state.stop.is_set():
                # An empty buffer means the worker is still leasing refs:
                # producer-bound wait. Time spent resolving an already
                # submitted future is transfer-bound wait.
                wait_started = time.perf_counter()
                try:
                    item = state.buffer.get(timeout=_PREFETCH_POLL_S)
                except queue_module.Empty:
                    self._perf_add(wait_producer_s=time.perf_counter() - wait_started)
                    continue
                if item is eos:
                    return
                if isinstance(item, BaseException):
                    raise item
                future, refs = item
                fetch_started = time.perf_counter()
                while not state.stop.is_set():
                    try:
                        try:
                            prepared = future.result(timeout=_PREFETCH_POLL_S)
                        except futures.TimeoutError:
                            if not future.done():
                                continue
                            # A completed fetch can itself raise TimeoutError.
                            # Resolve it outside the wait-timeout handler so it
                            # follows the real materialization failure path.
                            prepared = future.result()
                        batch = prepared.consume()
                        break
                    except _PrefetchStopped:
                        if state.stop.is_set():
                            return
                        raise
                    except Exception as exc:
                        state.stop.set()
                        failed_refs = state.settle_unyielded_fetch(future)
                        if failed_refs is not None:
                            state.mark_fetch_failed(failed_refs)
                            try:
                                self._settle_unyielded_refs(state, failed_refs, reason=f"materialize:{exc}")
                            except BaseException as cleanup_error:
                                # Keep exact outstanding refs for finally or a
                                # later close, and preserve the original fetch.
                                exc.add_note(f"prefetch ref settlement also failed: {cleanup_error}")
                        raise
                else:
                    return
                self._perf_add(wait_fetch_s=time.perf_counter() - fetch_started)
                # From this point the trainer owns the refs. In deferred-ack
                # mode its optimizer-boundary transaction decides their
                # outcome; loader shutdown must touch only batches that were
                # materialized ahead but never yielded.
                if state.claim_for_yield(future) is None:
                    return
                yield batch
                if self.ack:
                    self.queue.ack(refs)
        finally:
            primary = sys.exc_info()[1]
            try:
                self._shutdown_prefetch(state)
            except BaseException as cleanup_error:
                if primary is None or isinstance(primary, GeneratorExit):
                    raise
                primary.add_note(f"prefetch shutdown also failed: {cleanup_error}")

    def prefetch_shutdown_snapshot(self):
        """Read-only CPU progress for shutdown diagnostics and lifecycle probes."""
        with self._lifecycle_lock:
            state = self._prefetch_state
        if state is None:
            return dict(self._last_shutdown_diagnostics)
        value = state.snapshot(include_stacks=state.stop.is_set())
        value.update(batch_limit=self._max_batches, submitted_batches=self._batch_sequence,
                     drain_timeout_s=state.shutdown_budget_s or _PREFETCH_DRAIN_TIMEOUT_S,
                     diagnostic_interval_s=_PREFETCH_JOIN_TIMEOUT_S)
        return value

    def _prefetch_diagnostic(self, state, status, *, intervals=False):
        with state.progress_lock:
            state.status = status
            if intervals:
                state.diagnostic_intervals += 1
        value = state.snapshot(status=status, include_stacks=True)
        value.update(batch_limit=self._max_batches,
                     submitted_batches=self._batch_sequence,
                     drain_timeout_s=state.shutdown_budget_s,
                     diagnostic_interval_s=_PREFETCH_JOIN_TIMEOUT_S)
        self._last_shutdown_diagnostics = value
        return value

    def _shutdown_prefetch(self, state: _QueuePrefetchState) -> None:
        """Stop one worker and return every not-yet-yielded lease explicitly."""
        with state.shutdown_lock:
            if state.shutdown_complete:
                return
            state.stop.set()
            with state.progress_lock:
                if state.shutdown_started is None:
                    state.shutdown_started = time.monotonic()
                    state.shutdown_budget_s = _PREFETCH_DRAIN_TIMEOUT_S
                    state.shutdown_deadline = state.shutdown_started + state.shutdown_budget_s
            # A concurrent/reentrant close cannot silently grant a second
            # drain budget. Once expired it only succeeds after work is done.
            deadline = state.shutdown_deadline

            def wait_for_drain(wait, still_pending, kind):
                while still_pending():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._perf_add(prefetch_shutdown_timeouts=1.0)
                        value = self._prefetch_diagnostic(state, "TIMED_OUT")
                        raise RuntimeError(
                            f"FeatureDataLoader prefetch {kind} did not stop within "
                            f"{state.shutdown_budget_s:.1f}s; backend get/materialize "
                            f"is still blocked, outstanding_refs={state.outstanding_ids()}; "
                            f"progress={json.dumps(value, sort_keys=True)}"
                        )
                    wait(min(_PREFETCH_JOIN_TIMEOUT_S, remaining))
                    if still_pending() and time.monotonic() < deadline:
                        value = self._prefetch_diagnostic(state, "DRAINING", intervals=True)
                        logger.warning("FeatureDataLoader graceful drain pending: %s",
                                       json.dumps(value, sort_keys=True))

            worker = state.thread
            if worker is not None and worker is not threading.current_thread():
                wait_for_drain(lambda timeout: worker.join(timeout), worker.is_alive, "worker")
            live = state.live_fetches()
            # The submitter has stopped, so this snapshot includes every fetch
            # still owned by the loader. Normal EOF has already consumed them;
            # early close/error cancels only jobs that have not started.
            for future in live:
                future.cancel()
            # A cancelled Future need not have been dequeued by its executor
            # yet. Exclude done futures rather than waiting for the executor's
            # cancellation notification; only running transfers need draining.
            pending = [future for future in live if not future.done()]
            wait_for_drain(lambda timeout: futures.wait(pending, timeout=timeout),
                           lambda: any(not future.done() for future in pending), "fetch")

            # Per-sample progress is authoritative, including partial batches.
            # A cancelled future never started; a cooperative stop may have
            # safely consumed only a prefix. Never label the entire batch.
            for future in live:
                if future.cancelled():
                    state.settle_unyielded_fetch(future)
                    continue
                exc = future.exception()
                fetched_refs = state.settle_unyielded_fetch(future)
                if exc is not None and not isinstance(exc, _PrefetchStopped) and fetched_refs is not None:
                    state.mark_fetch_failed(fetched_refs)
                    self._settle_unyielded_refs(state, fetched_refs, reason=f"materialize:{exc}")

            self._settle_unyielded_refs(state, state.outstanding_ids())
            state.shutdown_complete = True
            self._prefetch_diagnostic(state, "STOPPED")

        with self._lifecycle_lock:
            if self._prefetch_state is state:
                self._prefetch_state = None

    def _settle_unyielded_refs(self, state, refs, *, reason=None):
        ids = {ref if isinstance(ref, str) else ref.sample_id for ref in refs}
        retryable, terminal = [], []
        retain = bool(getattr(self.store, "retain_on_release", False))
        for entry in state.outstanding_entries():
            if entry.ref.sample_id not in ids:
                continue
            can_retry = (not entry.failed and (not entry.touched or retain
                         or entry.ref.feature_store_uri.startswith("file://")))
            (retryable if can_retry else terminal).append(entry.ref)
        for group, retry, default_reason in (
            (retryable, True, "loader_prefetch_closed_before_yield"),
            (terminal, False, "loader_prefetch_closed_after_materialization_from_a_consume_once_store"),
        ):
            if group:
                self.queue.fail(group, reason=reason or default_reason, retryable=retry)
                state.mark_yielded_or_failed(group)

    def set_epoch(self, epoch: int) -> None:
        # hook for per-epoch shuffling of the offline ref set (no-op for now)
        self._epoch = epoch

    def close(self) -> None:
        """Stop online prefetch and settle every batch it never yielded.

        Fixed offline refs have no loader-owned background lifecycle and remain
        re-iterable after ``close``. Queue sources are consume-once, so closing
        them prevents a later iterator from starting a second worker.
        """
        with self._lifecycle_lock:
            if self._refs is None:
                self._queue_closed = True
            state = self._prefetch_state
        if state is not None:
            self._shutdown_prefetch(state)


__all__ = ["FeatureDataLoader", "PerSampleTransform", "CollateFn"]
