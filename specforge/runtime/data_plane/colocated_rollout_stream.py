# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
"""Bounded pull-through rollout for colocated online training."""

from __future__ import annotations

import time
from typing import List, Optional, Sequence

from specforge.runtime.contracts import SampleRef


class LocalRolloutStream:
    """Produce target features synchronously when the trainer requests data.

    Keeping capture on the trainer thread gives CUDA work deterministic ordering
    and bounds GPU-resident target features to one rank-local training batch.
    Cross-process retry/durability remains the disaggregated runtime's job.
    """

    loader_prefetch_safe = False

    def __init__(
        self,
        *,
        controller,
        workers: Sequence,
        feature_store,
        max_resident_samples: int,
        capture_batch_multiplier: int = 1,
        max_stalled_rounds: int = 3,
    ) -> None:
        if not workers:
            raise ValueError("local rollout requires at least one worker")
        if max_resident_samples < 1:
            raise ValueError("max_resident_samples must be >= 1")
        if capture_batch_multiplier < 1:
            raise ValueError("capture_batch_multiplier must be >= 1")
        if max_stalled_rounds < 1:
            raise ValueError("max_stalled_rounds must be >= 1")
        if not callable(getattr(feature_store, "abort_all", None)):
            raise TypeError(
                "local rollout requires a private feature store with abort_all()"
            )

        self.controller = controller
        self.workers = tuple(workers)
        self.feature_store = feature_store
        self.max_resident_samples = int(max_resident_samples)
        self.capture_batch_multiplier = int(capture_batch_multiplier)
        self.max_stalled_rounds = int(max_stalled_rounds)
        self._queue = controller.sample_queue
        self._started = False
        self._closed = False
        self._next_worker = 0
        self.produced_count = 0
        self.peak_resident_samples = 0
        self.peak_resident_bytes = 0
        self.capture_calls = 0
        self.capture_time_s = 0.0
        self._reported_capture_calls = 0
        self._reported_capture_time_s = 0.0
        self._reported_produced_count = 0

    def __enter__(self) -> "LocalRolloutStream":
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close(reason="training_failed" if exc_type else "training_finished")
        return False

    def _start(self) -> None:
        if self._started or self._closed:
            return
        self._started = True
        for worker in self.workers:
            worker.start()

    def _record_produced(self, refs: List[SampleRef]) -> None:
        health = self.feature_store.health()
        resident = int(health.get("resident_samples", 0))
        self.produced_count += len(refs)
        self.peak_resident_samples = max(self.peak_resident_samples, resident)
        self.peak_resident_bytes = max(
            self.peak_resident_bytes,
            int(health.get("resident_bytes", 0)),
        )
        if resident > self.max_resident_samples:
            raise RuntimeError(
                "local rollout exceeded its resident-sample bound: "
                f"{resident} > {self.max_resident_samples}"
            )

    @staticmethod
    def _raise_if_failed(status: dict) -> None:
        failed = int(status.get("prompts_failed", 0))
        if failed:
            raise RuntimeError(
                "local rollout ended with "
                f"{failed} terminally failed prompt(s); refusing partial training"
            )

    @staticmethod
    def _drained(status: dict) -> bool:
        return not status.get("prompts_pending") and not status.get("prompts_leased")

    def _pump_once(self, max_tasks: int) -> List[SampleRef]:
        resident = int(self.feature_store.health().get("resident_samples", 0))
        self.peak_resident_samples = max(self.peak_resident_samples, resident)
        capacity = self.max_resident_samples - resident
        if capacity <= 0:
            raise RuntimeError(
                "local rollout cannot progress: its feature store is at the "
                f"{self.max_resident_samples}-sample bound"
            )
        worker = self.workers[self._next_worker]
        self._next_worker = (self._next_worker + 1) % len(self.workers)
        local_capacity = min(max_tasks, capacity)
        capture_started = time.perf_counter()
        try:
            refs = worker.run_once(
                max_tasks=local_capacity * self.capture_batch_multiplier
            )
        finally:
            self.capture_calls += 1
            self.capture_time_s += time.perf_counter() - capture_started
        self._record_produced(refs)
        return refs

    def perf_metrics(self) -> dict[str, float]:
        """Return interval capture counters plus cumulative residency peaks."""
        capture_calls = self.capture_calls - self._reported_capture_calls
        capture_time_s = self.capture_time_s - self._reported_capture_time_s
        produced = self.produced_count - self._reported_produced_count
        self._reported_capture_calls = self.capture_calls
        self._reported_capture_time_s = self.capture_time_s
        self._reported_produced_count = self.produced_count
        return {
            "colocated_capture_calls": float(capture_calls),
            "colocated_capture_time_s": capture_time_s,
            "colocated_capture_samples_per_second": (
                produced / max(capture_time_s, 1e-12)
            ),
            "colocated_peak_resident_samples": float(self.peak_resident_samples),
            "colocated_peak_resident_gib": self.peak_resident_bytes / float(1 << 30),
        }

    def get(
        self,
        max_refs: int,
        timeout_s: Optional[float] = None,
        **_unused,
    ) -> List[SampleRef]:
        del timeout_s
        if max_refs < 1:
            return []
        if max_refs > self.max_resident_samples:
            raise ValueError(
                f"requested {max_refs} refs from a local rollout stream bounded "
                f"to {self.max_resident_samples} resident samples"
            )
        if self._closed:
            return []

        self._start()
        refs: List[SampleRef] = []
        stalled_calls = 0
        max_stalled_calls = self.max_stalled_rounds * len(self.workers)
        try:
            while len(refs) < max_refs:
                leased = self._queue.get(max_refs - len(refs), timeout_s=0.0)
                if leased:
                    refs.extend(leased)
                    stalled_calls = 0
                    continue
                status = self.controller.status()
                self._raise_if_failed(status)
                if self._drained(status):
                    return refs
                if self._pump_once(max_refs - len(refs)):
                    stalled_calls = 0
                    continue
                stalled_calls += 1
                if stalled_calls >= max_stalled_calls:
                    status = self.controller.status()
                    self._raise_if_failed(status)
                    raise RuntimeError(
                        "local rollout made no progress while prompts remain: "
                        f"pending={status.get('prompts_pending', 0)} "
                        f"leased={status.get('prompts_leased', 0)}"
                    )
            return refs
        except BaseException:
            if refs:
                self._queue.fail(
                    refs, reason="local_rollout_get_failed", retryable=True
                )
            raise

    def ack(self, refs: List[SampleRef]) -> None:
        self._queue.ack(refs)

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        self._queue.fail(refs, reason=reason, retryable=retryable)
        if not retryable:
            for ref in refs:
                self.feature_store.abort(ref.sample_id, reason=reason)

    def depth(self) -> int:
        return self._queue.depth()

    def in_flight(self) -> int:
        return self._queue.in_flight()

    def close(self, *, reason: str = "closed") -> None:
        if self._closed:
            return
        self._closed = True
        for worker in self.workers:
            worker.stop(reason=reason)
        self.feature_store.abort_all(reason=reason)


__all__ = ["LocalRolloutStream"]
