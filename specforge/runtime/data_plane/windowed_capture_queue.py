# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Ordered queue facade over one consumer's capture-window cursor."""

from __future__ import annotations

import threading
from typing import Callable, Optional, Sequence

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.windowed_capture_contracts import CaptureReadLease
from specforge.runtime.data_plane.windowed_capture_registry import (
    SQLiteWindowedCaptureRegistry,
)


class WindowedCaptureQueue:
    """Ordered ``SampleRefQueue`` facade for one logical consumer."""

    def __init__(
        self,
        registry: SQLiteWindowedCaptureRegistry,
        consumer_id: str,
        *,
        idle_timeout_s: Optional[float] = 1800.0,
        record_refs: Optional[Callable[[Sequence[SampleRef]], None]] = None,
    ) -> None:
        if idle_timeout_s is not None and idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be > 0 or None")
        snapshot = registry.snapshot()
        if consumer_id not in snapshot["consumers"]:
            raise KeyError(f"unknown consumer {consumer_id!r}")
        self.registry = registry
        self.consumer_id = consumer_id
        self.total_samples = int(snapshot["total_samples"])
        self.idle_timeout_s = idle_timeout_s
        if record_refs is not None and not callable(record_refs):
            raise TypeError("record_refs must be callable or None")
        self._record_refs = record_refs
        self._next_fetch = int(snapshot["consumers"][consumer_id]["cursor"])
        self._leases: dict[str, CaptureReadLease] = {}
        self._closed = False
        self._get_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float | int] = {
            "refs": 0,
            "ready_at_request_refs": 0,
            "demand_wait_s": 0.0,
            "max_demand_wait_s": 0.0,
        }

    def get(self, n: int, timeout_s: float = 0.0) -> list[SampleRef]:
        del timeout_s  # Registry waits use the explicit idle timeout.
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        with self._get_lock:
            while True:
                with self._state_lock:
                    if self._closed:
                        return []
                    start = self._next_fetch
                    if start >= self.total_samples:
                        if not self._leases:
                            self.registry.complete_consumer(self.consumer_id)
                            self._closed = True
                        return []
                    stop = min(self.total_samples, start + n)
                tickets = self.registry.request_many(
                    self.consumer_id, range(start, stop)
                )
                acquired: list[CaptureReadLease] = []
                try:
                    for ticket in tickets:
                        acquired.append(
                            self.registry.wait_ready(
                                ticket, timeout_s=self.idle_timeout_s
                            )
                        )
                except BaseException:
                    acquired_indices = {lease.source_index for lease in acquired}
                    for ticket in tickets:
                        if ticket.source_index not in acquired_indices:
                            try:
                                self.registry.cancel_acquire(ticket)
                            except RuntimeError:
                                pass
                    self.registry.abandon_leases(self.consumer_id, acquired)
                    raise
                refs = [lease.ref for lease in acquired]
                if len({ref.sample_id for ref in refs}) != len(refs):
                    self.registry.abandon_leases(self.consumer_id, acquired)
                    raise RuntimeError(
                        "windowed capture batch contains duplicate sample IDs"
                    )
                if self._record_refs is not None:
                    try:
                        self._record_refs(refs)
                    except BaseException:
                        self.registry.abandon_leases(self.consumer_id, acquired)
                        raise
                with self._state_lock:
                    if self._closed:
                        self.registry.abandon_leases(self.consumer_id, acquired)
                        return []
                    if self._next_fetch < start:
                        self.registry.abandon_leases(self.consumer_id, acquired)
                        continue
                    self._leases.update(
                        (lease.ref.sample_id, lease) for lease in acquired
                    )
                    self._next_fetch = max(self._next_fetch, stop)
                    for lease in acquired:
                        self._metrics["refs"] += 1
                        self._metrics["ready_at_request_refs"] += int(
                            lease.ready_at_request
                        )
                        self._metrics["demand_wait_s"] += lease.wait_s
                        self._metrics["max_demand_wait_s"] = max(
                            self._metrics["max_demand_wait_s"], lease.wait_s
                        )
                return refs

    def _resolve_leases(self, refs: Sequence[SampleRef]) -> list[CaptureReadLease]:
        missing = [ref.sample_id for ref in refs if ref.sample_id not in self._leases]
        if missing:
            raise RuntimeError(
                f"windowed queue references samples not leased: {missing}"
            )
        return [self._leases[ref.sample_id] for ref in refs]

    def ack(self, refs: list[SampleRef]) -> None:
        self.ack_ids([ref.sample_id for ref in refs])

    def ack_ids(self, sample_ids: list[str]) -> None:
        """Acknowledge the exact leased prefix after its durable train ACK."""
        if not sample_ids:
            return
        with self._state_lock:
            expected = list(self._leases)[: len(sample_ids)]
            if expected != list(sample_ids):
                raise RuntimeError(
                    "windowed queue acknowledgement is not the leased prefix: "
                    f"expected={expected}, got={list(sample_ids)}"
                )
            leases = [self._leases[sample_id] for sample_id in sample_ids]
            self.registry.release_and_advance(self.consumer_id, leases)
            for sample_id in sample_ids:
                self._leases.pop(sample_id)

    def fail(self, refs: list[SampleRef], reason: str, retryable: bool) -> None:
        del reason
        with self._state_lock:
            leases = self._resolve_leases(refs)
            self.registry.abandon_leases(self.consumer_id, leases)
            for ref in refs:
                self._leases.pop(ref.sample_id)
            if retryable and leases:
                self._next_fetch = min(
                    self._next_fetch,
                    min(lease.source_index for lease in leases),
                )

    def depth(self) -> int:
        with self._state_lock:
            return max(0, self.total_samples - self._next_fetch)

    def in_flight(self) -> int:
        with self._state_lock:
            return len(self._leases)

    def metrics(self) -> dict[str, float | int]:
        with self._state_lock:
            refs = int(self._metrics["refs"])
            return {
                **self._metrics,
                "ready_at_request_ratio": (
                    float(self._metrics["ready_at_request_refs"]) / refs
                    if refs
                    else 0.0
                ),
                "mean_demand_wait_s": (
                    float(self._metrics["demand_wait_s"]) / refs if refs else 0.0
                ),
                "next_fetch": self._next_fetch,
                "in_flight": len(self._leases),
            }

    def drained(self) -> bool:
        with self._state_lock:
            return self._next_fetch == self.total_samples and not self._leases

    def finalize(self) -> None:
        self.complete()

    def complete(self, *, allow_partial: bool = False) -> None:
        with self._state_lock:
            if not allow_partial and (
                self._next_fetch != self.total_samples or self._leases
            ):
                raise RuntimeError(
                    f"consumer {self.consumer_id!r} queue did not drain: "
                    f"next={self._next_fetch}/{self.total_samples}, "
                    f"leases={len(self._leases)}"
                )
            if allow_partial and self._leases:
                self.registry.abandon_leases(
                    self.consumer_id, list(self._leases.values())
                )
                self._leases.clear()
            self.registry.complete_consumer(
                self.consumer_id, allow_partial=allow_partial
            )
            self._closed = True

    def close(self, error: Optional[BaseException | str] = None) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._leases:
                self.registry.abandon_leases(
                    self.consumer_id, list(self._leases.values())
                )
                self._leases.clear()
            if error is not None:
                self.registry.fail_consumer(self.consumer_id, error)
            self._closed = True


__all__ = ["WindowedCaptureQueue"]
