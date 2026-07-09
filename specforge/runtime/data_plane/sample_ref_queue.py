# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SampleRefQueue: a metadata-only queue with lease / ack / fail semantics.

The current implementation is in-process; the lease/ack contract is present so a
durable queue (visibility timeout, replay) can be swapped in later without
touching callers. Carries no tensors — only ``SampleRef`` metadata.
"""

from __future__ import annotations

import hashlib
import threading
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

from specforge.runtime.contracts import SampleRef, assert_no_tensors


def dp_partition(sample_id: str, num_partitions: int) -> int:
    """Stable DP-rank assignment for a sample under a given partition count.

    The resharding contract (M6): partitioning is a *consumer* decision computed
    from a stable key (``sample_id``), NOT pinned by the producer. So the same
    committed pool re-distributes cleanly when the consumer's DP width changes —
    a sample produced under one layout is consumable under another. Hash-based so
    it is balanced and deterministic across ranks/restarts.
    """
    if num_partitions <= 1:
        return 0
    digest = hashlib.sha1(sample_id.encode("utf-8")).hexdigest()
    return int(digest, 16) % num_partitions


class SampleRefQueue:
    def __init__(self, *, lease_timeout_s: Optional[float] = None) -> None:
        self.lease_timeout_s = lease_timeout_s
        self._pending: "OrderedDict[str, SampleRef]" = OrderedDict()
        self._leased: "OrderedDict[str, tuple[SampleRef, float]]" = OrderedDict()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def put(
        self, refs: List[SampleRef], *, partition_key: Optional[str] = None
    ) -> None:
        # partition_key reserves the per-DP-rank queue partition seam for future
        # reshard support. Today there is a single partition, so it is accepted
        # and ignored.
        with self._cv:
            for ref in refs:
                assert_no_tensors(ref)  # structural no-tensor guard
                # Idempotent on sample_id (at-least-once delivery).
                if ref.sample_id in self._leased or ref.sample_id in self._pending:
                    continue
                self._pending[ref.sample_id] = ref
            self._cv.notify_all()

    def get(
        self,
        max_refs: int,
        timeout_s: Optional[float] = None,
        *,
        partition_key: Optional[str] = None,
        partition: Optional[Tuple[int, int]] = None,
    ) -> List[SampleRef]:
        """Lease up to ``max_refs`` pending refs; block until some arrive or
        ``timeout_s`` elapses.

        ``partition=(index, num_partitions)`` restricts the lease to the DP shard
        this consumer owns under its *current* layout (resharding contract), so
        changing ``num_partitions`` re-distributes the same committed pool. See
        :meth:`_lease_partition_locked` for the (non-blocking) shard semantics.

        ``partition_key`` is a separate, still-reserved seam (a producer-side
        routing hint); it is accepted and ignored today. ``partition`` is the
        consumer-side resharding control — the two are unrelated.
        """
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while True:
                self._reclaim_expired_locked()
                if self._pending:
                    if partition is None:
                        return self._lease_any_locked(max_refs)
                    return self._lease_partition_locked(max_refs, partition)
                if deadline is None:
                    return []
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cv.wait(timeout=remaining)

    def _lease_any_locked(self, max_refs: int) -> List[SampleRef]:
        """FIFO-lease up to ``max_refs`` pending refs (no partitioning)."""
        out: List[SampleRef] = []
        while self._pending and len(out) < max_refs:
            sid, ref = self._pending.popitem(last=False)
            self._leased[sid] = (ref, time.monotonic())
            out.append(ref)
        return out

    def _lease_partition_locked(
        self, max_refs: int, partition: Tuple[int, int]
    ) -> List[SampleRef]:
        """Lease only this shard's matching refs (resharding contract).

        Returns whatever this shard has pending — possibly empty — and never
        blocks waiting for a cross-partition match (that would starve one shard
        behind another).
        """
        index, num_partitions = partition
        out: List[SampleRef] = []
        for sid in list(self._pending.keys()):
            if len(out) >= max_refs:
                break
            if dp_partition(sid, num_partitions) == index:
                ref = self._pending.pop(sid)
                self._leased[sid] = (ref, time.monotonic())
                out.append(ref)
        return out

    def ack(self, refs: List[SampleRef]) -> None:
        with self._cv:
            for ref in refs:
                self._leased.pop(ref.sample_id, None)  # idempotent

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        with self._cv:
            for ref in refs:
                self._leased.pop(ref.sample_id, None)
                if retryable:
                    self._pending[ref.sample_id] = ref  # back to the tail
            if retryable:
                self._cv.notify_all()

    def depth(self) -> int:
        with self._lock:
            return len(self._pending)

    def in_flight(self) -> int:
        with self._lock:
            return len(self._leased)

    def _reclaim_expired_locked(self) -> None:
        if self.lease_timeout_s is None or not self._leased:
            return
        now = time.monotonic()
        expired = [
            sid
            for sid, (_, leased_at) in self._leased.items()
            if now - leased_at > self.lease_timeout_s
        ]
        for sid in expired:
            ref, _ = self._leased.pop(sid)
            self._pending[sid] = ref


__all__ = ["SampleRefQueue"]
