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

import threading
import time
from collections import OrderedDict
from typing import List, Optional

from specforge.runtime.contracts import SampleRef, assert_no_tensors


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
    ) -> List[SampleRef]:
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while True:
                self._reclaim_expired_locked()
                if self._pending:
                    out: List[SampleRef] = []
                    for _ in range(max_refs):
                        if not self._pending:
                            break
                        sid, ref = self._pending.popitem(last=False)
                        self._leased[sid] = (ref, time.monotonic())
                        out.append(ref)
                    return out
                if deadline is None:
                    return []
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return []
                self._cv.wait(timeout=remaining)

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
