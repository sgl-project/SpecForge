# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""In-process metadata-only staging queue for active online paths."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from typing import List, Optional

from specforge.runtime.contracts import SampleRef, assert_no_tensors


class SampleRefQueue:
    def __init__(self) -> None:
        self._pending: "OrderedDict[str, SampleRef]" = OrderedDict()
        self._leased: "OrderedDict[str, SampleRef]" = OrderedDict()
        self._lock = threading.Lock()
        self._cv = threading.Condition(self._lock)

    def put(self, refs: List[SampleRef]) -> None:
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
    ) -> List[SampleRef]:
        """Lease pending refs, optionally waiting up to ``timeout_s``."""
        deadline = None if timeout_s is None else time.monotonic() + timeout_s
        with self._cv:
            while True:
                if self._pending:
                    return self._lease_any_locked(max_refs)
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
            self._leased[sid] = ref
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

__all__ = ["SampleRefQueue"]
