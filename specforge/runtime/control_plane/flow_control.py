# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hysteretic producer flow control for the canonical streaming runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FlowControlLimits:
    """Reference/byte watermarks and the per-worker capture lease cap."""

    high_watermark_refs: int = 256
    low_watermark_refs: Optional[int] = None
    high_watermark_bytes: Optional[int] = None
    low_watermark_bytes: Optional[int] = None
    max_prompt_lease_per_worker: int = 8

    def __post_init__(self) -> None:
        low_refs = (
            self.high_watermark_refs
            if self.low_watermark_refs is None
            else self.low_watermark_refs
        )
        low_bytes = (
            self.high_watermark_bytes
            if self.low_watermark_bytes is None
            else self.low_watermark_bytes
        )
        if self.high_watermark_refs < 1:
            raise ValueError("high_watermark_refs must be >= 1")
        if low_refs < 0 or low_refs > self.high_watermark_refs:
            raise ValueError(
                "low_watermark_refs must be between 0 and high_watermark_refs"
            )
        if self.high_watermark_bytes is None and self.low_watermark_bytes is not None:
            raise ValueError(
                "low_watermark_bytes requires high_watermark_bytes"
            )
        if self.high_watermark_bytes is not None:
            if self.high_watermark_bytes < 1:
                raise ValueError("high_watermark_bytes must be >= 1")
            if low_bytes is None or not 0 <= low_bytes <= self.high_watermark_bytes:
                raise ValueError(
                    "low_watermark_bytes must be between 0 and "
                    "high_watermark_bytes"
                )
        if self.max_prompt_lease_per_worker < 1:
            raise ValueError("max_prompt_lease_per_worker must be >= 1")

    @property
    def resolved_low_watermark_refs(self) -> int:
        return (
            self.high_watermark_refs
            if self.low_watermark_refs is None
            else self.low_watermark_refs
        )

    @property
    def resolved_low_watermark_bytes(self) -> Optional[int]:
        if self.high_watermark_bytes is None:
            return None
        return (
            self.high_watermark_bytes
            if self.low_watermark_bytes is None
            else self.low_watermark_bytes
        )


class ProducerFlowControl:
    """Thread-safe pause/resume policy shared by all rollout workers."""

    def __init__(self, limits: FlowControlLimits) -> None:
        self.limits = limits
        self._paused = False
        self._lock = threading.Lock()
        self._stats = {
            "pause_transitions": 0,
            "resume_transitions": 0,
            "wait_checks": 0,
        }

    def prompt_lease(self, requested: int) -> int:
        """Cap a worker's capture request at the configured in-flight lease."""
        return max(0, min(int(requested), self.limits.max_prompt_lease_per_worker))

    def should_pause(self, *, in_flight_refs: int, resident_bytes: int = 0) -> bool:
        """Apply ref/byte hysteresis and return the current latched state."""
        refs = max(0, int(in_flight_refs))
        resident = max(0, int(resident_bytes))
        high_bytes = self.limits.high_watermark_bytes
        low_bytes = self.limits.resolved_low_watermark_bytes
        over_high = refs >= self.limits.high_watermark_refs or (
            high_bytes is not None and resident >= high_bytes
        )
        under_low = refs <= self.limits.resolved_low_watermark_refs and (
            low_bytes is None or resident <= low_bytes
        )
        with self._lock:
            if not self._paused and over_high:
                self._paused = True
                self._stats["pause_transitions"] += 1
            elif self._paused and under_low:
                self._paused = False
                self._stats["resume_transitions"] += 1
            if self._paused:
                self._stats["wait_checks"] += 1
            return self._paused

    def snapshot(self, *, in_flight_refs: int, resident_bytes: int = 0) -> Dict[str, Any]:
        with self._lock:
            return {
                "paused": self._paused,
                **self._stats,
                "in_flight_refs": int(in_flight_refs),
                "resident_bytes": int(resident_bytes),
                "high_watermark_refs": self.limits.high_watermark_refs,
                "low_watermark_refs": self.limits.resolved_low_watermark_refs,
                "high_watermark_bytes": self.limits.high_watermark_bytes,
                "low_watermark_bytes": self.limits.resolved_low_watermark_bytes,
                "max_prompt_lease_per_worker": (
                    self.limits.max_prompt_lease_per_worker
                ),
            }


__all__ = ["FlowControlLimits", "ProducerFlowControl"]
