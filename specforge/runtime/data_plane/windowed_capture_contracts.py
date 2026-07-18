# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Serializable identities and state shared by windowed capture components."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import Any, Mapping

from specforge.runtime.contracts import SampleRef


class CaptureState(str, Enum):
    ABSENT = "absent"
    QUEUED = "queued"
    CAPTURING = "capturing"
    COMMITTING = "committing"
    READY = "ready"
    EVICTING = "evicting"
    FAILED = "failed"


class CapturePriority(IntEnum):
    DEMAND = 0
    PREFETCH = 1


@dataclass(frozen=True, order=True)
class CaptureKey:
    source_sample_id: str
    contract_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_sample_id, str) or not self.source_sample_id:
            raise ValueError("source_sample_id must be a non-empty string")
        digest = self.contract_digest
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("contract_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class AcquireTicket:
    token: str
    consumer_id: str
    key: CaptureKey
    source_index: int
    ready_at_request: bool


@dataclass(frozen=True)
class CaptureReadLease:
    token: str
    consumer_id: str
    key: CaptureKey
    source_index: int
    generation: int
    ref: SampleRef
    ready_at_request: bool
    wait_s: float


@dataclass(frozen=True)
class CaptureRequest:
    key: CaptureKey
    source_index: int
    generation: int
    priority: CapturePriority
    demand_consumers: tuple[str, ...]
    reserved_bytes: int


@dataclass(frozen=True)
class EvictionCandidate:
    key: CaptureKey
    source_index: int
    generation: int
    ref: SampleRef


class CaptureFailedError(RuntimeError):
    """A requested capture reached a terminal failure."""


class ConsumerFailedError(RuntimeError):
    """A consumer was failed or expired while waiting for a capture."""


def _jsonable_contract(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_contract(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable_contract(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable_contract(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"capture contract contains unsupported {type(value).__name__}")


def capture_contract_digest(contract: Any) -> str:
    """Return a canonical digest separating incompatible capture payloads."""
    encoded = json.dumps(
        _jsonable_contract(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


__all__ = [
    "AcquireTicket",
    "CaptureFailedError",
    "CaptureKey",
    "CapturePriority",
    "CaptureReadLease",
    "CaptureRequest",
    "CaptureState",
    "ConsumerFailedError",
    "EvictionCandidate",
    "capture_contract_digest",
]
