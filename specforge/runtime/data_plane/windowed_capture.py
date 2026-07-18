# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Public facade for transactional, consumer-driven capture windows.

Payloads remain in ``FeatureStore``. The implementation is split by ownership:
serializable contracts, SQLite state transitions, and the consumer queue facade.
Imports from this original module remain stable for callers.
"""

from specforge.runtime.data_plane.windowed_capture_contracts import (
    AcquireTicket,
    CaptureFailedError,
    CaptureKey,
    CapturePriority,
    CaptureReadLease,
    CaptureRequest,
    CaptureState,
    ConsumerFailedError,
    EvictionCandidate,
    capture_contract_digest,
)
from specforge.runtime.data_plane.windowed_capture_queue import WindowedCaptureQueue
from specforge.runtime.data_plane.windowed_capture_registry import (
    SQLiteWindowedCaptureRegistry,
)

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
    "SQLiteWindowedCaptureRegistry",
    "WindowedCaptureQueue",
    "capture_contract_digest",
]
