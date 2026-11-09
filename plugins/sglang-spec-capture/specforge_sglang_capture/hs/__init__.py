"""Single-server content-keyed raw feature cache, frozen Plan A v1 API.

block_keys(namespace, tokens) returns complete canonical chunk identities.
stats() returns bounded storage/reservation and lifecycle counters.
"""

from .cache import PrefixFeatureCache
from .controller import CaptureController
from .protocol import (
    AdmissionBudgetError,
    CaptureCancelledError,
    CaptureCompletion,
    CaptureContractError,
    CapturePlan,
    CapturedSpan,
    ContentBlockKey,
    DrainHandle,
    ExecutionIdentity,
    FeatureSchema,
    FeatureSpec,
    FillReservation,
    MaterializedSample,
    OutputReservation,
    PrefixLease,
)

__all__ = [
    "AdmissionBudgetError",
    "CaptureCancelledError",
    "CaptureCompletion",
    "CaptureContractError",
    "CaptureController",
    "CapturePlan",
    "CapturedSpan",
    "ContentBlockKey",
    "DrainHandle",
    "ExecutionIdentity",
    "FeatureSchema",
    "FeatureSpec",
    "FillReservation",
    "MaterializedSample",
    "OutputReservation",
    "PrefixFeatureCache",
    "PrefixLease",
]
