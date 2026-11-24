"""Capture values, ownership handles, and canonical feature validation.

The capture runtime names the auxiliary output and post-norm last hidden state.
The sink assigns their physical object keys. Each ContentBlockKey keeps its
canonical token tuple locally for prefix validation.
"""

from __future__ import annotations

import hashlib
import json
import threading
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Callable, Mapping

import torch

DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}
DTYPE_BYTES = {"bfloat16": 2, "float16": 2, "float32": 4, "float64": 8}


class CaptureContractError(ValueError):
    """A malformed plan/span/schema, rather than a recoverable cache miss."""


class AdmissionBudgetError(CaptureContractError):
    """One request cannot fit even when all other reservations are released."""


class CaptureCancelledError(RuntimeError):
    """Execution fenced; all safely reclaimable resources have been drained."""


def checked_int(value: int, name: str, minimum: int = 0) -> int:
    if type(value) is not int or value < minimum:
        raise CaptureContractError(f"{name} must be an integer >= {minimum}")
    return value


def canonical_digest(value) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return hashlib.sha256(encoded.encode("ascii")).hexdigest()


def _normalized_error(error: BaseException) -> BaseException:
    """A safe error outcome cannot retain drained tensor/stream traceback locals."""
    try:
        clean = type(error)(str(error))
    except BaseException:
        clean = RuntimeError(f"{type(error).__name__}: {error}")
    clean.__traceback__ = None
    clean.__cause__ = None
    clean.__context__ = None
    clean.__suppress_context__ = True
    return clean


@dataclass(frozen=True)
class FeatureSpec:
    name: str
    width: int
    dtype: str
    layer_ids: tuple[int, ...]

    def __post_init__(self):
        if self.name not in ("aux", "last_hidden"):
            raise CaptureContractError(f"unknown feature artifact {self.name!r}")
        checked_int(self.width, "feature width", 1)
        if self.dtype not in DTYPES:
            raise CaptureContractError(f"unsupported canonical dtype {self.dtype!r}")
        if not isinstance(self.layer_ids, tuple):
            raise CaptureContractError("layer_ids must be an immutable tuple")
        for layer in self.layer_ids:
            checked_int(layer, "layer id")
        if len(set(self.layer_ids)) != len(self.layer_ids):
            raise CaptureContractError("duplicate capture layers")
        if self.name == "aux" and not self.layer_ids:
            raise CaptureContractError("aux needs ordered capture layer ids")
        if self.name == "last_hidden" and self.layer_ids:
            raise CaptureContractError("post-norm last_hidden uses layer_ids=()")


@dataclass(frozen=True)
class FeatureSchema:
    features: tuple[FeatureSpec, ...]
    signature: str = field(init=False)
    row_bytes: int = field(init=False)

    def __post_init__(self):
        if not isinstance(self.features, tuple) or not self.features:
            raise CaptureContractError("features must be a nonempty immutable tuple")
        if any(not isinstance(f, FeatureSpec) for f in self.features):
            raise CaptureContractError("schema requires FeatureSpec entries")
        if len({f.name for f in self.features}) != len(self.features):
            raise CaptureContractError("duplicate feature artifact")
        manifest = [(f.name, f.width, f.dtype, f.layer_ids) for f in self.features]
        object.__setattr__(
            self, "signature", canonical_digest(["spec-features-v1", manifest])
        )
        object.__setattr__(
            self,
            "row_bytes",
            sum(f.width * DTYPE_BYTES[f.dtype] for f in self.features),
        )

    def validate(
        self, features: Mapping[str, torch.Tensor], rows: int, device: str | None = None
    ) -> None:
        checked_int(rows, "feature rows")
        if set(features) != {f.name for f in self.features}:
            raise CaptureContractError("feature names do not match the complete schema")
        for spec in self.features:
            tensor = features[spec.name]
            if not isinstance(tensor, torch.Tensor):
                raise CaptureContractError(f"{spec.name} is not a tensor")
            if tuple(tensor.shape) != (rows, spec.width):
                raise CaptureContractError(
                    f"{spec.name} shape {tuple(tensor.shape)} != {(rows, spec.width)}"
                )
            if tensor.dtype != DTYPES[spec.dtype]:
                raise CaptureContractError(
                    f"{spec.name} dtype does not match {spec.dtype}"
                )
            if device is not None and tensor.device.type != device:
                raise CaptureContractError(f"{spec.name} must be on {device}")
            if not tensor.is_contiguous():
                raise CaptureContractError(f"{spec.name} must be contiguous")


@dataclass(frozen=True)
class ExecutionIdentity:
    server_epoch: str
    request_id: str
    attempt_id: str
    execution_epoch: int

    def __post_init__(self):
        if not all(
            isinstance(v, str) and v
            for v in (self.server_epoch, self.request_id, self.attempt_id)
        ):
            raise CaptureContractError(
                "execution identity fields must be nonempty strings"
            )
        checked_int(self.execution_epoch, "execution epoch")


@dataclass(frozen=True)
class ContentBlockKey:
    namespace: str
    start: int
    end: int
    digest: str
    parent_digest: str
    _tokens: tuple[int, ...] = field(repr=False)


@dataclass(frozen=True)
class PrefixLease:
    lease_id: str
    coverage_end: int


@dataclass(frozen=True)
class OutputReservation:
    reservation_id: str
    execution: ExecutionIdentity
    host_bytes: int
    pinned_bytes: int
    device_bytes: int
    N: int
    L: int
    passthrough_bytes: int


@dataclass(frozen=True)
class CapturePlan:
    execution: ExecutionIdentity
    namespace: str
    tokens: tuple[int, ...]
    N: int
    L: int
    schema_signature: str
    lease: PrefixLease
    reservation: OutputReservation
    plan_digest: str


@dataclass(frozen=True)
class CapturedSpan:
    execution: ExecutionIdentity
    start: int
    end: int
    features: Mapping[str, torch.Tensor]
    ready_event: torch.cuda.Event | None


class MaterializedSample:
    """Owned CPU output; release only after every sink reader has ended.

    A release drops this object's tensor references before returning its budget.
    Copies/views retained by the caller must have ended before invoking release.
    """

    def __init__(
        self,
        execution: ExecutionIdentity,
        features: Mapping[str, torch.Tensor],
        n_tokens: int,
        on_release: Callable[[], None],
    ) -> None:
        self.execution = execution
        self.n_tokens = n_tokens
        self._features = MappingProxyType(dict(features))
        self._on_release = on_release
        self._lock = threading.Lock()
        self._released = False

    @property
    def features(self) -> Mapping[str, torch.Tensor]:
        with self._lock:
            if self._released:
                raise CaptureContractError("sample already released")
            return self._features

    def release(self) -> None:
        with self._lock:
            if self._released:
                return
            self._released = True
            self._features = MappingProxyType({})
            callback, self._on_release = self._on_release, lambda: None
        callback()


@dataclass(frozen=True)
class CaptureCompletion:
    execution: ExecutionIdentity
    sample: MaterializedSample | None = None
    error: BaseException | None = None

    def __post_init__(self):
        if (self.sample is None) == (self.error is None):
            raise CaptureContractError("completion must have exactly one outcome")


class DrainHandle:
    """A timeout means storage is still owned, never permission to reclaim."""

    def __init__(self, event: threading.Event):
        self._event = event

    def done(self) -> bool:
        return self._event.is_set()

    def wait(self, timeout: float | None = None) -> bool:
        return self._event.wait(timeout)
