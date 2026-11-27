"""Scheduler-owned reservations for immutable capture attempts."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


class CaptureControlRejected(ValueError):
    # Structured marker also survives injected in-process control transports.
    capture_control_rejected = True


class AttemptCapacityDeclined(CaptureControlRejected):
    """An atomic begin operation has not granted any reservations."""


def scalar_error(exc: BaseException) -> RuntimeError:
    """Detach completed worker errors from frames that may retain tensor storage."""
    message = str(exc)
    exc.__traceback__ = exc.__cause__ = exc.__context__ = None
    exc.args = (message,)
    error = RuntimeError(message)
    if getattr(exc, "capture_writer_drained", True) is False:
        error.capture_writer_drained = False
    return error


def digest_request(tokens: Iterable[int], spec: Mapping[str, Any]) -> str:
    value = {
        "tokens": list(tokens),
        "features": spec.get("features", {}),
        "passthrough": spec.get("passthrough", []),
    }
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@dataclass
class Attempt:
    identity: tuple[str, str, int]
    attempt_id: str
    request_digest: str
    token: str
    sequence: int
    state: str = "RESERVED"
    revoked: bool = False
    claimed: bool = False
    request: Any = None
    execution_epoch: int = 0
    drains: list = field(default_factory=list)
    deliveries: int = 0
    last_plan: dict = field(default_factory=dict)
    object_key_layout: str | None = None
    transfer_certificate: dict = field(default_factory=dict)
    cleanup_owner: str | None = None
    cleanup_token: str | None = None
    readers: dict = field(default_factory=dict)
    released_readers: deque = field(default_factory=deque)


class AttemptRegistry:
    """Bounded active reservations. Unknown tokens never authorize generation."""

    def __init__(self, epoch: str, capacity: int = 4096) -> None:
        if capacity < 1:
            raise ValueError("attempt capacity must be positive")
        self.epoch, self.capacity = epoch, capacity
        self.active: dict[tuple[str, str, int], Attempt] = {}
        self.tokens: dict[str, Attempt] = {}
        self.retired: deque[Attempt] = deque(maxlen=capacity)
        self.sequence = 0

    @staticmethod
    def identity(body: Mapping[str, Any]) -> tuple[str, str, int]:
        identity = (
            str(body["store_id"]),
            str(body["sample_id"]),
            int(body.get("gen", 1)),
        )
        if identity[2] != 1:
            raise ValueError("capture protocol requires gen=1")
        return identity

    def begin(self, body: Mapping[str, Any], token: str | None = None) -> Attempt:
        identity = self.identity(body)
        old = self.active.get(identity)
        attempt_id, digest = str(body["attempt_id"]), str(body["request_digest"])
        if not attempt_id or not digest:
            raise ValueError("attempt id and request digest are required")
        if old:
            if old.attempt_id != attempt_id or old.request_digest != digest:
                raise ValueError("prior sample attempt is not retired")
            if old.revoked:
                raise ValueError(
                    "attempt is revoked; resolve and clean it before retry"
                )
            return old
        if len(self.active) >= self.capacity:
            raise AttemptCapacityDeclined("attempt registry capacity exhausted")
        self.sequence += 1
        entry = Attempt(
            identity, attempt_id, digest, token or uuid.uuid4().hex, self.sequence
        )
        entry.object_key_layout = (
            "attempt-v1" if body.get("publication_ownership") == "attempt-v1" else None
        )
        self.active[identity] = entry
        self.tokens[entry.token] = entry
        return entry

    def begin_batch(
        self, bodies: Sequence[Mapping[str, Any]], tokens: Sequence[str]
    ) -> list[Attempt]:
        trial, entries = self.prepare_begin_batch(bodies, tokens)
        self.commit_begin_batch(trial)
        return entries

    def prepare_begin_batch(
        self, bodies: Sequence[Mapping[str, Any]], tokens: Sequence[str]
    ) -> tuple[AttemptRegistry, list[Attempt]]:
        # Validate the complete batch before changing the live registry.
        if len(bodies) != len(tokens) or len(
            {self.identity(body) for body in bodies}
        ) != len(bodies):
            raise ValueError("duplicate begin_batch identity or token count mismatch")
        trial = AttemptRegistry(self.epoch, self.capacity)
        trial.active = self.active.copy()
        trial.tokens = self.tokens.copy()
        trial.sequence = self.sequence
        entries = [trial.begin(body, token) for body, token in zip(bodies, tokens)]
        return trial, entries

    def commit_begin_batch(self, trial: AttemptRegistry) -> None:
        self.active, self.tokens, self.sequence = (
            trial.active,
            trial.tokens,
            trial.sequence,
        )

    def lookup(self, body: Mapping[str, Any]) -> Attempt:
        token = str(body["reservation_token"])
        entry = self.tokens.get(token)
        if (
            entry is None
            or entry.identity != self.identity(body)
            or entry.attempt_id != str(body["attempt_id"])
        ):
            raise ValueError("unknown, retired or mismatched reservation token")
        return entry

    def claim(self, body: Mapping[str, Any], request_digest: str) -> Attempt:
        entry = self.lookup(body)
        if entry.revoked or entry.claimed:
            raise ValueError("reservation revoked or generation already claimed")
        if request_digest != entry.request_digest:
            raise ValueError("generate payload differs from reservation")
        entry.claimed, entry.state = True, "QUEUED"
        return entry

    def retire(self, entry: Attempt) -> None:
        if (
            entry.state not in ("WRITER_DRAINED", "PUBLISHED")
            or entry.deliveries
            or entry.readers
        ):
            raise ValueError("writer is not drained")
        if any(not handle.done() for handle in entry.drains):
            raise ValueError("runtime readers are not drained")
        entry.revoked, entry.state = True, "RETIRED"
        self.tokens.pop(entry.token, None)
        self.active.pop(entry.identity, None)
        entry.request = None
        entry.drains.clear()
        self.retired.append(entry)
