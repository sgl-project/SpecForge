# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Process-safe runtime loops for consumer-driven capture windows."""

from __future__ import annotations

import dataclasses
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Optional, Sequence

from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask, SampleRef, assert_no_tensors
from specforge.runtime.data_plane.windowed_capture import (
    CaptureRequest,
    SQLiteWindowedCaptureRegistry,
)


@dataclass
class WindowedConsumerControl:
    """Keep one logical consumer live across model initialization and training."""

    registry: SQLiteWindowedCaptureRegistry
    consumer_id: str
    heartbeat_interval_s: float
    _stop: threading.Event = field(default_factory=threading.Event, init=False)
    _ready: threading.Event = field(default_factory=threading.Event, init=False)
    _thread: Optional[threading.Thread] = field(default=None, init=False)
    _error: Optional[BaseException] = field(default=None, init=False)
    _started: bool = field(default=False, init=False)
    _terminal: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.heartbeat_interval_s <= 0:
            raise ValueError("heartbeat_interval_s must be > 0")

    def _heartbeat_loop(self) -> None:
        while not self._stop.wait(self.heartbeat_interval_s):
            try:
                self.registry.heartbeat(self.consumer_id, ready=self._ready.is_set())
            except BaseException as exc:  # exposed synchronously by ensure_healthy
                # WindowedCaptureQueue may observe EOF and complete the consumer
                # between heartbeat intervals. That terminal race is success.
                try:
                    state = self.registry.snapshot()["consumers"][self.consumer_id][
                        "state"
                    ]
                except BaseException:
                    state = None
                if state == "completed":
                    return
                self._error = exc
                return

    def start(self) -> "WindowedConsumerControl":
        if self._started:
            return self
        if self._terminal:
            raise RuntimeError("cannot start terminal windowed consumer control")
        self.registry.heartbeat(self.consumer_id, ready=False)
        self._thread = threading.Thread(
            target=self._heartbeat_loop,
            name=f"windowed-heartbeat-{self.consumer_id}",
            daemon=True,
        )
        self._started = True
        self._thread.start()
        return self

    def mark_ready(self) -> None:
        self.start()
        self.ensure_healthy()
        self._ready.set()
        self.registry.heartbeat(self.consumer_id, ready=True)

    def _stop_heartbeat(self) -> None:
        self._stop.set()
        if self._thread is None or not self._thread.is_alive():
            return
        self._thread.join(timeout=max(1.0, self.heartbeat_interval_s * 2))
        if self._thread.is_alive():
            raise TimeoutError(f"consumer {self.consumer_id!r} heartbeat did not stop")

    def ensure_healthy(self) -> None:
        if self._error is not None:
            raise RuntimeError(
                f"consumer {self.consumer_id!r} registry heartbeat failed"
            ) from self._error

    def complete(self, *, allow_partial: bool = False) -> None:
        if self._terminal:
            return
        self._stop_heartbeat()
        self.ensure_healthy()
        self.registry.complete_consumer(self.consumer_id, allow_partial=allow_partial)
        self._terminal = True

    def fail(self, error: BaseException | str) -> None:
        if self._terminal:
            return
        try:
            self._stop_heartbeat()
        except BaseException as stop_error:
            if isinstance(error, BaseException):
                error.add_note(f"failed to stop registry heartbeat: {stop_error!r}")
        self.registry.fail_consumer(self.consumer_id, error)
        self._terminal = True

    def close(self) -> None:
        self._stop_heartbeat()


def start_windowed_consumer_control(
    registry: SQLiteWindowedCaptureRegistry,
    consumer_id: str,
    *,
    lookbehind: int,
    lookahead: int,
    prefetch_depth: int,
    max_outstanding: int,
    heartbeat_interval_s: float,
    durable_cursor: Optional[int] = None,
) -> WindowedConsumerControl:
    """Register a new consumer or reconcile an interrupted active consumer."""
    snapshot = registry.snapshot()
    existing = snapshot["consumers"].get(consumer_id)
    cursor = 0 if durable_cursor is None else durable_cursor
    if existing is None:
        registry.register_consumer(
            consumer_id,
            lookbehind=lookbehind,
            lookahead=lookahead,
            prefetch_depth=prefetch_depth,
            max_outstanding=max_outstanding,
            cursor=cursor,
        )
    else:
        expected = (lookbehind, lookahead, prefetch_depth, max_outstanding)
        observed = tuple(
            int(existing[name])
            for name in (
                "lookbehind",
                "lookahead",
                "prefetch_depth",
                "max_outstanding",
            )
        )
        if observed != expected:
            raise RuntimeError(
                f"consumer {consumer_id!r} window mismatch: "
                f"expected={expected}, observed={observed}"
            )
        if existing["state"] in ("completed", "failed"):
            raise RuntimeError(
                f"cannot resume terminal consumer {consumer_id!r}: "
                f"{existing['state']}"
            )
        registry.resume_consumer(
            consumer_id,
            durable_cursor=(
                int(existing["cursor"]) if durable_cursor is None else cursor
            ),
        )
    return WindowedConsumerControl(
        registry=registry,
        consumer_id=consumer_id,
        heartbeat_interval_s=heartbeat_interval_s,
    ).start()


class WindowedCaptureService:
    """Demand-driven capture producer and sole payload-reclamation owner."""

    def __init__(
        self,
        registry: SQLiteWindowedCaptureRegistry,
        *,
        prompts: Sequence[PromptTask],
        feature_source: Any,
        capture: CaptureConfig,
        owner_store: Any,
        capture_batch_size: int = 8,
        batch_wait_s: float = 0.002,
        max_capture_retries: int = 2,
        retry_backoff_s: float = 0.05,
        consumer_registration_timeout_s: float = 600.0,
        consumer_heartbeat_timeout_s: float = 120.0,
        poll_s: float = 0.01,
        reclaim_batch_size: int = 64,
    ) -> None:
        for name, value in (
            ("capture_batch_size", capture_batch_size),
            ("reclaim_batch_size", reclaim_batch_size),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if max_capture_retries < 0:
            raise ValueError("max_capture_retries must be >= 0")
        for name, value in (
            ("batch_wait_s", batch_wait_s),
            ("retry_backoff_s", retry_backoff_s),
            ("consumer_registration_timeout_s", consumer_registration_timeout_s),
            ("consumer_heartbeat_timeout_s", consumer_heartbeat_timeout_s),
            ("poll_s", poll_s),
        ):
            if value < 0 or (
                name not in {"batch_wait_s", "retry_backoff_s"} and value == 0
            ):
                raise ValueError(f"{name} has invalid duration {value}")
        if not callable(getattr(feature_source, "produce_refs", None)):
            raise TypeError(
                "feature_source must expose produce_refs(tasks, capture=...)"
            )
        if not callable(getattr(owner_store, "reclaim", None)):
            raise TypeError("owner_store must expose reclaim(ref, reason=...)")
        if hasattr(owner_store, "lifetime_owner") and not owner_store.lifetime_owner:
            raise ValueError(
                "windowed producer must use the feature-store lifetime owner"
            )

        prompt_by_id: dict[str, PromptTask] = {}
        for prompt in prompts:
            if not isinstance(prompt, PromptTask):
                raise TypeError("prompts must contain PromptTask values")
            assert_no_tensors(prompt)
            if prompt.task_id in prompt_by_id:
                raise ValueError(f"duplicate prompt task_id {prompt.task_id!r}")
            prompt_by_id[prompt.task_id] = prompt
        if not prompt_by_id:
            raise ValueError("prompts must not be empty")

        self.registry = registry
        self.prompts = prompt_by_id
        self.feature_source = feature_source
        self.capture = capture
        self.owner_store = owner_store
        self.capture_batch_size = capture_batch_size
        self.batch_wait_s = batch_wait_s
        self.max_capture_retries = max_capture_retries
        self.retry_backoff_s = retry_backoff_s
        self.consumer_registration_timeout_s = consumer_registration_timeout_s
        self.consumer_heartbeat_timeout_s = consumer_heartbeat_timeout_s
        self.poll_s = poll_s
        self.reclaim_batch_size = reclaim_batch_size
        self.capture_calls = 0
        self.captured_refs = 0
        self.capture_failures = 0

    def _task(self, request: CaptureRequest) -> PromptTask:
        try:
            prompt = self.prompts[request.key.source_sample_id]
        except KeyError as exc:
            raise KeyError(
                f"capture registry requested unknown source "
                f"{request.key.source_sample_id!r}"
            ) from exc
        return dataclasses.replace(prompt, attempt=request.generation)

    def _fail_requests(
        self,
        requests: Sequence[CaptureRequest],
        error: BaseException | str,
        *,
        retryable: bool,
    ) -> bool:
        retried = False
        for request in requests:
            retried |= self.registry.fail_capture(
                request,
                error,
                retryable=retryable,
                max_retries=self.max_capture_retries,
            )
            self.capture_failures += 1
        return retried

    def _capture_requests(self, requests: Sequence[CaptureRequest]) -> None:
        tasks = [self._task(request) for request in requests]
        self.capture_calls += 1
        try:
            results = self.feature_source.produce_refs(tasks, capture=self.capture)
        except BaseException as exc:
            retried = self._fail_requests(requests, exc, retryable=True)
            if retried and self.retry_backoff_s:
                time.sleep(self.retry_backoff_s)
            return
        if len(results) != len(requests):
            self._fail_requests(
                requests,
                RuntimeError(
                    f"produce_refs returned {len(results)} results for "
                    f"{len(requests)} requests"
                ),
                retryable=False,
            )
            return

        retried = False
        for request, result in zip(requests, results):
            if not isinstance(result, SampleRef):
                result_task_id = getattr(
                    result, "task_id", request.key.source_sample_id
                )
                reason = getattr(result, "reason", repr(result))
                aligned = result_task_id == request.key.source_sample_id
                retried |= self._fail_requests(
                    (request,),
                    (
                        reason
                        if aligned
                        else f"misaligned capture result: {result_task_id!r}"
                    ),
                    retryable=aligned and bool(getattr(result, "retryable", False)),
                )
                continue
            result = dataclasses.replace(
                result,
                metadata={
                    **result.metadata,
                    "window_generation": request.generation,
                },
            )
            try:
                self.registry.mark_committing(request, result)
                self.registry.complete_capture(request, result)
                self.captured_refs += 1
            except BaseException as exc:
                try:
                    self.owner_store.reclaim(
                        result, reason="windowed-registry-commit-failed"
                    )
                except BaseException as cleanup_error:
                    exc.add_note(
                        f"failed to reclaim rejected capture: {cleanup_error!r}"
                    )
                retried |= self._fail_requests((request,), exc, retryable=False)
        if retried and self.retry_backoff_s:
            time.sleep(self.retry_backoff_s)

    @staticmethod
    def _all_consumers_terminal(snapshot: Mapping[str, Any]) -> bool:
        consumers = snapshot["consumers"]
        return set(consumers) == set(snapshot["expected_consumers"]) and all(
            value["state"] in ("completed", "failed") for value in consumers.values()
        )

    def _reclaim(self, *, pressure: bool = False) -> int:
        reclaimed = self.registry.reclaim(
            self.owner_store,
            limit=self.reclaim_batch_size,
            pressure=pressure,
            reason="window-pressure" if pressure else "window-expired",
        )
        gc = getattr(self.owner_store, "gc", None)
        if callable(gc):
            gc()
        return reclaimed

    def drive(
        self,
        *,
        should_stop: Optional[Callable[[], bool]] = None,
        max_rounds: int = 10_000_000,
    ) -> int:
        """Serve capture demand until every configured consumer is terminal."""
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if not self.registry.wait_for_consumers(self.consumer_registration_timeout_s):
            raise TimeoutError("windowed consumers did not register before deadline")

        for _ in range(max_rounds):
            if should_stop is not None and should_stop():
                raise RuntimeError("windowed capture service was stopped")
            self.registry.expire_consumers(self.consumer_heartbeat_timeout_s)
            progressed = bool(self._reclaim())
            snapshot = self.registry.snapshot()
            if self._all_consumers_terminal(snapshot):
                while self._reclaim():
                    pass
                self.registry.finalize_run()
                return self.captured_refs

            if snapshot["queued"] and self.batch_wait_s:
                time.sleep(self.batch_wait_s)
            requests = self.registry.claim_batch(self.capture_batch_size)
            if requests:
                self._capture_requests(requests)
                progressed = True
            elif snapshot["queued"]:
                progressed = bool(self._reclaim(pressure=True)) or progressed
            if not progressed:
                time.sleep(self.poll_s)
        raise RuntimeError(f"windowed capture exceeded max_rounds={max_rounds}")

    def snapshot(self) -> dict[str, Any]:
        return {
            "registry": self.registry.snapshot(),
            "capture_calls": self.capture_calls,
            "captured_refs": self.captured_refs,
            "capture_failures": self.capture_failures,
        }


__all__ = [
    "WindowedCaptureService",
    "WindowedConsumerControl",
    "start_windowed_consumer_control",
]
