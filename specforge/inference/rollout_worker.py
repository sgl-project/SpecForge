# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""RolloutWorker: PromptTask -> SampleRef commit.

The worker is deliberately small and algorithm-agnostic: it leases prompt tasks,
asks a server-side capture source for the ``SampleRef``s of the samples that
server already wrote (``produce_refs``), and commits that metadata to the
controller. It never hands a tensor to the controller. Strategy-specific
capture requirements live in ``CaptureConfig`` + the feature schema, not here.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Protocol

from specforge.inference.capture import CaptureConfig
from specforge.runtime.contracts import PromptTask, SampleRef

# health states: a worker REPORTS health; the controller decides scheduling.
HEALTH_STATES = ("starting", "ready", "paused", "draining", "unhealthy", "stopped")


class RefSource(Protocol):
    def produce_refs(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Any]: ...


class RolloutWorker:
    def __init__(
        self,
        controller,
        feature_source: RefSource,
        capture: CaptureConfig,
        *,
        worker_id: Optional[str] = None,
        strategy: str = "eagle3",
        draft_weight_version: Optional[str] = None,
    ) -> None:
        self.controller = controller
        self.feature_source = feature_source
        self.capture = capture
        self.strategy = strategy
        self.draft_weight_version = draft_weight_version
        self._health_lock = threading.Lock()
        self._state = "starting"
        self._inflight = 0
        self._recent_failures: List[str] = []
        self._last_commit_count = 0
        self.worker_id = controller.register_rollout_worker(
            {"worker_id": worker_id, "strategy": strategy, "role": "rollout"}
        )

    def start(self) -> None:
        with self._health_lock:
            self._state = "ready"

    def stop(self, reason: str = "stopped") -> None:
        # graceful: finish in-flight, then mark stopped (drain happens in run_once)
        with self._health_lock:
            self._state = "stopped"

    def _begin_batch(self, count: int) -> None:
        with self._health_lock:
            self._inflight += count
            if self._state not in ("stopped", "draining"):
                self._state = "ready"

    def _finish_batch(self, count: int) -> None:
        with self._health_lock:
            self._inflight = max(0, self._inflight - count)

    def _mark_unhealthy(self, reason: Optional[str] = None) -> None:
        with self._health_lock:
            if self._state not in ("stopped", "draining"):
                self._state = "unhealthy"
            if reason is not None:
                self._recent_failures.append(reason)

    def _record_failure(self, reason: str) -> None:
        with self._health_lock:
            self._recent_failures.append(reason)

    def _record_commit(self, count: int) -> None:
        with self._health_lock:
            self._last_commit_count += count

    def run_once(self, max_tasks: int) -> List[SampleRef]:
        with self._health_lock:
            if self._state in ("stopped", "draining"):
                return []
        tasks = self.controller.lease_prompt_tasks(self.worker_id, max_tasks)
        if not tasks:
            return []
        self._begin_batch(len(tasks))
        try:
            results = self.feature_source.produce_refs(tasks, capture=self.capture)
        except Exception as exc:
            self._mark_unhealthy(f"produce_refs: {exc}")
            self.controller.fail_prompt_tasks(
                self.worker_id,
                [t.task_id for t in tasks],
                reason=f"produce_refs:{exc}",
                retryable=True,
            )
            self._finish_batch(len(tasks))
            raise
        if len(results) != len(tasks):
            reason = (
                f"produce_refs returned {len(results)} records for {len(tasks)} tasks"
            )
            self._mark_unhealthy(reason)
            self.controller.fail_prompt_tasks(
                self.worker_id,
                [t.task_id for t in tasks],
                reason=reason,
                retryable=False,
            )
            self._finish_batch(len(tasks))
            raise ValueError(reason)

        refs: List[SampleRef] = []
        for result in results:
            if isinstance(result, SampleRef):
                refs.append(result)
                continue

            task_id = getattr(result, "task_id", None)
            reason = getattr(
                result,
                "reason",
                f"produce_refs returned {type(result).__name__}, not SampleRef",
            )
            retryable = bool(getattr(result, "retryable", True))
            if task_id is None:
                self._mark_unhealthy(reason)
                self.controller.fail_prompt_tasks(
                    self.worker_id,
                    [t.task_id for t in tasks],
                    reason=reason,
                    retryable=False,
                )
                self._finish_batch(len(tasks))
                raise TypeError(reason)
            self._record_failure(str(reason))
            self.controller.fail_prompt_tasks(
                self.worker_id, [task_id], reason=str(reason), retryable=retryable
            )

        if refs:
            self.controller.commit_samples(self.worker_id, refs)
            self._record_commit(len(refs))
        self._finish_batch(len(tasks))
        return refs

    def drain(self) -> None:
        """Stop leasing new work; in-flight is finished by the active run_once."""
        with self._health_lock:
            self._state = "draining"

    # Draft-weight hot update (update_weights -> adapter) is not yet supported.

    def health(self) -> Dict[str, Any]:
        with self._health_lock:
            return {
                "worker_id": self.worker_id,
                "state": self._state,
                "strategy": self.strategy,
                "draft_weight_version": self.draft_weight_version,
                "in_flight": self._inflight,
                "recent_failures": self._recent_failures[-5:],
                "committed": self._last_commit_count,
            }


__all__ = ["RolloutWorker", "RefSource", "HEALTH_STATES"]
