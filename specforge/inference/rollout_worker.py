# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""RolloutWorker: PromptTask -> features/refs -> SampleRef commit.

The worker is deliberately small and algorithm-agnostic: it leases prompt tasks,
asks an external-server ``feature_source`` for per-sample features,
verifies them against the typed ``CaptureConfig`` *before* writing, writes them
to the ``FeatureStore``, and commits the resulting ``SampleRef`` metadata to the
controller. A server-side capture source may instead return already-written
``SampleRef``s via ``produce_refs``. It never hands a tensor to the controller.
Strategy-specific capture requirements live in ``CaptureConfig`` + the feature
schema, not here.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Protocol

from specforge.inference.batch_partition import TargetBatchPartition
from specforge.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture,
)
from specforge.runtime.contracts import PromptTask, SampleRef

# health states: a worker REPORTS health; the controller decides scheduling.
HEALTH_STATES = ("starting", "ready", "paused", "draining", "unhealthy", "stopped")


class FeatureSource(Protocol):
    def generate_features(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Dict[str, Any]]: ...


class RefSource(Protocol):
    def produce_refs(
        self, tasks: List[PromptTask], *, capture: CaptureConfig
    ) -> List[Any]: ...


class RolloutWorker:
    def __init__(
        self,
        controller,
        feature_store,
        feature_source: FeatureSource,
        capture: CaptureConfig,
        *,
        run_id: str,
        worker_id: Optional[str] = None,
        strategy: str = "eagle3",
        target_model_version: str = "unknown",
        tokenizer_version: str = "unknown",
        draft_weight_version: Optional[str] = None,
        batch_partition: Optional[TargetBatchPartition] = None,
        feature_source_returns_local_batch: bool = False,
    ) -> None:
        self.controller = controller
        self.feature_store = feature_store
        self.feature_source = feature_source
        self.capture = capture
        self.run_id = run_id
        self.strategy = strategy
        self.target_model_version = target_model_version
        self.tokenizer_version = tokenizer_version
        self.draft_weight_version = draft_weight_version
        self.batch_partition = batch_partition or TargetBatchPartition()
        self.feature_source_returns_local_batch = bool(
            feature_source_returns_local_batch
        )
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

    def _sample_id(self, task: PromptTask) -> str:
        return f"{self.run_id}:{task.task_id}"

    def run_once(self, max_tasks: int) -> List[SampleRef]:
        with self._health_lock:
            if self._state in ("stopped", "draining"):
                return []
        tasks = self.controller.lease_prompt_tasks(self.worker_id, max_tasks)
        if not tasks:
            return []
        self._begin_batch(len(tasks))
        # Legacy colocated target-TP training used a drop-last target batch.
        # A partial target batch would give ranks different local batch sizes
        # and break the all-rank draft/FSDP step contract.
        if self.batch_partition.size > 1 and len(tasks) != max_tasks:
            self.controller.complete_prompt_tasks(
                self.worker_id, [task.task_id for task in tasks]
            )
            self._finish_batch(len(tasks))
            return []

        local_slice = self.batch_partition.local_slice(len(tasks))
        local_tasks = list(tasks[local_slice])
        peer_tasks = list(tasks[: local_slice.start]) + list(tasks[local_slice.stop :])
        produce_refs = getattr(self.feature_source, "produce_refs", None)
        if callable(produce_refs):
            if self.batch_partition.size > 1:
                raise NotImplementedError(
                    "target-TP batch partitioning is supported for colocated "
                    "feature capture only"
                )
            return self._run_ref_source(tasks, produce_refs)

        try:
            feats_list = self.feature_source.generate_features(
                tasks, capture=self.capture
            )
        except Exception as exc:  # rollout failure before any feature write
            self._mark_unhealthy(f"generate_features: {exc}")
            self.controller.fail_prompt_tasks(
                self.worker_id,
                [t.task_id for t in tasks],
                reason=f"generate_features:{exc}",
                retryable=True,
            )
            self._finish_batch(len(tasks))
            raise
        expected_count = (
            len(local_tasks) if self.feature_source_returns_local_batch else len(tasks)
        )
        if len(feats_list) != expected_count:
            reason = (
                f"generate_features returned {len(feats_list)} feature records "
                f"for {expected_count} expected records"
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

        local_feats = (
            feats_list
            if self.feature_source_returns_local_batch
            else list(feats_list[local_slice])
        )
        if peer_tasks:
            self.controller.complete_prompt_tasks(
                self.worker_id, [task.task_id for task in peer_tasks]
            )

        refs: List[SampleRef] = []
        capture_error: Optional[CaptureMismatchError] = None
        put_error: Optional[Exception] = None
        for task, feats in zip(local_tasks, local_feats):
            sample_id = self._sample_id(task)
            recorded = feats.pop("__aux_layer_ids__", None)
            try:
                verify_capture(
                    feats,
                    self.capture,
                    sample_id=sample_id,
                    recorded_aux_layer_ids=recorded,
                )
            except CaptureMismatchError as exc:
                # Loud failure: do not persist a corrupt sample, but keep this
                # batch's other prompt leases moving so no lease is stranded.
                self._record_failure(str(exc))
                self.controller.fail_prompt_tasks(
                    self.worker_id, [task.task_id], reason=str(exc), retryable=False
                )
                if capture_error is None:
                    capture_error = exc
                continue
            try:
                ref = self.feature_store.put(
                    feats,
                    sample_id=sample_id,
                    metadata=self._put_metadata(task),
                )
            except Exception as exc:  # partial write -> abort, report
                self.feature_store.abort(sample_id, reason=f"put_failed:{exc}")
                self.controller.fail_prompt_tasks(
                    self.worker_id,
                    [task.task_id],
                    reason=str(exc),
                    # A target-TP peer cannot recapture this one prompt alone
                    # without desynchronizing the next collective target batch.
                    retryable=self.batch_partition.size == 1,
                )
                if self.batch_partition.size > 1 and put_error is None:
                    put_error = exc
                continue
            refs.append(ref)

        if refs:
            self.controller.commit_samples(self.worker_id, refs)
            self._record_commit(len(refs))
        self._finish_batch(len(tasks))
        if capture_error is not None:
            self._mark_unhealthy()
            raise capture_error
        if put_error is not None:
            self._mark_unhealthy()
            raise RuntimeError("target-TP local feature write failed") from put_error
        return refs

    def _run_ref_source(self, tasks: List[PromptTask], produce_refs) -> List[SampleRef]:
        try:
            results = produce_refs(tasks, capture=self.capture)
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

    def _put_metadata(self, task: PromptTask) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "source_task_id": task.task_id,
            "strategy": self.strategy,
            "target_repr": self.capture.target_repr,
            "vocab_map_version": self.capture.vocab_map_version,
            "ttt_length": self.capture.extra.get("ttt_length"),
            "target_model_version": self.target_model_version,
            "tokenizer_version": self.tokenizer_version,
            "draft_weight_version": self.draft_weight_version,
            "num_tokens": int(task.metadata.get("num_tokens", 0)),
        }

    def drain(self) -> None:
        """Stop leasing new work; in-flight is finished by the active run_once."""
        with self._health_lock:
            self._state = "draining"

    # Draft-weight hot update (update_weights -> adapter) is not yet supported.
    # draft_weight_version is still recorded as rollout provenance on each sample.

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


__all__ = ["RolloutWorker", "FeatureSource", "RefSource", "HEALTH_STATES"]
