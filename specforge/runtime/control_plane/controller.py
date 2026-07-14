# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DataFlowController: the metadata-only scheduler / debug boundary.

The controller owns prompt and sample lifecycle, acknowledgements, and worker
registration. It NEVER touches tensors — every public method that accepts
a record runs ``assert_no_tensors`` (this is what ``test_controller_carries_no_tensor``
exercises). All large tensors travel through the data plane (FeatureStore);
Online ``commit_samples`` feeds the private local/distributor staging queue.
Offline refs remain a fixed list owned by ``FeatureDataLoader`` and never enter
this queue or ledger.

Committed-sample dedup and acknowledgements live behind a ``MetadataStore``;
online consumers use one SQLite ledger shared by their ranks.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional

from specforge.runtime.contracts import PromptTask, SampleRef, assert_no_tensors
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
)
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


class DataFlowController:
    def __init__(
        self,
        run_id: str,
        *,
        metadata_store: Optional[MetadataStore] = None,
        max_prompt_attempts: Optional[int] = None,
        enable_sample_queue: bool = True,
    ) -> None:
        self.run_id = run_id
        self.sample_queue = SampleRefQueue() if enable_sample_queue else None
        self.store = metadata_store or InMemoryMetadataStore()
        # Retryable-failure bound: a task failed this many attempts goes
        # terminal instead of requeueing (None = retry forever). Bounds a
        # poisoned prompt that would otherwise pin the pool non-empty forever.
        self.max_prompt_attempts = max_prompt_attempts
        self._prompts: "OrderedDict[str, PromptTask]" = OrderedDict()
        self._prompt_pending: Deque[str] = deque()
        self._prompt_leased: Dict[str, str] = {}  # task_id -> worker_id
        self._prompt_failed: Dict[str, str] = {}  # task_id -> terminal reason
        self._workers: Dict[str, Dict[str, Any]] = {}
        self._trainers: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()

    # -- registration ------------------------------------------------------
    def register_rollout_worker(self, info: Dict[str, Any]) -> str:
        assert_no_tensors(info)
        worker_id = info.get("worker_id") or f"rollout-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._workers[worker_id] = dict(info)
        return worker_id

    def register_trainer(self, info: Dict[str, Any]) -> str:
        assert_no_tensors(info)
        trainer_id = info.get("trainer_id") or f"trainer-{uuid.uuid4().hex[:8]}"
        with self._lock:
            self._trainers[trainer_id] = dict(info)
        return trainer_id

    # -- prompt lifecycle (online) ----------------------------------------
    def ingest_prompts(self, prompts: List[Dict[str, Any]]) -> List[str]:
        task_ids: List[str] = []
        with self._lock:
            for p in prompts:
                assert_no_tensors(p)
                task_id = p.get("task_id") or f"task-{uuid.uuid4().hex[:12]}"
                task = PromptTask(
                    task_id=task_id,
                    run_id=self.run_id,
                    source_id=str(p.get("source_id", "prompt_source")),
                    payload=p.get("payload", p),
                    max_length=int(p.get("max_length", 2048)),
                    chat_template=p.get("chat_template"),
                    loss_mask_policy=p.get("loss_mask_policy", {}),
                    target_model_version=str(p.get("target_model_version", "unknown")),
                    draft_weight_version=p.get("draft_weight_version"),
                    metadata=p.get("metadata", {}),
                )
                assert_no_tensors(task)
                self._prompts[task_id] = task
                self._prompt_pending.append(task_id)
                task_ids.append(task_id)
        return task_ids

    def lease_prompt_tasks(self, worker_id: str, max_tasks: int) -> List[PromptTask]:
        out: List[PromptTask] = []
        with self._lock:
            for _ in range(max_tasks):
                if not self._prompt_pending:
                    break
                task_id = self._prompt_pending.popleft()
                self._prompt_leased[task_id] = worker_id
                out.append(self._prompts[task_id])
        return out

    def fail_prompt_tasks(
        self, worker_id: str, task_ids: List[str], reason: str, retryable: bool
    ) -> None:
        """Release failed prompt leases and optionally requeue them.

        Prompt failures happen before a ``SampleRef`` exists, so they release or
        retire the prompt lease directly rather than touching sample staging.
        """
        with self._lock:
            for task_id in task_ids:
                owner = self._prompt_leased.get(task_id)
                if owner is not None and owner != worker_id:
                    continue
                self._prompt_leased.pop(task_id, None)
                task = self._prompts.get(task_id)
                if task is None:
                    continue
                attempts_left = (
                    self.max_prompt_attempts is None
                    or task.attempt + 1 < self.max_prompt_attempts
                )
                if retryable and attempts_left:
                    self._prompts[task_id] = dataclasses.replace(
                        task, attempt=task.attempt + 1
                    )
                    if task_id not in self._prompt_pending:
                        self._prompt_pending.append(task_id)
                elif retryable:
                    self._prompt_failed[task_id] = (
                        f"{reason} (attempts exhausted: {task.attempt + 1})"
                    )
                    self._prompts.pop(task_id, None)
                else:
                    self._prompt_failed[task_id] = reason
                    self._prompts.pop(task_id, None)

    def commit_samples(self, worker_id: str, refs: List[SampleRef]) -> None:
        fresh: List[SampleRef] = []
        for ref in refs:
            assert_no_tensors(ref)  # online no-tensor guard
            if not self.store.commit_sample(ref):
                continue  # idempotent on sample_id (at-least-once delivery)
            if ref.source_task_id is not None:
                with self._lock:
                    self._prompt_leased.pop(ref.source_task_id, None)
                    self._prompts.pop(ref.source_task_id, None)
            fresh.append(ref)
        if fresh and self.sample_queue is not None:
            self.sample_queue.put(fresh)

    # -- offline ingest ----------------------------------------------------
    # -- train-side durable ack -------------------------------------------
    def ack_train_refs(
        self,
        trainer_id: str,
        sample_ids: List[str],
        *,
        global_step: Optional[int] = None,
        optimizer_durable: bool = False,
    ) -> None:
        """Ack consumed refs at the trainer's optimizer-step boundary.

        Records the durable ack transaction, then releases the queue lease.
        """
        self.store.record_train_ack(
            sample_ids, global_step=global_step, optimizer_durable=optimizer_durable
        )
        refs = [
            r
            for r in (self.store.get_committed(s) for s in sample_ids)
            if r is not None
        ]
        if self.sample_queue is not None:
            self.sample_queue.ack(refs)

    # NOTE: weight publishing (publish_weight_version / latest_weight_version) is
    # not yet implemented; it lands with the rest of the weight-version lifecycle.

    def status(self) -> Dict[str, Any]:
        with self._lock:
            prompts = len(self._prompts)
            pending = len(self._prompt_pending)
            leased = len(self._prompt_leased)
            failed = len(self._prompt_failed)
            workers = len(self._workers)
            trainers = len(self._trainers)
        marker = self.store.durable_marker()
        committed = self.store.committed_count()
        acked = len(marker["acked"])
        status = {
            "run_id": self.run_id,
            "prompts": prompts,
            "prompts_pending": pending,
            "prompts_leased": leased,
            "prompts_failed": failed,
            "samples_committed": committed,
            # Samples produced but not yet durably acked by the trainer.
            "train_backlog": committed - acked,
            "queue_depth": self.sample_queue.depth() if self.sample_queue else 0,
            "queue_in_flight": (
                self.sample_queue.in_flight() if self.sample_queue else 0
            ),
            "rollout_workers": workers,
            "trainers": trainers,
            "durable_global_step": marker["global_step"],
            "durable_acked": acked,
        }
        return status


__all__ = ["DataFlowController"]
