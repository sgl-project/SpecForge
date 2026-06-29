# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DataFlowController: the metadata-only scheduler / debug boundary.

The controller owns prompt and sample lifecycle, leases, and worker registration.
It NEVER touches tensors — every public method that accepts
a record runs ``assert_no_tensors`` (this is what ``test_controller_carries_no_tensor``
exercises). All large tensors travel through the data plane (FeatureStore);
online ``commit_samples`` and offline ``enqueue_offline_refs`` converge onto the
same ``SampleRefQueue`` so the trainer path has no online/offline branch.

Recovery-critical state (committed-sample dedup and the durable ack transaction)
lives behind a ``MetadataStore`` so a durable backend (SQLite → Redis/DB) is a
swap, not a rewrite. The current backend is in-process; the public surface already
matches the durable controller shape so a later Ray/service deployment is mechanical.
"""

from __future__ import annotations

import dataclasses
import threading
import uuid
from collections import OrderedDict, deque
from typing import Any, Deque, Dict, List, Optional

from specforge.runtime.contracts import PromptTask, SampleRef, assert_no_tensors
from specforge.runtime.control_plane.backpressure import BackpressureController
from specforge.runtime.control_plane.metadata_store import (
    InMemoryMetadataStore,
    MetadataStore,
)
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue


class TrainLease:
    """A train-side lease client that routes lease/ack/fail through the controller.

    Exposes the loader-facing ``get/ack/fail`` shape so ``FeatureDataLoader`` can
    consume it interchangeably with a raw queue — but every op goes through the
    controller, so the durable ack transaction is recorded and a disaggregated
    (cross-node) trainer is a drop-in (it never holds a raw in-process queue).
    """

    def __init__(self, controller: "DataFlowController", trainer_id: str) -> None:
        self._controller = controller
        self._trainer_id = trainer_id

    def get(self, max_refs: int, timeout_s: Optional[float] = None) -> List[SampleRef]:
        return self._controller.lease_train_refs(self._trainer_id, max_refs, timeout_s)

    def ack(
        self,
        refs: List[SampleRef],
        *,
        global_step: Optional[int] = None,
        optimizer_durable: bool = False,
    ) -> None:
        self._controller.ack_train_refs(
            self._trainer_id,
            [r.sample_id for r in refs],
            global_step=global_step,
            optimizer_durable=optimizer_durable,
        )

    def fail(self, refs: List[SampleRef], reason: str, retryable: bool) -> None:
        self._controller.fail_refs(
            self._trainer_id, [r.sample_id for r in refs], reason, retryable
        )


class DataFlowController:
    def __init__(
        self,
        run_id: str,
        *,
        sample_queue: Optional[SampleRefQueue] = None,
        metadata_store: Optional[MetadataStore] = None,
        backpressure: Optional[BackpressureController] = None,
    ) -> None:
        self.run_id = run_id
        self.sample_queue = sample_queue or SampleRefQueue()
        self.store = metadata_store or InMemoryMetadataStore()
        # Optional backpressure policy. None = never pause / no caps (M1-M4
        # behavior). The controller owns the pause decision; the policy only
        # reads capacity (FeatureStore.health) — no tensors cross this seam.
        self.backpressure = backpressure
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

    def train_lease(self, trainer_id: Optional[str] = None) -> TrainLease:
        """Return a train-side lease client (lease + ack route through here)."""
        if trainer_id is None:
            trainer_id = self.register_trainer({"role": "trainer"})
        return TrainLease(self, trainer_id)

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
        # Backpressure: pause leasing entirely above the high watermark (a paused
        # rollout that asks for work is "rollout starvation" — logged distinctly
        # from trainer starvation). Consult the policy *outside* the controller
        # lock; it takes the store lock internally.
        if self.backpressure is not None and self.backpressure.should_pause_prompts():
            self.backpressure.note_rollout_starved()
            return []
        out: List[PromptTask] = []
        with self._lock:
            grant = max_tasks
            if self.backpressure is not None:
                inflight = sum(
                    1 for w in self._prompt_leased.values() if w == worker_id
                )
                grant = self.backpressure.cap_prompt_grant(inflight, max_tasks)
            for _ in range(grant):
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

        This is intentionally separate from ``fail_refs``. Prompt failures happen
        before a ``SampleRef`` exists, so routing them through the train-sample
        queue would be a no-op and would leave prompt leases stuck.
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
                if retryable:
                    self._prompts[task_id] = dataclasses.replace(
                        task, attempt=task.attempt + 1
                    )
                    if task_id not in self._prompt_pending:
                        self._prompt_pending.append(task_id)
                else:
                    self._prompt_failed[task_id] = reason

    def commit_samples(self, worker_id: str, refs: List[SampleRef]) -> None:
        fresh: List[SampleRef] = []
        for ref in refs:
            assert_no_tensors(ref)  # online no-tensor guard
            if not self.store.commit_sample(ref):
                continue  # idempotent on sample_id (at-least-once delivery)
            if ref.source_task_id is not None:
                with self._lock:
                    self._prompt_leased.pop(ref.source_task_id, None)
            fresh.append(ref)
        if fresh:
            self.sample_queue.put(fresh)

    # -- offline ingest ----------------------------------------------------
    def enqueue_offline_refs(self, refs: List[SampleRef]) -> None:
        fresh: List[SampleRef] = []
        for ref in refs:
            assert_no_tensors(ref)
            if self.store.commit_sample(ref):
                fresh.append(ref)
        if fresh:
            self.sample_queue.put(fresh)

    # -- train-side lease/ack ---------------------------------------------
    def lease_train_refs(
        self, trainer_id: str, max_refs: int, timeout_s: Optional[float] = None
    ) -> List[SampleRef]:
        if self.backpressure is not None:
            max_refs = self.backpressure.cap_train_lease(max_refs)
        refs = self.sample_queue.get(max_refs, timeout_s=timeout_s)
        # Empty lease == trainer outran rollout (or rollout is paused). Logged as
        # trainer starvation — the opposite condition from rollout starvation.
        if not refs and self.backpressure is not None:
            self.backpressure.note_trainer_starved()
        return refs

    def ack_train_refs(
        self,
        trainer_id: str,
        sample_ids: List[str],
        *,
        global_step: Optional[int] = None,
        optimizer_durable: bool = False,
    ) -> None:
        """Ack consumed refs at the trainer's optimizer-step boundary.

        Records the durable ``{acked sample_ids, global_step, optimizer-durable
        marker}`` transaction *then* releases the queue lease, so restart can
        derive release state from the single committed marker.
        """
        self.store.record_train_ack(
            sample_ids, global_step=global_step, optimizer_durable=optimizer_durable
        )
        refs = [
            r
            for r in (self.store.get_committed(s) for s in sample_ids)
            if r is not None
        ]
        self.sample_queue.ack(refs)

    def fail_refs(
        self, owner_id: str, sample_ids: List[str], reason: str, retryable: bool
    ) -> None:
        refs = [
            r
            for r in (self.store.get_committed(s) for s in sample_ids)
            if r is not None
        ]
        self.sample_queue.fail(refs, reason, retryable)

    # -- restart reconciliation (B4) --------------------------------------
    def reconcile_on_restart(
        self, feature_store: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Rebuild queue + release state from the single durable marker.

        Call once on a fresh controller backed by a *durable* metadata store
        (e.g. ``SQLiteMetadataStore`` reopened on the same DB file) after a crash.
        In-process lease state did not survive the crash, so release state is
        *derived* solely from the durable ``{acked, global_step}`` marker — never
        a separate fact that could disagree with the optimizer step.

        Per ``failure_recovery.md`` (B4), each committed sample lands in one row:

        | durable state                | action                              |
        |------------------------------|-------------------------------------|
        | acked AND step committed     | released (idempotent free)          |
        | committed, not durably acked | replay/requeue (at-least-once)      |
        | never committed (in-flight)  | not here; rollout retries on id     |

        Re-training a leased-but-unacked sample is allowed (idempotent effects);
        the guarantee is **no data loss** (every committed-unacked sample is
        requeued) and **no duplicate train** (an acked+stepped sample is never
        requeued — the exact crash window "after ack, before release" is safe
        because the durable ack already records it as trained).
        """
        marker = self.store.durable_marker()
        acked = marker["acked"]
        # Release eligibility is gated on the optimizer-durable marker, NOT on a
        # global_step scalar that the ack API leaves None by default — otherwise a
        # durable ack with global_step omitted would force every acked sample to
        # replay (duplicate train). (Coarse: optimizer_durable is a run-level
        # flag; a per-sample durability column is the future refinement.)
        step_committed = bool(marker["optimizer_durable"])
        requeued: List[str] = []
        released: List[str] = []
        for sid in self.store.all_committed_ids():
            if sid in acked and step_committed:
                released.append(sid)
                if feature_store is not None:
                    # idempotent free; safe if the sample was already freed
                    feature_store.abort(sid, reason="reconciled-released")
            else:
                ref = self.store.get_committed(sid)
                if ref is not None:
                    self.sample_queue.put([ref])  # idempotent on sample_id
                    requeued.append(sid)
        return {
            "requeued": requeued,
            "released": released,
            "global_step": marker["global_step"],
            "optimizer_durable": marker["optimizer_durable"],
        }

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
            # train_backlog = samples produced but not yet acked-as-trained: the
            # count-based "trainer behind rollout" signal for backpressure.
            "train_backlog": committed - acked,
            "queue_depth": self.sample_queue.depth(),
            "queue_in_flight": self.sample_queue.in_flight(),
            "rollout_workers": workers,
            "trainers": trainers,
            "durable_global_step": marker["global_step"],
            "durable_acked": acked,
        }
        if self.backpressure is not None:
            status["backpressure"] = self.backpressure.snapshot()
        return status


__all__ = ["DataFlowController", "TrainLease"]
