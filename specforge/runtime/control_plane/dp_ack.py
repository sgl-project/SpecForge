# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""DP-aware durable ack: gather every rank's sample_ids, record ONCE.

With data-parallel consumers each rank trains a disjoint shard, but the durable
``{acked, global_step, optimizer_durable}`` marker must have a single writer —
otherwise N ranks interleave partial ack sets into one ledger.
:class:`DPAckController` keeps the write single-authority without giving up the
existing trainer seam: ``TrainerController.fit`` already calls ``ack_fn`` on
EVERY rank at EVERY optimizer boundary (in lockstep), so ``ack_train_refs``
becomes a collective — all ranks contribute their shard's ids
(``all_gather_object``), only the authority (DP rank 0) records the union and
drives the ``ack_sink`` (the producer-backpressure counter).

Correctness rests on the lockstep invariant the :class:`RefDistributor`
enforces (equal per-rank ref counts): a rank that skipped a boundary would hang
the gather.

``process_group`` carries every ack object collective. With asynchronous acks
the online consumer passes a dedicated Gloo group
(:func:`new_durable_ack_process_group`); synchronous acks keep the default
group. Pickled object collectives over NCCL stage through the device and
``.item()``/``.cpu()`` drain the training stream at every boundary, while Gloo
stays on the host and lets the ack run on a background thread without
interleaving with the training thread's NCCL collectives.
"""

from __future__ import annotations

import functools
import logging
from collections import OrderedDict
from typing import Any, Callable, List, Optional

from specforge.runtime.control_plane.controller import DataFlowController

logger = logging.getLogger(__name__)


def new_durable_ack_process_group() -> Optional[Any]:
    """Create the host-only Gloo group for durable-ack object collectives.

    ``new_group`` is collective over the default group: every rank must call
    this at the same point of its setup. Returns ``None`` when there is nothing
    to isolate (no process group, a single rank) or Gloo is unavailable; the
    caller then keeps acks on the default group and on the training thread.
    """
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return None
    if not (dist.is_available() and dist.is_initialized()):
        return None
    if dist.get_world_size() == 1:
        return None
    if not dist.is_gloo_available():
        logger.warning(
            "torch.distributed has no Gloo backend; durable acks stay on the "
            "default process group and run synchronously"
        )
        return None
    return dist.new_group(backend="gloo")


def gather_id_union(ids: List[str], group: Optional[Any] = None) -> List[str]:
    """All-gather each rank's sample_ids; return the rank-ordered, deduped union.

    Identity when torch.distributed is absent/uninitialized/world=1. Dedup keeps
    first occurrence so SP-replicated shards (same ids on sp peers) collapse.
    """
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return list(ids)

    if not (dist.is_available() and dist.is_initialized()):
        return list(ids)
    world = dist.get_world_size(group)
    if world == 1:
        return list(ids)
    gathered: List[Optional[List[str]]] = [None] * world
    dist.all_gather_object(gathered, list(ids), group=group)
    out: List[str] = []
    seen = set()
    for rank_ids in gathered:
        for sid in rank_ids or ():
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _broadcast_authority_error(
    error: Optional[str], group: Optional[Any] = None
) -> Optional[str]:
    """Return rank 0's post-commit result on every rank."""
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return error
    if not (dist.is_available() and dist.is_initialized()):
        return error
    if dist.get_world_size(group) == 1:
        return error
    payload = [error]
    # The ack group spans every rank, so global rank 0 is the authority.
    dist.broadcast_object_list(payload, src=0, group=group)
    return payload[0]


def _gather_cleanup_errors(
    error: Optional[str], group: Optional[Any] = None
) -> Optional[str]:
    """Return every rank's local cleanup failure on every rank.

    Cleanup is intentionally a second collective after the authority's durable
    commit broadcast.  A non-authority rank owns a distinct Mooncake client and
    therefore can fail while rank 0 succeeds; all ranks must observe that
    failure before any caller advances its rank-local inbox counter.
    """
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return error
    if not (dist.is_available() and dist.is_initialized()):
        return error
    world = dist.get_world_size(group)
    if world == 1:
        return error
    gathered: List[Optional[str]] = [None] * world
    dist.all_gather_object(gathered, error, group=group)
    failures = [
        f"rank {rank}: {rank_error}"
        for rank, rank_error in enumerate(gathered)
        if rank_error is not None
    ]
    return "; ".join(failures) or None


class DPAckController(DataFlowController):
    """A :class:`DataFlowController` whose ``ack_train_refs`` is a DP collective.

    * ``is_authority=True`` (DP rank 0): holds the run's ONE durable store and
      records the gathered union.
    * ``is_authority=False`` (other ranks): participates in the gather (the
      collective needs every rank) and records nothing; give it a throwaway
      in-memory store.

    Every rank owns a separate feature-store client and may have materialized
    only its own DP shard.  Once rank 0 commits the union and broadcasts success,
    each rank deletes only its local ``sample_ids`` through its own store.  A
    second all-rank error collective completes before any caller may advance its
    inbox acknowledgement; the distributor then mirrors those
    optimizer-boundary counts onto the source counter.

    ``process_group`` (default: the default group) carries the gather and both
    error collectives unless explicit ``gather``/``sync_*`` callables are given.
    Every rank must issue ``ack_train_refs`` calls in the same order, from one
    thread at a time.
    """

    _CLEANUP_LAG_BOUNDARIES = 1
    _CLEANUP_ESCALATION_BOUNDARIES = 4

    def __init__(
        self,
        run_id: str,
        *,
        is_authority: bool = True,
        gather: Optional[Callable[[List[str]], List[str]]] = None,
        sync_error: Optional[Callable[[Optional[str]], Optional[str]]] = None,
        sync_cleanup_error: Optional[Callable[[Optional[str]], Optional[str]]] = None,
        feature_store=None,
        process_group: Optional[Any] = None,
        **kwargs,
    ) -> None:
        super().__init__(run_id, **kwargs)
        self.is_authority = is_authority
        self.process_group = process_group
        self._gather = gather or functools.partial(gather_id_union, group=process_group)
        self._sync_error = sync_error or functools.partial(
            _broadcast_authority_error, group=process_group
        )
        self._sync_cleanup_error = sync_cleanup_error or functools.partial(
            _gather_cleanup_errors, group=process_group
        )
        self.feature_store = feature_store
        self._cleanup_boundary = 0
        self._cleanup_pending: OrderedDict[str, int] = OrderedDict()

    def ack_train_refs(
        self,
        trainer_id: str,
        sample_ids: List[str],
        *,
        global_step: Optional[int] = None,
        optimizer_durable: bool = False,
    ) -> None:
        local_ids = list(dict.fromkeys(sample_ids))
        union = self._gather(local_ids)
        commit_error = None
        if self.is_authority:
            try:
                # The SQLite ack ids + optimizer marker are the one durable,
                # authority-owned fact. No rank may physically delete features
                # before this transaction succeeds.
                super().ack_train_refs(
                    trainer_id,
                    union,
                    global_step=global_step,
                    optimizer_durable=optimizer_durable,
                )
            except BaseException as exc:
                commit_error = f"{type(exc).__name__}: {exc}"
        commit_error = self._sync_error(commit_error)
        if commit_error is not None:
            raise RuntimeError(f"durable DP acknowledgement failed: {commit_error}")

        cleanup_error = None
        if optimizer_durable and self.feature_store is not None:
            self._cleanup_boundary += 1
            boundary = self._cleanup_boundary
            failures = []
            for sample_id in local_ids:
                try:
                    self.feature_store.abort(
                        sample_id, reason="optimizer-boundary-durable-ack"
                    )
                except BaseException as exc:
                    failures.append(f"{sample_id}: {type(exc).__name__}: {exc}")
                self._cleanup_pending.setdefault(sample_id, boundary)

            eligible_ids = [
                sample_id
                for sample_id, first_boundary in self._cleanup_pending.items()
                if boundary - first_boundary >= self._CLEANUP_LAG_BOUNDARIES
            ]
            try:
                from specforge.runtime.data_plane.feature_store import (
                    drain_feature_store_sample_removals,
                    retry_feature_store_sample_removals,
                )

                # Give short Mooncake read leases one optimizer window to
                # expire, then make one selective, no-sleep batched attempt.
                # A store-wide retry could delete prefetched refs that still
                # need crash replay.
                report = retry_feature_store_sample_removals(
                    self.feature_store, eligible_ids
                )
                remaining_ids = set(report.get("remaining_ids", ()))
                for sample_id in eligible_ids:
                    if sample_id not in remaining_ids:
                        self._cleanup_pending.pop(sample_id, None)
            except BaseException as exc:
                failures.append(
                    "optimizer-boundary selective retry: "
                    f"{type(exc).__name__}: {exc}"
                )

            overdue_ids = [
                sample_id
                for sample_id, first_boundary in self._cleanup_pending.items()
                if boundary - first_boundary >= self._CLEANUP_ESCALATION_BOUNDARIES
            ]
            if overdue_ids:
                try:
                    # Sustained failure is exceptional.  Use the existing
                    # bounded strong drain only for the old durable ids, never
                    # for this boundary's fresh or merely-prefetched samples.
                    drain_feature_store_sample_removals(self.feature_store, overdue_ids)
                    for sample_id in overdue_ids:
                        self._cleanup_pending.pop(sample_id, None)
                except BaseException as exc:
                    failures.append(
                        "optimizer-boundary selective drain: "
                        f"{type(exc).__name__}: {exc}"
                    )
            if failures:
                cleanup_error = ", ".join(failures)
        cleanup_error = self._sync_cleanup_error(cleanup_error)
        if cleanup_error is not None:
            raise RuntimeError(
                "durable DP acknowledgement committed, but rank-local feature "
                f"cleanup failed: {cleanup_error}"
            )


__all__ = ["DPAckController", "gather_id_union", "new_durable_ack_process_group"]
