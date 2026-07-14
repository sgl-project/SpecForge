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
"""

from __future__ import annotations

from typing import Callable, List, Optional

from specforge.runtime.control_plane.controller import DataFlowController


def gather_id_union(ids: List[str]) -> List[str]:
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
    world = dist.get_world_size()
    if world == 1:
        return list(ids)
    gathered: List[Optional[List[str]]] = [None] * world
    dist.all_gather_object(gathered, list(ids))
    out: List[str] = []
    seen = set()
    for rank_ids in gathered:
        for sid in rank_ids or ():
            if sid not in seen:
                seen.add(sid)
                out.append(sid)
    return out


def _broadcast_authority_error(error: Optional[str]) -> Optional[str]:
    """Return rank 0's post-commit result on every rank."""
    try:
        import torch.distributed as dist
    except ModuleNotFoundError:
        return error
    if not (dist.is_available() and dist.is_initialized()):
        return error
    payload = [error]
    dist.broadcast_object_list(payload, src=0)
    return payload[0]


class DPAckController(DataFlowController):
    """A :class:`DataFlowController` whose ``ack_train_refs`` is a DP collective.

    * ``is_authority=True`` (DP rank 0): holds the run's ONE durable store and
      records the gathered union and removes its retained features only after
      that durable transaction succeeds.
    * ``is_authority=False`` (other ranks): participates in the gather (the
      collective needs every rank) and records nothing; give it a throwaway
      in-memory store.

    Every rank must call ``ack_train_refs`` at every optimizer boundary — the
    trainer's ``ack_fn`` already does exactly that. Once rank 0 has committed
    and removed the gathered union, every rank is released from the result
    broadcast and may advance its inbox acknowledgement; the distributor then
    mirrors those optimizer-boundary counts onto the source counter.
    """

    def __init__(
        self,
        run_id: str,
        *,
        is_authority: bool = True,
        gather: Callable[[List[str]], List[str]] = gather_id_union,
        sync_error: Callable[
            [Optional[str]], Optional[str]
        ] = _broadcast_authority_error,
        feature_store=None,
        **kwargs,
    ) -> None:
        super().__init__(run_id, **kwargs)
        self.is_authority = is_authority
        self._gather = gather
        self._sync_error = sync_error
        self.feature_store = feature_store

    def ack_train_refs(
        self,
        trainer_id: str,
        sample_ids: List[str],
        *,
        global_step: Optional[int] = None,
        optimizer_durable: bool = False,
    ) -> None:
        union = self._gather(list(sample_ids))
        error = None
        if self.is_authority:
            try:
                # The SQLite ack ids + optimizer marker commit first. Only
                # after that durable fact exists may consume-once features be
                # physically removed.
                super().ack_train_refs(
                    trainer_id,
                    union,
                    global_step=global_step,
                    optimizer_durable=optimizer_durable,
                )
                if optimizer_durable and self.feature_store is not None:
                    for sample_id in union:
                        self.feature_store.abort(
                            sample_id, reason="optimizer-boundary-durable-ack"
                        )
            except BaseException as exc:
                error = f"{type(exc).__name__}: {exc}"
        error = self._sync_error(error)
        if error is not None:
            raise RuntimeError(f"durable DP acknowledgement failed: {error}")


__all__ = ["DPAckController", "gather_id_union"]
