# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MetadataStore: the durability seam for recovery-critical control-plane state.

The controller's *recovery-critical* state — committed sample dedup and the
at-least-once durable ack transaction (``{acked sample_ids, global_step,
optimizer-durable marker}``) — lives behind this interface rather than in inline
dicts. The current implementation ships ``InMemoryMetadataStore``; a SQLite
(dev) or Redis/DB (prod) backend is then a *new subclass*, not a
method-by-method rewrite of the controller. The single durable transaction
(``record_train_ack``) is the unit a restart reconciles release state from.

Dependency-light (stdlib only) so it stays importable without torch.
"""

from __future__ import annotations

import abc
import threading
from typing import Any, Dict, List, Optional, Set

from specforge.runtime.contracts import SampleRef


class MetadataStore(abc.ABC):
    # -- sample commit / dedup (at-least-once: idempotent on sample_id) ----
    @abc.abstractmethod
    def commit_sample(self, ref: SampleRef) -> bool:
        """Record a committed sample. Returns True if new, False if duplicate."""

    @abc.abstractmethod
    def is_committed(self, sample_id: str) -> bool: ...

    @abc.abstractmethod
    def get_committed(self, sample_id: str) -> Optional[SampleRef]: ...

    @abc.abstractmethod
    def committed_count(self) -> int: ...

    # -- durable ack transaction -------------------------------------------
    @abc.abstractmethod
    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        """Commit {acked sample_ids, global_step, optimizer-durable marker} atomically.

        Release state is *derived* from this on restart — it is the single
        transaction recovery reconciles against; never split it.
        """

    @abc.abstractmethod
    def durable_marker(self) -> Dict[str, Any]:
        """{acked: set[str], global_step: int|None, optimizer_durable: bool}."""

    # NOTE: a weight-version registry (put/latest/count) is not yet implemented;
    # it belongs with the rest of the published-weight lifecycle.


class InMemoryMetadataStore(MetadataStore):
    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._committed: Dict[str, SampleRef] = {}
        self._acked: Set[str] = set()
        self._global_step: Optional[int] = None
        self._optimizer_durable: bool = False

    def commit_sample(self, ref: SampleRef) -> bool:
        with self._lock:
            if ref.sample_id in self._committed:
                return False
            self._committed[ref.sample_id] = ref
            return True

    def is_committed(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._committed

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        with self._lock:
            return self._committed.get(sample_id)

    def committed_count(self) -> int:
        with self._lock:
            return len(self._committed)

    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        # one atomic update of {acked ids, global_step, optimizer marker}
        with self._lock:
            self._acked.update(sample_ids)
            if global_step is not None:
                self._global_step = global_step
            self._optimizer_durable = optimizer_durable

    def durable_marker(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "acked": set(self._acked),
                "global_step": self._global_step,
                "optimizer_durable": self._optimizer_durable,
            }


__all__ = ["MetadataStore", "InMemoryMetadataStore"]
