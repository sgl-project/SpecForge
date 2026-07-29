# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MetadataStore: the durable ledger for one disaggregated training attempt.

Committed-sample dedup and trainer acknowledgements live behind this interface
rather than in controller-local dictionaries. Online attempts use the SQLite
implementation as their single cross-process ledger; colocated runs use the
lightweight in-memory or no-op implementations.

Dependency-light (stdlib only) so it stays importable without torch.
"""

from __future__ import annotations

import abc
import json
import sqlite3
import threading
from typing import Any, Dict, List, Optional, Sequence, Set

from specforge.runtime.contracts import SampleRef
from specforge.runtime.data_plane.ref_serialization import ref_from_dict, ref_to_dict


class MetadataStore(abc.ABC):
    # -- sample commit / dedup (at-least-once: idempotent on sample_id) ----
    @abc.abstractmethod
    def commit_sample(self, ref: SampleRef) -> bool:
        """Record a committed sample. Returns True if new, False if duplicate."""

    def commit_samples(self, refs: Sequence[SampleRef]) -> List[bool]:
        """Record a batch and return one freshness result per input ref.

        The default preserves compatibility with stores that only implement the
        original single-ref API. Durable stores should override this method so
        the whole batch shares one transaction.
        """

        return [self.commit_sample(ref) for ref in refs]

    @abc.abstractmethod
    def is_committed(self, sample_id: str) -> bool: ...

    @abc.abstractmethod
    def get_committed(self, sample_id: str) -> Optional[SampleRef]: ...

    @abc.abstractmethod
    def committed_count(self) -> int: ...

    @abc.abstractmethod
    def all_committed_ids(self) -> List[str]: ...

    # -- durable ack transaction -------------------------------------------
    @abc.abstractmethod
    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        """Commit the ack ids and optimizer-step marker atomically."""

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
        return self.commit_samples([ref])[0]

    def commit_samples(self, refs: Sequence[SampleRef]) -> List[bool]:
        with self._lock:
            fresh = []
            for ref in refs:
                if ref.sample_id in self._committed:
                    fresh.append(False)
                    continue
                self._committed[ref.sample_id] = ref
                fresh.append(True)
            return fresh

    def is_committed(self, sample_id: str) -> bool:
        with self._lock:
            return sample_id in self._committed

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        with self._lock:
            return self._committed.get(sample_id)

    def committed_count(self) -> int:
        with self._lock:
            return len(self._committed)

    def all_committed_ids(self) -> List[str]:
        with self._lock:
            return list(self._committed)

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


class NoOpMetadataStore(MetadataStore):
    """MetadataStore implementation for ``local_colocated`` runs.

    It retains no committed refs or ack marker. ``commit_sample`` always reports
    a fresh ref so the controller still enqueues it. Use a retaining store for
    cross-process runs.
    """

    def commit_sample(self, ref: SampleRef) -> bool:
        return True

    def commit_samples(self, refs: Sequence[SampleRef]) -> List[bool]:
        return [True] * len(refs)

    def is_committed(self, sample_id: str) -> bool:
        return False

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        return None

    def committed_count(self) -> int:
        return 0

    def all_committed_ids(self) -> List[str]:
        return []

    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        pass

    def durable_marker(self) -> Dict[str, Any]:
        return {"acked": set(), "global_step": None, "optimizer_durable": False}


class SQLiteMetadataStore(MetadataStore):
    """SQLite ledger shared by the ranks of one online consumer attempt."""

    def __init__(self, path: str) -> None:
        # check_same_thread=False + a guard lock: one connection, serialized.
        self.path = path
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")  # durable + concurrent reads
        self._conn.execute(
            "PRAGMA synchronous=FULL"
        )  # ack survives power loss, not just process crash
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS committed "
                "(sample_id TEXT PRIMARY KEY, ref_json TEXT NOT NULL)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS acked (sample_id TEXT PRIMARY KEY)"
            )
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS marker (k TEXT PRIMARY KEY, v TEXT)"
            )
            self._conn.commit()

    def commit_sample(self, ref: SampleRef) -> bool:
        return self.commit_samples([ref])[0]

    def commit_samples(self, refs: Sequence[SampleRef]) -> List[bool]:
        if not refs:
            return []
        serialized = [(ref.sample_id, json.dumps(ref_to_dict(ref))) for ref in refs]
        with self._lock:
            fresh = []
            try:
                # One FULL-synchronous transaction amortizes the durability
                # barrier across every ref returned by a source-channel poll.
                self._conn.execute("BEGIN IMMEDIATE")
                for record in serialized:
                    cur = self._conn.execute(
                        "INSERT OR IGNORE INTO committed "
                        "(sample_id, ref_json) VALUES (?, ?)",
                        record,
                    )
                    fresh.append(cur.rowcount == 1)
                self._conn.commit()
            except BaseException:
                self._conn.rollback()
                raise
            return fresh

    def is_committed(self, sample_id: str) -> bool:
        with self._lock:
            row = self._conn.execute(
                "SELECT 1 FROM committed WHERE sample_id = ?", (sample_id,)
            ).fetchone()
            return row is not None

    def get_committed(self, sample_id: str) -> Optional[SampleRef]:
        with self._lock:
            row = self._conn.execute(
                "SELECT ref_json FROM committed WHERE sample_id = ?", (sample_id,)
            ).fetchone()
        return ref_from_dict(json.loads(row[0])) if row else None

    def committed_count(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM committed").fetchone()[0]

    def all_committed_ids(self) -> List[str]:
        with self._lock:
            return [
                row[0]
                for row in self._conn.execute(
                    "SELECT sample_id FROM committed ORDER BY rowid"
                ).fetchall()
            ]

    def record_train_ack(
        self,
        sample_ids: List[str],
        *,
        global_step: Optional[int],
        optimizer_durable: bool,
    ) -> None:
        # ONE transaction commits ack ids and the optimizer marker together.
        with self._lock:
            self._conn.executemany(
                "INSERT OR IGNORE INTO acked (sample_id) VALUES (?)",
                [(s,) for s in sample_ids],
            )
            if global_step is not None:
                self._conn.execute(
                    "INSERT OR REPLACE INTO marker (k, v) VALUES ('global_step', ?)",
                    (json.dumps(global_step),),
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO marker (k, v) VALUES ('optimizer_durable', ?)",
                (json.dumps(bool(optimizer_durable)),),
            )
            self._conn.commit()

    def durable_marker(self) -> Dict[str, Any]:
        with self._lock:
            acked = {
                r[0]
                for r in self._conn.execute("SELECT sample_id FROM acked").fetchall()
            }
            rows = dict(self._conn.execute("SELECT k, v FROM marker").fetchall())
        gs = json.loads(rows["global_step"]) if "global_step" in rows else None
        od = (
            json.loads(rows["optimizer_durable"])
            if "optimizer_durable" in rows
            else False
        )
        return {"acked": acked, "global_step": gs, "optimizer_durable": od}

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = [
    "MetadataStore",
    "InMemoryMetadataStore",
    "NoOpMetadataStore",
    "SQLiteMetadataStore",
]
