# coding=utf-8
"""Durable single-host lifecycle index for Mooncake-owned feature objects."""

from __future__ import annotations

import json
import os
import sqlite3
import threading
import time
from dataclasses import dataclass
from typing import Iterable, Optional


@dataclass(frozen=True)
class LifecycleRecord:
    sample_id: str
    generation: int
    feature_names: tuple[str, ...]
    estimated_bytes: int
    state: str


class SQLiteMooncakeLifecycleIndex:
    """Persistent owner inventory and cross-process tombstone authority.

    The database contains metadata only. Tensor payloads remain exclusively in
    Mooncake. SQLite is the single-host production tier used by the manifest
    supervisor; a cross-node deployment must provide an equivalent shared
    metadata service before sharing this contract across hosts.
    """

    _PLANNED = "planned"
    _LIVE = "resident"
    _TOMBSTONED = "tombstoned"
    _CLEANED = "cleaned"

    def __init__(self, path: str, *, store_id: str) -> None:
        self.path = os.path.abspath(path)
        self.store_id = store_id
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30.0)
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=FULL")
        self._lock = threading.RLock()
        with self._lock:
            self._conn.execute(
                "CREATE TABLE IF NOT EXISTS mooncake_objects ("
                "store_id TEXT NOT NULL, sample_id TEXT NOT NULL, "
                "generation INTEGER NOT NULL, feature_names_json TEXT NOT NULL, "
                "estimated_bytes INTEGER NOT NULL, state TEXT NOT NULL, "
                "reason TEXT, updated_at REAL NOT NULL, "
                "PRIMARY KEY (store_id, sample_id, generation))"
            )
            self._conn.commit()

    def record_planned(
        self,
        sample_id: str,
        generation: int,
        feature_names: Iterable[str],
        estimated_bytes: int,
    ) -> None:
        """Durably declare every key before the first hard-pinned write."""
        names_json = json.dumps(sorted(feature_names), separators=(",", ":"))
        with self._lock:
            row = self._conn.execute(
                "SELECT feature_names_json, estimated_bytes, state FROM "
                "mooncake_objects WHERE store_id=? AND sample_id=? AND generation=?",
                (self.store_id, sample_id, generation),
            ).fetchone()
            if row is not None:
                prior_names, prior_bytes, state = row
                if state in {self._TOMBSTONED, self._CLEANED}:
                    raise RuntimeError(
                        f"refusing to plan {sample_id} generation {generation} "
                        f"from lifecycle state {state!r}"
                    )
                if prior_names != names_json or int(prior_bytes) != int(
                    estimated_bytes
                ):
                    raise RuntimeError(
                        f"lifecycle identity changed for {sample_id} generation "
                        f"{generation}"
                    )
                return
            self._conn.execute(
                "INSERT INTO mooncake_objects "
                "(store_id, sample_id, generation, feature_names_json, "
                "estimated_bytes, state, reason, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    self.store_id,
                    sample_id,
                    generation,
                    names_json,
                    int(estimated_bytes),
                    self._PLANNED,
                    time.time(),
                ),
            )
            self._conn.commit()

    def record_resident(
        self,
        sample_id: str,
        generation: int,
        feature_names: Iterable[str],
        estimated_bytes: int,
    ) -> None:
        names_json = json.dumps(sorted(feature_names), separators=(",", ":"))
        with self._lock:
            row = self._conn.execute(
                "SELECT feature_names_json, estimated_bytes, state FROM "
                "mooncake_objects WHERE store_id=? AND "
                "sample_id=? AND generation=?",
                (self.store_id, sample_id, generation),
            ).fetchone()
            if row is not None and row[2] not in {self._PLANNED, self._LIVE}:
                raise RuntimeError(
                    f"refusing to resurrect {sample_id} generation {generation} "
                    f"from lifecycle state {row[2]!r}"
                )
            if row is not None and (
                row[0] != names_json or int(row[1]) != int(estimated_bytes)
            ):
                raise RuntimeError(
                    f"lifecycle identity changed for {sample_id} generation "
                    f"{generation}"
                )
            self._conn.execute(
                "INSERT OR REPLACE INTO mooncake_objects "
                "(store_id, sample_id, generation, feature_names_json, "
                "estimated_bytes, state, reason, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, NULL, ?)",
                (
                    self.store_id,
                    sample_id,
                    generation,
                    names_json,
                    int(estimated_bytes),
                    self._LIVE,
                    time.time(),
                ),
            )
            self._conn.commit()

    def tombstone(self, sample_id: str, generation: int, reason: str) -> None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mooncake_objects SET state=?, reason=?, updated_at=? "
                "WHERE store_id=? AND sample_id=? AND generation=?",
                (
                    self._TOMBSTONED,
                    reason,
                    time.time(),
                    self.store_id,
                    sample_id,
                    generation,
                ),
            )
            if cur.rowcount != 1:
                raise KeyError(
                    f"no lifecycle inventory for {sample_id} generation {generation}"
                )
            self._conn.commit()

    def mark_cleaned(self, sample_id: str, generation: int) -> None:
        with self._lock:
            cur = self._conn.execute(
                "UPDATE mooncake_objects SET state=?, updated_at=? WHERE "
                "store_id=? AND sample_id=? AND generation=?",
                (
                    self._CLEANED,
                    time.time(),
                    self.store_id,
                    sample_id,
                    generation,
                ),
            )
            if cur.rowcount != 1:
                raise KeyError(
                    f"no lifecycle inventory for {sample_id} generation {generation}"
                )
            self._conn.commit()

    def state(self, sample_id: str, generation: int) -> Optional[str]:
        with self._lock:
            row = self._conn.execute(
                "SELECT state FROM mooncake_objects WHERE store_id=? AND "
                "sample_id=? AND generation=?",
                (self.store_id, sample_id, generation),
            ).fetchone()
        return row[0] if row is not None else None

    def record(self, sample_id: str, generation: int) -> Optional[LifecycleRecord]:
        """Return one exact-generation inventory row, including cleaned state."""
        with self._lock:
            row = self._conn.execute(
                "SELECT feature_names_json, estimated_bytes, state FROM "
                "mooncake_objects WHERE store_id=? AND sample_id=? AND generation=?",
                (self.store_id, sample_id, generation),
            ).fetchone()
        if row is None:
            return None
        return LifecycleRecord(
            sample_id=sample_id,
            generation=generation,
            feature_names=tuple(json.loads(row[0])),
            estimated_bytes=int(row[1]),
            state=row[2],
        )

    def pending(self) -> tuple[LifecycleRecord, ...]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT sample_id, generation, feature_names_json, "
                "estimated_bytes, state FROM mooncake_objects WHERE store_id=? "
                "AND state != ? ORDER BY rowid",
                (self.store_id, self._CLEANED),
            ).fetchall()
        return tuple(
            LifecycleRecord(
                sample_id=row[0],
                generation=int(row[1]),
                feature_names=tuple(json.loads(row[2])),
                estimated_bytes=int(row[3]),
                state=row[4],
            )
            for row in rows
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


__all__ = ["LifecycleRecord", "SQLiteMooncakeLifecycleIndex"]
