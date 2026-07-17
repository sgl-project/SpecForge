# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Transactional, consumer-driven capture windows.

Tensor payloads remain in :class:`FeatureStore`.  This module persists only
capture identity, ``SampleRef`` metadata, consumer interests, and read leases.
It is a single-host control plane: every process opens its own registry object
against the same SQLite path.

Windows are soft cache interests.  Demand waiters and read leases are hard
interests.  Under capacity pressure, the producer may reclaim a window-only
capture without blocking another consumer; a later demand recaptures it with a
new generation.  Capture slots (and, when configured, bytes) are reserved in
the same transaction that claims work.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import re
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from enum import Enum, IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
)

from specforge.runtime.contracts import SampleRef, assert_no_tensors
from specforge.runtime.data_plane.ref_serialization import (
    ref_from_dict,
    ref_to_dict,
)

if TYPE_CHECKING:
    from specforge.runtime.data_plane.feature_store import FeatureStore


_CONSUMER_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


class CaptureState(str, Enum):
    ABSENT = "absent"
    QUEUED = "queued"
    CAPTURING = "capturing"
    COMMITTING = "committing"
    READY = "ready"
    EVICTING = "evicting"
    FAILED = "failed"


class CapturePriority(IntEnum):
    DEMAND = 0
    PREFETCH = 1


@dataclass(frozen=True, order=True)
class CaptureKey:
    source_sample_id: str
    contract_digest: str

    def __post_init__(self) -> None:
        if not isinstance(self.source_sample_id, str) or not self.source_sample_id:
            raise ValueError("source_sample_id must be a non-empty string")
        digest = self.contract_digest
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(char not in "0123456789abcdef" for char in digest)
        ):
            raise ValueError("contract_digest must be a lowercase SHA-256 digest")


@dataclass(frozen=True)
class AcquireTicket:
    token: str
    consumer_id: str
    key: CaptureKey
    source_index: int
    ready_at_request: bool


@dataclass(frozen=True)
class CaptureReadLease:
    token: str
    consumer_id: str
    key: CaptureKey
    source_index: int
    generation: int
    ref: SampleRef
    ready_at_request: bool
    wait_s: float


@dataclass(frozen=True)
class CaptureRequest:
    key: CaptureKey
    source_index: int
    generation: int
    priority: CapturePriority
    demand_consumers: tuple[str, ...]
    reserved_bytes: int


@dataclass(frozen=True)
class EvictionCandidate:
    key: CaptureKey
    source_index: int
    generation: int
    ref: SampleRef


class CaptureFailedError(RuntimeError):
    """A requested capture reached a terminal failure."""


class ConsumerFailedError(RuntimeError):
    """A consumer was failed or expired while waiting for a capture."""


def _jsonable_contract(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        value = dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): _jsonable_contract(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (set, frozenset)):
        return sorted(_jsonable_contract(item) for item in value)
    if isinstance(value, (tuple, list)):
        return [_jsonable_contract(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"capture contract contains unsupported {type(value).__name__}")


def capture_contract_digest(contract: Any) -> str:
    """Return a canonical digest separating incompatible capture payloads."""
    encoded = json.dumps(
        _jsonable_contract(contract),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


class SQLiteWindowedCaptureRegistry:
    """SQLite authority for reusable captures and independent cursors."""

    _ACTIVE_CONSUMER_STATES = ("initializing", "ready", "eof")
    _LIVE_CAPTURE_STATES = (
        CaptureState.CAPTURING.value,
        CaptureState.COMMITTING.value,
        CaptureState.READY.value,
        CaptureState.EVICTING.value,
    )

    def __init__(
        self,
        path: str,
        *,
        max_live_refs: int,
        max_live_bytes: Optional[int] = None,
        capture_reservation_bytes: Optional[int] = None,
        clock: Callable[[], float] = time.time,
        poll_s: float = 0.01,
    ) -> None:
        if isinstance(max_live_refs, bool) or not isinstance(max_live_refs, int):
            raise TypeError("max_live_refs must be an integer")
        if max_live_refs < 1:
            raise ValueError("max_live_refs must be >= 1")
        if (max_live_bytes is None) != (capture_reservation_bytes is None):
            raise ValueError(
                "max_live_bytes and capture_reservation_bytes must be set together"
            )
        if max_live_bytes is not None:
            for name, value in (
                ("max_live_bytes", max_live_bytes),
                ("capture_reservation_bytes", capture_reservation_bytes),
            ):
                if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                    raise ValueError(f"{name} must be a positive integer")
            if capture_reservation_bytes > max_live_bytes:
                raise ValueError("capture_reservation_bytes exceeds max_live_bytes")
        if poll_s <= 0:
            raise ValueError("poll_s must be > 0")

        self.path = os.path.abspath(path)
        parent = os.path.dirname(self.path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        self.max_live_refs = max_live_refs
        self.max_live_bytes = max_live_bytes
        self.capture_reservation_bytes = capture_reservation_bytes or 0
        self._clock = clock
        self.poll_s = poll_s
        self._lock = threading.RLock()
        self._closed = False
        self._conn = sqlite3.connect(
            self.path, check_same_thread=False, timeout=30.0, isolation_level=None
        )
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.execute("PRAGMA foreign_keys=ON")
        self._create_schema()

    def _create_schema(self) -> None:
        schema = """
        CREATE TABLE IF NOT EXISTS run_config (
            singleton INTEGER PRIMARY KEY CHECK(singleton = 1),
            run_id TEXT NOT NULL,
            contract_digest TEXT NOT NULL,
            source_digest TEXT NOT NULL,
            total_samples INTEGER NOT NULL,
            expected_consumers_json TEXT NOT NULL,
            max_live_refs INTEGER NOT NULL,
            max_live_bytes INTEGER,
            capture_reservation_bytes INTEGER NOT NULL,
            status TEXT NOT NULL,
            created_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS sources (
            source_index INTEGER PRIMARY KEY,
            source_sample_id TEXT NOT NULL UNIQUE
        );
        CREATE TABLE IF NOT EXISTS consumers (
            consumer_id TEXT PRIMARY KEY,
            cursor INTEGER NOT NULL,
            total_samples INTEGER NOT NULL,
            lookbehind INTEGER NOT NULL,
            lookahead INTEGER NOT NULL,
            prefetch_depth INTEGER NOT NULL,
            max_outstanding INTEGER NOT NULL,
            state TEXT NOT NULL,
            heartbeat_at REAL NOT NULL,
            failure TEXT,
            updated_at REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS entries (
            source_index INTEGER PRIMARY KEY,
            source_sample_id TEXT NOT NULL UNIQUE,
            generation INTEGER NOT NULL DEFAULT 0,
            materializations INTEGER NOT NULL DEFAULT 0,
            state TEXT NOT NULL,
            priority INTEGER,
            queued_at REAL,
            captured_at REAL,
            reserved_bytes INTEGER NOT NULL DEFAULT 0,
            estimated_bytes INTEGER NOT NULL DEFAULT 0,
            ref_json TEXT,
            retry_count INTEGER NOT NULL DEFAULT 0,
            last_error TEXT,
            prefetch_suppressed INTEGER NOT NULL DEFAULT 0,
            FOREIGN KEY(source_index) REFERENCES sources(source_index)
        );
        CREATE INDEX IF NOT EXISTS entries_schedule
            ON entries(state, priority, queued_at, source_index);
        CREATE TABLE IF NOT EXISTS interests (
            consumer_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            kind TEXT NOT NULL CHECK(kind IN ('window', 'demand')),
            created_at REAL NOT NULL,
            PRIMARY KEY(consumer_id, source_index, kind),
            FOREIGN KEY(consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE,
            FOREIGN KEY(source_index) REFERENCES sources(source_index)
        );
        CREATE TABLE IF NOT EXISTS waiters (
            token TEXT PRIMARY KEY,
            consumer_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            created_at REAL NOT NULL,
            ready_at_request INTEGER NOT NULL,
            UNIQUE(consumer_id, source_index),
            FOREIGN KEY(consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS read_leases (
            token TEXT PRIMARY KEY,
            consumer_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            generation INTEGER NOT NULL,
            created_at REAL NOT NULL,
            UNIQUE(consumer_id, source_index),
            FOREIGN KEY(consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS completed_samples (
            consumer_id TEXT NOT NULL,
            source_index INTEGER NOT NULL,
            PRIMARY KEY(consumer_id, source_index),
            FOREIGN KEY(consumer_id) REFERENCES consumers(consumer_id)
                ON DELETE CASCADE
        );
        CREATE TABLE IF NOT EXISTS registry_meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
        with self._lock:
            self._conn.executescript(schema)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.rollback()
                raise
            else:
                self._conn.commit()

    @staticmethod
    def _validate_consumer_id(consumer_id: str) -> None:
        if (
            not isinstance(consumer_id, str)
            or _CONSUMER_ID.fullmatch(consumer_id) is None
        ):
            raise ValueError("consumer_id must match [A-Za-z0-9][A-Za-z0-9._-]{0,127}")

    @staticmethod
    def _validate_non_negative(name: str, value: int) -> None:
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")

    def initialize_run(
        self,
        *,
        run_id: str,
        contract_digest: str,
        source_sample_ids: Sequence[str],
        expected_consumers: Sequence[str],
        recover_inflight: bool = False,
        recovery_store: Optional["FeatureStore"] = None,
    ) -> None:
        """Create or validate a run, optionally recovering interrupted work.

        ``recover_inflight`` is an owner-only operation.  Captures that have no
        published payload are fenced by advancing their generation.  A
        ``recovery_store`` is required for COMMITTING or EVICTING entries so the
        old payload is generation-safely reclaimed before work is requeued.
        """
        CaptureKey("validation", contract_digest)
        if not isinstance(run_id, str) or not run_id:
            raise ValueError("run_id must be a non-empty string")
        sources = tuple(source_sample_ids)
        if (
            not sources
            or any(not isinstance(item, str) or not item for item in sources)
            or len(sources) != len(set(sources))
        ):
            raise ValueError("source_sample_ids must be non-empty, unique strings")
        consumers = tuple(expected_consumers)
        if not consumers or len(consumers) != len(set(consumers)):
            raise ValueError("expected_consumers must be non-empty and unique")
        for consumer_id in consumers:
            self._validate_consumer_id(consumer_id)
        source_digest = hashlib.sha256(
            json.dumps(sources, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        expected_json = json.dumps(sorted(consumers), separators=(",", ":"))
        identity = (
            run_id,
            contract_digest,
            source_digest,
            len(sources),
            expected_json,
            self.max_live_refs,
            self.max_live_bytes,
            self.capture_reservation_bytes,
        )
        recovery_candidates: tuple[EvictionCandidate, ...] = ()
        with self._transaction() as conn:
            existing = conn.execute(
                "SELECT * FROM run_config WHERE singleton=1"
            ).fetchone()
            if existing is not None:
                observed = (
                    existing["run_id"],
                    existing["contract_digest"],
                    existing["source_digest"],
                    int(existing["total_samples"]),
                    existing["expected_consumers_json"],
                    int(existing["max_live_refs"]),
                    existing["max_live_bytes"],
                    int(existing["capture_reservation_bytes"]),
                )
                if observed != identity:
                    raise RuntimeError(
                        "windowed capture registry identity mismatch: "
                        f"expected={identity!r}, observed={observed!r}"
                    )
                interrupted = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM entries WHERE state IN (?,?,?)",
                        (
                            CaptureState.CAPTURING.value,
                            CaptureState.COMMITTING.value,
                            CaptureState.EVICTING.value,
                        ),
                    ).fetchone()[0]
                )
                physical = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM entries WHERE state IN (?,?)",
                        (
                            CaptureState.COMMITTING.value,
                            CaptureState.EVICTING.value,
                        ),
                    ).fetchone()[0]
                )
                if interrupted and not recover_inflight:
                    raise RuntimeError(
                        "registry has interrupted captures; owner must explicitly "
                        "set recover_inflight=True"
                    )
                if physical and recovery_store is None:
                    raise RuntimeError(
                        "recovery_store is required to resolve interrupted payloads"
                    )
                if interrupted:
                    recovery_candidates = self._recover_inflight_locked(conn)
            else:
                conn.execute(
                    "INSERT INTO run_config VALUES(1,?,?,?,?,?,?,?,?,?,?)",
                    (*identity, "active", self._clock()),
                )
                conn.executemany(
                    "INSERT INTO sources(source_index,source_sample_id) VALUES(?,?)",
                    enumerate(sources),
                )
                conn.executemany(
                    "INSERT INTO registry_meta(key,value) VALUES(?,?)",
                    (
                        ("scheduler_cursor", ""),
                        ("capture_count", "0"),
                        ("recapture_count", "0"),
                        ("peak_live_refs", "0"),
                        ("peak_live_bytes", "0"),
                    ),
                )

        if recovery_candidates:
            assert recovery_store is not None
            completed: list[EvictionCandidate] = []
            try:
                for candidate in recovery_candidates:
                    recovery_store.reclaim(
                        candidate.ref, reason="interrupted-capture-recovery"
                    )
                    completed.append(candidate)
            except BaseException:
                if completed:
                    self.finish_evictions(completed)
                raise
            self.finish_evictions(completed)

    def _recover_inflight_locked(
        self, conn: sqlite3.Connection
    ) -> tuple[EvictionCandidate, ...]:
        rows = conn.execute(
            "SELECT * FROM entries WHERE state IN (?,?,?) ORDER BY source_index",
            (
                CaptureState.CAPTURING.value,
                CaptureState.COMMITTING.value,
                CaptureState.EVICTING.value,
            ),
        ).fetchall()
        candidates: list[EvictionCandidate] = []
        for row in rows:
            state = CaptureState(row["state"])
            if state in (CaptureState.COMMITTING, CaptureState.EVICTING):
                if not row["ref_json"]:
                    raise RuntimeError(
                        f"interrupted {state.value} capture has no payload metadata"
                    )
                if state == CaptureState.COMMITTING:
                    conn.execute(
                        "UPDATE entries SET state=? WHERE source_index=?",
                        (CaptureState.EVICTING.value, row["source_index"]),
                    )
                candidates.append(
                    EvictionCandidate(
                        key=self._key(conn, int(row["source_index"])),
                        source_index=int(row["source_index"]),
                        generation=int(row["generation"]),
                        ref=ref_from_dict(json.loads(row["ref_json"])),
                    )
                )
                continue

            demand = conn.execute(
                "SELECT 1 FROM interests WHERE source_index=? AND kind='demand' "
                "LIMIT 1",
                (row["source_index"],),
            ).fetchone()
            interested = conn.execute(
                "SELECT 1 FROM interests WHERE source_index=? LIMIT 1",
                (row["source_index"],),
            ).fetchone()
            if interested is None:
                conn.execute(
                    "UPDATE entries SET state=?,priority=NULL,queued_at=NULL,"
                    "reserved_bytes=0,last_error=? WHERE source_index=?",
                    (
                        CaptureState.ABSENT.value,
                        "producer interrupted",
                        row["source_index"],
                    ),
                )
                continue
            priority = CapturePriority.DEMAND if demand else CapturePriority.PREFETCH
            conn.execute(
                "UPDATE entries SET state=?,generation=generation+1,priority=?,"
                "queued_at=?,captured_at=NULL,reserved_bytes=0,retry_count="
                "retry_count+1,last_error=? WHERE source_index=?",
                (
                    CaptureState.QUEUED.value,
                    int(priority),
                    self._clock(),
                    "producer interrupted",
                    row["source_index"],
                ),
            )
        return tuple(candidates)

    def wait_initialized(self, timeout_s: float) -> dict[str, Any]:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        deadline = time.monotonic() + timeout_s
        while True:
            with self._lock:
                row = self._conn.execute(
                    "SELECT * FROM run_config WHERE singleton=1"
                ).fetchone()
            if row is not None:
                return {
                    "run_id": row["run_id"],
                    "contract_digest": row["contract_digest"],
                    "total_samples": int(row["total_samples"]),
                    "expected_consumers": tuple(
                        json.loads(row["expected_consumers_json"])
                    ),
                    "status": row["status"],
                    "max_live_refs": int(row["max_live_refs"]),
                    "max_live_bytes": row["max_live_bytes"],
                    "capture_reservation_bytes": int(row["capture_reservation_bytes"]),
                }
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"registry {self.path!r} was not initialized within "
                    f"{timeout_s:.1f}s"
                )
            time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))

    def _run(self, conn: sqlite3.Connection) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM run_config WHERE singleton=1").fetchone()
        if row is None:
            raise RuntimeError("windowed capture run is not initialized")
        return row

    def _source(self, conn: sqlite3.Connection, source_index: int) -> sqlite3.Row:
        if isinstance(source_index, bool) or not isinstance(source_index, int):
            raise TypeError("source_index must be an integer")
        run = self._run(conn)
        if source_index < 0 or source_index >= int(run["total_samples"]):
            raise IndexError(
                f"source_index {source_index} outside [0, {run['total_samples']})"
            )
        row = conn.execute(
            "SELECT * FROM sources WHERE source_index=?", (source_index,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing source catalog entry {source_index}")
        return row

    def _ensure_entry(self, conn: sqlite3.Connection, source_index: int) -> sqlite3.Row:
        source = self._source(conn, source_index)
        conn.execute(
            "INSERT OR IGNORE INTO entries(source_index,source_sample_id,state) "
            "VALUES(?,?,?)",
            (source_index, source["source_sample_id"], CaptureState.ABSENT.value),
        )
        return conn.execute(
            "SELECT * FROM entries WHERE source_index=?", (source_index,)
        ).fetchone()

    def _key(self, conn: sqlite3.Connection, source_index: int) -> CaptureKey:
        source = self._source(conn, source_index)
        run = self._run(conn)
        return CaptureKey(source["source_sample_id"], run["contract_digest"])

    def _queue_entry(
        self,
        conn: sqlite3.Connection,
        row: sqlite3.Row,
        priority: CapturePriority,
    ) -> sqlite3.Row:
        state = CaptureState(row["state"])
        if state == CaptureState.QUEUED:
            if int(row["priority"]) > int(priority):
                conn.execute(
                    "UPDATE entries SET priority=?,prefetch_suppressed=0 "
                    "WHERE source_index=?",
                    (int(priority), row["source_index"]),
                )
            return conn.execute(
                "SELECT * FROM entries WHERE source_index=?", (row["source_index"],)
            ).fetchone()
        if state not in (CaptureState.ABSENT, CaptureState.FAILED):
            return row
        if state == CaptureState.FAILED:
            waiters = conn.execute(
                "SELECT COUNT(*) FROM waiters WHERE source_index=?",
                (row["source_index"],),
            ).fetchone()[0]
            if waiters:
                return row
        if priority == CapturePriority.PREFETCH and row["prefetch_suppressed"]:
            return row
        conn.execute(
            "UPDATE entries SET state=?,generation=generation+1,priority=?,"
            "queued_at=?,captured_at=NULL,reserved_bytes=0,last_error=NULL,"
            "prefetch_suppressed=0 WHERE source_index=?",
            (
                CaptureState.QUEUED.value,
                int(priority),
                self._clock(),
                row["source_index"],
            ),
        )
        return conn.execute(
            "SELECT * FROM entries WHERE source_index=?", (row["source_index"],)
        ).fetchone()

    def _window_indices(self, consumer: sqlite3.Row) -> range:
        cursor = int(consumer["cursor"])
        low = max(0, cursor - int(consumer["lookbehind"]))
        high = min(
            int(consumer["total_samples"]), cursor + int(consumer["lookahead"]) + 1
        )
        return range(low, high)

    def _refresh_window_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        consumer = conn.execute(
            "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
        ).fetchone()
        if consumer is None:
            raise KeyError(f"unknown consumer {consumer_id!r}")
        desired = set(self._window_indices(consumer))
        existing = {
            int(row[0])
            for row in conn.execute(
                "SELECT source_index FROM interests WHERE consumer_id=? "
                "AND kind='window'",
                (consumer_id,),
            ).fetchall()
        }
        now = self._clock()
        for source_index in sorted(desired - existing):
            self._ensure_entry(conn, source_index)
            conn.execute(
                "INSERT INTO interests VALUES(?,?,'window',?)",
                (consumer_id, source_index, now),
            )
            conn.execute(
                "UPDATE entries SET prefetch_suppressed=0 WHERE source_index=?",
                (source_index,),
            )
        for source_index in sorted(existing - desired):
            conn.execute(
                "DELETE FROM interests WHERE consumer_id=? AND source_index=? "
                "AND kind='window'",
                (consumer_id, source_index),
            )
        self._prune_orphans_locked(conn)
        self._top_up_prefetch_locked(conn, consumer_id)

    def _top_up_prefetch_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        consumer = conn.execute(
            "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
        ).fetchone()
        if consumer is None or consumer["state"] not in self._ACTIVE_CONSUMER_STATES:
            return
        depth = int(consumer["prefetch_depth"])
        if depth == 0:
            return
        cursor = int(consumer["cursor"])
        stop = min(int(consumer["total_samples"]), cursor + depth)
        for source_index in range(cursor, stop):
            row = self._ensure_entry(conn, source_index)
            self._queue_entry(conn, row, CapturePriority.PREFETCH)

    def _drop_demand_if_idle_locked(
        self, conn: sqlite3.Connection, consumer_id: str, source_index: int
    ) -> None:
        outstanding = conn.execute(
            "SELECT (SELECT COUNT(*) FROM waiters WHERE consumer_id=? AND "
            "source_index=?) + (SELECT COUNT(*) FROM read_leases WHERE "
            "consumer_id=? AND source_index=?)",
            (consumer_id, source_index, consumer_id, source_index),
        ).fetchone()[0]
        if not outstanding:
            conn.execute(
                "DELETE FROM interests WHERE consumer_id=? AND source_index=? "
                "AND kind='demand'",
                (consumer_id, source_index),
            )

    def _prune_orphans_locked(self, conn: sqlite3.Connection) -> int:
        rows = conn.execute(
            "SELECT source_index FROM entries e WHERE e.state IN (?,?) AND "
            "NOT EXISTS(SELECT 1 FROM interests i WHERE i.source_index="
            "e.source_index) AND NOT EXISTS(SELECT 1 FROM waiters w WHERE "
            "w.source_index=e.source_index)",
            (CaptureState.QUEUED.value, CaptureState.FAILED.value),
        ).fetchall()
        for row in rows:
            conn.execute(
                "UPDATE entries SET state=?,priority=NULL,queued_at=NULL,"
                "reserved_bytes=0,last_error=NULL WHERE source_index=?",
                (CaptureState.ABSENT.value, row["source_index"]),
            )
        return len(rows)

    def register_consumer(
        self,
        consumer_id: str,
        *,
        lookbehind: int = 0,
        lookahead: int = 0,
        prefetch_depth: int = 0,
        max_outstanding: int = 1,
        cursor: int = 0,
    ) -> None:
        """Register one stable logical consumer and its bounded cache window."""
        self._validate_consumer_id(consumer_id)
        for name, value in (
            ("lookbehind", lookbehind),
            ("lookahead", lookahead),
            ("prefetch_depth", prefetch_depth),
            ("max_outstanding", max_outstanding),
            ("cursor", cursor),
        ):
            self._validate_non_negative(name, value)
        if max_outstanding < 1:
            raise ValueError("max_outstanding must be >= 1")
        if prefetch_depth > lookahead + 1:
            raise ValueError("prefetch_depth cannot exceed lookahead + 1")

        with self._transaction() as conn:
            run = self._run(conn)
            expected = set(json.loads(run["expected_consumers_json"]))
            if consumer_id not in expected:
                raise ValueError(
                    f"consumer {consumer_id!r} is not expected; expected={sorted(expected)}"
                )
            total = int(run["total_samples"])
            if cursor > total:
                raise ValueError("consumer cursor exceeds total samples")
            identity = (
                cursor,
                total,
                lookbehind,
                lookahead,
                prefetch_depth,
                max_outstanding,
            )
            existing = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if existing is not None:
                observed = tuple(
                    int(existing[name])
                    for name in (
                        "cursor",
                        "total_samples",
                        "lookbehind",
                        "lookahead",
                        "prefetch_depth",
                        "max_outstanding",
                    )
                )
                if observed != identity or existing["state"] in ("completed", "failed"):
                    raise RuntimeError(
                        f"consumer {consumer_id!r} registration mismatch: "
                        f"expected={identity}, observed={observed}, "
                        f"state={existing['state']!r}"
                    )
                return
            now = self._clock()
            state = "eof" if cursor == total else "initializing"
            conn.execute(
                "INSERT INTO consumers VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (
                    consumer_id,
                    cursor,
                    total,
                    lookbehind,
                    lookahead,
                    prefetch_depth,
                    max_outstanding,
                    state,
                    now,
                    None,
                    now,
                ),
            )
            self._refresh_window_locked(conn, consumer_id)

    def resume_consumer(self, consumer_id: str, *, durable_cursor: int) -> None:
        """Reconcile transient state to an optimizer-durable prefix cursor.

        The cursor may move in either direction.  A rewind deliberately replays
        captures acknowledged only by the crashed process; a fast-forward is
        accepted only because the caller explicitly supplies durable progress.
        """
        self._validate_non_negative("durable_cursor", durable_cursor)
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] == "failed":
                raise RuntimeError(f"cannot resume failed consumer {consumer_id!r}")
            total = int(consumer["total_samples"])
            if durable_cursor > total:
                raise ValueError("durable_cursor exceeds total samples")
            self._release_consumer_transients_locked(conn, consumer_id)
            conn.execute(
                "DELETE FROM completed_samples WHERE consumer_id=?", (consumer_id,)
            )
            now = self._clock()
            state = "eof" if durable_cursor == total else "initializing"
            conn.execute(
                "UPDATE consumers SET cursor=?,state=?,failure=NULL,heartbeat_at=?,"
                "updated_at=? WHERE consumer_id=?",
                (durable_cursor, state, now, now, consumer_id),
            )
            self._refresh_window_locked(conn, consumer_id)

    def heartbeat(self, consumer_id: str, *, ready: bool = False) -> None:
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] not in self._ACTIVE_CONSUMER_STATES:
                raise RuntimeError(f"consumer {consumer_id!r} is {consumer['state']!r}")
            state = consumer["state"]
            if ready and state == "initializing":
                state = "ready"
            now = self._clock()
            conn.execute(
                "UPDATE consumers SET state=?,heartbeat_at=?,updated_at=? "
                "WHERE consumer_id=?",
                (state, now, now, consumer_id),
            )

    def consumer_cursor(self, consumer_id: str) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT cursor FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
        if row is None:
            raise KeyError(f"unknown consumer {consumer_id!r}")
        return int(row["cursor"])

    def request_acquire(self, consumer_id: str, source_index: int) -> AcquireTicket:
        token = uuid.uuid4().hex
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] not in self._ACTIVE_CONSUMER_STATES:
                raise RuntimeError(f"consumer {consumer_id!r} is {consumer['state']!r}")
            if source_index not in self._window_indices(consumer):
                window = self._window_indices(consumer)
                raise ValueError(
                    f"source_index {source_index} outside consumer {consumer_id!r} "
                    f"window [{window.start}, {window.stop})"
                )
            duplicate = conn.execute(
                "SELECT (SELECT COUNT(*) FROM waiters WHERE consumer_id=? AND "
                "source_index=?) + (SELECT COUNT(*) FROM read_leases WHERE "
                "consumer_id=? AND source_index=?)",
                (consumer_id, source_index, consumer_id, source_index),
            ).fetchone()[0]
            if duplicate:
                raise RuntimeError(
                    f"consumer {consumer_id!r} already has source_index "
                    f"{source_index} outstanding"
                )
            outstanding = conn.execute(
                "SELECT (SELECT COUNT(*) FROM waiters WHERE consumer_id=?) + "
                "(SELECT COUNT(*) FROM read_leases WHERE consumer_id=?)",
                (consumer_id, consumer_id),
            ).fetchone()[0]
            if int(outstanding) >= int(consumer["max_outstanding"]):
                raise RuntimeError(
                    f"consumer {consumer_id!r} reached max_outstanding="
                    f"{consumer['max_outstanding']}"
                )
            row = self._ensure_entry(conn, source_index)
            ready = CaptureState(row["state"]) == CaptureState.READY
            now = self._clock()
            conn.execute(
                "INSERT OR IGNORE INTO interests VALUES(?,?,'demand',?)",
                (consumer_id, source_index, now),
            )
            self._queue_entry(conn, row, CapturePriority.DEMAND)
            conn.execute(
                "INSERT INTO waiters VALUES(?,?,?,?,?)",
                (token, consumer_id, source_index, now, int(ready)),
            )
            key = self._key(conn, source_index)
        return AcquireTicket(token, consumer_id, key, source_index, ready)

    def request_many(
        self, consumer_id: str, source_indices: Iterable[int]
    ) -> tuple[AcquireTicket, ...]:
        tickets: list[AcquireTicket] = []
        try:
            for source_index in source_indices:
                tickets.append(self.request_acquire(consumer_id, source_index))
        except BaseException:
            for ticket in tickets:
                self.cancel_acquire(ticket)
            raise
        return tuple(tickets)

    def wait_ready(
        self, ticket: AcquireTicket, *, timeout_s: Optional[float]
    ) -> CaptureReadLease:
        if timeout_s is not None and timeout_s <= 0:
            raise ValueError("timeout_s must be > 0 or None")
        started = time.monotonic()
        deadline = None if timeout_s is None else started + timeout_s
        while True:
            with self._lock:
                waiter = self._conn.execute(
                    "SELECT 1 FROM waiters WHERE token=?", (ticket.token,)
                ).fetchone()
                entry = self._conn.execute(
                    "SELECT state FROM entries WHERE source_index=?",
                    (ticket.source_index,),
                ).fetchone()
                consumer = self._conn.execute(
                    "SELECT state,failure FROM consumers WHERE consumer_id=?",
                    (ticket.consumer_id,),
                ).fetchone()
            if waiter is None:
                if consumer is not None and consumer["state"] == "failed":
                    raise ConsumerFailedError(
                        f"consumer {ticket.consumer_id!r} failed: "
                        f"{consumer['failure'] or 'unknown failure'}"
                    )
                raise RuntimeError(f"acquire ticket {ticket.token} is no longer active")
            if entry is None:
                raise RuntimeError(f"capture entry disappeared for {ticket.key}")
            if CaptureState(entry["state"]) in (
                CaptureState.READY,
                CaptureState.FAILED,
            ):
                lease = self._finish_wait(ticket, started)
                if lease is not None:
                    return lease
            if deadline is not None and time.monotonic() >= deadline:
                self.cancel_acquire(ticket)
                raise TimeoutError(
                    f"capture {ticket.key.source_sample_id} did not become ready "
                    f"within {timeout_s:.1f}s"
                )
            sleep_s = self.poll_s
            if deadline is not None:
                sleep_s = min(sleep_s, max(0.0, deadline - time.monotonic()))
            time.sleep(sleep_s)

    def _finish_wait(
        self, ticket: AcquireTicket, started: float
    ) -> Optional[CaptureReadLease]:
        failure: Optional[str] = None
        lease: Optional[CaptureReadLease] = None
        with self._transaction() as conn:
            waiter = conn.execute(
                "SELECT 1 FROM waiters WHERE token=?", (ticket.token,)
            ).fetchone()
            if waiter is None:
                raise RuntimeError(f"acquire ticket {ticket.token} is no longer active")
            row = conn.execute(
                "SELECT * FROM entries WHERE source_index=?", (ticket.source_index,)
            ).fetchone()
            state = CaptureState(row["state"])
            if state not in (CaptureState.READY, CaptureState.FAILED):
                return None
            if state == CaptureState.FAILED:
                conn.execute("DELETE FROM waiters WHERE token=?", (ticket.token,))
                self._drop_demand_if_idle_locked(
                    conn, ticket.consumer_id, ticket.source_index
                )
                failure = (
                    f"capture {ticket.key.source_sample_id} generation "
                    f"{row['generation']} failed: {row['last_error'] or 'unknown error'}"
                )
            else:
                if not row["ref_json"]:
                    raise RuntimeError("READY capture has no SampleRef metadata")
                lease_token = uuid.uuid4().hex
                conn.execute(
                    "INSERT INTO read_leases VALUES(?,?,?,?,?)",
                    (
                        lease_token,
                        ticket.consumer_id,
                        ticket.source_index,
                        int(row["generation"]),
                        self._clock(),
                    ),
                )
                conn.execute("DELETE FROM waiters WHERE token=?", (ticket.token,))
                lease = CaptureReadLease(
                    token=lease_token,
                    consumer_id=ticket.consumer_id,
                    key=ticket.key,
                    source_index=ticket.source_index,
                    generation=int(row["generation"]),
                    ref=ref_from_dict(json.loads(row["ref_json"])),
                    ready_at_request=ticket.ready_at_request,
                    wait_s=time.monotonic() - started,
                )
        if failure is not None:
            raise CaptureFailedError(failure)
        return lease

    def cancel_acquire(self, ticket: AcquireTicket) -> None:
        with self._transaction() as conn:
            conn.execute("DELETE FROM waiters WHERE token=?", (ticket.token,))
            self._drop_demand_if_idle_locked(
                conn, ticket.consumer_id, ticket.source_index
            )
            self._prune_orphans_locked(conn)

    def _live_usage_locked(self, conn: sqlite3.Connection) -> tuple[int, int]:
        placeholders = ",".join("?" for _ in self._LIVE_CAPTURE_STATES)
        row = conn.execute(
            f"SELECT COUNT(*) AS refs,COALESCE(SUM(CASE WHEN state IN (?,?) "
            f"THEN estimated_bytes ELSE reserved_bytes END),0) AS bytes "
            f"FROM entries WHERE state IN ({placeholders})",
            (
                CaptureState.READY.value,
                CaptureState.EVICTING.value,
                *self._LIVE_CAPTURE_STATES,
            ),
        ).fetchone()
        return int(row["refs"]), int(row["bytes"])

    def _claim_capacity_locked(self, conn: sqlite3.Connection) -> int:
        live_refs, live_bytes = self._live_usage_locked(conn)
        capacity = max(0, self.max_live_refs - live_refs)
        if self.max_live_bytes is not None:
            byte_capacity = max(0, self.max_live_bytes - live_bytes)
            capacity = min(capacity, byte_capacity // self.capture_reservation_bytes)
        return capacity

    @staticmethod
    def _meta_int(conn: sqlite3.Connection, key: str) -> int:
        row = conn.execute(
            "SELECT value FROM registry_meta WHERE key=?", (key,)
        ).fetchone()
        if row is None:
            raise RuntimeError(f"missing registry metadata {key!r}")
        return int(row["value"])

    @staticmethod
    def _set_meta_int(conn: sqlite3.Connection, key: str, value: int) -> None:
        conn.execute("UPDATE registry_meta SET value=? WHERE key=?", (str(value), key))

    def _record_peaks_locked(self, conn: sqlite3.Connection) -> None:
        live_refs, live_bytes = self._live_usage_locked(conn)
        if live_refs > self._meta_int(conn, "peak_live_refs"):
            self._set_meta_int(conn, "peak_live_refs", live_refs)
        if live_bytes > self._meta_int(conn, "peak_live_bytes"):
            self._set_meta_int(conn, "peak_live_bytes", live_bytes)

    def claim_batch(self, max_requests: int) -> tuple[CaptureRequest, ...]:
        """Claim demand-first work while reserving authoritative capacity."""
        if isinstance(max_requests, bool) or not isinstance(max_requests, int):
            raise TypeError("max_requests must be an integer")
        if max_requests < 1:
            raise ValueError("max_requests must be >= 1")
        with self._transaction() as conn:
            limit = min(max_requests, self._claim_capacity_locked(conn))
            if limit == 0:
                return ()
            rows = conn.execute(
                "SELECT * FROM entries WHERE state=? ORDER BY priority,queued_at,"
                "source_index",
                (CaptureState.QUEUED.value,),
            ).fetchall()
            if not rows:
                return ()
            demand_rows = [
                row
                for row in rows
                if int(row["priority"]) == int(CapturePriority.DEMAND)
            ]
            prefetch_rows = [
                row
                for row in rows
                if int(row["priority"]) == int(CapturePriority.PREFETCH)
            ]
            owners: dict[int, tuple[str, ...]] = {}
            by_consumer: dict[str, list[sqlite3.Row]] = {}
            for row in demand_rows:
                source_index = int(row["source_index"])
                demanders = tuple(
                    owner["consumer_id"]
                    for owner in conn.execute(
                        "SELECT consumer_id FROM interests WHERE source_index=? "
                        "AND kind='demand' ORDER BY consumer_id",
                        (source_index,),
                    ).fetchall()
                )
                owners[source_index] = demanders
                for consumer_id in demanders:
                    by_consumer.setdefault(consumer_id, []).append(row)

            cursor_row = conn.execute(
                "SELECT value FROM registry_meta WHERE key='scheduler_cursor'"
            ).fetchone()
            last_consumer = cursor_row["value"] if cursor_row else ""
            consumer_order = sorted(by_consumer)
            if last_consumer in consumer_order:
                start = (consumer_order.index(last_consumer) + 1) % len(consumer_order)
                consumer_order = consumer_order[start:] + consumer_order[:start]

            selected: list[sqlite3.Row] = []
            selected_indices: set[int] = set()
            last_selected = last_consumer
            while consumer_order and len(selected) < limit:
                made_progress = False
                for consumer_id in consumer_order:
                    pending = by_consumer[consumer_id]
                    while pending:
                        row = pending.pop(0)
                        source_index = int(row["source_index"])
                        if source_index not in selected_indices:
                            selected.append(row)
                            selected_indices.add(source_index)
                            last_selected = consumer_id
                            made_progress = True
                            break
                    if len(selected) >= limit:
                        break
                if not made_progress:
                    break
            for row in prefetch_rows:
                if len(selected) >= limit:
                    break
                source_index = int(row["source_index"])
                if source_index not in selected_indices:
                    selected.append(row)
                    selected_indices.add(source_index)

            now = self._clock()
            requests: list[CaptureRequest] = []
            for row in selected:
                source_index = int(row["source_index"])
                priority = CapturePriority(int(row["priority"]))
                updated = conn.execute(
                    "UPDATE entries SET state=?,captured_at=?,reserved_bytes=? "
                    "WHERE source_index=? AND state=?",
                    (
                        CaptureState.CAPTURING.value,
                        now,
                        self.capture_reservation_bytes,
                        source_index,
                        CaptureState.QUEUED.value,
                    ),
                )
                if updated.rowcount != 1:
                    raise RuntimeError(
                        f"capture claim lost transactional ownership for {source_index}"
                    )
                requests.append(
                    CaptureRequest(
                        key=self._key(conn, source_index),
                        source_index=source_index,
                        generation=int(row["generation"]),
                        priority=priority,
                        demand_consumers=owners.get(source_index, ()),
                        reserved_bytes=self.capture_reservation_bytes,
                    )
                )
            if last_selected:
                conn.execute(
                    "UPDATE registry_meta SET value=? WHERE key='scheduler_cursor'",
                    (last_selected,),
                )
            self._record_peaks_locked(conn)
            return tuple(requests)

    def _validated_ref_json(
        self, request: CaptureRequest, ref: SampleRef
    ) -> tuple[str, int]:
        assert_no_tensors(ref)
        # Storage backends own ``generation`` as the physical object locator.
        # Recapture scheduling is a separate namespace: runtime adapters attach
        # ``window_generation`` without rewriting the store's generation.  The
        # fallback preserves refs produced directly against the registry.
        generation = ref.metadata.get(
            "window_generation", ref.metadata.get("generation")
        )
        if (
            isinstance(generation, bool)
            or not isinstance(generation, int)
            or generation != request.generation
        ):
            raise ValueError(
                f"capture ref generation {generation!r} does not match "
                f"claimed generation {request.generation}"
            )
        if ref.source_task_id != request.key.source_sample_id:
            raise ValueError(
                f"capture ref source_task_id={ref.source_task_id!r} does not match "
                f"{request.key.source_sample_id!r}"
            )
        with self._lock:
            run = self._conn.execute(
                "SELECT run_id FROM run_config WHERE singleton=1"
            ).fetchone()
        if run is None or ref.run_id != run["run_id"]:
            raise ValueError(
                f"capture ref run_id={ref.run_id!r} does not match registry run "
                f"{None if run is None else run['run_id']!r}"
            )
        estimated = ref.estimated_bytes
        if (
            isinstance(estimated, bool)
            or not isinstance(estimated, int)
            or estimated < 0
        ):
            raise ValueError("SampleRef.estimated_bytes must be a non-negative integer")
        if self.max_live_bytes is not None and estimated > request.reserved_bytes:
            raise ValueError(
                f"capture estimated_bytes={estimated} exceeds reserved upper bound "
                f"{request.reserved_bytes}; reclaim the payload and fail the request"
            )
        return json.dumps(ref_to_dict(ref), separators=(",", ":")), estimated

    def mark_committing(self, request: CaptureRequest, ref: SampleRef) -> None:
        """Persist payload identity before exposing the READY transition."""
        ref_json, estimated = self._validated_ref_json(request, ref)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE source_index=?", (request.source_index,)
            ).fetchone()
            if (
                row is None
                or CaptureState(row["state"]) != CaptureState.CAPTURING
                or int(row["generation"]) != request.generation
                or row["source_sample_id"] != request.key.source_sample_id
            ):
                raise RuntimeError(
                    f"capture commit transition mismatch for {request.key}"
                )
            conn.execute(
                "UPDATE entries SET state=?,estimated_bytes=?,ref_json=? "
                "WHERE source_index=?",
                (
                    CaptureState.COMMITTING.value,
                    estimated,
                    ref_json,
                    request.source_index,
                ),
            )

    def complete_capture(self, request: CaptureRequest, ref: SampleRef) -> None:
        """Expose a previously persisted COMMITTING payload to consumers."""
        ref_json, estimated = self._validated_ref_json(request, ref)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE source_index=?", (request.source_index,)
            ).fetchone()
            if (
                row is None
                or CaptureState(row["state"]) != CaptureState.COMMITTING
                or int(row["generation"]) != request.generation
            ):
                raise RuntimeError(
                    f"capture READY transition mismatch for {request.key}"
                )
            if row["ref_json"] != ref_json:
                raise RuntimeError(
                    f"capture payload identity changed while committing {request.key}"
                )
            recapture = int(row["materializations"]) > 0
            conn.execute(
                "UPDATE entries SET state=?,materializations=materializations+1,"
                "priority=NULL,queued_at=NULL,reserved_bytes=0,estimated_bytes=?,"
                "retry_count=0,last_error=NULL WHERE source_index=?",
                (
                    CaptureState.READY.value,
                    estimated,
                    request.source_index,
                ),
            )
            self._set_meta_int(
                conn, "capture_count", self._meta_int(conn, "capture_count") + 1
            )
            if recapture:
                self._set_meta_int(
                    conn,
                    "recapture_count",
                    self._meta_int(conn, "recapture_count") + 1,
                )
            self._record_peaks_locked(conn)

    def fail_capture(
        self,
        request: CaptureRequest,
        error: BaseException | str,
        *,
        retryable: bool,
        max_retries: int,
    ) -> bool:
        """Fail one generation and return whether a retry was queued."""
        if (
            isinstance(max_retries, bool)
            or not isinstance(max_retries, int)
            or max_retries < 0
        ):
            raise ValueError("max_retries must be a non-negative integer")
        message = str(error)
        with self._transaction() as conn:
            row = conn.execute(
                "SELECT * FROM entries WHERE source_index=?", (request.source_index,)
            ).fetchone()
            if (
                row is None
                or CaptureState(row["state"])
                not in (CaptureState.CAPTURING, CaptureState.COMMITTING)
                or int(row["generation"]) != request.generation
            ):
                raise RuntimeError(
                    f"capture failure transition mismatch for {request.key}"
                )
            failures = int(row["retry_count"]) + 1
            demand = conn.execute(
                "SELECT 1 FROM interests WHERE source_index=? AND kind='demand' "
                "LIMIT 1",
                (request.source_index,),
            ).fetchone()
            interested = conn.execute(
                "SELECT 1 FROM interests WHERE source_index=? LIMIT 1",
                (request.source_index,),
            ).fetchone()
            should_retry = bool(
                retryable and failures <= max_retries and interested is not None
            )
            if should_retry:
                priority = (
                    CapturePriority.DEMAND if demand else CapturePriority.PREFETCH
                )
                conn.execute(
                    "UPDATE entries SET state=?,generation=generation+1,priority=?,"
                    "queued_at=?,captured_at=NULL,reserved_bytes=0,retry_count=?,"
                    "last_error=? WHERE source_index=?",
                    (
                        CaptureState.QUEUED.value,
                        int(priority),
                        self._clock(),
                        failures,
                        message,
                        request.source_index,
                    ),
                )
            else:
                conn.execute(
                    "UPDATE entries SET state=?,priority=NULL,queued_at=NULL,"
                    "captured_at=NULL,reserved_bytes=0,retry_count=?,last_error=? "
                    "WHERE source_index=?",
                    (
                        CaptureState.FAILED.value,
                        failures,
                        message,
                        request.source_index,
                    ),
                )
                self._prune_orphans_locked(conn)
            return should_retry

    def release_and_advance(
        self, consumer_id: str, leases: Sequence[CaptureReadLease]
    ) -> int:
        """Release read protection and advance only the contiguous ACK prefix."""
        if not leases:
            return self.consumer_cursor(consumer_id)
        if len({lease.token for lease in leases}) != len(leases):
            raise ValueError("leases contains duplicate tokens")
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] not in self._ACTIVE_CONSUMER_STATES:
                raise RuntimeError(f"consumer {consumer_id!r} is {consumer['state']!r}")
            for lease in leases:
                row = conn.execute(
                    "SELECT * FROM read_leases WHERE token=?", (lease.token,)
                ).fetchone()
                observed = None
                if row is not None:
                    observed = (
                        row["consumer_id"],
                        int(row["source_index"]),
                        int(row["generation"]),
                    )
                expected = (consumer_id, lease.source_index, lease.generation)
                if observed != expected:
                    raise RuntimeError(
                        f"unknown, released, or mismatched lease {lease.token}"
                    )
                conn.execute("DELETE FROM read_leases WHERE token=?", (lease.token,))
                conn.execute(
                    "INSERT OR IGNORE INTO completed_samples VALUES(?,?)",
                    (consumer_id, lease.source_index),
                )
                self._drop_demand_if_idle_locked(conn, consumer_id, lease.source_index)

            cursor = int(consumer["cursor"])
            total = int(consumer["total_samples"])
            while cursor < total:
                completed = conn.execute(
                    "SELECT 1 FROM completed_samples WHERE consumer_id=? AND "
                    "source_index=?",
                    (consumer_id, cursor),
                ).fetchone()
                if completed is None:
                    break
                conn.execute(
                    "DELETE FROM completed_samples WHERE consumer_id=? AND "
                    "source_index=?",
                    (consumer_id, cursor),
                )
                cursor += 1
            state = "eof" if cursor == total else consumer["state"]
            now = self._clock()
            conn.execute(
                "UPDATE consumers SET cursor=?,state=?,updated_at=? "
                "WHERE consumer_id=?",
                (cursor, state, now, consumer_id),
            )
            self._refresh_window_locked(conn, consumer_id)
            return cursor

    def abandon_leases(
        self, consumer_id: str, leases: Sequence[CaptureReadLease]
    ) -> None:
        """Release read protection without recording consumer progress."""
        if not leases:
            return
        with self._transaction() as conn:
            for lease in leases:
                row = conn.execute(
                    "SELECT * FROM read_leases WHERE token=?", (lease.token,)
                ).fetchone()
                if row is None:
                    continue
                if (
                    row["consumer_id"] != consumer_id
                    or int(row["source_index"]) != lease.source_index
                    or int(row["generation"]) != lease.generation
                ):
                    raise RuntimeError(
                        f"read lease identity mismatch while abandoning {lease.token}"
                    )
                conn.execute("DELETE FROM read_leases WHERE token=?", (lease.token,))
                self._drop_demand_if_idle_locked(conn, consumer_id, lease.source_index)
            self._prune_orphans_locked(conn)

    def begin_evictions(
        self, *, limit: int = 64, pressure: bool = False
    ) -> tuple[EvictionCandidate, ...]:
        """Atomically claim reclaimable captures.

        Normal eviction requires no interest.  Pressure eviction may drop soft
        window cache entries, but never a demand waiter or read lease.
        """
        if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
            raise ValueError("limit must be a positive integer")
        with self._transaction() as conn:
            if pressure:
                interest_clause = (
                    "NOT EXISTS(SELECT 1 FROM interests i WHERE i.source_index="
                    "e.source_index AND i.kind='demand')"
                )
            else:
                interest_clause = (
                    "NOT EXISTS(SELECT 1 FROM interests i WHERE i.source_index="
                    "e.source_index)"
                )
            rows = conn.execute(
                f"SELECT e.* FROM entries e WHERE e.state=? AND {interest_clause} "
                "AND NOT EXISTS(SELECT 1 FROM waiters w WHERE w.source_index="
                "e.source_index) AND NOT EXISTS(SELECT 1 FROM read_leases l "
                "WHERE l.source_index=e.source_index) ORDER BY e.source_index "
                "LIMIT ?",
                (CaptureState.READY.value, limit),
            ).fetchall()
            candidates: list[EvictionCandidate] = []
            for row in rows:
                if not row["ref_json"]:
                    raise RuntimeError("READY capture selected for eviction has no ref")
                conn.execute(
                    "UPDATE entries SET state=?,prefetch_suppressed=? "
                    "WHERE source_index=?",
                    (
                        CaptureState.EVICTING.value,
                        int(pressure),
                        row["source_index"],
                    ),
                )
                candidates.append(
                    EvictionCandidate(
                        key=self._key(conn, int(row["source_index"])),
                        source_index=int(row["source_index"]),
                        generation=int(row["generation"]),
                        ref=ref_from_dict(json.loads(row["ref_json"])),
                    )
                )
            return tuple(candidates)

    def finish_evictions(self, candidates: Sequence[EvictionCandidate]) -> None:
        if not candidates:
            return
        with self._transaction() as conn:
            affected: set[str] = set()
            for candidate in candidates:
                row = conn.execute(
                    "SELECT * FROM entries WHERE source_index=?",
                    (candidate.source_index,),
                ).fetchone()
                if (
                    row is None
                    or CaptureState(row["state"]) != CaptureState.EVICTING
                    or int(row["generation"]) != candidate.generation
                    or row["source_sample_id"] != candidate.key.source_sample_id
                ):
                    raise RuntimeError(
                        f"eviction completion mismatch for {candidate.key}"
                    )
                conn.execute(
                    "UPDATE entries SET state=?,priority=NULL,queued_at=NULL,"
                    "captured_at=NULL,reserved_bytes=0,estimated_bytes=0,"
                    "ref_json=NULL WHERE source_index=?",
                    (CaptureState.ABSENT.value, candidate.source_index),
                )
                interests = conn.execute(
                    "SELECT consumer_id,kind FROM interests WHERE source_index=?",
                    (candidate.source_index,),
                ).fetchall()
                if any(item["kind"] == "demand" for item in interests):
                    updated = conn.execute(
                        "SELECT * FROM entries WHERE source_index=?",
                        (candidate.source_index,),
                    ).fetchone()
                    self._queue_entry(conn, updated, CapturePriority.DEMAND)
                affected.update(item["consumer_id"] for item in interests)
            for consumer_id in affected:
                self._top_up_prefetch_locked(conn, consumer_id)

    def cancel_evictions(
        self, candidates: Sequence[EvictionCandidate], error: BaseException | str
    ) -> None:
        if not candidates:
            return
        with self._transaction() as conn:
            for candidate in candidates:
                row = conn.execute(
                    "SELECT state,generation FROM entries WHERE source_index=?",
                    (candidate.source_index,),
                ).fetchone()
                if (
                    row is not None
                    and CaptureState(row["state"]) == CaptureState.EVICTING
                    and int(row["generation"]) == candidate.generation
                ):
                    conn.execute(
                        "UPDATE entries SET state=?,last_error=? WHERE source_index=?",
                        (
                            CaptureState.READY.value,
                            str(error),
                            candidate.source_index,
                        ),
                    )

    def reclaim(
        self,
        store: "FeatureStore",
        *,
        limit: int = 64,
        pressure: bool = False,
        reason: str = "window-expired",
    ) -> int:
        """Reclaim eligible payloads and commit metadata eviction afterward."""
        candidates = self.begin_evictions(limit=limit, pressure=pressure)
        completed: list[EvictionCandidate] = []
        try:
            for candidate in candidates:
                store.reclaim(candidate.ref, reason=reason)
                completed.append(candidate)
        except BaseException as error:
            if completed:
                self.finish_evictions(completed)
            pending = candidates[len(completed) :]
            self.cancel_evictions(pending, error)
            raise
        self.finish_evictions(completed)
        return len(completed)

    def cancel_queued_prefetch(self, limit: Optional[int] = None) -> int:
        if limit is not None and (
            isinstance(limit, bool) or not isinstance(limit, int) or limit < 0
        ):
            raise ValueError("limit must be a non-negative integer or None")
        with self._transaction() as conn:
            sql = (
                "SELECT source_index FROM entries WHERE state=? AND priority=? "
                "ORDER BY queued_at DESC"
            )
            params: list[Any] = [
                CaptureState.QUEUED.value,
                int(CapturePriority.PREFETCH),
            ]
            if limit is not None:
                sql += " LIMIT ?"
                params.append(limit)
            rows = conn.execute(sql, params).fetchall()
            for row in rows:
                conn.execute(
                    "UPDATE entries SET state=?,priority=NULL,queued_at=NULL,"
                    "prefetch_suppressed=1 WHERE source_index=?",
                    (CaptureState.ABSENT.value, row["source_index"]),
                )
            return len(rows)

    def resume_prefetch(self) -> None:
        with self._transaction() as conn:
            conn.execute("UPDATE entries SET prefetch_suppressed=0")
            consumers = conn.execute(
                "SELECT consumer_id FROM consumers WHERE state IN (?,?,?)",
                self._ACTIVE_CONSUMER_STATES,
            ).fetchall()
            for row in consumers:
                self._top_up_prefetch_locked(conn, row["consumer_id"])

    def _release_consumer_transients_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        conn.execute("DELETE FROM waiters WHERE consumer_id=?", (consumer_id,))
        conn.execute("DELETE FROM read_leases WHERE consumer_id=?", (consumer_id,))
        conn.execute(
            "DELETE FROM interests WHERE consumer_id=? AND kind='demand'",
            (consumer_id,),
        )
        self._prune_orphans_locked(conn)

    def _release_consumer_locked(
        self, conn: sqlite3.Connection, consumer_id: str
    ) -> None:
        self._release_consumer_transients_locked(conn, consumer_id)
        conn.execute("DELETE FROM interests WHERE consumer_id=?", (consumer_id,))
        conn.execute(
            "DELETE FROM completed_samples WHERE consumer_id=?", (consumer_id,)
        )
        self._prune_orphans_locked(conn)

    def complete_consumer(
        self, consumer_id: str, *, allow_partial: bool = False
    ) -> None:
        """Mark one consumer complete and release all of its cache interests.

        The default remains strict and requires the canonical stream to drain.
        ``allow_partial`` is reserved for an explicit launcher step budget: the
        consumer completed its configured work, even though other consumers may
        continue farther through the shared source stream.
        """
        if not isinstance(allow_partial, bool):
            raise TypeError("allow_partial must be a bool")
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] == "completed":
                return
            if consumer["state"] == "failed":
                raise RuntimeError(f"cannot complete failed consumer {consumer_id!r}")
            outstanding = conn.execute(
                "SELECT (SELECT COUNT(*) FROM waiters WHERE consumer_id=?) + "
                "(SELECT COUNT(*) FROM read_leases WHERE consumer_id=?)",
                (consumer_id, consumer_id),
            ).fetchone()[0]
            if not allow_partial and int(consumer["cursor"]) != int(
                consumer["total_samples"]
            ):
                raise RuntimeError(
                    f"consumer {consumer_id!r} completed at cursor "
                    f"{consumer['cursor']}/{consumer['total_samples']}"
                )
            if outstanding and not allow_partial:
                raise RuntimeError(
                    f"consumer {consumer_id!r} completed with {outstanding} "
                    "outstanding acquisitions"
                )
            self._release_consumer_locked(conn, consumer_id)
            now = self._clock()
            conn.execute(
                "UPDATE consumers SET state='completed',heartbeat_at=?,updated_at=? "
                "WHERE consumer_id=?",
                (now, now, consumer_id),
            )

    def fail_consumer(self, consumer_id: str, error: BaseException | str) -> None:
        with self._transaction() as conn:
            consumer = conn.execute(
                "SELECT * FROM consumers WHERE consumer_id=?", (consumer_id,)
            ).fetchone()
            if consumer is None:
                raise KeyError(f"unknown consumer {consumer_id!r}")
            if consumer["state"] == "failed":
                return
            if consumer["state"] == "completed":
                raise RuntimeError(f"cannot fail completed consumer {consumer_id!r}")
            self._release_consumer_locked(conn, consumer_id)
            now = self._clock()
            conn.execute(
                "UPDATE consumers SET state='failed',failure=?,heartbeat_at=?,"
                "updated_at=? WHERE consumer_id=?",
                (str(error), now, now, consumer_id),
            )

    def expire_consumers(self, timeout_s: float) -> tuple[str, ...]:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        cutoff = self._clock() - timeout_s
        with self._transaction() as conn:
            rows = conn.execute(
                "SELECT consumer_id FROM consumers WHERE state IN (?,?,?) AND "
                "heartbeat_at < ? ORDER BY consumer_id",
                (*self._ACTIVE_CONSUMER_STATES, cutoff),
            ).fetchall()
            expired = tuple(row["consumer_id"] for row in rows)
            for consumer_id in expired:
                self._release_consumer_locked(conn, consumer_id)
                conn.execute(
                    "UPDATE consumers SET state='failed',failure=?,updated_at=? "
                    "WHERE consumer_id=?",
                    ("heartbeat expired", self._clock(), consumer_id),
                )
            return expired

    def wait_for_consumers(self, timeout_s: float) -> bool:
        if timeout_s <= 0:
            raise ValueError("timeout_s must be > 0")
        deadline = time.monotonic() + timeout_s
        while True:
            snapshot = self.snapshot()
            if set(snapshot["consumers"]) == set(snapshot["expected_consumers"]):
                return True
            if time.monotonic() >= deadline:
                return False
            time.sleep(min(self.poll_s, max(0.0, deadline - time.monotonic())))

    def finalize_run(self) -> str:
        """Persist terminal status only after consumers and payloads drain."""
        with self._transaction() as conn:
            run = self._run(conn)
            consumers = conn.execute(
                "SELECT consumer_id,state FROM consumers ORDER BY consumer_id"
            ).fetchall()
            expected = set(json.loads(run["expected_consumers_json"]))
            observed = {row["consumer_id"] for row in consumers}
            if observed != expected:
                raise RuntimeError(
                    f"cannot finalize before all consumers register: "
                    f"missing={sorted(expected - observed)}"
                )
            nonterminal = [
                row["consumer_id"]
                for row in consumers
                if row["state"] not in ("completed", "failed")
            ]
            if nonterminal:
                raise RuntimeError(f"nonterminal consumers remain: {nonterminal}")
            inventory = conn.execute(
                "SELECT state,COUNT(*) AS count FROM entries WHERE state != ? "
                "GROUP BY state",
                (CaptureState.ABSENT.value,),
            ).fetchall()
            outstanding = conn.execute(
                "SELECT (SELECT COUNT(*) FROM waiters) + "
                "(SELECT COUNT(*) FROM read_leases) + "
                "(SELECT COUNT(*) FROM interests)"
            ).fetchone()[0]
            if inventory or outstanding:
                detail = {row["state"]: int(row["count"]) for row in inventory}
                raise RuntimeError(
                    f"registry inventory did not drain: entries={detail}, "
                    f"outstanding={outstanding}"
                )
            status = (
                "completed_with_failures"
                if any(row["state"] == "failed" for row in consumers)
                else "completed"
            )
            conn.execute("UPDATE run_config SET status=? WHERE singleton=1", (status,))
            return status

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            run = self._conn.execute(
                "SELECT * FROM run_config WHERE singleton=1"
            ).fetchone()
            if run is None:
                raise RuntimeError("windowed capture run is not initialized")
            consumers = self._conn.execute(
                "SELECT * FROM consumers ORDER BY consumer_id"
            ).fetchall()
            entry_counts = self._conn.execute(
                "SELECT state,COUNT(*) AS count FROM entries GROUP BY state"
            ).fetchall()
            queued_counts = self._conn.execute(
                "SELECT priority,COUNT(*) AS count FROM entries WHERE state=? "
                "GROUP BY priority",
                (CaptureState.QUEUED.value,),
            ).fetchall()
            live_refs, live_bytes = self._live_usage_locked(self._conn)
            waiters = int(
                self._conn.execute("SELECT COUNT(*) FROM waiters").fetchone()[0]
            )
            leases = int(
                self._conn.execute("SELECT COUNT(*) FROM read_leases").fetchone()[0]
            )
            interests = int(
                self._conn.execute("SELECT COUNT(*) FROM interests").fetchone()[0]
            )
            capture_count = self._meta_int(self._conn, "capture_count")
            recapture_count = self._meta_int(self._conn, "recapture_count")
            peak_live_refs = self._meta_int(self._conn, "peak_live_refs")
            peak_live_bytes = self._meta_int(self._conn, "peak_live_bytes")
        priority_names = {
            int(CapturePriority.DEMAND): "demand",
            int(CapturePriority.PREFETCH): "prefetch",
        }
        return {
            "run_id": run["run_id"],
            "status": run["status"],
            "total_samples": int(run["total_samples"]),
            "expected_consumers": tuple(json.loads(run["expected_consumers_json"])),
            "consumers": {
                row["consumer_id"]: {
                    "cursor": int(row["cursor"]),
                    "state": row["state"],
                    "lookbehind": int(row["lookbehind"]),
                    "lookahead": int(row["lookahead"]),
                    "prefetch_depth": int(row["prefetch_depth"]),
                    "max_outstanding": int(row["max_outstanding"]),
                    "failure": row["failure"],
                }
                for row in consumers
            },
            "entries": {row["state"]: int(row["count"]) for row in entry_counts},
            "queued": {
                priority_names[int(row["priority"])]: int(row["count"])
                for row in queued_counts
            },
            "waiters": waiters,
            "leases": leases,
            "interests": interests,
            "live_refs": live_refs,
            "live_bytes": live_bytes,
            "max_live_refs": self.max_live_refs,
            "max_live_bytes": self.max_live_bytes,
            "capture_count": capture_count,
            "recapture_count": recapture_count,
            "peak_live_refs": peak_live_refs,
            "peak_live_bytes": peak_live_bytes,
        }

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._conn.close()
            self._closed = True


class WindowedCaptureQueue:
    """Ordered ``SampleRefQueue`` facade for one logical consumer."""

    def __init__(
        self,
        registry: SQLiteWindowedCaptureRegistry,
        consumer_id: str,
        *,
        idle_timeout_s: Optional[float] = 1800.0,
        record_refs: Optional[Callable[[Sequence[SampleRef]], None]] = None,
    ) -> None:
        if idle_timeout_s is not None and idle_timeout_s <= 0:
            raise ValueError("idle_timeout_s must be > 0 or None")
        snapshot = registry.snapshot()
        if consumer_id not in snapshot["consumers"]:
            raise KeyError(f"unknown consumer {consumer_id!r}")
        self.registry = registry
        self.consumer_id = consumer_id
        self.total_samples = int(snapshot["total_samples"])
        self.idle_timeout_s = idle_timeout_s
        if record_refs is not None and not callable(record_refs):
            raise TypeError("record_refs must be callable or None")
        self._record_refs = record_refs
        self._next_fetch = int(snapshot["consumers"][consumer_id]["cursor"])
        self._leases: dict[str, CaptureReadLease] = {}
        self._closed = False
        self._get_lock = threading.Lock()
        self._state_lock = threading.RLock()
        self._metrics: dict[str, float | int] = {
            "refs": 0,
            "ready_at_request_refs": 0,
            "demand_wait_s": 0.0,
            "max_demand_wait_s": 0.0,
        }

    def get(self, n: int, timeout_s: float = 0.0) -> list[SampleRef]:
        del timeout_s  # Registry waits use the explicit idle timeout.
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError("n must be a positive integer")
        with self._get_lock:
            with self._state_lock:
                if self._closed:
                    return []
                start = self._next_fetch
                if start >= self.total_samples:
                    if not self._leases:
                        self.registry.complete_consumer(self.consumer_id)
                        self._closed = True
                    return []
                stop = min(self.total_samples, start + n)
            tickets = self.registry.request_many(self.consumer_id, range(start, stop))
            acquired: list[CaptureReadLease] = []
            try:
                for ticket in tickets:
                    acquired.append(
                        self.registry.wait_ready(ticket, timeout_s=self.idle_timeout_s)
                    )
            except BaseException:
                acquired_indices = {lease.source_index for lease in acquired}
                for ticket in tickets:
                    if ticket.source_index not in acquired_indices:
                        try:
                            self.registry.cancel_acquire(ticket)
                        except RuntimeError:
                            pass
                self.registry.abandon_leases(self.consumer_id, acquired)
                raise
            refs = [lease.ref for lease in acquired]
            if len({ref.sample_id for ref in refs}) != len(refs):
                self.registry.abandon_leases(self.consumer_id, acquired)
                raise RuntimeError(
                    "windowed capture batch contains duplicate sample IDs"
                )
            if self._record_refs is not None:
                try:
                    self._record_refs(refs)
                except BaseException:
                    self.registry.abandon_leases(self.consumer_id, acquired)
                    raise
            with self._state_lock:
                if self._closed:
                    self.registry.abandon_leases(self.consumer_id, acquired)
                    return []
                self._leases.update((lease.ref.sample_id, lease) for lease in acquired)
                self._next_fetch = stop
                for lease in acquired:
                    self._metrics["refs"] += 1
                    self._metrics["ready_at_request_refs"] += int(
                        lease.ready_at_request
                    )
                    self._metrics["demand_wait_s"] += lease.wait_s
                    self._metrics["max_demand_wait_s"] = max(
                        self._metrics["max_demand_wait_s"], lease.wait_s
                    )
            return refs

    def _resolve_leases(self, refs: Sequence[SampleRef]) -> list[CaptureReadLease]:
        missing = [ref.sample_id for ref in refs if ref.sample_id not in self._leases]
        if missing:
            raise RuntimeError(
                f"windowed queue references samples not leased: {missing}"
            )
        return [self._leases[ref.sample_id] for ref in refs]

    def ack(self, refs: list[SampleRef]) -> None:
        with self._state_lock:
            leases = self._resolve_leases(refs)
            self.registry.release_and_advance(self.consumer_id, leases)
            for ref in refs:
                self._leases.pop(ref.sample_id)

    def fail(self, refs: list[SampleRef], reason: str, retryable: bool) -> None:
        del reason
        with self._state_lock:
            leases = self._resolve_leases(refs)
            self.registry.abandon_leases(self.consumer_id, leases)
            for ref in refs:
                self._leases.pop(ref.sample_id)
            if retryable and leases:
                self._next_fetch = min(
                    self._next_fetch,
                    min(lease.source_index for lease in leases),
                )

    def depth(self) -> int:
        with self._state_lock:
            return max(0, self.total_samples - self._next_fetch)

    def in_flight(self) -> int:
        with self._state_lock:
            return len(self._leases)

    def metrics(self) -> dict[str, float | int]:
        with self._state_lock:
            refs = int(self._metrics["refs"])
            return {
                **self._metrics,
                "ready_at_request_ratio": (
                    float(self._metrics["ready_at_request_refs"]) / refs
                    if refs
                    else 0.0
                ),
                "mean_demand_wait_s": (
                    float(self._metrics["demand_wait_s"]) / refs if refs else 0.0
                ),
                "next_fetch": self._next_fetch,
                "in_flight": len(self._leases),
            }

    def drained(self) -> bool:
        with self._state_lock:
            return self._next_fetch == self.total_samples and not self._leases

    def finalize(self) -> None:
        self.complete()

    def complete(self, *, allow_partial: bool = False) -> None:
        with self._state_lock:
            if not allow_partial and (
                self._next_fetch != self.total_samples or self._leases
            ):
                raise RuntimeError(
                    f"consumer {self.consumer_id!r} queue did not drain: "
                    f"next={self._next_fetch}/{self.total_samples}, "
                    f"leases={len(self._leases)}"
                )
            if allow_partial and self._leases:
                self.registry.abandon_leases(
                    self.consumer_id, list(self._leases.values())
                )
                self._leases.clear()
            self.registry.complete_consumer(
                self.consumer_id, allow_partial=allow_partial
            )
            self._closed = True

    def close(self, error: Optional[BaseException | str] = None) -> None:
        with self._state_lock:
            if self._closed:
                return
            if self._leases:
                self.registry.abandon_leases(
                    self.consumer_id, list(self._leases.values())
                )
                self._leases.clear()
            if error is not None:
                self.registry.fail_consumer(self.consumer_id, error)
            self._closed = True


__all__ = [
    "AcquireTicket",
    "CaptureFailedError",
    "CaptureKey",
    "CapturePriority",
    "CaptureReadLease",
    "CaptureRequest",
    "CaptureState",
    "ConsumerFailedError",
    "EvictionCandidate",
    "SQLiteWindowedCaptureRegistry",
    "WindowedCaptureQueue",
    "capture_contract_digest",
]
