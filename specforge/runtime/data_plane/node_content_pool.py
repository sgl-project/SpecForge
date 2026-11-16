"""A node-wide pageable content pool with process-shared storage and directory.

SQLite transactions coordinate processes, not Python tensor dictionaries. Native
transport writes always land in the caller's registered staging slots; only a
completed CPU copy may publish a shared slot. A crashed fetcher cannot leave DMA
writing to a slot that a new generation will reuse.
"""
from __future__ import annotations

import fcntl
import mmap
import os
import sqlite3
import threading
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch


class PoolFetchFailed(RuntimeError):
    pass


def _alive(pid):
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False


class PoolPin:
    def __init__(self, pool, key, generation, token, tensor):
        self.pool, self.key, self.generation, self.token = pool, key, generation, token
        self.tensor = tensor

    def release(self):
        self.tensor = None
        if self.token is not None:
            token, self.token = self.token, None
            self.pool.release(self.key, self.generation, token)


class NodeContentPool:
    VERSION = 'node-content-v1'

    def __init__(self, directory, capacity_bytes=8 << 30, *, wait_timeout_s=60.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.total_capacity = int(capacity_bytes)
        # Directory and rollback journal get a bounded share of the node budget.
        # Tiny CPU test pools use an unbudgeted small directory for adversarial
        # slot tests; production-sized pools always reserve this headroom.
        self.metadata_budget = min(32 << 20, self.total_capacity // 128) if self.total_capacity >= (64 << 20) else 0
        self.capacity = self.total_capacity - self.metadata_budget
        self.max_entries = max(1, (self.metadata_budget - (256 << 10)) // 4096) if self.metadata_budget else None
        if self.capacity < 4096:
            raise ValueError('node pool capacity must be at least one 4096-byte page')
        self.wait_timeout = float(wait_timeout_s)
        self._lock = threading.RLock()
        self._closed = False
        self.client = uuid.uuid4().hex
        self.pid = os.getpid()
        self._gate = open(self.directory / 'pool.lock', 'a+b')
        with self._file_lock():
            self._db = sqlite3.connect(self.directory / 'directory.sqlite', timeout=30,
                                       isolation_level=None, check_same_thread=False)
            self._db.execute('PRAGMA journal_mode=DELETE')
            if self.metadata_budget:
                self._db.execute(f'PRAGMA max_page_count={max(32, (self.metadata_budget - 4096) // 8192)}')
            self._db.executescript('''
                CREATE TABLE IF NOT EXISTS config(version TEXT, capacity INTEGER);
                CREATE TABLE IF NOT EXISTS clients(id TEXT PRIMARY KEY, pid INTEGER);
                CREATE TABLE IF NOT EXISTS entries(key TEXT PRIMARY KEY, generation TEXT,
                    offset INTEGER, extent INTEGER, nbytes INTEGER, state TEXT,
                    writer TEXT, pid INTEGER, touched REAL, error TEXT);
                CREATE TABLE IF NOT EXISTS pins(token TEXT PRIMARY KEY, key TEXT,
                    generation TEXT, client TEXT, pid INTEGER);
                CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY, value INTEGER);
            ''')
            config = self._db.execute('SELECT version,capacity FROM config').fetchone()
            if config is None:
                self._db.execute('INSERT INTO config VALUES (?,?)', (self.VERSION, self.total_capacity))
            elif config != (self.VERSION, self.total_capacity):
                raise ValueError('node pool layout/capacity differs across processes')
            fd = os.open(self.directory / 'data.mmap', os.O_RDWR | os.O_CREAT, 0o600)
            try:
                os.ftruncate(fd, self.capacity)
                self._mapping = mmap.mmap(fd, self.capacity, access=mmap.ACCESS_WRITE)
            finally:
                os.close(fd)
            self._db.executemany('INSERT OR IGNORE INTO counters VALUES (?,0)', ((name,) for name in ('hits','cross_process_hits','misses','singleflight_waits','fallbacks','cpu_copy_bytes','evictions','fetch_failures','storage_peak_bytes')))
            self._db.execute('INSERT INTO clients VALUES (?,?)', (self.client, self.pid))
        self._last_health = {}

    @contextmanager
    def _file_lock(self):
        fcntl.flock(self._gate, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(self._gate, fcntl.LOCK_UN)

    @contextmanager
    def _transaction(self):
        with self._lock:
            if self._closed:
                raise RuntimeError('node content pool is closed')
            self._db.execute('BEGIN IMMEDIATE')
            try:
                yield self._db
                self._db.execute('COMMIT')
            except BaseException:
                self._db.execute('ROLLBACK')
                raise

    @staticmethod
    def _count(db, name, value=1):
        db.execute('INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value',
                   (name, int(value)))

    def _reap(self, db):
        pids = {row[0] for row in db.execute('SELECT pid FROM clients')}
        for pid in pids:
            if _alive(pid):
                continue
            db.execute('DELETE FROM pins WHERE pid=?', (pid,))
            db.execute('DELETE FROM clients WHERE pid=?', (pid,))
            db.execute("UPDATE entries SET state='FAILED', error='fetch process exited',extent=0 WHERE state='WRITING' AND pid=?", (pid,))

    def _reserve(self, db, extent):
        # Evict only READY, unpinned generations. Lookup and pin share this tx.
        while True:
            if self.max_entries is not None and db.execute('SELECT COUNT(*) FROM entries').fetchone()[0] >= self.max_entries:
                victim = db.execute("SELECT key FROM entries e WHERE state='READY' AND NOT EXISTS (SELECT 1 FROM pins p WHERE p.key=e.key AND p.generation=e.generation) ORDER BY touched LIMIT 1").fetchone()
                if victim is None:
                    return None
                db.execute('DELETE FROM entries WHERE key=?', victim)
                self._count(db, 'evictions')
            rows = list(db.execute("SELECT offset,extent FROM entries WHERE state IN ('READY','WRITING') ORDER BY offset"))
            offset = 0
            for start, amount in rows:
                if start - offset >= extent:
                    return offset
                offset = max(offset, start + amount)
            if self.capacity - offset >= extent:
                return offset
            victim = db.execute("SELECT key FROM entries e WHERE state='READY' AND NOT EXISTS (SELECT 1 FROM pins p WHERE p.key=e.key AND p.generation=e.generation) ORDER BY touched LIMIT 1").fetchone()
            if victim is None:
                return None
            db.execute('DELETE FROM entries WHERE key=?', victim)
            self._count(db, 'evictions')

    def acquire(self, key, nbytes, fetch):
        """Atomically lookup+pin, or singleflight one completed CPU fetch.

        Return None when all capacity is held by readers/writers. The caller
        then fetches one segment through staging directly into its final batch.
        """
        nbytes = int(nbytes)
        extent = (nbytes + 4095) // 4096 * 4096
        deadline = time.monotonic() + self.wait_timeout
        observed = None
        waited = False
        while True:
            writer = False
            with self._transaction() as db:
                self._reap(db)
                row = db.execute('SELECT generation,offset,nbytes,state,error,pid FROM entries WHERE key=?', (key,)).fetchone()
                if row is not None and row[2] != nbytes:
                    raise ValueError('content key changed byte layout')
                if row is not None and row[3] == 'READY':
                    generation, offset = row[:2]
                    token = uuid.uuid4().hex
                    db.execute('INSERT INTO pins VALUES (?,?,?,?,?)', (token, key, generation, self.client, self.pid))
                    db.execute('UPDATE entries SET touched=? WHERE key=?', (time.monotonic(), key))
                    self._count(db, 'hits')
                    if row[5] != self.pid:
                        self._count(db, 'cross_process_hits')
                    tensor = torch.frombuffer(self._mapping, dtype=torch.uint8, count=nbytes, offset=offset)
                    return PoolPin(self, key, generation, token, tensor)
                if row is not None and row[3] == 'FAILED':
                    raise PoolFetchFailed(row[4])
                if row is not None and row[3] == 'WRITING':
                    observed = row[0]
                    if not waited:
                        self._count(db, 'singleflight_waits')
                        waited = True
                else:
                    if extent > self.capacity:
                        self._count(db, 'fallbacks')
                        return None
                    offset = self._reserve(db, extent)
                    if offset is None:
                        self._count(db, 'fallbacks')
                        return None
                    generation = uuid.uuid4().hex
                    db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (key, generation, offset, extent, nbytes, 'WRITING', self.client, self.pid, time.monotonic(), ''))
                    self._count(db, 'misses')
                    current = db.execute("SELECT COALESCE(SUM(extent),0) FROM entries WHERE state IN ('READY','WRITING')").fetchone()[0]
                    old = db.execute("SELECT value FROM counters WHERE name='storage_peak_bytes'").fetchone()
                    if old is None or old[0] < current:
                        db.execute("INSERT OR REPLACE INTO counters VALUES ('storage_peak_bytes',?)", (current,))
                    writer = True
            if writer:
                shared = None
                source_release = None
                try:
                    source = fetch()
                    if isinstance(source, tuple):
                        source, source_release = source
                    if source.device.type != 'cpu' or not source.is_contiguous() or source.numel() * source.element_size() != nbytes:
                        raise ValueError('node pool fetch must return completed contiguous CPU storage')
                    shared = torch.frombuffer(self._mapping, dtype=torch.uint8, count=nbytes, offset=offset)
                    shared.copy_(source.view(torch.uint8).reshape(-1), non_blocking=False)
                    source = None
                    if source_release is not None:
                        source_release()
                        source_release = None
                    with self._transaction() as db:
                        changed = db.execute("UPDATE entries SET state='READY' WHERE key=? AND generation=? AND state='WRITING' AND writer=?", (key, generation, self.client)).rowcount
                        if changed != 1:
                            raise RuntimeError('shared writer generation was fenced')
                        token = uuid.uuid4().hex
                        db.execute('INSERT INTO pins VALUES (?,?,?,?,?)', (token, key, generation, self.client, self.pid))
                        self._count(db, 'cpu_copy_bytes', nbytes)
                        committed = (self.directory / 'data.mmap').stat().st_blocks * 512
                        old_peak = db.execute("SELECT value FROM counters WHERE name='storage_peak_bytes'").fetchone()[0]
                        if committed > old_peak:
                            db.execute("UPDATE counters SET value=? WHERE name='storage_peak_bytes'", (committed,))
                    return PoolPin(self, key, generation, token, shared)
                except BaseException as exc:
                    shared = None
                    source = None
                    if source_release is not None:
                        source_release()
                        source_release = None
                    with self._transaction() as db:
                        db.execute("UPDATE entries SET state='FAILED',extent=0,error=? WHERE key=? AND generation=? AND writer=?", (str(exc), key, generation, self.client))
                        self._count(db, 'fetch_failures')
                    raise
            if time.monotonic() >= deadline:
                raise TimeoutError('node content singleflight did not complete')
            time.sleep(0.01)

    def release(self, key, generation, token):
        with self._transaction() as db:
            db.execute('DELETE FROM pins WHERE token=? AND key=? AND generation=? AND client=?',
                       (token, key, generation, self.client))

    def health(self):
        if self._closed:
            return {**self._last_health, 'node_pool_closed': True}
        with self._transaction() as db:
            self._reap(db)
            result = {f'node_pool_{name}': value for name, value in db.execute('SELECT name,value FROM counters')}
            result.update(node_pool_enabled=True, node_pool_capacity_bytes=self.total_capacity,
                          node_pool_data_capacity_bytes=self.capacity, node_pool_metadata_budget_bytes=self.metadata_budget,
                          node_pool_committed_file_bytes=(self.directory / 'data.mmap').stat().st_blocks * 512,
                          node_pool_metadata_file_bytes=sum(path.stat().st_size for path in self.directory.iterdir() if path.name.startswith('directory.sqlite')),
                          node_pool_storage_bytes=db.execute("SELECT COALESCE(SUM(extent),0) FROM entries WHERE state IN ('READY','WRITING')").fetchone()[0],
                          node_pool_inflight_bytes=db.execute("SELECT COALESCE(SUM(extent),0) FROM entries WHERE state='WRITING'").fetchone()[0],
                          node_pool_active_pins=db.execute('SELECT COUNT(*) FROM pins').fetchone()[0],
                          node_pool_clients=db.execute('SELECT COUNT(*) FROM clients').fetchone()[0],
                          node_pool_quarantined_bytes=0, node_pool_closed=False,
                          node_pool_backing='file-mmap-pageable')
            return result

    def close(self):
        if self._closed:
            return
        with self._file_lock():
            with self._transaction() as db:
                self._reap(db)
                if db.execute('SELECT COUNT(*) FROM pins WHERE client=?', (self.client,)).fetchone()[0]:
                    raise RuntimeError('cannot close node content pool with active pins')
                if db.execute("SELECT COUNT(*) FROM entries WHERE state='WRITING' AND writer=?", (self.client,)).fetchone()[0]:
                    raise RuntimeError('cannot close node content pool with active writers')
                db.execute('DELETE FROM clients WHERE id=?', (self.client,))
                remaining = db.execute('SELECT COUNT(*) FROM clients').fetchone()[0]
                if not remaining:
                    db.execute('DELETE FROM entries')
            self._last_health = self.health()
            self._mapping.close()
            self._db.close()
            self._closed = True
            if not remaining:
                for name in ('data.mmap', 'directory.sqlite', 'directory.sqlite-journal'):
                    (self.directory / name).unlink(missing_ok=True)
                self._last_health['node_pool_committed_file_bytes'] = 0
                self._last_health['node_pool_metadata_file_bytes'] = 0
                # Keep the empty lock inode until all potential launchers are done;
                # removing a locked inode would permit a second coordinator.
        self._gate.close()
