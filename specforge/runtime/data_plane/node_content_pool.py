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

from .pool_diagnostics import get_diagnostics


def node_pool_budgets(capacity_bytes):
    """Partition the *total* node budget, including directory and proof costs.

    At >=8 GiB, entries cover one-MiB blocks filling the data region. Arbitrary
    tiny tail blocks can still exhaust the entry cap before data bytes. Pools
    below 64 MiB retain the unbudgeted-directory CPU-test exception; they are not
    production capacity experiments. Smaller budgeted pools scale proof space
    down rather than subtracting a fixed 14 MiB from insufficient metadata.
    """
    total = int(capacity_bytes)
    if total < 4096:
        raise ValueError('node pool capacity must be at least one 4096-byte page')
    metadata = total // 128 if total >= (64 << 20) else 0
    headroom = metadata // 8
    proof = min(14 << 20, (metadata - headroom) // 2)
    directory = metadata - headroom - proof
    return {
        'capacity_bytes': total,
        'data_capacity_bytes': total - metadata,
        'metadata_budget_bytes': metadata,
        'directory_budget_bytes': directory,
        'terminal_proof_budget_bytes': proof,
        'metadata_runtime_headroom_bytes': headroom,
        'max_entries': max(1, (directory - (256 << 10)) // 4096) if metadata else None,
        # A rollback page record includes eight bytes beyond the database page.
        'max_page_count': max(8, (directory - (64 << 10)) // (2 * 4096 + 8)) if metadata else None,
        'budget_mode': 'total-including-metadata' if metadata else 'tiny-cpu-test-unbudgeted-directory',
    }


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
    # Different metadata/data partition: never join a baseline630 directory.
    VERSION = 'node-content-diagnostics-v1'

    def __init__(self, directory, capacity_bytes=8 << 30, *, wait_timeout_s=60.0):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.budgets = node_pool_budgets(capacity_bytes)
        self.total_capacity = self.budgets['capacity_bytes']
        self.metadata_budget = self.budgets['metadata_budget_bytes']
        self.capacity = self.budgets['data_capacity_bytes']
        self.metadata_runtime_headroom = self.budgets['metadata_runtime_headroom_bytes']
        self.directory_budget = self.budgets['directory_budget_bytes']
        self.terminal_proof_budget = self.budgets['terminal_proof_budget_bytes']
        self.max_entries = self.budgets['max_entries']
        self.max_page_count = self.budgets['max_page_count']
        self._diag = get_diagnostics().component('node_pool')
        self._transaction_sequence = 0
        self._observed_metadata = {}
        self.wait_timeout = float(wait_timeout_s)
        self._lock = threading.RLock()
        self._closed = False
        self.client = uuid.uuid4().hex
        self.pid = os.getpid()
        self._gate = open(self.directory / 'pool.lock', 'a+b')
        with self._file_lock():
            self._db = sqlite3.connect(self.directory / 'directory.sqlite', timeout=30,
                                       isolation_level=None, check_same_thread=False)
            self._db.execute('PRAGMA page_size=4096')
            if self._db.execute('PRAGMA page_size').fetchone()[0] != 4096:
                raise ValueError('node directory requires 4096-byte SQLite pages')
            self._db.execute('PRAGMA cache_size=-64')
            self._db.execute('PRAGMA journal_mode=DELETE')
            if self.metadata_budget:
                self._db.execute(f'PRAGMA max_page_count={self.max_page_count}')
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
    def _transaction(self, scope='unspecified'):
        # Disabled diagnostics retain the original transaction path. Only
        # explicit transactions are counted; initialization uses autocommit.
        if not self._diag.enabled:
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
            return
        diag = self._diag
        diag.count('transaction_calls')
        diag.count(f'{scope}.transaction_calls')
        lock_start = time.perf_counter_ns()
        with self._lock:
            diag.add_ns('python_lock_wait', time.perf_counter_ns() - lock_start)
            if self._closed:
                raise RuntimeError('node content pool is closed')
            begin_start = time.perf_counter_ns()
            diag.count('begin_calls')
            try:
                self._db.execute('BEGIN IMMEDIATE')
            except BaseException:
                diag.count('begin_errors')
                raise
            finally:
                diag.add_ns('begin', time.perf_counter_ns() - begin_start)
            diag.count('begin_success')
            self._transaction_sequence += 1
            body_start = time.perf_counter_ns()
            try:
                try:
                    yield self._db
                finally:
                    diag.add_ns('transaction_body', time.perf_counter_ns() - body_start)
                if self._transaction_sequence % 64 == 0:
                    self._sample_metadata(self._db, 'periodic')
                commit_start = time.perf_counter_ns()
                diag.count('commit_calls')
                diag.count(f'{scope}.commit_calls')
                try:
                    self._db.execute('COMMIT')
                except BaseException:
                    diag.count('commit_errors')
                    raise
                finally:
                    diag.add_ns('commit', time.perf_counter_ns() - commit_start)
                diag.count('commit_success')
                diag.count(f'{scope}.commit_success')
            except BaseException:
                diag.count('transaction_errors')
                rollback_start = time.perf_counter_ns()
                diag.count('rollback_calls')
                try:
                    self._db.execute('ROLLBACK')
                finally:
                    diag.add_ns('rollback', time.perf_counter_ns() - rollback_start)
                raise

    def _sample_metadata(self, db, source):
        """Observe within an existing transaction; never add a transaction.

        File/journal/page peaks are sampled observed maxima, not exhaustive
        peaks. Periodic sampling runs before COMMIT every 64 successful BEGINs;
        health observations are counted separately and also precede COMMIT.
        """
        if not self._diag.enabled:
            return
        start = time.perf_counter_ns()
        self._diag.count(f'metadata_{source}_samples')
        try:
            values = {}
            for name, filename in (('directory_file_bytes', 'directory.sqlite'),
                                   ('journal_file_bytes', 'directory.sqlite-journal')):
                try:
                    values[name] = (self.directory / filename).stat().st_size
                except FileNotFoundError:
                    values[name] = 0
            values['metadata_file_bytes'] = values['directory_file_bytes'] + values['journal_file_bytes']
            values['directory_page_count'] = db.execute('PRAGMA page_count').fetchone()[0]
            values['entry_count'] = db.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
            for name, value in values.items():
                self._diag.max(name + '_observed_max', value)
                self._observed_metadata[name] = max(self._observed_metadata.get(name, 0), value)
        except Exception:
            # Observation failure must not change writer/pin/transaction state.
            self._diag.count('metadata_observation_errors')
        finally:
            self._diag.add_ns('metadata_observation', time.perf_counter_ns() - start)

    @staticmethod
    def _count(db, name, value=1):
        db.execute('INSERT INTO counters VALUES (?,?) ON CONFLICT(name) DO UPDATE SET value=value+excluded.value',
                   (name, int(value)))

    def _reap(self, db):
        pids = {row[0] for row in db.execute('SELECT pid FROM clients')}
        if self._diag.enabled:
            self._diag.count('reap_calls')
            self._diag.count('reap_pids_checked', len(pids))
        for pid in pids:
            if _alive(pid):
                continue
            deleted_pins = db.execute('DELETE FROM pins WHERE pid=?', (pid,)).rowcount
            db.execute('DELETE FROM clients WHERE pid=?', (pid,))
            failed_writers = db.execute("UPDATE entries SET state='FAILED', error='fetch process exited',extent=0 WHERE state='WRITING' AND pid=?", (pid,)).rowcount
            if self._diag.enabled:
                self._diag.count('reaped_pids')
                self._diag.count('reaped_pins', deleted_pins)
                self._diag.count('reaped_writers', failed_writers)

    def _victim(self, db, reason):
        start = time.perf_counter_ns() if self._diag.enabled else 0
        if self._diag.enabled:
            self._diag.count('victim_queries')
            self._diag.count(f'victim_queries_{reason}')
        try:
            victim = db.execute("SELECT key FROM entries e WHERE state='READY' AND NOT EXISTS (SELECT 1 FROM pins p WHERE p.key=e.key AND p.generation=e.generation) ORDER BY touched LIMIT 1").fetchone()
            if victim is None and self._diag.enabled:
                self._diag.count(f'no_eligible_victim_{reason}')
            return victim
        finally:
            if self._diag.enabled:
                self._diag.add_ns('victim_query', time.perf_counter_ns() - start)

    def _evict(self, db, victim, reason):
        db.execute('DELETE FROM entries WHERE key=?', victim)
        self._count(db, 'evictions')
        if self._diag.enabled:
            self._diag.count('evictions')
            self._diag.count(f'evictions_{reason}')

    def _reserve(self, db, extent):
        # Evict only READY, unpinned generations. Lookup and pin share this tx.
        if self._diag.enabled:
            self._diag.count('reserve_calls')
        start_time = time.perf_counter_ns() if self._diag.enabled else 0
        try:
            return self._reserve_impl(db, extent)
        finally:
            if self._diag.enabled:
                self._diag.add_ns('reserve', time.perf_counter_ns() - start_time)

    def _reserve_impl(self, db, extent):
        while True:
            if self._diag.enabled:
                self._diag.count('reserve_iterations')
            if self.max_entries is not None:
                entries = db.execute('SELECT COUNT(*) FROM entries').fetchone()[0]
                if self._diag.enabled:
                    self._diag.max('entry_count_observed_max', entries)
                if entries >= self.max_entries:
                    if self._diag.enabled:
                        self._diag.count('reserve_pressure_entry_cap')
                    victim = self._victim(db, 'entry_cap')
                    if victim is None:
                        return None
                    self._evict(db, victim, 'entry_cap')
                    entries -= 1
            query_start = time.perf_counter_ns() if self._diag.enabled else 0
            rows = list(db.execute("SELECT offset,extent FROM entries WHERE state IN ('READY','WRITING') ORDER BY offset"))
            if self._diag.enabled:
                self._diag.count('offset_queries')
                self._diag.count('offset_rows_returned', len(rows))
                self._diag.add_ns('offset_query', time.perf_counter_ns() - query_start)
            scan_start = time.perf_counter_ns() if self._diag.enabled else 0
            offset = 0
            if not self._diag.enabled:
                # Keep the baseline630 scan loop when observations are off.
                for start, amount in rows:
                    if start - offset >= extent:
                        return offset
                    offset = max(offset, start + amount)
            else:
                visited = 0
                for visited, (start, amount) in enumerate(rows, 1):
                    if start - offset >= extent:
                        self._diag.count('offset_rows_visited', visited)
                        self._diag.add_ns('offset_scan', time.perf_counter_ns() - scan_start)
                        if self.max_entries is not None:
                            self._entry_count_before_insert = entries
                        return offset
                    offset = max(offset, start + amount)
                self._diag.count('offset_rows_visited', visited)
                self._diag.add_ns('offset_scan', time.perf_counter_ns() - scan_start)
            if self.capacity - offset >= extent:
                if self._diag.enabled and self.max_entries is not None:
                    self._entry_count_before_insert = entries
                return offset
            reason = 'disabled'
            if self._diag.enabled:
                # This additional observation pass is needed only after the
                # original scan found no adequate hole, never on a hit/fit.
                classification_start = time.perf_counter_ns()
                self._diag.count('reason_classification_calls')
                occupied = sum(amount for _, amount in rows)
                reason = 'fragmentation' if self.capacity - occupied >= extent else 'capacity'
                self._diag.count(f'reserve_pressure_{reason}')
                self._diag.add_ns('reason_classification', time.perf_counter_ns() - classification_start)
            victim = self._victim(db, reason)
            if victim is None:
                return None
            self._evict(db, victim, reason)

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
        if self._diag.enabled:
            self._diag.count('acquire_calls')
        while True:
            writer = False
            if self._diag.enabled:
                self._diag.count('acquire_iterations')
            with self._transaction('lookup') as db:
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
                    if self._diag.enabled:
                        self._diag.count('writing_poll_iterations')
                    if not waited:
                        self._count(db, 'singleflight_waits')
                        waited = True
                else:
                    if extent > self.capacity:
                        self._count(db, 'fallbacks')
                        if self._diag.enabled:
                            self._diag.count('fallbacks_oversize')
                        return None
                    offset = self._reserve(db, extent)
                    if offset is None:
                        self._count(db, 'fallbacks')
                        if self._diag.enabled:
                            self._diag.count('fallbacks_no_reservation')
                        return None
                    generation = uuid.uuid4().hex
                    db.execute('INSERT OR REPLACE INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)',
                               (key, generation, offset, extent, nbytes, 'WRITING', self.client, self.pid, time.monotonic(), ''))
                    self._count(db, 'misses')
                    if self._diag.enabled and self.max_entries is not None:
                        self._diag.max('entry_count_high_water', self._entry_count_before_insert + 1)
                    current = db.execute("SELECT COALESCE(SUM(extent),0) FROM entries WHERE state IN ('READY','WRITING')").fetchone()[0]
                    if self._diag.enabled:
                        self._diag.max('reserved_data_bytes_observed_max', current)
                    old = db.execute("SELECT value FROM counters WHERE name='storage_peak_bytes'").fetchone()
                    if old is None or old[0] < current:
                        db.execute("INSERT OR REPLACE INTO counters VALUES ('storage_peak_bytes',?)", (current,))
                    writer = True
            if writer:
                shared = None
                source_release = None
                try:
                    fetch_start = time.perf_counter_ns() if self._diag.enabled else 0
                    if self._diag.enabled:
                        self._diag.count('fetch_calls')
                    try:
                        source = fetch()
                    finally:
                        if self._diag.enabled:
                            self._diag.add_ns('fetch', time.perf_counter_ns() - fetch_start)
                    if isinstance(source, tuple):
                        source, source_release = source
                    if source.device.type != 'cpu' or not source.is_contiguous() or source.numel() * source.element_size() != nbytes:
                        raise ValueError('node pool fetch must return completed contiguous CPU storage')
                    shared = torch.frombuffer(self._mapping, dtype=torch.uint8, count=nbytes, offset=offset)
                    copy_start = time.perf_counter_ns() if self._diag.enabled else 0
                    if self._diag.enabled:
                        self._diag.count('shared_cpu_copy_calls')
                    try:
                        shared.copy_(source.view(torch.uint8).reshape(-1), non_blocking=False)
                    finally:
                        if self._diag.enabled:
                            self._diag.add_ns('shared_cpu_copy', time.perf_counter_ns() - copy_start)
                    if self._diag.enabled:
                        self._diag.count('shared_cpu_copy_completed_bytes', nbytes)
                    source = None
                    if source_release is not None:
                        source_release()
                        source_release = None
                    with self._transaction('fill') as db:
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
                    with self._transaction('failure') as db:
                        db.execute("UPDATE entries SET state='FAILED',extent=0,error=? WHERE key=? AND generation=? AND writer=?", (str(exc), key, generation, self.client))
                        self._count(db, 'fetch_failures')
                    raise
            if time.monotonic() >= deadline:
                if self._diag.enabled:
                    self._diag.count('singleflight_timeouts')
                raise TimeoutError('node content singleflight did not complete')
            sleep_start = time.perf_counter_ns() if self._diag.enabled else 0
            if self._diag.enabled:
                self._diag.count('writing_poll_sleeps')
            time.sleep(0.01)
            if self._diag.enabled:
                self._diag.add_ns('writing_poll_sleep', time.perf_counter_ns() - sleep_start)

    def release(self, key, generation, token):
        with self._transaction('release') as db:
            db.execute('DELETE FROM pins WHERE token=? AND key=? AND generation=? AND client=?',
                       (token, key, generation, self.client))

    def health(self):
        if self._closed:
            return {**self._last_health, 'node_pool_closed': True}
        with self._transaction('health') as db:
            self._reap(db)
            result = {f'node_pool_{name}': value for name, value in db.execute('SELECT name,value FROM counters')}
            metadata_file_bytes = 0
            for path in self.directory.glob('directory.sqlite*'):
                try:
                    metadata_file_bytes += path.stat().st_size
                except FileNotFoundError:
                    pass
            self._sample_metadata(db, 'health')
            result.update(node_pool_enabled=True, node_pool_capacity_bytes=self.total_capacity,
                          node_pool_data_capacity_bytes=self.capacity, node_pool_metadata_budget_bytes=self.metadata_budget,
                          node_pool_directory_budget_bytes=self.directory_budget, node_pool_terminal_proof_budget_bytes=self.terminal_proof_budget,
                          node_pool_metadata_runtime_headroom_bytes=self.metadata_runtime_headroom,
                          node_pool_budget_mode=self.budgets['budget_mode'],
                          node_pool_effective_max_entries=self.max_entries,
                          node_pool_effective_max_page_count=self.max_page_count,
                          node_pool_diagnostic_observed_maxima=dict(self._observed_metadata),
                          node_pool_diagnostic_scope='process-local; explicit COMMIT only; initialization autocommit excluded',
                          node_pool_metadata_sampling='before COMMIT every 64 successful BEGINs and on health; observed maxima, not exhaustive peaks',
                          node_pool_committed_file_bytes=(self.directory / 'data.mmap').stat().st_blocks * 512,
                          node_pool_metadata_file_bytes=metadata_file_bytes,
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
            with self._transaction('close') as db:
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
