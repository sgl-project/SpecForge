"""Bounded metadata windows; transport and copies never hold a SQLite tx."""
from concurrent.futures import CancelledError
import os
import threading
import time
import uuid

import torch

from .node_content_pool import PoolFetchFailed, PoolPin


NORMAL_BLOCK_BYTES = 8 << 20
HARD_WINDOW_BYTES = 64 << 20
HARD_WINDOW_BLOCKS = 8


def batch_limits():
    prefix = 'SPECFORGE_TRAINER_NODE_POOL_'
    size = int(os.environ.get(prefix + 'WINDOW_BYTES', HARD_WINDOW_BYTES))
    blocks = int(os.environ.get(prefix + 'WINDOW_BLOCKS', HARD_WINDOW_BLOCKS))
    inflight = int(os.environ.get(prefix + 'WINDOW_INFLIGHT', 1))
    if not 4096 <= size <= HARD_WINDOW_BYTES or not 1 <= blocks <= HARD_WINDOW_BLOCKS or inflight != 1:
        raise ValueError('node-pool window requires bytes<=64MiB, blocks<=8, inflight=1')
    return {'extent_bytes': size, 'blocks': blocks, 'inflight': inflight,
            'normal_block_bytes': NORMAL_BLOCK_BYTES, 'oversize_block_hard_bytes': min(size, HARD_WINDOW_BYTES)}


class _Gate:
    def __init__(self):
        self.pid = os.getpid()
        self.condition = threading.Condition()
        self.window = None

    def enter(self, window):
        with self.condition:
            while self.window is not None:
                if self.window.poisoned:
                    raise RuntimeError('previous batch window requires proven cleanup before new admission')
                window.check_cancel()
                remaining = window.deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('node-pool local batch gate did not drain')
                self.condition.wait(min(0.01, remaining))
            window.check_cancel()
            self.window = window

    def leave(self, window):
        with self.condition:
            if self.window is not window:
                raise RuntimeError('batch gate owner changed')
            self.window = None
            self.condition.notify_all()

    def current(self):
        with self.condition:
            return self.window


_gate = None
_gate_lock = threading.Lock()


def process_gate():
    global _gate
    with _gate_lock:
        if _gate is None or _gate.pid != os.getpid():
            _gate = _Gate()
        return _gate


class BatchPin(PoolPin):
    def release(self):
        # PoolPin's legacy release clears its token before SQL. Batch-owned
        # claims instead survive failed metadata commits and can be retried.
        self.pool.release_many([self])


class _Claim:
    """A committed token whose PoolPin construction has not yet succeeded."""
    def __init__(self, window, entry):
        self.pool, self._window = window.pool, window
        self.key, self.generation, self.token = entry['key'], entry['generation'], entry['token']
        self.tensor = None


class BatchWindow:
    def __init__(self, pool, requests, *, cancelled=None):
        self.pool, self.cancelled = pool, cancelled
        self.deadline = time.monotonic() + pool.wait_timeout
        self.id = uuid.uuid4().hex
        self.entries, self.order = {}, []
        self.charge = 0
        for key, nbytes, fetch in requests:
            if len(self.order) >= pool.batch_limits['blocks']:
                raise ValueError('batch window block bound exceeded')
            nbytes = int(nbytes)
            if not isinstance(key, str) or not key or nbytes <= 0 or not callable(fetch):
                raise ValueError('invalid batch block descriptor')
            extent = (nbytes + 4095) // 4096 * 4096
            if extent > pool.batch_limits['extent_bytes'] or self.charge + extent > pool.batch_limits['extent_bytes']:
                raise ValueError('batch window extent bound exceeded before fetch')
            self.charge += extent
            entry = self.entries.setdefault(key, {'key': key, 'nbytes': nbytes, 'extent': extent,
                'fetch': fetch, 'aliases': 0, 'state': 'new', 'token': None, 'pin': None,
                'generation': None, 'offset': None, 'waited': False, 'registered': False,
                'writer_committed': False})
            if entry['nbytes'] != nbytes:
                raise ValueError('duplicate batch key changed byte layout')
            entry['aliases'] += 1
            self.order.append(entry)
        if not self.order:
            raise ValueError('empty batch window')
        if any(e['nbytes'] > NORMAL_BLOCK_BYTES for e in self.order) and len(self.order) != 1:
            raise ValueError('oversize block requires a single-item window')
        self.pins = [None] * len(self.order)
        self.external = set()
        self.deferred = set()
        self.source_releases = []
        self.poisoned = self.closed = self.closing = False
        self.entered = False
        self._attempted = False

    def check_cancel(self):
        if self.cancelled is not None and self.cancelled():
            raise CancelledError('node-pool batch cancelled at a safe boundary')

    def snapshot(self):
        active = self.entered and not self.closed
        pins = [e for e in self.entries.values() if e['token'] is not None and e['registered']]
        writers = [e for e in self.entries.values() if e['state'] in ('own', 'copied') and e['writer_committed']]
        return {'active_windows': int(active), 'blocks': len(self.order) if active else 0,
                'window_extent_bytes': self.charge if active else 0,
                'registered_pin_tokens': len(pins), 'registered_pin_extent_bytes': sum(e['extent'] for e in pins),
                'unconfirmed_pin_claims': sum(e['token'] is not None and not e['registered'] for e in self.entries.values()),
                'unconfirmed_writer_claims': sum(e['state'] in ('own','copied') and not e['writer_committed'] for e in self.entries.values()),
                'own_writing_blocks': len(writers), 'own_writing_extent_bytes': sum(e['extent'] for e in writers),
                'retained_windows': int(active and bool(self.external or self.deferred or self.source_releases or self.poisoned)),
                'external_dependencies': sorted(self.external), 'quarantined_blocks': len(self.deferred),
                'oversize_fallback_bytes': self.order[0]['nbytes'] if active and self.order[0]['nbytes'] > NORMAL_BLOCK_BYTES else 0}

    def observe(self):
        snapshot = self.snapshot()
        for name, value in snapshot.items():
            if type(value) is int:
                self.pool._batch_high_water[name] = max(self.pool._batch_high_water.get(name, 0), value)
                self.pool._diag.max('batch_' + name + '_high_water', value)

    def hold_external(self, reason):
        self.external.add(reason)
        self.poisoned = True
        self.observe()

    def resolve_external(self, reason):
        self.external.discard(reason)
        self.close()

    def defer(self, index):
        self.deferred.add(index)
        self.poisoned = True
        self.observe()

    def resolve_deferred(self, index):
        self.deferred.discard(index)
        self.close()

    def receive_only(self):
        return self.external == {'receive'} and not self.deferred and not self.source_releases and not any(
            e['token'] is not None or e['state'] in ('own', 'copied') for e in self.entries.values())

    def __enter__(self):
        with self.pool._lock:
            if self._attempted or self.entered or self.closed:
                raise RuntimeError('batch window is a one-shot context; create a new window')
            self._attempted = True
        try:
            process_gate().enter(self)
            with self.pool._lock:
                if self.pool._closed or self.pool._batch_closing:
                    raise RuntimeError('node-pool batch admission is closed')
                self.pool._batch_windows[self.id] = self
                self.entered = True
            self.pool._diag.count('batch_windows')
            self.pool._diag.count('batch_requested_blocks', len(self.order))
            self.pool._diag.count('batch_requested_extent_bytes', self.charge)
            self.observe()
            self._admit()
            self._fill_owned()
            self._wait_foreign()
            self.pins = [entry['pin'] for entry in self.order]
            return self
        except BaseException as exc:
            try:
                if self.id in self.pool._batch_windows:
                    self.entered = True
                    self.close(error=exc)
                elif process_gate().current() is self:
                    process_gate().leave(self)
            except BaseException as cleanup:
                exc.add_note(f'batch acquisition cleanup also failed: {cleanup!r}; window {self.id} retained')
            raise

    def __exit__(self, kind, value, traceback):
        try:
            self.close(error=value)
        except BaseException as cleanup:
            if value is None:
                raise
            value.add_note(f'batch window cleanup also failed: {cleanup!r}; window {self.id} retained')
        return False

    def _pin_row(self, db, entry, row):
        generation, offset, _, _, _, pid = row
        token = uuid.uuid4().hex
        db.execute('INSERT INTO pins VALUES (?,?,?,?,?)', (token, entry['key'], generation, self.pool.client, self.pool.pid))
        db.execute('UPDATE entries SET touched=? WHERE key=?', (time.monotonic(), entry['key']))
        self.pool._count(db, 'hits', entry['aliases'])
        if pid != self.pool.pid:
            self.pool._count(db, 'cross_process_hits', entry['aliases'])
        entry.update(generation=generation, offset=offset, token=token, state='ready', registered=False)

    def _make_pins(self):
        for entry in self.entries.values():
            if entry['token'] is not None and entry['registered'] and entry['pin'] is None:
                tensor = None
                try:
                    tensor = torch.frombuffer(self.pool._mapping, dtype=torch.uint8,
                                              count=entry['nbytes'], offset=entry['offset'])
                    pin = BatchPin(self.pool, entry['key'], entry['generation'], entry['token'], tensor)
                    pin._window = self
                    entry['pin'] = pin
                finally:
                    tensor = None
        self.observe()

    def _admit(self):
        with self.pool._transaction('batch_lookup') as db:
            self.pool._reap(db)
            # Pin all existing READY keys before reserving any new key. A
            # reserve cannot evict a later hot key in this same window.
            for entry in self.entries.values():
                if entry['nbytes'] > NORMAL_BLOCK_BYTES:
                    entry['state'] = 'fallback'
                    self.pool._count(db, 'fallbacks')
                    self.pool._diag.count('batch_oversize_fallback_blocks')
                    continue
                row = db.execute('SELECT generation,offset,nbytes,state,error,pid FROM entries WHERE key=?', (entry['key'],)).fetchone()
                if row is not None and row[2] != entry['nbytes']:
                    raise ValueError('content key changed byte layout')
                if row is not None and row[3] == 'FAILED':
                    raise PoolFetchFailed(row[4])
                if row is not None and row[3] == 'READY':
                    self._pin_row(db, entry, row)
                elif row is not None:
                    entry.update(state='wait', generation=row[0], waited=True)
                    self.pool._count(db, 'singleflight_waits')
            for entry in self.entries.values():
                if entry['state'] != 'new':
                    continue
                offset = self.pool._reserve(db, entry['extent']) if entry['extent'] <= self.pool.capacity else None
                if offset is None:
                    entry['state'] = 'fallback'
                    self.pool._count(db, 'fallbacks', entry['aliases'])
                    continue
                generation = uuid.uuid4().hex
                entry.update(state='own', generation=generation, offset=offset)
                db.execute('INSERT INTO entries VALUES (?,?,?,?,?,?,?,?,?,?)', (entry['key'], generation,
                           offset, entry['extent'], entry['nbytes'], 'WRITING', self.pool.client, self.pool.pid, time.monotonic(), ''))
                self.pool._count(db, 'misses')
                if self.pool._diag.enabled and self.pool.max_entries is not None:
                    self.pool._diag.max('entry_count_high_water', self.pool._entry_count_before_insert + 1)
                current = db.execute("SELECT COALESCE(SUM(extent),0) FROM entries WHERE state IN ('READY','WRITING')").fetchone()[0]
                self.pool._diag.max('reserved_data_bytes_observed_max', current)
                old = db.execute("SELECT value FROM counters WHERE name='storage_peak_bytes'").fetchone()[0]
                if current > old:
                    db.execute("UPDATE counters SET value=? WHERE name='storage_peak_bytes'", (current,))
        for entry in self.entries.values():
            if entry['state'] == 'own':
                entry['writer_committed'] = True
            if entry['token'] is not None:
                entry['registered'] = True
        self._make_pins()
        self.observe()

    def _fill_owned(self):
        for entry in self.entries.values():
            if entry['state'] != 'own':
                continue
            self.check_cancel()
            shared = source = release = None
            primary = None
            try:
                with self.pool._diag.span('fetch'):
                    source = entry['fetch']()
                if isinstance(source, tuple):
                    source, release = source
                if source.device.type != 'cpu' or not source.is_contiguous() or source.numel() * source.element_size() != entry['nbytes']:
                    raise ValueError('node pool fetch must return completed contiguous CPU storage')
                shared = torch.frombuffer(self.pool._mapping, dtype=torch.uint8, count=entry['nbytes'], offset=entry['offset'])
                with self.pool._diag.span('shared_cpu_copy'):
                    shared.copy_(source.view(torch.uint8).reshape(-1), non_blocking=False)
                self.pool._diag.count('shared_cpu_copy_completed_bytes', entry['nbytes'])
                entry['state'] = 'copied'
                shared = source = None
            except BaseException as exc:
                primary = exc
                raise
            finally:
                shared = source = None
                if release is not None:
                    try:
                        release()
                    except BaseException as cleanup:
                        self.source_releases.append(release)
                        if primary is None:
                            raise
                        primary.add_note(f'batch source release also failed: {cleanup!r}')
        # Publish every own writer before waiting on any foreign writer.
        self._publish_owned()

    def _publish_owned(self, error=None):
        pending = [entry for entry in self.entries.values() if entry['state'] in ('own', 'copied')]
        if not pending:
            return
        with self.pool._transaction('batch_publish') as db:
            for entry in pending:
                if entry['state'] == 'copied':
                    changed = db.execute("UPDATE entries SET state='READY' WHERE key=? AND generation=? AND state='WRITING' AND writer=?",
                        (entry['key'], entry['generation'], self.pool.client)).rowcount
                    if changed != 1:
                        raise RuntimeError('shared batch writer generation was fenced')
                    entry['token'] = uuid.uuid4().hex
                    db.execute('INSERT INTO pins VALUES (?,?,?,?,?)', (entry['token'], entry['key'], entry['generation'], self.pool.client, self.pool.pid))
                    self.pool._count(db, 'cpu_copy_bytes', entry['nbytes'])
                    if entry['aliases'] > 1:
                        self.pool._count(db, 'hits', entry['aliases'] - 1)
                else:
                    if error is None:
                        raise RuntimeError('batch publish contains an unfinished writer')
                    db.execute("UPDATE entries SET state='FAILED',extent=0,error=? WHERE key=? AND generation=? AND writer=?",
                               (str(error), entry['key'], entry['generation'], self.pool.client))
                    self.pool._count(db, 'fetch_failures')
            committed = (self.pool.directory / 'data.mmap').stat().st_blocks * 512
            old = db.execute("SELECT value FROM counters WHERE name='storage_peak_bytes'").fetchone()[0]
            if committed > old:
                db.execute("UPDATE counters SET value=? WHERE name='storage_peak_bytes'", (committed,))
        for entry in pending:
            entry['state'] = 'ready' if entry['state'] == 'copied' else 'failed'
            entry['writer_committed'] = False
            if entry['token'] is not None:
                entry['registered'] = True
        self._make_pins()
        self.observe()

    def _wait_foreign(self):
        while any(entry['state'] == 'wait' for entry in self.entries.values()):
            self.check_cancel()
            with self.pool._transaction('batch_poll') as db:
                self.pool._reap(db)
                for entry in self.entries.values():
                    if entry['state'] != 'wait':
                        continue
                    row = db.execute('SELECT generation,offset,nbytes,state,error,pid FROM entries WHERE key=?', (entry['key'],)).fetchone()
                    if row is None:
                        entry['state'] = 'fallback'
                        self.pool._count(db, 'fallbacks', entry['aliases'])
                    elif row[0] != entry['generation'] or row[2] != entry['nbytes']:
                        raise PoolFetchFailed('foreign batch writer generation/layout changed')
                    elif row[3] == 'FAILED':
                        raise PoolFetchFailed(row[4])
                    elif row[3] == 'READY':
                        self._pin_row(db, entry, row)
                    else:
                        self.pool._diag.count('writing_poll_iterations')
            for entry in self.entries.values():
                if entry['token'] is not None:
                    entry['registered'] = True
            self._make_pins()
            if not any(entry['state'] == 'wait' for entry in self.entries.values()):
                return
            if time.monotonic() >= self.deadline:
                raise TimeoutError('node content batch singleflight did not complete')
            self.pool._diag.count('writing_poll_sleeps')
            started = time.perf_counter_ns()
            time.sleep(0.01)
            self.pool._diag.add_ns('writing_poll_sleep', time.perf_counter_ns() - started)

    def pin_released(self, key, token):
        entry = self.entries[key]
        if entry['token'] == token:
            entry['token'] = None
            entry['registered'] = False
        self.observe()
        self._maybe_done()

    def _maybe_done(self):
        if not self.closing or self.closed or self.external or self.deferred or self.source_releases:
            return
        if any(e['token'] is not None or e['state'] in ('own', 'copied') for e in self.entries.values()):
            return
        self.closed = True
        self.pool._batch_windows.pop(self.id, None)
        process_gate().leave(self)

    def close(self, error=None):
        if not self.entered or self.closed:
            return
        self.closing = True
        try:
            self._publish_owned(error=error or RuntimeError('batch window cancelled before publish'))
            for release in list(self.source_releases):
                release()
                self.source_releases.remove(release)
            deferred_keys = {self.order[index]['key'] for index in self.deferred}
            claims = [entry['pin'] or _Claim(self, entry) for entry in self.entries.values()
                      if entry['token'] is not None and entry['key'] not in deferred_keys]
            self.pool.release_many(claims)
            self.poisoned = bool(self.external or self.deferred or self.source_releases)
            self._maybe_done()
            self.observe()
        except BaseException as exc:
            self.poisoned = True
            try:
                self.observe()
            except BaseException as observation:
                exc.add_note(f'batch cleanup observation also failed: {observation!r}')
            raise
