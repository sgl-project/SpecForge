"""Opt-in bounded observations; never changes data-plane synchronization.

Counters are process-local, durations may nest, and traces use one host's monotonic
clock. No per-block fsync. A dropped/error trace is explicitly incomplete.
"""
from __future__ import annotations

import atexit
from collections import defaultdict
from contextlib import contextmanager
import json
import os
from pathlib import Path
import threading
import time


class Component:
    def __init__(self, owner, prefix):
        self.owner, self.prefix = owner, prefix + '.'
        self.enabled = owner.enabled

    def count(self, name, n=1):
        self.owner.count(self.prefix + name, n)

    def add_ns(self, name, ns):
        self.owner.count(self.prefix + name + '_ns', ns)

    def max(self, name, value):
        self.owner.maximum(self.prefix + name, value)

    @contextmanager
    def span(self, name):
        if not self.enabled:
            yield
            return
        start = time.perf_counter_ns()
        try:
            yield
        finally:
            self.add_ns(name, time.perf_counter_ns() - start)
            self.count(name + '_calls')


class Diagnostics:
    def __init__(self, directory=None, *, trace_limit_bytes=128 << 20,
                 trace_limit_keys=100000, buffer_bytes=256 << 10):
        self.enabled = bool(directory)
        self.pid = os.getpid()
        self.rank = int(os.environ.get('RANK', os.environ.get('LOCAL_RANK', '-1')))
        self.lock = threading.RLock()
        self.local = threading.local()
        self.values = defaultdict(int)
        self.keys = {}
        self.components = {}
        self.buffer = []
        self.buffer_size = 0
        self.trace_bytes = 0
        self.trace_limit = int(trace_limit_bytes)
        self.key_limit = int(trace_limit_keys)
        self.buffer_limit = int(buffer_bytes)
        self.sequence = 0
        self.phase_name = "unspecified"
        self.errors = []
        self.file = None
        self.directory = Path(directory) if directory else None
        if self.enabled:
            try:
                self.directory.mkdir(parents=True, exist_ok=True)
                self.path = self.directory / f'rank{self.rank}-pid{self.pid}.jsonl'
                self.file = self.path.open('x', buffering=1 << 20)
                self._record({'type': 'header', 'version': 1, 'pid': self.pid,
                              'rank': self.rank, 'hostname': os.uname().nodename,
                              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                              'monotonic_ns': time.monotonic_ns(),
                              'columns': ['event', 'monotonic_ns', 'operation', 'key_id', 'batch_index', 'phase', 'detail'],
                              'trace_limit_bytes': self.trace_limit,
                              'trace_limit_keys': self.key_limit,
                              'buffer_limit_bytes': self.buffer_limit})
                atexit.register(self.flush)
            except Exception as exc:
                self.errors.append(repr(exc))

    def component(self, name):
        if name not in self.components:
            self.components[name] = Component(self, name)
        return self.components[name]

    def count(self, name, n=1):
        if self.enabled:
            with self.lock:
                self.values[name] += n

    def maximum(self, name, value):
        if self.enabled:
            with self.lock:
                self.values[name] = max(self.values[name], value)

    @contextmanager
    def batch(self, index):
        old = getattr(self.local, 'batch_index', None)
        self.local.batch_index = index
        try:
            yield
        finally:
            self.local.batch_index = old

    def _record(self, record):
        # Called under RLock except constructor. Logging errors never mask native
        # failures; analysis must reject incomplete traces instead.
        try:
            encoded = json.dumps(record, separators=(',', ':')) + '\n'
            size = len(encoded.encode())
            if self.trace_bytes + self.buffer_size + size > self.trace_limit:
                self.values['trace_dropped_records'] += 1
                return False
            self.buffer.append(encoded)
            self.buffer_size += size
            self.values['trace_buffer_peak_bytes'] = max(self.values['trace_buffer_peak_bytes'], self.buffer_size)
            if self.buffer_size >= self.buffer_limit:
                self._flush_buffer()
            return True
        except Exception as exc:
            if len(self.errors) < 8:
                self.errors.append(repr(exc))
            self.values['trace_errors'] += 1
            return False

    def _flush_buffer(self):
        start = time.perf_counter_ns()
        try:
            if self.file is None:
                raise RuntimeError('trace file unavailable')
            self.file.writelines(self.buffer)
            self.file.flush()  # buffered write, intentionally no fsync
            self.trace_bytes += self.buffer_size
            self.buffer.clear()
            self.buffer_size = 0
            self.values['trace_flush_calls'] += 1
        finally:
            self.values['trace_flush_ns'] += time.perf_counter_ns() - start

    def begin_block(self, identity):
        if not self.enabled:
            return None
        start = time.perf_counter_ns()
        with self.lock:
            try:
                canonical = json.dumps(identity, sort_keys=True, separators=(',', ':'))
                key_id = self.keys.get(canonical)
                if key_id is None:
                    if len(self.keys) >= self.key_limit:
                        self.values['trace_dropped_records'] += 1
                        return None
                    key_id = len(self.keys)
                    if not self._record({'type': 'key', 'id': key_id, **identity}):
                        return None
                    self.keys[canonical] = key_id
                    self.values['trace_identity_bytes'] += len(canonical.encode())
                self.sequence += 1
                op = (self.sequence, key_id, getattr(self.local, 'batch_index', None), self.phase_name)
                self.event('lookup', op)
                return op
            except Exception as exc:
                if len(self.errors) < 8:
                    self.errors.append(repr(exc))
                self.values['trace_errors'] += 1
                return None
            finally:
                self.values['trace_encode_ns'] += time.perf_counter_ns() - start

    def event(self, event, op, detail=None):
        if not self.enabled or op is None:
            return
        start = time.perf_counter_ns()
        with self.lock:
            self._record([event, time.monotonic_ns(), *op, detail])
            self.values['trace_event_calls'] += 1
            self.values['trace_event_ns'] += time.perf_counter_ns() - start

    def snapshot(self):
        with self.lock:
            return {'enabled': self.enabled, 'pid': self.pid, 'rank': self.rank,
                    'metrics': dict(self.values), 'trace_bytes_written': self.trace_bytes,
                    'trace_bytes_buffered': self.buffer_size,
                    'trace_keys': len(self.keys), 'trace_errors': list(self.errors),
                    'trace_complete': not self.errors and not self.values.get('trace_dropped_records', 0),
                    'duration_semantics': 'process-local inclusive/nested; do not sum into wall time'}

    def flush(self):
        if not self.enabled:
            return
        with self.lock:
            try:
                if self.buffer:
                    self._flush_buffer()
                path = self.directory / f'rank{self.rank}-pid{self.pid}-summary.json'
                path.write_text(json.dumps(self.snapshot(), indent=2) + '\n')
            except Exception as exc:
                if len(self.errors) < 8:
                    self.errors.append(repr(exc))
                self.values['trace_errors'] += 1


_instance = None
_instance_lock = threading.Lock()


def get_diagnostics():
    global _instance
    if _instance is None or _instance.pid != os.getpid():
        with _instance_lock:
            if _instance is None or _instance.pid != os.getpid():
                _instance = Diagnostics(os.environ.get('SPECFORGE_POOL_DIAGNOSTICS_DIR'),
                                        trace_limit_bytes=int(os.environ.get('SPECFORGE_POOL_TRACE_LIMIT_BYTES', str(128 << 20))))
    return _instance
