"""Optional per-Teacher monotonic publication observations, never a data-plane fence."""
from collections import Counter
from contextlib import contextmanager
import json
import os
from pathlib import Path
import socket
import threading
import time
import uuid


IDENTITY = ('store_id', 'sample_id', 'gen', 'server_epoch', 'attempt_id', 'reservation_token')
NATIVE = {'register_buffer':'native_register', 'unregister_buffer':'native_unregister',
          'put_from':'native_put', 'batch_put_from':'native_put', 'remove':'native_remove',
          'get_into':'native_ack_read', 'is_exist':'native_check', 'batch_is_exist':'native_check'}
LEDGER = {name:'ledger_' + name for name in ('reserve','state','forget')}


class TimedCalls:
    """Delegate the real signature, result and exceptions. Missing optional APIs stay missing."""
    def __init__(self, target, trace, methods):
        self.target, self.trace, self.methods = target, trace, methods

    def __getattr__(self, name):
        fn = getattr(self.target, name)
        phase = self.methods.get(name)
        if phase is None or not callable(fn):
            return fn

        def measured(*args, **kwargs):
            totals = getattr(self.trace.local, 'totals', None)
            if totals is None:
                return fn(*args, **kwargs)
            start = time.monotonic_ns()
            try:
                return fn(*args, **kwargs)
            finally:
                totals[phase] += time.monotonic_ns() - start
        setattr(self, name, measured)
        return measured


class PublicationTrace:
    def __init__(self, directory, limit=4096):
        self.local = threading.local()
        self.stream = uuid.uuid4().hex
        self.path = Path(directory) / f'publication-timing-{self.stream}.jsonl'
        self.limit = limit
        self.guard = threading.Lock()
        self.wake = threading.Event()
        self.pending = []
        self.dropped = 0
        self.write_errors = 0
        self.closed = False
        self.source = dict(host=socket.gethostname(), pid=os.getpid(), stream=self.stream,
                           clock='monotonic_ns', schema=1)
        self.writer = threading.Thread(target=self._write, daemon=True, name='publication-trace')
        self.writer.start()

    def emit(self, kind, **fields):
        with self.guard:
            if self.closed or len(self.pending) >= self.limit:
                self.dropped += 1
                return
            self.pending.append(dict(event=kind, **fields))
            if len(self.pending) >= 64:
                self.wake.set()

    def _write(self):
        while True:
            self.wake.wait(0.5)
            with self.guard:
                records, self.pending = self.pending, []
                closed, dropped = self.closed, self.dropped
                self.wake.clear()
            if records or closed:
                try:
                    with self.path.open('a') as f:
                        f.write(json.dumps(dict(**self.source, rows=records, dropped=dropped,
                                                final=closed), separators=(',', ':')) + '\n')
                except (OSError, ValueError, TypeError):
                    with self.guard:
                        self.dropped += len(records)
                        self.write_errors += 1
            if closed:
                return

    def close(self):
        with self.guard:
            self.closed = True
            self.wake.set()
        self.writer.join(timeout=2)

    def begin_cleanup(self, token):
        if self.local.cleanup_start is None:
            self.local.cleanup_start = (token, time.monotonic_ns())

    def end_cleanup(self):
        if self.local.cleanup_start is not None:
            token, start = self.local.cleanup_start
            self.local.cleanup_start = None
            if len(self.local.cleanups) < 64:
                self.local.cleanups.append((token, start, time.monotonic_ns()))
            else:
                self.local.cleanup_dropped += 1

    @contextmanager
    def hold(self, lock, owner, phase, spec=None):
        identity = {k:spec.get(k) for k in IDENTITY} if spec is not None else None
        sample = getattr(self.local, 'sample', None) if spec is not None else None
        totals = Counter()
        previous = getattr(self.local, 'totals', None)
        attempt_ns = time.monotonic_ns()
        end_ns = None
        try:
            with lock:
                acquired_ns = time.monotonic_ns()
                self.local.totals = totals
                if phase == 'progress':
                    self.local.cleanups = []
                    self.local.cleanup_dropped = 0
                    self.local.cleanup_start = None
                try:
                    yield
                finally:
                    if phase == 'progress':
                        self.end_cleanup()
                    self.local.totals = previous
                    end_ns = time.monotonic_ns()
        finally:
            if end_ns is not None:
                self.emit('hold', phase=phase, owner=owner, identity=identity, sample=sample,
                          tid=threading.get_native_id(), attempt_ns=attempt_ns, acquired_ns=acquired_ns,
                          end_ns=end_ns, component_ns=dict(totals),
                          cleanups=self.local.cleanups if phase == 'progress' else [],
                          cleanup_dropped=self.local.cleanup_dropped if phase == 'progress' else 0)

