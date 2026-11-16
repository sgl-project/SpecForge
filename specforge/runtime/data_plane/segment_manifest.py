"""Versioned HS segment manifests and physical remote-object accounting."""
from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
import time
import uuid
from pathlib import Path

import torch

MANIFEST_LAYOUT = 'hs-manifest-v1'
MANIFEST_SCHEMA_VERSION = 2
MARKER_BYTES = 32


def digest_manifest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def validate_manifest(manifest, specs, num_tokens):
    if not isinstance(manifest, dict) or manifest.get('layout') != MANIFEST_LAYOUT:
        raise ValueError('unknown hidden-state manifest layout')
    if not manifest.get('namespace') or not manifest.get('owner_epoch') or not manifest.get('token') or not manifest.get('schema_signature'):
        raise ValueError('manifest omitted namespace/owner/attempt identity')
    body = {name: value for name, value in manifest.items() if name != 'digest'}
    if manifest.get('digest') != digest_manifest(body):
        raise ValueError('manifest digest mismatch')
    if not manifest.get('ack_key') or not manifest.get('receipt_key'):
        raise ValueError('manifest omitted exact ACK/receipt objects')
    features = manifest.get('features')
    expected = {name for name,spec in specs.items() if len(spec.shape) == 3 and spec.dtype in ('float16','bfloat16','float32','float64')}
    if not isinstance(features, dict) or set(features) != expected:
        raise ValueError('manifest names do not match feature schema')
    for name, segments in features.items():
        spec = specs[name]
        if len(spec.shape) != 3 or spec.shape[:2] != (1, num_tokens):
            raise ValueError('manifest feature must have logical shape [1,N,width]')
        position = 0
        for segment in segments:
            start, end = segment.get('start'), segment.get('end')
            if type(start) is not int or type(end) is not int or start != position or not start < end <= num_tokens:
                raise ValueError('manifest coverage has a gap, overlap, or wrong position')
            shape = [1, end - start, spec.shape[-1]]
            dtype = getattr(torch, spec.dtype, None)
            if (segment.get('shape') != shape or segment.get('dtype') != spec.dtype
                    or not isinstance(dtype, torch.dtype)
                    or segment.get('nbytes') != (end - start) * spec.shape[-1] * torch.empty((), dtype=dtype).element_size()
                    or not segment.get('logical_key') or not segment.get('key')
                    or not segment.get('publication_generation')):
                raise ValueError('manifest segment byte layout/identity mismatch')
            logical = segment['logical_key']
            expected_key = 'specforge-hs-block/' + hashlib.sha256(logical.encode()).hexdigest() + '/' + segment['publication_generation']
            if (not logical.startswith(manifest['namespace'] + '/' + manifest['schema_signature'] + '/')
                    or segment['key'] != expected_key):
                raise ValueError('manifest physical generation does not match its logical identity')
            position = end
        if position != num_tokens:
            raise ValueError('manifest does not cover the complete logical input')
    return manifest


def remote_allocation_charge(nbytes):
    # Fixed 96 GiB offset-allocator segment: multiplier_bits=5, 3-bit mantissa.
    # A 4 KiB minimum conservatively includes tiny ACK/receipt/header objects.
    units = (max(1, int(nbytes)) + 31) // 32
    quantum = 1 << max(0, units.bit_length() - 1 - 3)
    return max(4096, ((units + quantum - 1) // quantum) * quantum * 32)


class RemoteCapacityError(RuntimeError):
    pass


class RemoteObjectLedger:
    """One physical-byte budget shared by all Teacher writers on a node."""
    def __init__(self, directory, capacity, owner):
        self.path = Path(directory)
        self.path.mkdir(parents=True, exist_ok=True)
        self.capacity, self.owner = int(capacity), owner
        self.alignment = 4096
        self.closed = False
        self._last_health = {}
        if self.capacity <= 0:
            raise ValueError('remote object budget must be positive')
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path / 'remote.sqlite', isolation_level=None,
                                  timeout=30, check_same_thread=False)
        self.db.executescript('''CREATE TABLE IF NOT EXISTS config(capacity INTEGER);
            CREATE TABLE IF NOT EXISTS counters(name TEXT PRIMARY KEY,value INTEGER);
            CREATE TABLE IF NOT EXISTS objects(key TEXT PRIMARY KEY, owner TEXT,
                bytes INTEGER, state TEXT, pid INTEGER);''')
        self.db.execute('BEGIN IMMEDIATE')
        try:
            old = self.db.execute('SELECT capacity FROM config').fetchone()
            if old is None:
                self.db.execute('INSERT INTO config VALUES (?)', (self.capacity,))
            elif old[0] != self.capacity:
                raise ValueError('remote budget differs across Teacher endpoints')
            for pid, in self.db.execute('SELECT DISTINCT pid FROM objects'):
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    raise RuntimeError('remote manifest owner exited with charged objects; explicit recovery required') from None
            self.db.execute('COMMIT')
        except BaseException:
            self.db.execute('ROLLBACK')
            raise

    def reserve(self, objects):
        # Conservative allocation-class charge, including minimum-size metadata.
        objects = {key: remote_allocation_charge(value) for key, value in objects.items()}
        with self.lock:
            self.db.execute('BEGIN IMMEDIATE')
            try:
                current = self.db.execute('SELECT COALESCE(SUM(bytes),0) FROM objects').fetchone()[0]
                if current + sum(objects.values()) > self.capacity:
                    raise RemoteCapacityError('remote physical object budget exhausted; admitted window cannot silently wait')
                self.db.executemany('INSERT INTO objects VALUES (?,?,?,?,?)',
                                    ((key, self.owner, value, 'RESERVED', os.getpid()) for key, value in objects.items()))
                peak = self.db.execute("SELECT value FROM counters WHERE name='peak_charged_bytes'").fetchone()
                now = current + sum(objects.values())
                if peak is None or peak[0] < now:
                    self.db.execute("INSERT OR REPLACE INTO counters VALUES ('peak_charged_bytes',?)", (now,))
                self.db.execute('COMMIT')
            except BaseException:
                self.db.execute('ROLLBACK')
                raise

    def state(self, keys, state):
        with self.lock:
            self.db.executemany('UPDATE objects SET state=? WHERE key=? AND owner=?',
                                ((state, key, self.owner) for key in keys))

    def forget(self, key):
        with self.lock:
            self.db.execute('DELETE FROM objects WHERE key=? AND owner=?', (key, self.owner))

    def health(self):
        with self.lock:
            if self.closed:
                return dict(self._last_health)
            states = dict(self.db.execute('SELECT state,SUM(bytes) FROM objects GROUP BY state'))
            own = dict(self.db.execute('SELECT state,SUM(bytes) FROM objects WHERE owner=? GROUP BY state', (self.owner,)))
            return {'remote_node_budget_bytes': self.capacity,
                    'remote_charge_policy': 'offset96g-3mantissa-min4096',
                    'remote_node_charged_bytes': sum(states.values()),
                    'remote_node_peak_charged_bytes': (self.db.execute("SELECT value FROM counters WHERE name='peak_charged_bytes'").fetchone() or [0])[0],
                    'remote_owner_charged_bytes': sum(own.values()),
                    **{f'remote_{name}_bytes': own.get(state, 0) for name, state in
                       (('reserved', 'RESERVED'), ('inflight', 'WRITING'), ('unique', 'READY'),
                        ('pending_delete', 'DELETING'), ('quarantined', 'QUARANTINED'))},
                    'remote_idle_bytes': 0}

    def close(self):
        with self.lock:
            if self.db.execute('SELECT COUNT(*) FROM objects WHERE owner=?', (self.owner,)).fetchone()[0]:
                raise RuntimeError('remote manifest owner still has charged physical objects')
            self._last_health = self.health()
            self.db.close()
            self.closed = True


class RemoteManifestOwner:
    """Hard-pin owner: dependencies precede manifest visibility, ACKs release one.

    No HTTP calls are made by the Consumer. Immutable marker keys carry ACKs
    through the object store. A receipt waits only for objects whose *last*
    dependency this ACK removed, never for a later manifest's completion.
    """
    def __init__(self, store, put, *, directory, capacity, hard_pin):
        if hard_pin is not True:
            raise RuntimeError('segmented publication requires verified native hard pin')
        self.store, self.put = store, put
        self.epoch = uuid.uuid4().hex
        self.ledger = RemoteObjectLedger(directory, capacity, self.epoch)
        self.lock = threading.RLock()
        self.objects, self.manifests = {}, {}
        self.stats = dict.fromkeys(('manifest_published', 'manifest_dependencies', 'block_new',
                                    'block_reused', 'logical_feature_bytes', 'physical_put_bytes',
                                    'manifest_acked', 'receipt_completed', 'publication_failures'), 0)
        self.stop = threading.Event()
        self.error = None
        self.closed = False
        self._marker_quarantine = []
        self.thread = threading.Thread(target=self._worker, name='hs-manifest-owner', daemon=True)
        self.thread.start()

    def _worker(self):
        while not self.stop.wait(0.05):
            try:
                self.progress()
            except BaseException as exc:
                self.error = RuntimeError(f'manifest owner progress failed: {exc}')
                return

    def _absent_after_remove(self, key):
        try:
            try:
                rc = self.store.remove(key, force=True)
            except TypeError:
                rc = self.store.remove(key)
            if rc is not None and int(rc) not in (0, -704):
                return False
            # rc=0 may still mean lease-deferred. Never return budget on that.
            return int(self.store.is_exist(key)) == 0
        except Exception:
            return False

    def publish(self, spec, sample, schema, namespace, sample_key):
        with self.lock:
            if self.error:
                raise self.error
            token = str(spec['reservation_token'])
            if token in self.manifests:
                raise ValueError('manifest attempt already published')
            base = sample_key(spec, '__manifest').rsplit('/', 1)[0]
            ack_key, receipt_key = base + '/__segment_ack', base + '/__segment_receipt'
            new, dependencies, features = {}, set(), {}
            tensors = {}
            full_bytes = 0
            for feature in schema.features:
                name = spec['features'][feature.name]
                descriptors = []
                for block in sample.segments:
                    tensor = block.features[feature.name].unsqueeze(0)
                    nbytes = tensor.numel() * tensor.element_size()
                    full_bytes += nbytes
                    logical = f'{namespace}/{schema.signature}/{block.digest}/{feature.name}'
                    if not block.complete:
                        logical += '/' + token
                    obj = self.objects.get(logical)
                    if obj is not None and obj['state'] == 'READY':
                        if int(self.store.is_exist(obj['key'])) != 1:
                            raise RuntimeError('protected shared block disappeared; remote owner cannot recover old manifests')
                        self.stats['block_reused'] += 1
                    else:
                        generation = uuid.uuid4().hex
                        key = 'specforge-hs-block/' + hashlib.sha256(logical.encode()).hexdigest() + '/' + generation
                        obj = {'key': key, 'generation': generation, 'nbytes': nbytes,
                               'state': 'WRITING', 'dependencies': set(), 'logical': logical}
                        new[logical] = obj
                        tensors[key] = tensor
                    dependencies.add(obj['key'])
                    descriptors.append({'logical_key': logical, 'key': obj['key'],
                                        'publication_generation': obj['generation'],
                                        'start': block.start, 'end': block.end,
                                        'shape': list(tensor.shape), 'dtype': feature.dtype,
                                        'nbytes': nbytes})
                features[name] = descriptors
            manifest = {'layout': MANIFEST_LAYOUT, 'namespace': namespace,
                        'owner_epoch': self.epoch, 'token': token, 'features': features,
                        'schema_signature': schema.signature,
                        'ack_key': ack_key, 'receipt_key': receipt_key}
            manifest['digest'] = digest_manifest(manifest)
            encoded = json.dumps(manifest, sort_keys=True, separators=(',', ':')).encode()
            result_features, private = {}, []
            for feature in schema.features:
                name = spec['features'][feature.name]
                key = sample_key(spec, name)
                tensors[key] = torch.tensor(list(encoded), dtype=torch.uint8)
                private.append(key)
                result_features[name] = {'key': key, 'shape': [1, sample.n_tokens, feature.width], 'dtype': feature.dtype}
            for item in spec.get('passthrough', ()):
                name = item['name']
                key = sample_key(spec, name)
                tensors[key] = torch.tensor(item['data'], dtype=getattr(torch, item.get('dtype', 'int64'))).reshape(item['shape'])
                private.append(key)
                result_features[name] = {'key': key, 'shape': list(item['shape']), 'dtype': item.get('dtype', 'int64')}
            reservation = {key: tensor.numel() * tensor.element_size() for key, tensor in tensors.items()}
            reservation.update({ack_key: MARKER_BYTES, receipt_key: MARKER_BYTES})
            self.ledger.reserve(reservation)
            rec = {'manifest': manifest, 'dependencies': dependencies, 'private': private,
                   'delete': set(), 'phase': 'PUBLISHING', 'ack_key': ack_key, 'receipt_key': receipt_key}
            self.manifests[token] = rec
            for logical, obj in new.items():
                self.objects[logical] = obj
            for obj in self.objects.values():
                if obj['key'] in dependencies:
                    obj['dependencies'].add(token)
            try:
                block_tensors = {obj['key']: tensors[obj['key']] for obj in new.values()}
                self.ledger.state(block_tensors, 'WRITING')
                if block_tensors:
                    self.stats['physical_put_bytes'] += sum(t.numel()*t.element_size() for t in block_tensors.values())
                    self.put(block_tensors)
                    self.ledger.state(block_tensors, 'READY')
                for obj in new.values():
                    obj['state'] = 'READY'
                headers = {key: tensors[key] for key in private}
                self.ledger.state(headers, 'WRITING')
                self.stats['physical_put_bytes'] += sum(t.numel()*t.element_size() for t in headers.values())
                self.put(headers)
                self.ledger.state(headers, 'READY')
                rec['phase'] = 'READY'
                self.stats['block_new'] += len(new)
                self.stats['manifest_published'] += 1
                self.stats['manifest_dependencies'] += len(dependencies)
                self.stats['logical_feature_bytes'] += full_bytes

            except BaseException as exc:
                self.stats['publication_failures'] += 1
                if getattr(exc, 'capture_writer_drained', True) is False:
                    self.ledger.state(reservation, 'QUARANTINED')
                    rec['phase'] = 'QUARANTINED'
                    for obj in new.values():
                        obj['state'] = 'QUARANTINED'
                else:
                    self._detach(token, rec)
                    rec['phase'] = 'ABORTED'
                raise
            return {'sample_id': str(spec['sample_id']), 'store_id': str(spec['store_id']),
                    'gen': int(spec['gen']), 'features': result_features,
                    'aux_layer_ids': list(schema.features[0].layer_ids),
                    'feature_layout': MANIFEST_LAYOUT, 'segment_manifest': manifest}

    def _detach(self, token, rec):
        rec['delete'].update(rec['private'])
        rec['delete'].add(rec['ack_key'])
        for logical, obj in tuple(self.objects.items()):
            if token in obj['dependencies']:
                obj['dependencies'].remove(token)
                if not obj['dependencies']:
                    obj['state'] = 'DELETING'
                    rec['delete'].add(obj['key'])
                    # A subsequent manifest gets a different physical generation.
                    self.objects.pop(logical)
        self.ledger.state(rec['delete'], 'DELETING')

    def cancel(self, token):
        with self.lock:
            rec = self.manifests.get(token)
            if rec is None:
                return
            if rec['phase'] == 'QUARANTINED':
                raise RuntimeError('cannot cancel a manifest with an unknown native writer')
            if rec['phase'] in ('PUBLISHING', 'READY'):
                self._detach(token, rec)
                rec['phase'] = 'ABORTED'

    def _ack_matches(self, rec):
        value = torch.empty(MARKER_BYTES, dtype=torch.uint8)
        ptr = value.data_ptr()
        rc = self.store.register_buffer(ptr, MARKER_BYTES)
        if rc is not None and int(rc) != 0:
            raise RuntimeError('manifest ACK read registration failed')
        drained = False
        try:
            rc = self.store.get_into(rec['ack_key'], ptr, MARKER_BYTES)
            drained = rc is not None and int(rc) == MARKER_BYTES
            if not drained:
                raise RuntimeError('manifest ACK read completion is unknown')
            if bytes(value.tolist()).hex() != rec['manifest']['digest']:
                raise RuntimeError('manifest ACK digest differs from the exact attempt')
            return True
        finally:
            if drained:
                try:
                    rc = self.store.unregister_buffer(ptr)
                except BaseException:
                    rc = -1
                if rc is not None and int(rc) != 0:
                    self._marker_quarantine.append(value)
                    self.ledger.state([rec['ack_key']], 'QUARANTINED')
                    raise RuntimeError('manifest ACK registration drain failed')
            else:
                self._marker_quarantine.append(value)
                self.ledger.state([rec['ack_key']], 'QUARANTINED')

    def progress(self):
        with self.lock:
            for token, rec in tuple(self.manifests.items()):
                if rec['phase'] == 'READY' and int(self.store.is_exist(rec['ack_key'])) == 1:
                    self._ack_matches(rec)
                    self.ledger.state([rec['ack_key']], 'READY')
                    self._detach(token, rec)
                    rec['phase'] = 'ACKED'
                    self.stats['manifest_acked'] += 1
                if rec['phase'] in ('ACKED', 'ABORTED'):
                    for key in tuple(rec['delete']):
                        if self._absent_after_remove(key):
                            self.ledger.forget(key)
                            rec['delete'].remove(key)
                    if rec['delete']:
                        continue
                    if rec['phase'] == 'ABORTED':
                        self.ledger.forget(rec['receipt_key'])
                        self.manifests.pop(token)
                        continue
                    value = torch.tensor(list(bytes.fromhex(rec['manifest']['digest'])), dtype=torch.uint8)
                    self.ledger.state([rec['receipt_key']], 'WRITING')
                    self.stats['physical_put_bytes'] += MARKER_BYTES
                    try:
                        self.put({rec['receipt_key']: value})
                    except BaseException:
                        # The writer callback retains uncertain native sources.
                        # This exact control object remains charged until an
                        # explicit recovery proves its writer/registration drain.
                        self.ledger.state([rec['receipt_key']], 'QUARANTINED')
                        rec['phase'] = 'QUARANTINED'
                        raise
                    self.ledger.state([rec['receipt_key']], 'READY')
                    rec['phase'] = 'RECEIPT'
                elif rec['phase'] == 'RECEIPT' and int(self.store.is_exist(rec['receipt_key'])) == 0:
                    self.ledger.forget(rec['receipt_key'])
                    self.manifests.pop(token)
                    self.stats['receipt_completed'] += 1

    def health(self):
        with self.lock:
            return {**self.stats, **self.ledger.health(), 'active_manifests': len(self.manifests),
                    'active_shared_objects': len(self.objects),
                    'active_dependencies': sum(len(obj['dependencies']) for obj in self.objects.values()),
                    'owner_epoch': self.epoch, 'owner_error': str(self.error) if self.error else None,
                    'owner_thread_alive': self.thread.is_alive()}

    def close(self):
        if self.closed:
            return
        self.progress()
        with self.lock:
            if self.error:
                raise self.error
            if self.manifests or self.objects:
                raise RuntimeError('manifest owner cannot close before ACK/receipt drains')
        self.stop.set()
        self.thread.join(timeout=5)
        if self.thread.is_alive():
            raise RuntimeError('manifest owner did not stop')
        self.ledger.close()
        self.closed = True
