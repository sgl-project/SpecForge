"""Version-one capture control client and serializable sample writer fences."""
from __future__ import annotations

import hashlib
import json
import threading
import time
import uuid
from dataclasses import dataclass, field


class CaptureControlRejected(RuntimeError):
    """The server explicitly rejected the operation without granting a permit."""


def request_digest(tokens, spec):
    value = {"tokens": list(tokens), "features": spec.get("features", {}),
             "passthrough": spec.get("passthrough", [])}
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def http_control(url, json_body, timeout):
    import requests
    response = requests.post(url, json=json_body, timeout=timeout)
    response.raise_for_status()
    return response.json()


class CaptureProtocol:
    def __init__(self, base_url, post, timeout):
        self.url = base_url.rstrip('/') + '/spec_capture/control'
        self.post = post
        self.timeout = timeout
        self.negotiated = False
        self._capability = None
        self._capability_known = False
        self._capability_lock = threading.RLock()

    def call(self, body):
        try:
            result = self.post(self.url, json_body={"protocol_version": 1, **body}, timeout=self.timeout)
        except Exception as exc:
            if body.get('action') != 'capability' and getattr(getattr(exc, 'response', None), 'status_code', None) == 409:
                # Refresh discovery only. Never replay an uncertain mutation.
                try:
                    self.capability(refresh=True)
                except Exception:
                    pass  # Keep the original rejection recognizable to its owner.
            raise
        if not isinstance(result, dict) or result.get('error'):
            raise RuntimeError('capture control failed: ' + repr(result))
        return result

    def capability(self, *, refresh=False):
        with self._capability_lock:
            if self._capability_known and not refresh:
                return dict(self._capability) if self._capability is not None else None
            return self._discover_capability()

    def invalidate_epoch(self, server_epoch):
        if self._capability is not None and self._capability['server_epoch'] != server_epoch:
            self.capability(refresh=True)

    def _discover_capability(self):
        try:
            result = self.call({'action': 'capability'})
        except Exception as exc:
            status = getattr(getattr(exc, 'response', None), 'status_code', None)
            if status == 404 and not self.negotiated:
                self._capability = None
                self._capability_known = True
                return None
            raise
        if result.get('protocol_version') != 1:
            if self.negotiated:
                raise RuntimeError('negotiated capture protocol disappeared')
            self._capability = None
            self._capability_known = True
            return None
        required = ('server_epoch', 'capture_namespace', 'capture_signature', 'schema', 'cache_enabled')
        if any(name not in result for name in required):
            raise RuntimeError('incomplete capture capability')
        self.negotiated = True
        self._capability_known = True
        self._capability = dict(result)
        return dict(result)

    def begin(self, capability, spec, tokens, attempt_id):
        reply = self.call({'action': 'begin', **self._begin_body(capability, spec, tokens, attempt_id)})
        return self._begin_wire(capability, spec, attempt_id, reply)

    def _begin_body(self, capability, spec, tokens, attempt_id):
        return {**{name: spec[name] for name in ('store_id', 'sample_id', 'gen')},
                'server_epoch': capability['server_epoch'], 'attempt_id': attempt_id,
                'request_digest': request_digest(tokens, spec),
                'publication_ownership': capability.get('published_ownership')}

    def begin_batch(self, capability, requests):
        if not capability.get('begin_batch'):
            return [self.begin(capability, spec, tokens, attempt_id) for spec, tokens, attempt_id in requests]
        tasks = [self._begin_body(capability, *request) for request in requests]
        result = self.call({'action': 'begin_batch', 'server_epoch': capability['server_epoch'], 'tasks': tasks})
        replies = result.get('attempts')
        if not isinstance(replies, list) or len(replies) != len(requests):
            raise RuntimeError('capture begin_batch count mismatch')
        return [self._begin_wire(capability, spec, attempt_id, reply)
                for (spec, _, attempt_id), reply in zip(requests, replies)]

    def _begin_wire(self, capability, spec, attempt_id, reply):
        identity = {name: spec[name] for name in ('store_id', 'sample_id', 'gen')}
        for name, value in {**identity, 'attempt_id': attempt_id, 'server_epoch': capability['server_epoch']}.items():
            if reply.get(name) != value:
                raise RuntimeError('capture begin identity mismatch: ' + name)
        if not reply.get('reservation_token'):
            raise RuntimeError('capture begin omitted server reservation')
        return {**identity, 'protocol_version': 1, 'server_epoch': capability['server_epoch'],
                'attempt_id': attempt_id, 'reservation_token': reply['reservation_token'],
                'control_url': self.url, 'timeout_s': self.timeout,
                **({'object_key_layout': 'attempt-v1'} if reply.get('object_key_layout') == 'attempt-v1' else {})}


@dataclass
class WriterFence:
    """A sample's writer identity; serializable fields travel on every SampleRef."""
    wire: dict
    names: tuple
    post: object = http_control
    writer_drained: bool = False
    transferred: bool = False
    removed: bool = False
    acknowledged: bool = False
    adopted: bool = False
    cleanup_owner: str = field(default_factory=lambda: uuid.uuid4().hex)
    cleanup_token: str | None = None
    remote_retired: bool = False
    lock: object = field(default_factory=threading.RLock)

    @property
    def token(self):
        return self.wire['reservation_token']

    def control(self, action, **values):
        body = {name: self.wire[name] for name in ('server_epoch', 'store_id', 'sample_id', 'gen', 'attempt_id', 'reservation_token')}
        try:
            result = self.post(self.wire['control_url'], json_body={'protocol_version': 1, 'action': action, **body, **values},
                               timeout=min(float(self.wire.get('timeout_s', 30.0)), 30.0))
        except Exception as exc:
            if getattr(exc, 'capture_control_rejected', False) or getattr(getattr(exc, 'response', None), 'status_code', None) == 409:
                raise CaptureControlRejected(str(exc)) from None
            raise
        if isinstance(result, dict) and result.get('error'):
            raise CaptureControlRejected('capture fence control rejected: ' + str(result['error']))
        if not isinstance(result, dict):
            raise RuntimeError('capture fence control returned invalid response')
        if any(result.get(name) != value for name, value in body.items()):
            raise RuntimeError('capture fence identity changed')
        return result

    def accept_transfer(self, certificate):
        if certificate.get('publication_ownership') != 'attempt-v1':
            return False
        if self.wire.get('object_key_layout') != 'attempt-v1' or certificate.get('state') != 'PUBLISHED' or certificate.get('writer_drained') is not True:
            raise RuntimeError('capture ownership transfer before writer drain')
        if any(certificate.get(name) != self.wire[name] for name in
               ('server_epoch', 'store_id', 'sample_id', 'gen', 'attempt_id', 'reservation_token')):
            raise RuntimeError('capture ownership certificate identity mismatch')
        expected = attempt_keys(self.wire, self.names)
        if certificate.get('physical_keys') != expected:
            raise RuntimeError('capture ownership certificate keys mismatch')
        self.wire.update(publication_ownership='attempt-v1', physical_keys=expected,
                         writer_drained=True, state='PUBLISHED')
        self.transferred = self.writer_drained = True
        return True

    def drain(self, *, cancel=True):
        if self.writer_drained:
            return
        deadline = time.monotonic() + float(self.wire.get('timeout_s', 30.0))
        while True:
            result = self.control('resolve', cancel=cancel)
            if result.get('writer_drained'):
                self.accept_transfer(result.get("publication_certificate", result))
                self.writer_drained = True
                return
            if time.monotonic() >= deadline:
                raise TimeoutError('capture writer has not drained; sample retained')
            time.sleep(0.01)

    def acquire_read(self, reader_owner):
        reply = self.control('read_acquire', reader_owner=reader_owner)
        if not reply.get('reader_token') or reply.get('reader_owner') != reader_owner:
            raise RuntimeError('server did not grant exact read ownership')
        return reply['reader_token']

    def release_read(self, reader_owner, reader_token):
        reply = self.control('read_release', reader_owner=reader_owner, reader_token=reader_token)
        if not reply.get('reader_released') or reply.get('reader_token') != reader_token:
            raise RuntimeError('server did not release exact read ownership')

    def acquire_cleanup(self):
        # Only a complete immutable-key ownership certificate bypasses permits.
        self.drain(cancel=True)
        if self.transferred:
            return
        reply = self.control('cleanup_acquire', cleanup_owner=self.cleanup_owner)
        if reply.get('state') == 'RETIRED':
            self.remote_retired = self.removed = self.acknowledged = True
            return
        if not reply.get('cleanup_token') or reply.get('cleanup_owner') != self.cleanup_owner:
            raise RuntimeError('server did not grant exact cleanup ownership')
        self.cleanup_token = reply['cleanup_token']

    def acknowledge(self):
        if self.acknowledged:
            return
        if not self.writer_drained or not self.removed:
            raise RuntimeError('cleanup acknowledgement before writer/remove drain')
        if self.transferred:
            self.acknowledged = True
            return
        if not self.cleanup_token:
            raise RuntimeError('cleanup acknowledgement without server permit')
        result = self.control('cleanup_done', cleanup_token=self.cleanup_token)
        if result.get('state') != 'RETIRED':
            raise RuntimeError('capture cleanup was not retired')
        self.acknowledged = True


def attempt_keys(wire, names):
    """Immutable physical identity; a stale process can only address its attempt."""
    base = f"{wire['store_id']}/{wire['sample_id']}/g{int(wire['gen'])}/attempt/{wire['server_epoch']}/{wire['reservation_token']}"
    return {str(name): base + '/' + str(name) for name in names}
