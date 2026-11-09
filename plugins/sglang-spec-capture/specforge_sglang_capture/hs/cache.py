"""Immutable content chunks with atomic continuous coverage and read leases."""

from __future__ import annotations

import hashlib
import struct
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Mapping

import torch

from .protocol import (
    CaptureContractError, ContentBlockKey, ExecutionIdentity, FeatureSchema,
    FillReservation, PrefixLease, checked_int, _normalized_error,
)


def _token_tuple(tokens) -> tuple[int, ...]:
    if not isinstance(tokens, tuple):
        raise CaptureContractError("tokens must be an immutable tuple")
    if any(type(t) is not int or not 0 <= t < 2**63 for t in tokens):
        raise CaptureContractError("tokens must be nonnegative canonical int64 ids")
    return tokens


@dataclass
class _StorageOwner:
    nbytes: int
    features: dict[str, torch.Tensor]
    keys: set[ContentBlockKey] = field(default_factory=set)
    leases: int = 0


@dataclass
class _Entry:
    key: ContentBlockKey
    nbytes: int
    state: str = "WRITING"
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    leases: int = 0
    owner: _StorageOwner | None = None


@dataclass
class _Lease:
    handle: PrefixLease
    namespace: str
    entries: tuple[_Entry, ...]
    readers: int = 0
    released: bool = False


@dataclass
class _Fill:
    handle: FillReservation
    entry: _Entry
    writer_active: bool = False
    abort_requested: bool = False


class PrefixFeatureCache:
    def __init__(self, schema: FeatureSchema, chunk_tokens: int, capacity_bytes: int):
        if not isinstance(schema, FeatureSchema):
            raise CaptureContractError("cache requires a FeatureSchema")
        self.schema = schema
        self.chunk_tokens = checked_int(chunk_tokens, "chunk_tokens", 1)
        self.capacity_bytes = checked_int(capacity_bytes, "capacity_bytes")
        self._chunk_bytes = self.chunk_tokens * self.schema.row_bytes
        self._lock = threading.RLock()
        self._entries: dict[ContentBlockKey, _Entry] = {}
        self._lru: OrderedDict[ContentBlockKey, None] = OrderedDict()
        self._leases: dict[str, _Lease] = {}
        self._fills: dict[str, _Fill] = {}
        self._bytes = self._peak_bytes = self._evictions = self._fill_skips = 0
        self._logical_peak_bytes = self._adopted_samples = self._adoption_skips = 0
        self._owned_bytes = self._owned_peak_bytes = self._owned_groups = 0

    def _root_digest(self, namespace: str) -> str:
        if not isinstance(namespace, str) or not namespace:
            raise CaptureContractError("namespace must be a nonempty server-validated string")
        encoded = namespace.encode("utf-8")
        return hashlib.sha256(b"spec-prefix-root-v1\0" + struct.pack("<QQ", len(encoded), self.chunk_tokens) + encoded).hexdigest()

    def _digest(self, namespace: str, parent: str, start: int, end: int,
                tokens: tuple[int, ...]) -> str:
        encoded = namespace.encode("utf-8")
        try:
            parent_bytes = bytes.fromhex(parent)
        except (TypeError, ValueError) as exc:
            raise CaptureContractError("invalid parent SHA256") from exc
        if len(parent_bytes) != 32 or len(parent) != 64:
            raise CaptureContractError("parent must be a complete SHA256")
        body = (b"spec-prefix-block-v1\0" + struct.pack("<QQQQQ", len(encoded), self.chunk_tokens,
                start, end, len(tokens)) + encoded + parent_bytes)
        body += b"".join(struct.pack("<q", token) for token in tokens)
        return hashlib.sha256(body).hexdigest()

    def block_keys(self, namespace: str, tokens: tuple[int, ...]) -> tuple[ContentBlockKey, ...]:
        """Complete chunks only; no persistent lookup-side token table."""
        _token_tuple(tokens)
        parent = self._root_digest(namespace)
        result = []
        for start in range(0, len(tokens) - self.chunk_tokens + 1, self.chunk_tokens):
            end = start + self.chunk_tokens
            chunk = tokens[start:end]
            digest = self._digest(namespace, parent, start, end, chunk)
            result.append(ContentBlockKey(namespace, start, end, digest, parent, chunk))
            parent = digest
        return tuple(result)

    def _validate_key(self, key: ContentBlockKey) -> None:
        if not isinstance(key, ContentBlockKey):
            raise CaptureContractError("fill needs a content block key")
        self._root_digest(key.namespace)
        checked_int(key.start, "block start")
        checked_int(key.end, "block end")
        if key.start % self.chunk_tokens or key.end - key.start != self.chunk_tokens:
            raise CaptureContractError("only aligned complete chunks can be cached")
        _token_tuple(key._tokens)
        if len(key._tokens) != self.chunk_tokens:
            raise CaptureContractError("block tokens do not match its complete range")
        if key.start == 0 and key.parent_digest != self._root_digest(key.namespace):
            raise CaptureContractError("first block has an invalid root identity")
        if key.digest != self._digest(key.namespace, key.parent_digest, key.start, key.end, key._tokens):
            raise CaptureContractError("content key does not match its tokens/range")

    def _complete(self, entry: _Entry) -> bool:
        if entry.state != "READY":
            return False
        try:
            self.schema.validate(entry.features, self.chunk_tokens, "cpu")
            return True
        except CaptureContractError:
            return False

    def lookup_and_pin(self, namespace: str, tokens: tuple[int, ...], limit: int) -> PrefixLease:
        _token_tuple(tokens)
        checked_int(limit, "coverage limit")
        keys = self.block_keys(namespace, tokens[: min(len(tokens), limit)])
        with self._lock:
            entries = []
            for key in keys:
                entry = self._entries.get(key)
                if entry is None or not self._complete(entry):
                    break
                entries.append(entry)
            # The query and every refcount update are one metadata transaction.
            for entry in entries:
                entry.leases += 1
                if entry.owner is not None:
                    entry.owner.leases += 1
                self._lru.move_to_end(entry.key)
            end = len(entries) * self.chunk_tokens
            handle = PrefixLease(uuid.uuid4().hex, end)
            self._leases[handle.lease_id] = _Lease(handle, namespace, tuple(entries))
            return handle

    def _checked_lease(self, lease: PrefixLease, end: int,
                       namespace: str | None = None, tokens: tuple[int, ...] | None = None) -> _Lease:
        checked_int(end, "lease read end")
        rec = self._leases.get(getattr(lease, "lease_id", None))
        if rec is None or rec.handle is not lease or rec.released:
            raise CaptureContractError("lease is stale, foreign, or released")
        if end > lease.coverage_end:
            raise CaptureContractError("lease does not cover the requested prefix")
        if namespace is not None and namespace != rec.namespace:
            raise CaptureContractError("lease namespace differs from plan")
        if tokens is not None:
            expected = self.block_keys(rec.namespace, tokens)
            needed = (end + self.chunk_tokens - 1) // self.chunk_tokens
            if tuple(e.key for e in rec.entries[:needed]) != expected[:needed]:
                raise CaptureContractError("lease content differs from plan")
        return rec

    def _validate_plan_lease(self, lease, end, namespace, tokens) -> None:
        with self._lock:
            self._checked_lease(lease, end, namespace, tokens)

    def shrink_lease(self, lease: PrefixLease, end: int) -> PrefixLease:
        with self._lock:
            rec = self._checked_lease(lease, end)
            if rec.readers:
                raise CaptureContractError("cannot shrink a lease while a reader is active")
            keep = (end + self.chunk_tokens - 1) // self.chunk_tokens
            for entry in rec.entries[keep:]:
                entry.leases -= 1
                if entry.owner is not None:
                    entry.owner.leases -= 1
            self._leases.pop(lease.lease_id)
            handle = PrefixLease(uuid.uuid4().hex, end)
            self._leases[handle.lease_id] = _Lease(handle, rec.namespace, rec.entries[:keep])
            return handle

    def _finish_release(self, rec: _Lease) -> None:
        for entry in rec.entries:
            entry.leases -= 1
            if entry.owner is not None:
                entry.owner.leases -= 1
                assert entry.owner.leases >= 0
            assert entry.leases >= 0
        self._leases.pop(rec.handle.lease_id, None)

    def release_lease(self, lease: PrefixLease) -> None:
        with self._lock:
            rec = self._leases.get(getattr(lease, "lease_id", None))
            if rec is None or rec.handle is not lease or rec.released:
                return
            rec.released = True
            if not rec.readers:
                self._finish_release(rec)

    def copy_prefix_into(self, lease: PrefixLease, end: int,
                         dest: Mapping[str, torch.Tensor]) -> None:
        with self._lock:
            rec = self._checked_lease(lease, end)
            if set(dest) != {f.name for f in self.schema.features}:
                raise CaptureContractError("destination does not contain the complete schema")
            rows = {int(t.shape[0]) if isinstance(t, torch.Tensor) and t.ndim == 2 else -1 for t in dest.values()}
            if len(rows) != 1 or next(iter(rows)) < end:
                raise CaptureContractError("destination cannot hold the leased prefix")
            self.schema.validate(dest, next(iter(rows)), "cpu")
            entries = rec.entries[: (end + self.chunk_tokens - 1) // self.chunk_tokens]
            if any(not self._complete(e) for e in entries):
                raise CaptureContractError("leased cache block is missing or corrupt")
            rec.readers += 1
        failure = None
        try:
            spans = []
            for entry in entries:
                start, stop = entry.key.start, min(end, entry.key.end)
                if entry.owner is not None and spans and spans[-1][2] is entry.owner:
                    old_start, _, owner, _ = spans[-1]
                    spans[-1] = (old_start, stop, owner, None)
                else:
                    spans.append((start, stop, entry.owner, entry.features))
            with torch.no_grad():
                for start, stop, owner, block in spans:
                    for spec in self.schema.features:
                        source = (owner.features[spec.name][start:stop] if owner is not None
                                  else block[spec.name][:stop - start])
                        dest[spec.name][start:stop].copy_(source)
            # Drop source views before releasing the reader's lease protection.
            source = owner = block = None
            spans.clear()
        except BaseException as exc:
            failure = _normalized_error(exc)
        finally:
            source = owner = block = None
            spans.clear()
            with self._lock:
                rec.readers -= 1
                if rec.released and not rec.readers:
                    self._finish_release(rec)
        if failure is not None:
            raise failure from None

    def reserve_fill(self, key: ContentBlockKey,
                     execution: ExecutionIdentity) -> FillReservation | None:
        self._validate_key(key)
        if not isinstance(execution, ExecutionIdentity):
            raise CaptureContractError("fill requires an execution identity")
        with self._lock:
            if key in self._entries or self._chunk_bytes > self.capacity_bytes:
                self._fill_skips += 1
                return None
            if not self._make_room_locked(self._chunk_bytes):
                self._fill_skips += 1
                return None
            handle = FillReservation(uuid.uuid4().hex, key, execution, self._chunk_bytes)
            entry = _Entry(key, self._chunk_bytes)
            self._entries[key] = entry
            self._fills[handle.reservation_id] = _Fill(handle, entry)
            self._bytes += self._chunk_bytes
            self._peak_bytes = max(self._peak_bytes, self._bytes)
            self._record_logical_peak_locked()
            return handle

    def _record_logical_peak_locked(self):
        self._logical_peak_bytes = max(self._logical_peak_bytes,
                                       len(self._entries) * self._chunk_bytes)

    def _remove_entry_locked(self, entry):
        entry.features.clear()
        self._entries.pop(entry.key, None)
        self._lru.pop(entry.key, None)
        owner = entry.owner
        if owner is None:
            self._bytes -= entry.nbytes
        else:
            owner.keys.discard(entry.key)
            if not owner.keys:
                # Drop the final owner references before making storage budget
                # available. A chunk view never charges only its logical rows.
                owner.features.clear()
                self._bytes -= owner.nbytes
                self._owned_bytes -= owner.nbytes
                self._owned_groups -= 1
            entry.owner = None

    def _make_room_locked(self, amount):
        if amount > self.capacity_bytes:
            return False
        while self._bytes + amount > self.capacity_bytes:
            victim = None
            for key in self._lru:
                entry = self._entries[key]
                if entry.leases:
                    continue
                if entry.owner is not None and entry.owner.leases:
                    continue
                victim = entry
                break
            if victim is None:
                return False
            # Evict an entire owner group. Partial eviction cannot free its
            # backing allocation and only creates holes in prefix coverage.
            keys = tuple(victim.owner.keys) if victim.owner is not None else (victim.key,)
            for key in keys:
                self._remove_entry_locked(self._entries[key])
                self._evictions += 1
        return True

    def _adopt_released_sample(self, keys, features) -> bool:
        """Take exclusive CPU ownership after every MaterializedSample reader ends.

        Only the controller calls this internal path. Key hashes are prepared
        before release. No CPU tensor allocation/copy occurs here; every chunk
        view belongs to a group charged for its complete underlying storages,
        including any reused prefix, padding, and incomplete tail.
        """
        if not keys:
            return False
        rows = next(iter(features.values())).shape[0]
        self.schema.validate(features, rows, "cpu")
        storages = {(str(t.device), t.untyped_storage().data_ptr()): t.untyped_storage().nbytes()
                    for t in features.values()}
        nbytes = sum(storages.values())
        owner = _StorageOwner(nbytes, dict(features))
        pending = []
        failure = None
        published = charged = False
        entry = tensor = None
        try:
            with self._lock:
                if not any(key not in self._entries for key in keys):
                    self._adoption_skips += 1
                    self._fill_skips += len(keys)
                elif not self._make_room_locked(nbytes):
                    self._adoption_skips += 1
                    self._fill_skips += len(keys)
                else:
                    # Capacity eviction may itself have removed these keys.
                    # Register every view with pending before constructing the
                    # next one, so partial metadata errors drop all references.
                    for key in keys:
                        if key not in self._entries:
                            entry = _Entry(key, self._chunk_bytes, "READY", owner=owner)
                            pending.append(entry)
                            for name, tensor in features.items():
                                entry.features[name] = tensor[key.start:key.end]
                    owner.keys.update(entry.key for entry in pending)
                    self._bytes += nbytes
                    self._peak_bytes = max(self._peak_bytes, self._bytes)
                    self._owned_bytes += nbytes
                    self._owned_groups += 1
                    charged = True
                    self._owned_peak_bytes = max(self._owned_peak_bytes, self._owned_bytes)
                    for entry in pending:
                        self._entries[entry.key] = entry
                        self._lru[entry.key] = None
                    self._record_logical_peak_locked()
                    self._adopted_samples += 1
                    self._fill_skips += len(keys) - len(pending)
                    published = True
        except BaseException as exc:
            failure = _normalized_error(exc)
        finally:
            tensor = entry = None
            if not published:
                with self._lock:
                    for entry in pending:
                        if self._entries.get(entry.key) is entry:
                            self._entries.pop(entry.key)
                            self._lru.pop(entry.key, None)
                        entry.features.clear()
                        entry.owner = None
                    if charged:
                        self._bytes -= nbytes
                        self._owned_bytes -= nbytes
                        self._owned_groups -= 1
                pending.clear()
                owner.features.clear()
                owner.keys.clear()
                entry = None
        if failure is not None:
            raise failure from None
        return published

    def _drop_fill(self, rec: _Fill) -> None:
        if self._entries.get(rec.entry.key) is rec.entry:
            self._remove_entry_locked(rec.entry)
        self._fills.pop(rec.handle.reservation_id, None)

    def publish_ready(self, fill: FillReservation,
                      features: Mapping[str, torch.Tensor]) -> bool:
        with self._lock:
            rec = self._fills.get(getattr(fill, "reservation_id", None))
            if rec is None or rec.handle is not fill or rec.writer_active or rec.abort_requested:
                return False
            rec.writer_active = True
        owned = {}
        tensor = None
        error = None
        try:
            self.schema.validate(features, self.chunk_tokens, "cpu")
            with torch.no_grad():
                for spec in self.schema.features:
                    tensor = torch.empty((self.chunk_tokens, spec.width), dtype=features[spec.name].dtype, device="cpu")
                    tensor.copy_(features[spec.name].detach())
                    owned[spec.name] = tensor
            # No caller tensor/view becomes durable cache storage.
        except BaseException as exc:
            error = exc
        # A caught exception traceback can retain a failed copy's arguments.
        # Normalize it and drop all owned locals before reusing WRITING bytes.
        failure = _normalized_error(error) if error is not None else None
        error = None
        tensor = None
        with self._lock:
            rec.writer_active = False
            if failure is not None or rec.abort_requested:
                owned.clear()
                self._drop_fill(rec)
                if failure is not None:
                    raise failure from None
                return False
            rec.entry.features = owned
            rec.entry.state = "READY"
            self._fills.pop(fill.reservation_id)
            self._lru[fill.key] = None
            return True

    def abort_fill(self, fill: FillReservation) -> None:
        with self._lock:
            rec = self._fills.get(getattr(fill, "reservation_id", None))
            if rec is None or rec.handle is not fill:
                return
            if rec.writer_active:
                rec.abort_requested = True
            else:
                self._drop_fill(rec)

    def stats(self) -> dict[str, int]:
        with self._lock:
            ready_bytes = sum(e.nbytes for e in self._entries.values()
                              if e.state == "READY" and e.owner is None)
            ready_bytes += self._owned_bytes
            return {
                "ready_blocks": sum(e.state == "READY" for e in self._entries.values()),
                "writing_blocks": sum(e.state == "WRITING" for e in self._entries.values()),
                "ready_bytes": ready_bytes,
                "writing_bytes": sum(e.nbytes for e in self._entries.values() if e.state == "WRITING"),
                "cache_bytes": self._bytes, "cache_peak_bytes": self._peak_bytes,
                "cache_logical_bytes": len(self._entries) * self._chunk_bytes,
                "cache_logical_peak_bytes": self._logical_peak_bytes,
                "cache_owned_bytes": self._owned_bytes,
                "cache_owned_groups": self._owned_groups,
                "cache_owned_peak_bytes": self._owned_peak_bytes,
                "cache_adopted_samples": self._adopted_samples,
                "cache_adoption_skips": self._adoption_skips,
                "active_leases": len(self._leases), "fill_reservations": len(self._fills),
                "leased_blocks": sum(e.leases > 0 for e in self._entries.values()),
                "evictions": self._evictions, "fill_skips": self._fill_skips,
            }
