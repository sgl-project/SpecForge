"""Immutable CPU feature chunks and leases for contiguous prefixes."""

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
    CaptureContractError,
    ContentBlockKey,
    FeatureSchema,
    PrefixLease,
    _normalized_error,
    checked_int,
)


def _token_tuple(tokens: object) -> tuple[int, ...]:
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
    owner: _StorageOwner
    features: dict[str, torch.Tensor] = field(default_factory=dict)
    leases: int = 0


@dataclass
class _Lease:
    handle: PrefixLease
    namespace: str
    entries: tuple[_Entry, ...]
    readers: int = 0
    released: bool = False


class PrefixFeatureCache:
    """Owns released CPU samples and leases their contiguous prefix blocks."""

    def __init__(
        self, schema: FeatureSchema, chunk_tokens: int, capacity_bytes: int
    ) -> None:
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
        self._bytes = 0
        self._peak_bytes = 0
        self._evictions = 0
        self._logical_peak_bytes = 0
        self._owned_groups = 0

    @staticmethod
    def _encode_namespace(namespace: str) -> bytes:
        if not isinstance(namespace, str) or not namespace:
            raise CaptureContractError(
                "namespace must be a nonempty server-validated string"
            )
        return namespace.encode("utf-8")

    def _digest(
        self, encoded: bytes, parent: str, start: int, end: int, tokens: tuple[int, ...]
    ) -> str:
        try:
            parent_bytes = bytes.fromhex(parent)
        except (TypeError, ValueError) as exc:
            raise CaptureContractError("invalid parent SHA256") from exc
        if len(parent_bytes) != 32 or len(parent) != 64:
            raise CaptureContractError("parent must be a complete SHA256")
        body = (
            b"spec-prefix-block-v1\0"
            + struct.pack(
                "<QQQQQ", len(encoded), self.chunk_tokens, start, end, len(tokens)
            )
            + encoded
            + parent_bytes
        )
        body += b"".join(struct.pack("<q", token) for token in tokens)
        return hashlib.sha256(body).hexdigest()

    def block_keys(
        self, namespace: str, tokens: tuple[int, ...]
    ) -> tuple[ContentBlockKey, ...]:
        """Return the canonical keys for the complete chunks in the token prefix."""
        _token_tuple(tokens)
        encoded = self._encode_namespace(namespace)
        parent = hashlib.sha256(
            b"spec-prefix-root-v1\0"
            + struct.pack("<QQ", len(encoded), self.chunk_tokens)
            + encoded
        ).hexdigest()
        result = []
        for start in range(0, len(tokens) - self.chunk_tokens + 1, self.chunk_tokens):
            end = start + self.chunk_tokens
            chunk = tokens[start:end]
            digest = self._digest(encoded, parent, start, end, chunk)
            result.append(ContentBlockKey(namespace, start, end, digest, parent, chunk))
            parent = digest
        return tuple(result)

    def _complete(self, entry: _Entry) -> bool:
        try:
            self.schema.validate(entry.features, self.chunk_tokens, "cpu")
            return True
        except CaptureContractError:
            return False

    def lookup_and_pin(
        self, namespace: str, tokens: tuple[int, ...], limit: int
    ) -> PrefixLease:
        """Pin the longest complete contiguous prefix within the coverage limit."""
        _token_tuple(tokens)
        checked_int(limit, "coverage limit")
        if self.capacity_bytes == 0 or min(len(tokens), limit) < self.chunk_tokens:
            # Validate the namespace even when lookup cannot yield a complete block.
            self._encode_namespace(namespace)
            keys = ()
        else:
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
                entry.owner.leases += 1
                self._lru.move_to_end(entry.key)
            end = len(entries) * self.chunk_tokens
            handle = PrefixLease(uuid.uuid4().hex, end)
            self._leases[handle.lease_id] = _Lease(handle, namespace, tuple(entries))
            return handle

    def _checked_lease(
        self,
        lease: PrefixLease,
        end: int,
        namespace: str | None = None,
        tokens: tuple[int, ...] | None = None,
    ) -> _Lease:
        checked_int(end, "lease read end")
        rec = self._leases.get(getattr(lease, "lease_id", None))
        if rec is None or rec.handle is not lease or rec.released:
            raise CaptureContractError("lease is stale, foreign, or released")
        if end > lease.coverage_end:
            raise CaptureContractError("lease does not cover the requested prefix")
        if namespace is not None and namespace != rec.namespace:
            raise CaptureContractError("lease namespace differs from plan")
        if tokens is not None and not end:
            _token_tuple(tokens)
        if tokens is not None and end:
            expected = self.block_keys(rec.namespace, tokens)
            needed = (end + self.chunk_tokens - 1) // self.chunk_tokens
            if tuple(e.key for e in rec.entries[:needed]) != expected[:needed]:
                raise CaptureContractError("lease content differs from plan")
        return rec

    def _validate_plan_lease(
        self, lease: PrefixLease, end: int, namespace: str, tokens: tuple[int, ...]
    ) -> None:
        with self._lock:
            self._checked_lease(lease, end, namespace, tokens)

    def shrink_lease(self, lease: PrefixLease, end: int) -> PrefixLease:
        """Consume the old handle and retain just the blocks needed up to end."""
        with self._lock:
            rec = self._checked_lease(lease, end)
            if rec.readers:
                raise CaptureContractError(
                    "cannot shrink a lease while a reader is active"
                )
            keep = (end + self.chunk_tokens - 1) // self.chunk_tokens
            for entry in rec.entries[keep:]:
                entry.leases -= 1
                entry.owner.leases -= 1
            self._leases.pop(lease.lease_id)
            handle = PrefixLease(uuid.uuid4().hex, end)
            self._leases[handle.lease_id] = _Lease(
                handle, rec.namespace, rec.entries[:keep]
            )
            return handle

    def _finish_release(self, rec: _Lease) -> None:
        for entry in rec.entries:
            entry.leases -= 1
            entry.owner.leases -= 1
            assert entry.owner.leases >= 0
            assert entry.leases >= 0
        self._leases.pop(rec.handle.lease_id, None)

    def release_lease(self, lease: PrefixLease) -> None:
        """Release the lease once all copies already using it have returned."""
        with self._lock:
            rec = self._leases.get(getattr(lease, "lease_id", None))
            if rec is None or rec.handle is not lease or rec.released:
                return
            rec.released = True
            if not rec.readers:
                self._finish_release(rec)

    def copy_prefix_into(
        self, lease: PrefixLease, end: int, dest: Mapping[str, torch.Tensor]
    ) -> None:
        """Copy the requested leased rows into caller-owned CPU tensors."""
        with self._lock:
            rec = self._checked_lease(lease, end)
            if set(dest) != {f.name for f in self.schema.features}:
                raise CaptureContractError(
                    "destination does not contain the complete schema"
                )
            rows = {
                int(t.shape[0]) if isinstance(t, torch.Tensor) and t.ndim == 2 else -1
                for t in dest.values()
            }
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
                if spans and spans[-1][2] is entry.owner:
                    old_start, _, owner = spans[-1]
                    spans[-1] = (old_start, stop, owner)
                else:
                    spans.append((start, stop, entry.owner))
            with torch.no_grad():
                for start, stop, owner in spans:
                    for spec in self.schema.features:
                        source = owner.features[spec.name][start:stop]
                        dest[spec.name][start:stop].copy_(source)
        except BaseException as exc:
            failure = _normalized_error(exc)
        finally:
            # Drop source views before releasing the reader's lease protection.
            source = owner = None
            spans.clear()
            with self._lock:
                rec.readers -= 1
                if rec.released and not rec.readers:
                    self._finish_release(rec)
        if failure is not None:
            raise failure from None

    def _record_logical_peak_locked(self) -> None:
        self._logical_peak_bytes = max(
            self._logical_peak_bytes, len(self._entries) * self._chunk_bytes
        )

    def _remove_entry_locked(self, entry: _Entry) -> None:
        entry.features.clear()
        self._entries.pop(entry.key)
        self._lru.pop(entry.key)
        owner = entry.owner
        owner.keys.remove(entry.key)
        if not owner.keys:
            # Drop the backing storage references before returning the budget.
            owner.features.clear()
            self._bytes -= owner.nbytes
            self._owned_groups -= 1

    def _make_room_locked(self, amount: int) -> bool:
        if amount > self.capacity_bytes:
            return False
        while self._bytes + amount > self.capacity_bytes:
            victim = None
            for key in self._lru:
                entry = self._entries[key]
                if entry.owner.leases:
                    continue
                victim = entry
                break
            if victim is None:
                return False
            # Evict an entire owner group. Partial eviction cannot free its
            # backing allocation and only creates holes in prefix coverage.
            for key in tuple(victim.owner.keys):
                self._remove_entry_locked(self._entries[key])
                self._evictions += 1
        return True

    def _adopt_released_sample(
        self, keys: tuple[ContentBlockKey, ...], features: Mapping[str, torch.Tensor]
    ) -> bool:
        """Take ownership of full CPU storages after the sample readers finish.

        The controller prepares the complete block keys before release. Each
        block is a view, and the cache charges the entire backing allocation.
        """
        if not keys:
            return False
        rows = next(iter(features.values())).shape[0]
        self.schema.validate(features, rows, "cpu")
        storages = {
            (
                str(t.device),
                t.untyped_storage().data_ptr(),
            ): t.untyped_storage().nbytes()
            for t in features.values()
        }
        nbytes = sum(storages.values())
        with self._lock:
            if all(key in self._entries for key in keys) or not self._make_room_locked(
                nbytes
            ):
                return False

            # Eviction can remove keys for earlier blocks of the same prefix.
            owner = _StorageOwner(nbytes, dict(features))
            pending: list[_Entry] = []
            try:
                for key in keys:
                    if key not in self._entries:
                        entry = _Entry(key, owner)
                        pending.append(entry)
                        for name, tensor in features.items():
                            entry.features[name] = tensor[key.start : key.end]
                owner.keys.update(entry.key for entry in pending)
                for entry in pending:
                    self._entries[entry.key] = entry
                    self._lru[entry.key] = None
            except BaseException as exc:
                failure = _normalized_error(exc)
                for entry in pending:
                    if self._entries.get(entry.key) is entry:
                        self._entries.pop(entry.key)
                        self._lru.pop(entry.key, None)
                    entry.features.clear()
                owner.features.clear()
            else:
                self._bytes += nbytes
                self._peak_bytes = max(self._peak_bytes, self._bytes)
                self._owned_groups += 1
                self._record_logical_peak_locked()
                return True
        raise failure from None

    def stats(self) -> dict[str, int]:
        """Report backing-storage usage, indexed rows, leases, and evictions."""
        with self._lock:
            return {
                "ready_blocks": len(self._entries),
                "cache_bytes": self._bytes,
                "cache_peak_bytes": self._peak_bytes,
                "cache_logical_bytes": len(self._entries) * self._chunk_bytes,
                "cache_logical_peak_bytes": self._logical_peak_bytes,
                "cache_owned_bytes": self._bytes,
                "cache_owned_groups": self._owned_groups,
                "active_leases": len(self._leases),
                "leased_blocks": sum(e.leases > 0 for e in self._entries.values()),
                "evictions": self._evictions,
            }
