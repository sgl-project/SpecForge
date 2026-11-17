# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake-backed tensor store for the disaggregated runtime.

``SharedDirFeatureStore`` (``disaggregated.py``) locked down the disaggregation
*contract* over a shared POSIX directory. ``MooncakeFeatureStore`` swaps that
transport for the Mooncake distributed object store (RDMA zero-copy across
nodes) **behind the exact same FeatureStore API** — the control/training planes
see nothing new. The producer (rollout/ingest) ``put()``s feature tensors into
the Mooncake store on one node; the consumer (trainer) ``get()``s them on
another, peer-to-peer, with no shared filesystem. That is what makes this a
genuine network object store rather than a shared mount.

The wire contract is intentionally singular: every tensor is transferred as a
raw buffer with ``put_from``/``get_into``. Shape and dtype travel in the
metadata-only :class:`SampleRef`; no serialized tensor blob is accepted or
produced. Construction fails immediately when the installed Mooncake client
does not expose that API.

Contract carried from the reference backend:

* **B5 — no use-after-free.** ``get()`` after ``release``/``abort`` raises
  ``KeyError``; a generation guard rejects a stale ref after a re-``put``;
  every ``get()`` returns fresh caller-owned tensors, so the loader skips its
  otherwise-default defensive clone.
* **B9 — auth in disaggregated mode** (shared-secret :class:`AuthPolicy`).

Lifetime: Mooncake's default eviction is approximate-LRU for a KV *cache*, which
would silently drop a committed-but-unacked feature when the trainer lags hours
(turning ``get()`` into a ``KeyError`` and violating the controller's
no-data-loss guarantee). When Mooncake exposes ``with_hard_pin``, SpecForge
therefore **hard-pins** every object on ``put`` and frees it only by explicit
``remove()`` on consume/abort — SpecForge is the sole lifetime authority, not
Mooncake's LRU. Older and Ascend Mooncake clients may not expose that field; the
store logs a warning and inherits their default pin behavior, so deployments
that require the strict no-eviction guarantee must use a hard-pin-capable
client. Because ``remove()`` is a real (fallible) RPC, ``release()`` parks a
failed free in ``_release_pending`` and ``gc()`` retries up to
``max_release_attempts`` during steady state. Lifecycle shutdown calls
:meth:`drain_pending_removals`, a separate bounded retry that raises if physical
removal never succeeds; failed removals are never silently dropped from
bookkeeping.

Concurrency: ``release``/``abort``/``gc`` hold ``self._lock`` across the
``remove()`` RPC. The lock is what makes consume-once free race-free against a
concurrent ``get()`` (it prevents a re-lease between "decide to free" and the
remote delete), exactly as ``SharedDirFeatureStore`` holds its lock across
``os.remove``. For the offline single-consumer path this is fine; a high-fanout
online deployment that wants the ``remove()`` RPC off the critical section needs
a tombstone-then-free protocol — a follow-up tied to the shared metadata index.
"""

from __future__ import annotations

from contextlib import contextmanager, ExitStack
from dataclasses import dataclass

import logging
import os
import threading
import time
import uuid
import weakref
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from specforge.runtime.data_plane.pool_diagnostics import get_diagnostics
from specforge.runtime.contracts import SCHEMA_VERSION, FeatureHandle, SampleRef
from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.feature_store import (
    DEFAULT_PENDING_DRAIN_MAX_ATTEMPTS,
    DEFAULT_PENDING_DRAIN_RETRY_INTERVAL_S,
    DEFAULT_SAMPLE_DRAIN_MAX_ATTEMPTS,
    DEFAULT_SAMPLE_DRAIN_RETRY_INTERVAL_S,
    FeatureStore,
    spec_from_tensor,
)
from specforge.runtime.data_plane.mooncake_rdma import (
    check_rdma_environment,
    resolve_rdma_devices,
)

logger = logging.getLogger(__name__)

# Defaults for MooncakeDistributedStore.setup(); override via ``setup_kwargs``.
#: Mooncake ``ErrorCode`` values the store reasons about explicitly.
MOONCAKE_OBJECT_NOT_FOUND = -704  # remove()/get() on an already-freed key
MOONCAKE_OBJECT_HAS_LEASE = -706  # remove() during a live read lease

_MOONCAKE_SETUP_DEFAULTS = {
    "global_segment_size": 1 << 30,  # 1 GiB per-node segment
    "local_buffer_size": 1 << 30,
    # Every client of one store must use the same protocol, and SpecForge
    # cannot see an external server's, so the default stays explicit tcp.
    "protocol": "tcp",
    # Empty under rdma selects this node's usable HCAs (see _connect_store).
    "rdma_devices": "",
}
_RDMA_OPT_OUT = (
    "set MOONCAKE_PROTOCOL=tcp (deployment.disaggregated.mooncake_protocol or "
    "managed_local.mooncake.protocol) for every capture server and trainer"
)

_GET_RETRY_DELAYS_S = (2.0, 4.0, 8.0)
# TRANSFER_FAIL and a lease that expired during transfer are retryable with a
# fresh get_into call. Object/lifecycle errors and short reads are not.
_RETRYABLE_GET_STATUSES = frozenset({-800, -707})
_DEFAULT_MAX_QUARANTINED_BYTES = 8 << 30


class _MooncakeGetError(KeyError):
    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


class _InjectedReplicateConfig:
    """Minimal config object for an explicitly injected test backend."""

    def __init__(
        self,
        replica_num: int = 1,
        with_hard_pin: bool = True,
        with_soft_pin: bool = False,
    ) -> None:
        self.replica_num = replica_num
        self.with_hard_pin = with_hard_pin
        self.with_soft_pin = with_soft_pin


# Ascend's CUDA_VISIBLE_DEVICES equivalent; its presence marks an Ascend
# host even before torch_npu is imported.
_ASCEND_VISIBLE_DEVICE_ENVS = ("ASCEND_RT_VISIBLE_DEVICES", "ASCEND_VISIBLE_DEVICES")


def _ascend_runtime_available() -> bool:
    """Whether a usable Ascend NPU runtime is present.

    ``torch.npu`` exists only after ``torch_npu`` is imported; do that here,
    but only on hosts already selecting Ascend devices via env var, so the
    import never fires on CUDA/CPU hosts.
    """
    if getattr(torch, "npu", None) is None:
        if not any(os.environ.get(name) for name in _ASCEND_VISIBLE_DEVICE_ENVS):
            return False
        try:
            import torch_npu  # noqa: F401
        except Exception:
            return False
    npu = getattr(torch, "npu", None)
    try:
        return npu is not None and bool(npu.is_available())
    except Exception:  # pragma: no cover - defensive against driver faults
        return False


def _bind_transport_device() -> None:
    """Bind this process's local NPU before Mooncake installs its transport.

    Ascend's transport calls ``aclrtGetDevice`` in ``setup()`` and fails with
    ``ACL_ERROR_RT_CONTEXT_NULL`` when no device context exists — the case in
    a capture producer, which never runs ``init_distributed``. No-op on CUDA,
    where the transport defaults to device 0.
    """
    from specforge.utils import get_device_type

    device_type = get_device_type()
    if device_type == "cuda":
        return
    if device_type == "npu" or (device_type == "cpu" and _ascend_runtime_available()):
        from specforge.distributed import _bind_local_device

        _bind_local_device("npu")


def _connect_store(setup_kwargs: Dict[str, Any]) -> Tuple[Any, Any]:
    """Construct a real store and return its required config type."""
    try:
        from mooncake.store import (  # type: ignore
            MooncakeDistributedStore,
            ReplicateConfig,
        )
    except Exception as e:  # pragma: no cover - exercised only without mooncake
        raise RuntimeError(
            "MooncakeFeatureStore could not load the required Mooncake zero-copy "
            f"API: {type(e).__name__}: {e}. Install or upgrade the matching "
            "official wheel (`mooncake-transfer-engine` for CUDA < 13, or "
            "`mooncake-transfer-engine-cuda13` for CUDA >= 13)."
        ) from e
    setup_kwargs = dict(setup_kwargs)
    if setup_kwargs.get("protocol") == "rdma":
        # Mooncake would auto-discover HCAs for an empty list and fall back to
        # TCP silently, and drops a listed device it cannot use without a warning.
        setup_kwargs["rdma_devices"] = resolve_rdma_devices(
            setup_kwargs.get("rdma_devices") or None,
            selected_by="MOONCAKE_PROTOCOL is rdma",
            devices_setting="MOONCAKE_RDMA_DEVICES",
            opt_out=_RDMA_OPT_OUT,
        )
        check_rdma_environment(
            os.environ,
            where="this process's environment",
            opt_out=_RDMA_OPT_OUT,
            devices=setup_kwargs["rdma_devices"],
        )
    # Ascend's transport needs a bound device context (see _bind_transport_device).
    _bind_transport_device()
    if _ascend_runtime_available():
        # Ascend rejects the wildcard-location staging-buffer registration
        # ("location:* is not supported"); zero-copy clients can drop it.
        setup_kwargs["local_buffer_size"] = 0
    store = MooncakeDistributedStore()
    rc = store.setup(**setup_kwargs)
    if rc is not None and int(rc) != 0:
        raise RuntimeError(
            f"Mooncake setup failed (status {rc}); kwargs={setup_kwargs}"
        )
    return store, ReplicateConfig


def _require_store_api(store: Any) -> None:
    """Reject clients that cannot implement the canonical tensor wire path."""
    required = ("is_exist", "remove", "put_from", "get_into")
    missing = [name for name in required if not callable(getattr(store, name, None))]
    if not missing:
        return
    raise RuntimeError(
        "MooncakeFeatureStore requires callable is_exist/remove and the zero-copy "
        "MooncakeDistributedStore.put_from/get_into tensor API; backend "
        f"{type(store).__name__} is missing: {', '.join(missing)}. Upgrade the "
        "matching official Mooncake wheel (`mooncake-transfer-engine` for CUDA "
        "< 13, or `mooncake-transfer-engine-cuda13` for CUDA >= 13). The old "
        "serialized put/get transport is not supported."
    )


# FeatureSpec.dtype (a string) -> torch dtype, for allocating receive
# tensors from the ref alone (the ref carries shape+dtype, so get() needs no
# serialized header).
_TORCH_DTYPES = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int64": torch.int64,
    "int32": torch.int32,
    "int16": torch.int16,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


def _alloc_from_spec(spec) -> torch.Tensor:
    """Allocate a fresh contiguous receive tensor matching a FeatureSpec."""
    dtype = _TORCH_DTYPES.get(spec.dtype)
    if dtype is None:
        raise KeyError(f"unsupported feature dtype {spec.dtype!r} for Mooncake get")
    return torch.empty(tuple(int(d) for d in spec.shape), dtype=dtype)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _stays_on_host(dtype: torch.dtype) -> bool:
    """Whether a pooled device read still delivers this feature to the host.

    Integer features (token ids, loss and attention masks) are a few bytes per
    token, and strategies read them on the host: DFlash-family anchor sizing
    counts valid anchors from the CPU loss mask. Delivering them on the device
    forces a CUDA synchronization per micro-step, while their non-blocking H2D
    in the strategy is negligible. Hidden states stay on the device.
    """
    return not (dtype.is_floating_point or dtype.is_complex)


RECEIVE_BUFFER_KINDS = ("pageable", "pinned", "cuda")
DEFAULT_RECEIVE_POOL_BYTES = 8 << 30


def _check_get_result(key: str, rc: Optional[int], nbytes: int) -> None:
    if rc is None:
        raise _MooncakeGetError(f"mooncake get_into failed (status {rc}) for {key}")
    status = int(rc)
    if status < 0:
        hint = (
            "; increase deployment.disaggregated.client_buffer_size to cover "
            "concurrent device reads"
            if status == -200
            else ""
        )
        raise _MooncakeGetError(
            f"mooncake get_into failed (status {status}) for {key}{hint}",
            status=status,
        )
    if status != nbytes:
        raise _MooncakeGetError(
            f"mooncake get_into short read for {key}: got {status} of {nbytes} bytes"
        )


class _PoolSlot:
    __slots__ = ("storage", "capacity", "registered", "quarantined")

    def __init__(self, storage: torch.Tensor, registered: bool) -> None:
        self.storage = storage  # flat uint8 buffer
        self.capacity = storage.numel()
        self.registered = registered
        self.quarantined = False


# Last-resort retention when a pool is abandoned with writes still possible.
# Leaking these buffers is safer than returning DMA targets to the allocator.
_UNSAFE_RECEIVE_BUFFERS: List[Any] = []


def _unregister_slot(store: Any, slot: _PoolSlot) -> None:
    if slot.registered:
        rc = store.unregister_buffer(slot.storage.data_ptr())
        if rc is not None and int(rc) != 0:
            raise RuntimeError(f"Mooncake receive buffer unregistration failed ({rc})")
        slot.registered = False


def _finalize_receive_pool(store, free, active, quarantined) -> None:
    unsafe = list(active) + list(quarantined)
    for slot in free:
        try:
            _unregister_slot(store, slot)
        except Exception:
            logger.exception("Unable to unregister abandoned Mooncake receive buffer")
            unsafe.append(slot)
    if unsafe:
        _UNSAFE_RECEIVE_BUFFERS.append((store, unsafe))
        logger.warning(
            "Retaining %d unsafe receive buffers from an unclosed Mooncake pool; "
            "close the feature store after stopping its readers",
            len(unsafe),
        )


class ReceiveBufferPool:
    """Reusable, once-registered receive buffers for ``get_into``.

    Retained slots fit within ``max_bytes``; concurrent overflow uses one-off
    slots. Failed transfers retain their storage until the transport is stopped,
    since it can still write after returning an error, even after pool destruction.
    """

    def __init__(
        self,
        store: Any,
        *,
        kind: str,
        max_bytes: int = DEFAULT_RECEIVE_POOL_BYTES,
        max_quarantined_bytes: int = _DEFAULT_MAX_QUARANTINED_BYTES,
    ) -> None:
        if kind not in ("pinned", "cuda"):
            raise ValueError(f"receive pool kind must be pinned or cuda, got {kind!r}")
        if max_bytes <= 0:
            raise ValueError("receive_pool_bytes must be positive")
        if max_quarantined_bytes < 0:
            raise ValueError("max_quarantined_bytes must be >= 0")
        self.kind = kind
        self.max_bytes = int(max_bytes)
        self.max_quarantined_bytes = int(max_quarantined_bytes)
        self._store = store
        self._lock = threading.Lock()
        self._registration_disabled = False
        self._free: List[_PoolSlot] = []
        self._active: List[_PoolSlot] = []
        self._quarantined: List[_PoolSlot] = []
        self._closed = False
        self._finalizer = weakref.finalize(
            self,
            _finalize_receive_pool,
            store,
            self._free,
            self._active,
            self._quarantined,
        )
        self._quarantined_bytes = 0
        self._allocated_bytes = 0
        self._streams = threading.local()
        self.stats = {"hits": 0, "grown": 0, "overflow": 0, "quarantined": 0}

    def _device_for(self, device) -> torch.device:
        if self.kind == "cuda":
            target = torch.device(device)
            if target.type != "cuda":
                raise ValueError(
                    "receive_buffers=cuda needs a device-side consumer on CUDA; "
                    f"got {target}"
                )
            if target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            return target
        return torch.device("cpu")

    def _new_slot(self, nbytes: int, device: torch.device) -> _PoolSlot:
        if self.kind == "cuda":
            storage = torch.empty(nbytes, dtype=torch.uint8, device=device)
            # The caching allocator can return memory still used on this stream.
            torch.cuda.current_stream(device).synchronize()
        else:
            pin = torch.cuda.is_available()
            storage = torch.empty(nbytes, dtype=torch.uint8, pin_memory=pin)
        registered = False
        if not self._registration_disabled:
            try:
                rc = self._store.register_buffer(storage.data_ptr(), nbytes)
            except Exception as exc:  # pragma: no cover - some builds auto-register
                rc = exc
            if rc is None or (isinstance(rc, int) and rc == 0):
                registered = True
            else:
                if self.kind != "cuda":
                    raise RuntimeError(
                        f"Mooncake receive buffer registration failed ({rc}) for "
                        f"{nbytes} host bytes; RDMA host reads require registered memory"
                    )
                # Mooncake 0.3.x stages device reads even for registered slots.
                self._registration_disabled = True
                logger.warning(
                    "receive pool: register_buffer(%s bytes, %s) failed (%s); "
                    "using Mooncake's device-read staging path for subsequent slots",
                    nbytes,
                    storage.device,
                    rc,
                )
        return _PoolSlot(storage, registered)

    def acquire(self, nbytes: int, device) -> Tuple[_PoolSlot, bool]:
        """Return ``(slot, pooled)``; ``pooled`` False means a one-off buffer."""
        target = self._device_for(device)
        with self._lock:
            if self._closed:
                raise RuntimeError("Mooncake receive pool is closed")
            best = None
            for slot in self._free:
                if (
                    slot.capacity >= nbytes
                    and slot.storage.device == target
                    and (best is None or slot.capacity < best.capacity)
                ):
                    best = slot
            if best is not None:
                self._free.remove(best)
                self._active.append(best)
                self.stats["hits"] += 1
                return best, True
            pooled = self._allocated_bytes + nbytes <= self.max_bytes
            # Serialize registration and account only for successful allocations.
            slot = self._new_slot(nbytes, target)
            self._active.append(slot)
            if not pooled:
                self.stats["overflow"] += 1
            else:
                self._allocated_bytes += nbytes
                self.stats["grown"] += 1
        return slot, pooled

    def release(
        self, slot: _PoolSlot, pooled: bool, *, quarantine: bool = False
    ) -> None:
        if quarantine:
            with self._lock:
                self._active.remove(slot)
                self.stats["quarantined"] += 1
                slot.quarantined = True
                self._quarantined.append(slot)
                self._quarantined_bytes += slot.capacity
                if self._quarantined_bytes > self.max_quarantined_bytes:
                    raise MemoryError(
                        "Mooncake failed-transfer quarantine exceeded "
                        f"{self.max_quarantined_bytes} bytes; refusing unbounded "
                        "receive-pool memory growth because an in-flight transfer "
                        "may still write into each quarantined buffer"
                    )
            return
        with self._lock:
            self._active.remove(slot)
            # Keep ownership until unregister succeeds, including overflow.
            self._free.append(slot)
            if not pooled:
                try:
                    _unregister_slot(self._store, slot)
                except Exception:
                    # Do not reuse an overflow slot whose registration state
                    # is now uncertain, or bypass the pool's memory budget.
                    self._closed = True
                    raise
                self._free.remove(slot)

    def close(self, *, transport_stopped: bool = False) -> None:
        """Seal the pool and release idle registrations; safe to retry.

        Call only after stopping readers. Quarantined storage remains owned
        unless the backend owner confirms that its transport has been stopped.
        """
        with self._lock:
            self._closed = True
            if self._active:
                raise RuntimeError(
                    "Cannot close Mooncake receive pool with active readers"
                )
            for slot in list(self._free):
                if not transport_stopped:
                    _unregister_slot(self._store, slot)
                self._free.remove(slot)
            if transport_stopped:
                self._quarantined.clear()
                self._quarantined_bytes = 0
            self._allocated_bytes = self._quarantined_bytes

    def copy_stream(self, device: torch.device):
        """Per-thread side stream for copies out of receive slots."""
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        stream = getattr(self._streams, "stream", None)
        if stream is None or stream.device != device:
            stream = torch.cuda.Stream(device=device)
            self._streams.stream = stream
        return stream

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "receive_buffers": self.kind,
                "receive_pool_bytes": self._allocated_bytes,
                "receive_pool_free_slots": len(self._free),
                "receive_pool_quarantined_bytes": self._quarantined_bytes,
                **{f"receive_pool_{k}": v for k, v in self.stats.items()},
            }


@dataclass(frozen=True)
class _CaptureBeginIntent:
    """Stable retry identity and its batch-scoped singleflight owner."""
    protocol: Any
    capability: dict
    payload: dict
    tokens: tuple
    attempt_id: str
    names: tuple
    flight: Any


class MooncakeFeatureStore(FeatureStore):
    """A disaggregated :class:`FeatureStore` backed by the Mooncake store.

    **Zero-copy transport.** One Mooncake object per
    *tensor*, keyed ``{store_id}/{sample_id}/g{gen}/{name}``. ``put()`` writes each
    tensor straight from its storage with ``put_from(ptr)``; ``get()`` reads each
    straight into a tensor allocated from the ref's ``FeatureSpec`` with
    ``get_into(ptr)``. Tensors are never serialized on the wire: shape/dtype
    travel on the ref, while each object's value is the raw tensor buffer. The
    generation lives in the key (like ``SharedDirFeatureStore``'s filename
    generation), so a re-put supersedes the old key set and a stale ref's keys
    are gone -> ``get()`` raises (B5).

    ``store`` may be injected (any object exposing the Mooncake method subset:
    ``is_exist``/``remove``/``put_from``/``get_into``) so the contract is
    unit-testable without a running master. An incompatible backend is rejected
    during construction rather than selected as a different transport.
    An injected backend's owner must ensure its transport supports device reads
    when selecting ``receive_buffers="cuda"``; setup kwargs apply only to owned
    connections, whose effective protocol is validated here.
    """

    # Every read lands in a fresh spec-allocated tensor or is copied out of its
    # pooled receive slot before the slot is reused (``_fetch_tensor_once``).
    get_returns_fresh_tensors = True

    def __init__(
        self,
        *,
        store_id: Optional[str] = None,
        store: Optional[Any] = None,
        setup_kwargs: Optional[Dict[str, Any]] = None,
        auth: Optional[AuthPolicy] = None,
        credential: Optional[str] = None,
        max_resident_bytes: Optional[int] = None,
        max_hold_age_s: Optional[float] = None,
        retain_on_release: bool = False,
        max_release_attempts: int = 3,
        max_quarantined_bytes: int = _DEFAULT_MAX_QUARANTINED_BYTES,
        replica_num: int = 1,
        hard_pin: bool = True,
        clock: Callable[[], float] = time.monotonic,
        receive_buffers: str = "pageable",
        receive_pool_bytes: int = DEFAULT_RECEIVE_POOL_BYTES,
        node_pool_enabled: Optional[bool] = None,
    ) -> None:
        self.auth = auth or AuthPolicy()
        self._credential = credential
        self.auth.check(credential)  # attach-time gate (B9)
        self.store_id = store_id or uuid.uuid4().hex[:8]
        if os.environ.get('SPECFORGE_HS_SEGMENTED', '0') == '1' and not os.environ.get('SPECFORGE_TRAINER_NODE_POOL_DIR'):
            raise ValueError('segmented clients require NODE_POOL_DIR as their shared terminal-proof anchor, even with caching disabled')
        self._owns_store = store is None
        self._closed = False
        self._closing = False
        self._active_get_calls = 0
        self._close_lock = threading.Lock()
        if store is None:
            kw = dict(_MOONCAKE_SETUP_DEFAULTS)
            kw.update(setup_kwargs or {})
            if receive_buffers == "cuda" and kw["protocol"] != "rdma":
                raise ValueError(
                    "receive_buffers=cuda needs an RDMA Mooncake transport; "
                    f"the {kw['protocol']!r} transport cannot write into device memory"
                )
            store, replicate_config_type = _connect_store(kw)
            put_config = replicate_config_type()
        else:
            # Injected stores are a unit-test seam and do not require importing
            # the optional Mooncake package merely to construct its config type.
            put_config = _InjectedReplicateConfig()
        _require_store_api(store)
        self._store = store
        if max_quarantined_bytes < 0:
            raise ValueError("max_quarantined_bytes must be >= 0")
        self.max_quarantined_bytes = int(max_quarantined_bytes)
        # Failed pageable-buffer reads are retained because a timed-out transfer
        # may still write into them. Pooled modes maintain the equivalent
        # quarantine inside ReceiveBufferPool.
        self._quarantined_buffers: List[torch.Tensor] = []
        self._quarantined_bytes = 0
        self._quarantine_lock = threading.Lock()
        self._h2d_quarantined = []
        self._segment_manifests = {}
        self._segment_ack_sent = set()
        self._segment_ack_unsafe = set()
        self._segment_receipt_seen = set()
        self._segment_counters = dict.fromkeys(('segment_get_bytes', 'segment_get_objects',
            'segment_materialized_features', 'node_pool_staging_copy_bytes',
            'transport_get_requested_bytes', 'transport_get_completed_bytes',
            'receive_to_pageable_copy_bytes', 'manifest_ack_puts', 'manifest_receipt_drains'), 0)
        self._diagnostics = get_diagnostics()
        self._diag = self._diagnostics.component("store")
        self._node_pool = None
        self._terminal_proofs = None
        self._terminal_proof_owner = uuid.uuid4().hex
        self._terminal_proof_health = {}
        use_node_pool = (os.environ.get('SPECFORGE_TRAINER_NODE_POOL', '0') == '1'
                         if node_pool_enabled is None else bool(node_pool_enabled))
        if use_node_pool:
            from specforge.runtime.data_plane.node_content_pool import NodeContentPool
            directory = os.environ.get('SPECFORGE_TRAINER_NODE_POOL_DIR')
            if not directory:
                raise ValueError('node content pool requires an explicit node-shared directory')
            self._node_pool = NodeContentPool(directory, int(os.environ.get('SPECFORGE_TRAINER_NODE_POOL_BYTES', str(8 << 30))))
        if receive_buffers not in RECEIVE_BUFFER_KINDS:
            raise ValueError(
                f"receive_buffers={receive_buffers!r} not in {RECEIVE_BUFFER_KINDS}"
            )
        self.receive_buffers = receive_buffers
        self._receive_pool: Optional[ReceiveBufferPool] = (
            ReceiveBufferPool(
                store,
                kind=receive_buffers,
                max_bytes=receive_pool_bytes,
                max_quarantined_bytes=max_quarantined_bytes,
            )
            if receive_buffers != "pageable"
            else None
        )
        put_config.replica_num = replica_num
        # Prefer true hard pinning when the installed Mooncake supports it.
        # Older ROCm builds expose only `with_soft_pin`; that is a best-effort
        # fallback rather than the same no-eviction guarantee. Some older
        # Ascend builds expose neither field and must use the store default.
        if hasattr(put_config, "with_hard_pin"):
            put_config.with_hard_pin = hard_pin
        elif hasattr(put_config, "with_soft_pin"):
            put_config.with_soft_pin = hard_pin
            if hard_pin:
                logger.warning(
                    "Mooncake ReplicateConfig has no with_hard_pin field; "
                    "falling back to with_soft_pin"
                )
        elif hard_pin:
            logger.warning(
                "Mooncake ReplicateConfig exposes neither with_hard_pin nor "
                "with_soft_pin; objects use the store's default pin behavior"
            )
        self._put_config = put_config
        self.max_resident_bytes = max_resident_bytes
        self.max_hold_age_s = max_hold_age_s
        # Offline re-iterable mode: release() must NOT free (multi-epoch); mirrors
        # SharedDirFeatureStore / LocalFeatureStore file:// no-op release.
        self.retain_on_release = retain_on_release
        self.max_release_attempts = max_release_attempts
        self._clock = clock
        # in-process liveness index (single-host; see module docstring)
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._sample_bytes: Dict[str, int] = {}
        # feature names per resident sample -> the per-tensor keys to remove on
        # free. Cached on both put() (producer) and get()
        # (consumer) so each side can free the sample it owns/consumed without the
        # ref in hand at release() time.
        self._sample_names: Dict[str, List[str]] = {}
        # Server capture registers deterministic keys before issuing HTTP. If
        # the response is lost, no SampleRef exists to adopt/abort them. Keep a
        # shared (multi-adapter) provisional index so terminal producer cleanup
        # can reclaim those provisional objects; a successful adopt clears it.
        self._external_provisional: Dict[Tuple[str, int], List[str]] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._capture_fences = {}
        self._capture_quarantined_registrations = set()
        self._capture_handle_tokens = {}
        self._capture_ack_pending = {}
        self._capture_pending_begin = {}
        self._terminal_cleanup_authority = None
        self._terminal_cleanup_lock = threading.RLock()
        self._capture_cleanup_local = threading.local()
        self._capture_read_local = threading.local()
        self._capture_read_intents = {}
        self._capture_reads = {}
        # Samples whose remote remove() failed. gc() performs bounded
        # steady-state retries; lifecycle drain either removes them or raises.
        self._release_pending: Dict[str, int] = {}
        # Failed local puts may leave objects from a generation that was never
        # published. Keep their exact physical keys separate from the live generation.
        self._partial_put_pending: Dict[Tuple[str, int], Tuple[List[str], int]] = {}
        # (sample_id, generation) logically freed in THIS process. Mooncake's
        # remove() is lease-deferred (an object keeps a short read-lease), so the
        # bytes can linger after release/abort; this makes the B5 "no
        # use-after-free" guarantee immediate — get() of a freed ref raises even
        # while physical reclamation is still pending. Grows with consume-once
        # frees within a run (empty in retain_on_release/offline mode); a durable
        # shared index would own this in the online multi-node follow-up.
        self._freed: set = set()
        self._freed_capture_attempts: set = set()
        self._lock = threading.RLock()
        self._counter = 0
        self._gen_counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}

    def _begin_capture_read(self, ref):
        fence = self._capture_from_ref(ref)
        if fence is None:
            return None
        owner = uuid.uuid4().hex
        intent = {"fence": fence, "owner": owner, "token": None, "unsafe_fetch": False,
                  "handle": None, "release_requested": False, "active": True}
        with self._lock:
            if self._capture_fences.get(ref.sample_id) is not fence or self._capture_ref_freed_locked(ref.sample_id, fence.wire):
                raise KeyError('capture read ownership was released')
            self._capture_read_intents[owner] = intent
        # Keep intent before HTTP: lost acquire replies recover the same owner.
        try:
            with fence.lock:
                intent["token"] = owner if fence.transferred else fence.acquire_read(owner)
        except BaseException as exc:
            from specforge.inference.capture_protocol import CaptureControlRejected
            intent["active"] = False
            if isinstance(exc, CaptureControlRejected):
                with self._lock:
                    self._capture_read_intents.pop(owner, None)
            raise
        return intent

    def _release_capture_read(self, intent):
        if intent["unsafe_fetch"]:
            raise RuntimeError("capture receive drain unknown; read ownership retained")
        fence = intent["fence"]
        with fence.lock:
            if not fence.transferred:
                if intent["token"] is None:
                    intent["token"] = fence.acquire_read(intent["owner"])
                fence.release_read(intent["owner"], intent["token"])
        with self._lock:
            self._capture_read_intents.pop(intent["owner"], None)
            handle = intent["handle"]
            if handle is not None:
                self._capture_reads.pop(handle.lease_token, None)
                if intent["release_requested"]:
                    self._active_leases.pop(handle.lease_token, None)
                    self._capture_handle_tokens.pop(handle.lease_token, None)
                    sid = handle.sample_id
                    if (not self.retain_on_release and self._capture_fences.get(sid) is fence
                            and not self._still_leased_locked(sid, handle.generation)
                            and not any(other["fence"] is fence for other in self._capture_read_intents.values())):
                        self._mark_freed_locked(sid, handle.generation)
                        self._release_pending.setdefault(sid, 0)

    def _recover_capture_reads(self, *, require_drained=False):
        from specforge.inference.capture_protocol import CaptureControlRejected
        with self._lock:
            intents = list(self._capture_read_intents.values())
        failures = []
        for intent in intents:
            if intent.get("active"):
                continue
            if intent["unsafe_fetch"]:
                failures.append(intent["owner"] + ": receive drain unknown")
                continue
            if intent["handle"] is None or intent["release_requested"]:
                try:
                    self._release_capture_read(intent)
                except CaptureControlRejected as exc:
                    if intent["token"] is None:
                        # No native read started; authoritative refusal proves
                        # this owner has no active permit. Retire this intent only.
                        with self._lock:
                            self._capture_read_intents.pop(intent["owner"], None)
                    else:
                        intent["recovery_error"] = str(exc)
                        failures.append(intent["owner"] + ": " + str(exc))
                except Exception as exc:
                    intent["recovery_error"] = str(exc)
                    failures.append(intent["owner"] + ": " + str(exc))
        if require_drained and failures:
            raise RuntimeError("capture read ownership not drained; permits retained: " + repr(failures))

    def begin_capture_attempts(self, protocol, capability, requests):
        """Reserve one batch while unrelated identities contact servers in parallel.

        A flight belongs to these immutable intents, not to the shared store.
        Recovery can wait for this exact batch without serializing other servers.
        """
        flight = threading.RLock()
        # Prepare long token snapshots outside the metadata critical section.
        intents = [(str(payload["sample_id"]), _CaptureBeginIntent(
            protocol, dict(capability), dict(payload), tuple(tokens), attempt_id,
            tuple(names), flight)) for payload, tokens, attempt_id, names in requests]
        ids = [sid for sid, _ in intents]
        with flight:
            with self._lock:
                if len(set(ids)) != len(ids) or any(sid in self._capture_pending_begin for sid in ids):
                    raise RuntimeError("duplicate or pending capture begin intent")
                if any(sid in self._capture_fences or sid in self._capture_ack_pending or sid in self._release_pending for sid in ids):
                    raise RuntimeError("prior capture attempt has not relinquished ownership")
                self._capture_pending_begin.update(intents)
            if not capability.get('begin_batch'):
                return [self._recover_capture_begin(sid) for sid in ids]
            from specforge.inference.capture_protocol import CaptureCapacityDeclined
            try:
                replies = protocol.begin_batch(capability, [(intent.payload, intent.tokens, intent.attempt_id) for _, intent in intents])
            except CaptureCapacityDeclined:
                # The typed atomic refusal proves none of this batch acquired a
                # server writer. Unknown/lost replies must retain their intents.
                with self._lock:
                    for sid, intent in intents:
                        if self._capture_pending_begin.get(sid) is intent:
                            self._capture_pending_begin.pop(sid)
                raise
            for (sid, intent), wire in zip(intents, replies):
                self._finish_capture_begin(sid, intent, wire)
            return replies

    def begin_capture_attempt(self, protocol, capability, payload, tokens, attempt_id, feature_names):
        return self.begin_capture_attempts(protocol, capability, [(payload, tokens, attempt_id, feature_names)])[0]

    def _finish_capture_begin(self, sid, intent, wire):
        # Caller holds intent.flight. No waiter may retarget this operation to
        # a replacement endpoint/epoch merely because it has the same sample id.
        with self._lock:
            if self._capture_pending_begin.get(sid) is not intent:
                raise RuntimeError("capture begin intent changed before completion")
        self.track_capture_fence(wire, intent.names, control_fn=intent.protocol.post)
        self.track_external_attempt(sid, generation=1, feature_names=list(intent.names),
                                    server_url=intent.protocol.url.removesuffix("/spec_capture/control"))
        with self._lock:
            if self._capture_pending_begin.get(sid) is intent:
                self._capture_pending_begin.pop(sid, None)

    def _recover_capture_begin(self, sid, *, expected_intent=None):
        with self._lock:
            intent = self._capture_pending_begin.get(sid)
            if expected_intent is not None and intent is not expected_intent:
                return None
        if intent is None:
            return None  # Fresh samples never wait for another endpoint's HTTP.
        with intent.flight:
            with self._lock:
                if self._capture_pending_begin.get(sid) is not intent:
                    return None
            wire = intent.protocol.begin(intent.capability, intent.payload, intent.tokens, intent.attempt_id)
            self._finish_capture_begin(sid, intent, wire)
            return wire

    @contextmanager
    def _capture_cleanup_scope(self, sample_ids, *, expected_records=None):
        # The initiating caller may already have crossed a network operation;
        # carry its original identity rather than snapshotting a replacement.
        with self._lock:
            records = (dict(expected_records) if expected_records is not None else
                       {sid: self._capture_fences.get(sid) for sid in sorted(set(sample_ids))})
        local = self._capture_cleanup_local
        previous = getattr(local, "records", {})
        depth = getattr(local, "depth", 0)
        with ExitStack() as stack:
            for sid, fence in records.items():
                if fence is None:
                    continue
                stack.enter_context(fence.lock)
                with self._lock:
                    still_current = self._capture_fences.get(sid) is fence
                if still_current:
                    if fence.transferred:
                        with self._lock:
                            if any(intent['fence'] is fence for intent in self._capture_read_intents.values()):
                                raise RuntimeError("capture local readers or receive transfers have not drained")
                    try:
                        fence.acquire_cleanup()
                    except BaseException:
                        with self._lock:
                            if self._capture_fences.get(sid) is fence:
                                self._generation.setdefault(sid, int(fence.wire["gen"]))
                                self._sample_names.setdefault(sid, list(fence.names))
                                self._sample_bytes.setdefault(sid, 0)
                                self._release_pending.setdefault(sid, 0)
                        raise
            local.records = {**previous, **records}
            local.depth = depth + 1
            try:
                yield records
            finally:
                local.records, local.depth = previous, depth
                if depth == 0:
                    self._flush_capture_acks(sample_ids=records)

    def _capture_cleanup_allows(self, sid):
        records = getattr(self._capture_cleanup_local, "records", {})
        if sid not in records:
            return self._capture_fences.get(sid) is None
        return self._capture_fences.get(sid) is records[sid]

    def track_capture_fence(self, wire, feature_names, *, control_fn=None):
        from specforge.inference.capture_protocol import WriterFence, http_control
        sid = str(wire["sample_id"])
        if str(wire["store_id"]) != str(self.store_id) or int(wire["gen"]) != 1:
            raise ValueError("capture fence store/generation mismatch")
        with self._lock:
            old = self._capture_fences.get(sid)
            if old is not None:
                if old.token != wire["reservation_token"]:
                    raise RuntimeError("capture sample still has an unclean attempt")
                return old
            if any(h.sample_id == sid for h in self._active_leases.values()):
                raise RuntimeError("capture sample has active consumer leases")
            fence = WriterFence(dict(wire), tuple(feature_names), post=control_fn or http_control)
            if wire.get('publication_ownership') == 'attempt-v1':
                fence.accept_transfer(wire)
            self._capture_fences[sid] = fence
            # Only a server-authorized new reservation can retire the old g1 tombstone.
            self._freed.discard((sid, 1))
            return fence

    def _capture_from_ref(self, ref, *, validate_remote=False):
        wire = ref.metadata.get("capture_fence")
        layout = ref.metadata.get('feature_layout', 'full-v1')
        if layout not in ('full-v1', 'hs-manifest-v1'):
            raise ValueError('unknown capture feature layout')
        if layout == 'hs-manifest-v1':
            if not os.environ.get('SPECFORGE_TRAINER_NODE_POOL_DIR'):
                raise ValueError('segmented clients require NODE_POOL_DIR as their shared terminal-proof anchor, even with caching disabled')
            from specforge.runtime.data_plane.segment_manifest import validate_manifest
            if ref.schema_version != 2 or wire is None or wire.get('feature_layout') != layout:
                raise ValueError('segmented capture requires explicit versioned ownership')
            if getattr(self._put_config, 'with_hard_pin', False) is not True:
                raise RuntimeError('manifest ACK protocol requires true native hard pin')
            manifest = validate_manifest(ref.metadata['segment_manifest'], ref.feature_specs, ref.num_tokens)
            if wire.get('segment_manifest') != manifest or manifest['token'] != wire.get('reservation_token'):
                raise ValueError('manifest differs from immutable ownership certificate')
        elif ref.schema_version != SCHEMA_VERSION:
            raise ValueError('unknown full capture schema version')
        if wire is None:
            return
        sid = ref.sample_id
        if (wire.get("sample_id") != sid or str(wire.get("store_id")) != str(self.store_id)
                or ref.metadata.get("generation") is None
                or int(wire.get("gen", -1)) != int(ref.metadata["generation"])):
            raise ValueError("capture ref sample/store/generation identity mismatch")
        with self._lock:
            old = self._capture_fences.get(sid)
            if old is not None and old.token != wire["reservation_token"]:
                raise KeyError("stale capture attempt ref")
            if old is None and self._capture_ref_freed_locked(sid, wire):
                raise KeyError("released capture attempt ref")
        fence = self.track_capture_fence(wire, list(ref.feature_keys))
        if wire.get('publication_ownership') == 'attempt-v1':
            fence.accept_transfer(wire)
            if ref.feature_keys != wire['physical_keys']:
                raise RuntimeError('capture ref physical keys mismatch')
        if validate_remote and not fence.transferred:
            status = fence.control("resolve", cancel=False)
            if status.get("state") != "PUBLISHED" or not status.get("writer_drained"):
                from specforge.inference.capture_protocol import CaptureControlRejected
                raise CaptureControlRejected("cannot adopt an attempt that is not currently PUBLISHED")
        # A returned complete SampleRef is emitted only after writer completion.
        fence.writer_drained = True
        fence.adopted = True
        if layout == 'hs-manifest-v1':
            with self._lock:
                self._segment_manifests[sid] = manifest
        return fence

    def _resolve_capture_writer(self, sid, expected_token=None):
        with self._lock:
            fence = self._capture_fences.get(sid)
            if fence is None:
                return
            if expected_token is not None and fence.token != expected_token:
                return
        with fence.lock:
            fence.drain(cancel=True)
            # Lost /generate responses have no SampleRef. The exact transfer
            # certificate must still seed manifest cleanup, including shared deps.
            if fence.wire.get('feature_layout') == 'hs-manifest-v1':
                manifest = fence.wire.get('segment_manifest')
                from specforge.runtime.data_plane.segment_manifest import digest_manifest
                if (not isinstance(manifest, dict) or manifest.get('token') != fence.token
                        or manifest.get('digest') != digest_manifest({name:value for name,value in manifest.items() if name != 'digest'})):
                    raise RuntimeError('cleanup certificate contains an invalid manifest')
                with self._lock:
                    self._segment_manifests[sid] = manifest

    def _flush_capture_acks(self, sample_ids=None):
        if getattr(self._capture_cleanup_local, "depth", 0):
            return
        with self._lock:
            pending = sorted((sid, fence) for sid, fence in self._capture_ack_pending.items()
                             if sample_ids is None or (sid in sample_ids and (not isinstance(sample_ids, dict) or sample_ids[sid] is fence)))
        for sid, fence in pending:
            with fence.lock:
                fence.acknowledge()
            with self._lock:
                if self._capture_ack_pending.get(sid) is fence:
                    self._capture_ack_pending.pop(sid, None)
                if self._capture_fences.get(sid) is fence:
                    self._capture_fences.pop(sid, None)

    def discard_capture_attempt(self, sample_id, *, server_epoch, attempt_id,
                                control_url, reservation_token=None):
        """Retire only the original prepared writer, even after a peer retries.

        An unknown begin reply can be recovered only through its matching
        immutable intent. The captured fence then stays the cleanup authority
        through resolve, physical removal and acknowledgement.
        """
        sid = str(sample_id)
        with self._lock:
            intent = self._capture_pending_begin.get(sid)
            if intent is not None and not (
                intent.protocol.url == control_url
                and intent.capability["server_epoch"] == server_epoch
                and intent.attempt_id == attempt_id
            ):
                intent = None
        if intent is not None:
            self._recover_capture_begin(sid, expected_intent=intent)
        with self._lock:
            fence = self._capture_fences.get(sid)
            if fence is None or any(fence.wire.get(name) != value for name, value in (
                ("server_epoch", server_epoch), ("attempt_id", attempt_id),
                ("control_url", control_url),
            )) or (reservation_token is not None and fence.token != reservation_token):
                return
            if fence.adopted and sid in self._generation and sid not in self._release_pending:
                raise RuntimeError("cannot discard an adopted live capture attempt")
        with self._capture_cleanup_scope([sid], expected_records={sid: fence}):
            with self._lock:
                if self._capture_fences.get(sid) is not fence:
                    return
                if fence.adopted and sid in self._generation and sid not in self._release_pending:
                    raise RuntimeError("cannot discard an adopted live capture attempt")
            self._abort_impl(sid, reason="capture-prepared-discard", attempt_token=fence.token)
            self.drain_sample_removals([sid])
        with self._lock:
            if (self._capture_fences.get(sid) is fence
                or self._capture_ack_pending.get(sid) is fence):
                raise RuntimeError("prepared capture cleanup has not drained")

    def prepare_capture_retry(self, sample_id):
        sid = str(sample_id)
        self._recover_capture_begin(sid)
        with self._lock:
            fence = self._capture_fences.get(sid)
            if any(h.sample_id == sid for h in self._active_leases.values()):
                raise RuntimeError("cannot replace capture with active consumer lease")
            adopted_live = fence is not None and fence.adopted and sid in self._generation and sid not in self._release_pending
        if adopted_live and fence.transferred:
            raise RuntimeError("cannot replace an adopted live capture sample before ownership handoff")
        if adopted_live:
            with fence.lock:
                state = fence.control("resolve", cancel=False)
                if state.get("state") != "RETIRED":
                    raise RuntimeError("cannot replace an adopted live capture sample")
                fence.remote_retired = fence.writer_drained = fence.removed = fence.acknowledged = True
        if fence is not None:
            self._resolve_capture_writer(sid, fence.token)
            self.abort(sid, reason="capture-retry", attempt_token=fence.token)
            self.drain_sample_removals([sid])
        self._flush_capture_acks()
        with self._lock:
            if sid in self._capture_fences or sid in self._capture_ack_pending or sid in self._release_pending:
                raise RuntimeError("prior capture cleanup has not drained")

    # -- keys --------------------------------------------------------------
    def _tkey(self, sample_id: str, gen: int, name: str) -> str:
        # generation lives in the key (like SharedDirFeatureStore's filename gen):
        # a re-put writes a new-gen key set and removes the old, so a stale ref's
        # keys are gone -> get() raises (B5), no payload-carried gen needed.
        fence = self._capture_fences.get(sample_id)
        if fence is not None and fence.wire.get('object_key_layout') == 'attempt-v1':
            from specforge.inference.capture_protocol import attempt_keys
            if name not in fence.names:
                raise KeyError("feature not owned by capture attempt")
            return attempt_keys(fence.wire, (name,))[name]
        return f"{self.store_id}/{sample_id}/g{gen}/{name}"

    # -- store wrappers (status-code aware) --------------------------------
    def _store_exists(self, key: str) -> bool:
        return int(self._store.is_exist(key)) == 1

    def _store_put_tensor(self, key: str, t: torch.Tensor) -> None:
        """Zero-copy publish, requesting a hard pin when the client supports it.

        ``t`` must be contiguous + CPU (caller stages it). The bytes are the raw
        tensor buffer; shape/dtype travel on the ref's FeatureSpec, so get()
        needs no header. The source is registered with the transfer engine for
        the duration of the put -- RDMA transfers it by DMA and rejects an
        unregistered address (AddressNotRegistered); TCP ignores the
        registration.
        """
        nb = _nbytes(t)
        try:
            self._store.register_buffer(t.data_ptr(), nb)
        except Exception:  # pragma: no cover - some builds auto-register
            pass
        try:
            rc = self._store.put_from(key, t.data_ptr(), nb, self._put_config)
        finally:
            try:
                self._store.unregister_buffer(t.data_ptr())
            except Exception:  # pragma: no cover
                pass
        if rc is not None and int(rc) < 0:
            raise RuntimeError(f"mooncake put_from failed (status {rc}) for {key}")

    def consumer_device(self) -> Optional[torch.device]:
        """Request device tensors so pooled copies run in the loader worker.

        Only floating-point features are placed on this device; integer
        features stay on the host (see :func:`_stays_on_host`).
        """
        if self.receive_buffers == "pageable":
            return None
        if self.receive_buffers == "pinned":
            from specforge.utils import get_device_type

            # Visible CUDA devices do not override an explicit CPU/NPU trainer.
            if get_device_type() != "cuda" or not torch.cuda.is_available():
                return None
        # Mirror init_distributed: the launcher's LOCAL_RANK names this rank's
        # device even before the process group has selected it.
        local_rank = os.environ.get("LOCAL_RANK")
        index = (
            int(local_rank) if local_rank is not None else torch.cuda.current_device()
        )
        return torch.device("cuda", index)

    def close(self) -> None:
        with self._close_lock:
            self._close_impl()

    def _close_impl(self) -> None:
        """Release receive memory after loaders and removal drains have stopped.

        An injected/shared backend is never closed here. Its failed-transfer
        buffers stay quarantined because its owner may still be using transport.
        """
        with self._lock:
            if self._closed:
                return
            # This fence is installed atomically with read admission. It stays
            # closed after failed shutdown; existing reads may drain and close
            # may be retried, but no new native receive can begin.
            self._closing = True
            if self._active_get_calls or any(intent["active"] or (intent["handle"] is not None and not intent["release_requested"])
                   for intent in self._capture_read_intents.values()):
                raise RuntimeError("cannot close capture transport with live readers")
        # CUDA H2D completion is independent of native transport shutdown.
        self._drain_h2d_quarantine()
        if self._terminal_cleanup_authority is not None:
            self._terminal_cleanup_authority.close()
        if self._terminal_proofs is not None:
            if self._terminal_proofs.owned_claims(self._terminal_proof_owner):
                raise RuntimeError('cannot close with unresolved terminal ACK claims')
            self._terminal_proof_health = self._terminal_proofs.health()
            self._terminal_proofs.close()
        if self._node_pool is not None:
            self._node_pool.close()
        if self._receive_pool is not None:
            self._receive_pool.close()
        if self._owns_store:
            if not getattr(self, "_transport_stopped", False):
                rc = self._store.close()
                if rc is not None and int(rc) != 0:
                    raise RuntimeError(f"Mooncake transport close failed ({rc})")
                self._transport_stopped = True
            if self._receive_pool is not None:
                self._receive_pool.close(transport_stopped=True)
            self._quarantined_buffers.clear()
            self._quarantined_bytes = 0
            self._capture_quarantined_registrations.clear()
            # Successful transport shutdown proves every quarantined read has
            # physically drained. An injected/shared transport has no such proof.
            with self._lock:
                for intent in self._capture_read_intents.values():
                    intent["unsafe_fetch"] = False
            self._recover_capture_reads(require_drained=True)
        self._closed = True

    def _fetch_tensor(self, key: str, spec, device="cpu") -> torch.Tensor:
        """Retry transient reads into fresh storage; return caller-owned tensors."""
        if self._closed:
            raise RuntimeError("Mooncake feature store is closed")
        attempts = len(_GET_RETRY_DELAYS_S) + 1
        for attempt in range(attempts):
            try:
                return self._fetch_tensor_once(key, spec, device)
            except _MooncakeGetError as exc:
                if getattr(self._capture_read_local, "intent", None) is not None:
                    # A successful retry cannot prove a prior transfer drained.
                    # Keep the exact read permission until transport shutdown.
                    raise
                if exc.status not in _RETRYABLE_GET_STATUSES or attempt + 1 == attempts:
                    raise
                delay = _GET_RETRY_DELAYS_S[attempt]
                logger.warning(
                    "%s; retry %d/%d in %.0fs",
                    exc,
                    attempt + 1,
                    len(_GET_RETRY_DELAYS_S),
                    delay,
                )
                time.sleep(delay)
        raise AssertionError("unreachable Mooncake fetch retry state")

    def _fetch_tensor_once(self, key: str, spec, device) -> torch.Tensor:
        destination = getattr(self._capture_read_local, "destinations", {}).get(key)
        stream = None
        pool = self._receive_pool
        if pool is None:
            dst = _alloc_from_spec(spec)
            try:
                self._store_get_tensor(key, dst)
            except BaseException as exc:
                # Native get_into can raise before confirming physical drain.
                # A failed fenced read keeps both storage and registration.
                if isinstance(exc, _MooncakeGetError) or getattr(self._capture_read_local, "intent", None) is not None:
                    self._quarantine_buffer(dst)
                raise
            if destination is not None:
                self._copy_segment_into(dst, destination)
                return destination
            return dst
        dtype = _TORCH_DTYPES.get(spec.dtype)
        if dtype is None:
            raise KeyError(f"unsupported feature dtype {spec.dtype!r} for Mooncake get")
        shape = tuple(int(d) for d in spec.shape)
        nbytes = torch.empty(0, dtype=dtype).element_size() * int(
            torch.Size(shape).numel()
        )
        slot, pooled = pool.acquire(max(nbytes, 1), device)
        try:
            self._segment_counters["transport_get_requested_bytes"] += nbytes
            with self._diag.span("rdma_get_into"):
                rc = self._store.get_into(key, slot.storage.data_ptr(), nbytes)
        except Exception:
            pool.release(slot, pooled, quarantine=True)
            raise
        if rc is None or int(rc) < 0 or int(rc) != nbytes:
            pool.release(slot, pooled, quarantine=True)
            _check_get_result(key, rc, nbytes)
        self._diag.count("rdma_get_into_completed_bytes", nbytes)
        self._segment_counters["transport_get_completed_bytes"] += nbytes
        view = slot.storage[:nbytes].view(dtype).view(shape)
        target = torch.device(device)
        if _stays_on_host(dtype):
            target = torch.device("cpu")
        try:
            if destination is not None:
                out = destination
                if out.device.type == "cuda":
                    stream = pool.copy_stream(out.device)
                    stream.wait_stream(torch.cuda.current_stream(out.device))
                    out.record_stream(stream)
                    with torch.cuda.stream(stream):
                        out.copy_(view, non_blocking=True)
                    # Source and registered receive slot remain owned until DMA ends.
                    with self._diag.span("h2d_existing_sync"):
                        stream.synchronize()
                    out.record_stream(torch.cuda.current_stream(out.device))
                else:
                    out.copy_(view, non_blocking=False)
            elif target.type == "cpu" and not view.is_cuda:
                with self._diag.span("receive_to_pageable_copy"):
                    out = view.clone()
                self._diag.count("receive_to_pageable_copy_bytes", nbytes)
                self._segment_counters["receive_to_pageable_copy_bytes"] += nbytes
            else:
                stream = pool.copy_stream(view.device if view.is_cuda else target)
                with torch.cuda.stream(stream):
                    out = view.to(target, non_blocking=True, copy=True)
                # RDMA cannot wait on a CUDA event before reusing this slot.
                with self._diag.span("h2d_existing_sync"):
                    stream.synchronize()
                if out.is_cuda:
                    out.record_stream(torch.cuda.current_stream(target))
        except Exception:
            if destination is not None and destination.is_cuda and stream is not None:
                self._quarantine_h2d(destination, view, stream)
            pool.release(slot, pooled, quarantine=True)
            raise
        else:
            pool.release(slot, pooled)
        return out

    def _quarantine_h2d(self, destination, source, stream, pin=None):
        with self._quarantine_lock:
            self._h2d_quarantined.append({'destination': destination, 'source': source,
                                          'stream': stream, 'pin': pin})

    def _drain_h2d_quarantine(self):
        with self._quarantine_lock:
            for item in self._h2d_quarantined:
                item['stream'].synchronize()
                if item['pin'] is not None:
                    item['pin'].release()
            self._h2d_quarantined.clear()

    def _copy_segment_into(self, source, destination, pin=None):
        pool = self._receive_pool
        if not destination.is_cuda:
            destination.copy_(source, non_blocking=False)
            return
        if pool is None or pool.kind != 'pinned':
            stream = torch.cuda.current_stream(destination.device)
            destination.record_stream(stream)
            try:
                destination.copy_(source, non_blocking=False)
            except BaseException:
                self._quarantine_h2d(destination, source, stream, pin)
                raise
            return
        nbytes = source.numel() * source.element_size()
        slot, pooled = (None, False) if source.is_pinned() else pool.acquire(max(1, nbytes), destination.device)
        stream = pool.copy_stream(destination.device)
        try:
            if slot is None:
                staged = source
            else:
                staged = slot.storage[:nbytes].view(source.dtype).view(source.shape)
                with self._diag.span("pageable_to_pinned_copy"):
                    staged.copy_(source, non_blocking=False)
                self._diag.count("pageable_to_pinned_copy_bytes", nbytes)
                self._segment_counters['node_pool_staging_copy_bytes'] += nbytes
            stream.wait_stream(torch.cuda.current_stream(destination.device))
            destination.record_stream(stream)
            with torch.cuda.stream(stream):
                with self._diag.span("h2d_enqueue"):
                    destination.copy_(staged, non_blocking=True)
                self._diag.count("h2d_enqueued_bytes", nbytes)
            with self._diag.span("h2d_existing_sync"):
                stream.synchronize()
        except BaseException:
            self._quarantine_h2d(destination, source, stream, pin)
            if slot is not None:
                pool.release(slot, pooled, quarantine=True)
            raise
        else:
            if slot is not None:
                pool.release(slot, pooled)

    def _fetch_segment_source(self, key, spec):
        """Lease existing pinned receive storage; no per-miss pageable clone."""
        pool = self._receive_pool
        if pool is None or pool.kind != 'pinned':
            return self._fetch_tensor(key, spec, 'cpu')
        dtype = _TORCH_DTYPES[spec.dtype]
        nbytes = torch.empty((), dtype=dtype).element_size() * int(torch.Size(spec.shape).numel())
        slot, pooled = pool.acquire(max(1, nbytes), 'cpu')
        try:
            self._segment_counters['transport_get_requested_bytes'] += nbytes
            with self._diag.span("rdma_get_into"):
                rc = self._store.get_into(key, slot.storage.data_ptr(), nbytes)
            _check_get_result(key, rc, nbytes)
            self._diag.count('rdma_get_into_completed_bytes', nbytes)
            self._segment_counters['transport_get_completed_bytes'] += nbytes
        except BaseException:
            pool.release(slot, pooled, quarantine=True)
            raise
        tensor = slot.storage[:nbytes].view(dtype).view(spec.shape)
        released = False
        def release(*, quarantine=False):
            nonlocal released
            if not released:
                released = True
                pool.release(slot, pooled, quarantine=quarantine)
        return tensor, release

    def _get_segments(self, ref, name, destination):
        from dataclasses import replace
        manifest = self._segment_manifests[ref.sample_id]
        for segment in manifest['features'][name]:
            spec = replace(ref.feature_specs[name], shape=tuple(segment['shape']))
            op = self._diagnostics.begin_block({
                'namespace': manifest['namespace'],
                'load_epoch': ref.metadata.get('capture_fence', {}).get('server_epoch'),
                'owner_epoch': manifest['owner_epoch'],
                'schema_signature': manifest['schema_signature'], 'feature': name,
                'logical_key': segment['logical_key'], 'physical_key': segment['key'],
                'publication_generation': segment['publication_generation'],
                'nbytes': segment['nbytes'], 'dtype': spec.dtype, 'shape': list(spec.shape),
            })
            fetched = False
            def fetch():
                nonlocal fetched
                fetched = True
                self._diagnostics.event('miss_fetch', op)
                self._segment_counters['segment_get_bytes'] += segment['nbytes']
                self._segment_counters['segment_get_objects'] += 1
                return self._fetch_segment_source(segment['key'], spec)
            pin = None
            source = None
            source_release = None
            completed = False
            try:
                pin = self._node_pool.acquire(segment['logical_key'], segment['nbytes'], fetch) if self._node_pool is not None else None
                self._diagnostics.event('pin', op, {'pooled': pin is not None, 'fetched': fetched,
                                                   'generation': pin.generation if pin is not None else None})
                if pin is not None:
                    source = pin.tensor.view(_TORCH_DTYPES[spec.dtype]).view(spec.shape)
                else:
                    source = fetch()
                    if isinstance(source, tuple):
                        source, source_release = source
                self._diag.count('lookup_bytes', segment['nbytes'])
                self._diag.count('miss_bytes' if fetched else 'hit_bytes', segment['nbytes'])
                self._diag.count('miss_blocks' if fetched else 'hit_blocks')
                self._diag.count('fallback_blocks', int(pin is None))
                self._copy_segment_into(source, destination[:,segment['start']:segment['end']], pin)
                completed = True
                self._diagnostics.event('copy_complete', op)
            finally:
                if source_release is not None:
                    uncertain = any(item['source'] is source for item in self._h2d_quarantined)
                    source_release(quarantine=uncertain)
                source = None
                if pin is not None and not any(item['pin'] is pin for item in self._h2d_quarantined):
                    pin.release()
                self._diagnostics.event('finish', op, {'copy_completed': completed,
                    'pin_released': pin is None or pin.token is None})
        self._segment_counters['segment_materialized_features'] += 1

    def _put_manifest_ack(self, manifest):
        tensor = torch.tensor(list(bytes.fromhex(manifest['digest'])), dtype=torch.uint8)
        ptr, size = tensor.data_ptr(), tensor.numel()
        rc = self._store.register_buffer(ptr, size)
        if rc is not None and int(rc) != 0:
            raise RuntimeError('ACK marker registration failed')
        drained = False
        try:
            rc = self._store.put_from(manifest['ack_key'], ptr, size, self._put_config)
            drained = rc is None or int(rc) != -800
            if rc is not None and int(rc) < 0:
                raise RuntimeError('ACK marker publication failed')
        finally:
            if drained:
                try:
                    rc = self._store.unregister_buffer(ptr)
                except BaseException:
                    rc = -1
                if rc is not None and int(rc) != 0:
                    self._segment_ack_unsafe.add(manifest['token'])
                    self._capture_quarantined_registrations.add(ptr)
                    self._quarantine_buffer(tensor)
                    raise RuntimeError('ACK marker registration drain failed')
            else:
                self._segment_ack_unsafe.add(manifest['token'])
                self._capture_quarantined_registrations.add(ptr)
                self._quarantine_buffer(tensor)

    def _receipt_proofs(self):
        if self._terminal_proofs is None:
            from pathlib import Path
            from specforge.runtime.data_plane.terminal_receipt_proofs import TerminalReceiptProofs
            directory = os.environ.get('SPECFORGE_TRAINER_NODE_POOL_DIR')
            if not directory:
                raise ValueError('segmented clients require NODE_POOL_DIR as their shared terminal-proof anchor, even with caching disabled')
            root = Path(directory)
            capacity = int(os.environ.get('SPECFORGE_TRAINER_NODE_POOL_BYTES', str(8 << 30)))
            from specforge.runtime.data_plane.node_content_pool import node_pool_budgets
            proof_budget = node_pool_budgets(capacity)['terminal_proof_budget_bytes'] or (1 << 20)  # existing tiny CPU-test exception
            self._terminal_proofs = TerminalReceiptProofs(root.parent / (root.name + '.terminal.sqlite'), proof_budget)
        return self._terminal_proofs

    def _try_manifest_free(self, sample_id):
        manifest = self._segment_manifests[sample_id]
        proofs = self._receipt_proofs()
        claim = proofs.claim(self.store_id, sample_id, manifest, self._terminal_proof_owner)
        if claim == 'BUSY':
            return False  # Another exact claimant alone may publish ACK/proof.
        if claim != 'PROVED':
            if manifest['token'] in self._segment_ack_unsafe:
                raise RuntimeError('ACK native completion is unknown; exact attempt retained')
            if not self._store_exists(manifest['receipt_key']):
                if sample_id not in self._segment_ack_sent:
                    try:
                        self._put_manifest_ack(manifest)
                    except BaseException:
                        proofs.unsafe(self.store_id, sample_id, manifest, self._terminal_proof_owner)
                        raise
                    self._segment_ack_sent.add(sample_id)
                    self._segment_counters['manifest_ack_puts'] += 1
                if not self._store_exists(manifest['receipt_key']):
                    return False
            from specforge.runtime.contracts import FeatureSpec
            value = self._fetch_tensor(manifest['receipt_key'], FeatureSpec('__receipt', (32,), 'uint8'), 'cpu')
            if bytes(value.tolist()).hex() != manifest['digest']:
                proofs.unsafe(self.store_id, sample_id, manifest, self._terminal_proof_owner)
                raise RuntimeError('manifest ACK receipt digest mismatch')
            proofs.prove(self.store_id, sample_id, manifest, self._terminal_proof_owner)
        # A durable exact receipt proof also serves a fresh terminal authority
        # after the original consumer removed the receipt and the owner retired.
        # No client may issue another ACK after this state is durable.
        if not self._store_remove(manifest['receipt_key'], force=True):
            return False
        if self._store_exists(manifest['receipt_key']):
            return False
        self._segment_receipt_seen.add(sample_id)
        self._segment_counters['manifest_receipt_drains'] += 1
        return True

    def _quarantine_buffer(self, tensor: torch.Tensor) -> None:
        size = _nbytes(tensor)
        with self._quarantine_lock:
            self._quarantined_buffers.append(tensor)
            self._quarantined_bytes += size
            if self._quarantined_bytes > self.max_quarantined_bytes:
                raise MemoryError(
                    "Mooncake failed-transfer quarantine exceeded "
                    f"{self.max_quarantined_bytes} bytes; refusing unbounded "
                    "host-memory growth because an in-flight transfer may still "
                    "write into each quarantined buffer"
                )

    def _store_get_tensor(self, key: str, out: torch.Tensor) -> None:
        """Zero-copy fetch into a pre-allocated tensor. Raises KeyError if absent.

        The receive buffer is registered with the transfer engine for the get_into
        (required by the raw-buffer path), then unregistered.
        """
        nb = _nbytes(out)
        try:
            self._store.register_buffer(out.data_ptr(), nb)
        except Exception:  # pragma: no cover - some builds auto-register
            pass
        drained = False
        try:
            self._segment_counters["transport_get_requested_bytes"] += nb
            with self._diag.span("rdma_get_into"):
                rc = self._store.get_into(key, out.data_ptr(), nb)
            _check_get_result(key, rc, nb)
            self._diag.count("rdma_get_into_completed_bytes", nb)
            self._segment_counters["transport_get_completed_bytes"] += nb
            drained = True
        finally:
            if drained or getattr(self._capture_read_local, "intent", None) is None:
                try:
                    self._store.unregister_buffer(out.data_ptr())
                except Exception:  # pragma: no cover
                    pass
            else:
                # Unregistering before an uncertain DMA completes is unsafe.
                # The owned transport's successful close is the release proof.
                self._capture_quarantined_registrations.add(out.data_ptr())

    def _store_remove(self, key: str, *, force: bool = False) -> bool:
        """Best-effort physical free. Returns True once the key is gone.

        Recent Mooncake bindings expose ``remove(key, force=True)`` so a
        lifecycle authority can reclaim an object after all application-level
        leases have closed without waiting for Mooncake's (potentially
        minutes-long) KV lease TTL.  Older bindings only accept ``key``; keep
        those usable and let their normal bounded retry behavior apply.

        ``OBJECT_NOT_FOUND`` is a completed removal, not a failure: a sample's
        tensors are freed one key at a time, and a retry after a partial free
        (some keys removed, others still under a read lease) must not keep the
        sample pending until the bounded drain probes the absent keys.
        """
        try:
            if force:
                try:
                    rc = self._store.remove(key, force=True)
                except TypeError:
                    rc = self._store.remove(key)
            else:
                rc = self._store.remove(key)
        except Exception:  # pragma: no cover - transient RPC failure
            return False
        return rc is None or int(rc) in (0, MOONCAKE_OBJECT_NOT_FOUND)

    # -- write -------------------------------------------------------------
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        self.auth.check(self._credential)
        if not tensors:
            raise ValueError("put requires at least one tensor")
        staged = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        nbytes = sum(_nbytes(t) for t in staged.values())
        with self._lock:
            if sample_id in self._capture_fences:
                raise RuntimeError("cannot put over a fenced capture sample")
            if (
                self.max_resident_bytes is not None
                and sum(self._sample_bytes.values()) + nbytes > self.max_resident_bytes
            ):
                raise MemoryError(
                    f"MooncakeFeatureStore {self.store_id} over budget "
                    f"({self.max_resident_bytes} bytes): consumer is behind"
                )
            self._gen_counter += 1
            gen = self._gen_counter
            prior_gen = self._generation.get(sample_id)
            keys = {name: self._tkey(sample_id, gen, name) for name in staged}
            prior_keys = [
                self._tkey(sample_id, prior_gen, name)
                for name in self._sample_names.get(sample_id, [])
            ] if prior_gen is not None else []
        # One object per tensor, DMA'd straight from its storage. The shared put
        # config requests hard pinning when the Mooncake client supports it.
        # staged keeps the source tensors alive across the synchronous puts.
        attempted_keys: List[str] = []
        try:
            for name, t in staged.items():
                # A failed RPC may still have published its key.
                key = keys[name]
                attempted_keys.append(key)
                self._store_put_tensor(key, t)
        except BaseException:
            with self._lock:
                attempt = (sample_id, gen)
                self._partial_put_pending[attempt] = (attempted_keys, 0)
                try:
                    self._try_partial_put_free_locked(attempt)
                except Exception:
                    # Preserve both the original put error and cleanup ownership.
                    logger.exception(
                        "Failed to roll back partial Mooncake put %s", attempt
                    )
            raise
        # Overwrite-safe: drop the prior generation's tensor keys so a stale
        # ref's keys are gone (its get() then raises -> no use-after-free).
        if prior_gen is not None and prior_gen != gen:
            leaked = [key for key in prior_keys if not self._store_remove(key)]
            if leaked:
                logger.warning(
                    "MooncakeFeatureStore re-put of %s gen %s: removing prior "
                    "generation %s tensors %s failed; scheduling retry via gc()",
                    sample_id,
                    gen,
                    prior_gen,
                    leaked,
                )
                with self._lock:
                    self._partial_put_pending[(sample_id, prior_gen)] = (leaked, 0)
        with self._lock:
            self._generation[sample_id] = gen
            self._put_time[sample_id] = self._clock()
            self._sample_bytes[sample_id] = nbytes
            self._sample_names[sample_id] = list(staged)
        return SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", "unknown")),
            source_task_id=metadata.get("source_task_id"),
            feature_store_uri=f"mooncake://{self.store_id}/{sample_id}",
            feature_keys={k: f"{sample_id}/{k}" for k in staged},
            feature_specs=specs,
            strategy=metadata.get("strategy", "eagle3"),
            schema_version=int(metadata.get("schema_version", SCHEMA_VERSION)),
            target_model_version=str(metadata.get("target_model_version", "unknown")),
            draft_weight_version=metadata.get("draft_weight_version"),
            tokenizer_version=str(metadata.get("tokenizer_version", "unknown")),
            num_tokens=int(metadata.get("num_tokens", 0)),
            estimated_bytes=nbytes,
            metadata={
                **{k: v for k, v in metadata.items() if k != "num_tokens"},
                "generation": gen,  # travels with the ref for the staleness guard
            },
        )

    def adopt(self, sample_ref: SampleRef) -> None:
        """Register an externally-produced sample for lifecycle management.

        The server-capture transport writes tensors into this store's key
        namespace from ANOTHER process (the SGLang server's sink), so this
        instance has no put-side bookkeeping for them. ``adopt()`` records the
        ref's generation / feature names / size so ``release``/``abort``/``gc``
        can free the server-written objects exactly like locally-put ones.
        """
        gen = sample_ref.metadata.get("generation")
        if gen is None:
            raise ValueError(
                f"cannot adopt {sample_ref.sample_id}: ref carries no generation"
            )
        self._capture_from_ref(sample_ref, validate_remote=True)
        self._locate(sample_ref)
        with self._lock:
            gen = int(gen)
            self._generation[sample_ref.sample_id] = gen
            self._sample_names[sample_ref.sample_id] = list(
                sample_ref.feature_keys.keys()
            )
            self._sample_bytes[sample_ref.sample_id] = int(
                sample_ref.estimated_bytes or 0
            )
            self._put_time[sample_ref.sample_id] = self._clock()
            self._external_provisional.pop((sample_ref.sample_id, gen), None)

    def track_external_attempt(
        self,
        sample_id: str,
        *,
        generation: int,
        feature_names: List[str],
        server_url: Optional[str] = None,
    ) -> None:
        """Track server-owned keys before an HTTP response makes a ref adoptable."""
        names = list(dict.fromkeys(str(name) for name in feature_names))
        if not names:
            raise ValueError("external capture attempt must name at least one feature")
        with self._lock:
            track = getattr(self._store, "track_capture_attempt", None)
            if track is not None:
                if server_url is None:
                    raise ValueError("arena capture attempts require a server_url")
                track(
                    [
                        self._tkey(str(sample_id), int(generation), name)
                        for name in names
                    ],
                    server_url,
                )
            self._external_provisional[(str(sample_id), int(generation))] = names

    def discard_external_attempts(self, *, reason="unadopted-external-capture"):
        self._recover_capture_reads(require_drained=True)
        with self._lock:
            pending_begins = list(self._capture_pending_begin)
        for sid in pending_begins:
            self._recover_capture_begin(sid)
        with self._lock:
            snapshot = [(identity, names, self._capture_fences.get(identity[0]))
                        for identity, names in self._external_provisional.items()]
        discarded = 0
        legacy = []
        for (sid, gen), names, expected in snapshot:
            if expected is None:
                legacy.append(((sid, gen), names))
                continue
            with expected.lock:
                with self._lock:
                    if self._capture_fences.get(sid) is not expected or (sid, gen) not in self._external_provisional:
                        continue
                self.abort(sid, reason=reason, attempt_token=expected.token)
                discarded += 1
        return discarded + (self._discard_external_attempts_legacy(reason=reason, expected_attempts=legacy) if legacy else 0)

    def _discard_external_attempts_legacy(
        self, *, reason: str = "unadopted-external-capture", expected_attempts=()
    ) -> int:
        """Abort server writes that never produced an adoptable response.

        The index belongs to the shared store rather than an adapter, so a retry
        that succeeds through another capture server clears the provisional
        entry in :meth:`adopt` and cannot be deleted by the failed adapter's
        shutdown path. Physical-remove failures remain visible to
        :meth:`drain_pending_removals`.
        """
        attempts = expected_attempts

        errors: Dict[str, str] = {}
        discarded = 0
        for (sample_id, generation), names in attempts:
            with self._lock:
                fence = self._capture_fences.get(sample_id)
            if fence is not None:
                continue
            with self._lock:
                attempt_key = (sample_id, generation)
                # A response can be adopted after the snapshot but before this
                # attempt is visited. adopt() removes the provisional entry
                # under the same lock, so never delete that now-live sample.
                if self._external_provisional.get(attempt_key) is not names or self._capture_fences.get(sample_id) is not None:
                    continue
                current = self._generation.get(sample_id)
                if current is not None:
                    if current != generation:
                        errors[sample_id] = (
                            f"cannot discard provisional sample {sample_id!r} "
                            f"generation {generation}: adopted generation is {current}"
                        )
                        continue
                else:
                    self._generation[sample_id] = generation
                    self._sample_names[sample_id] = names
                    self._sample_bytes[sample_id] = 0
                    self._put_time[sample_id] = self._clock()
                try:
                    # RLock keeps adopt() from racing between the ownership
                    # check above and this removal attempt. abort() reacquires
                    # the same lock and either frees the keys or records them
                    # in _release_pending for lifecycle drain.
                    self.abort(sample_id, reason=reason)
                except BaseException as exc:
                    # Preserve both explicit provisional ownership and a
                    # retryable removal entry.  Terminal drain can now reclaim
                    # the keys even when a Mooncake metadata probe itself
                    # failed, rather than losing every remaining snapshot item.
                    self._release_pending.setdefault(sample_id, 0)
                    errors[sample_id] = f"{type(exc).__name__}: {exc}"
                    continue
                self._external_provisional.pop(attempt_key, None)
                discarded += 1
        self._flush_capture_acks()
        if errors:
            raise RuntimeError(
                "could not discard all provisional external captures: " f"{errors}"
            )
        return discarded

    # -- read --------------------------------------------------------------
    def get_into(self, sample_ref, destinations):
        """Fill caller-owned final-batch views without intermediate GPU tensors."""
        self._capture_from_ref(sample_ref)
        mapping = {}
        for name, tensor in destinations.items():
            spec = sample_ref.feature_specs[name]
            if tuple(tensor.shape) != tuple(spec.shape) or str(tensor.dtype).removeprefix("torch.") != spec.dtype:
                raise ValueError(f"destination layout differs for {name!r}")
            if not tensor.is_contiguous():
                raise ValueError("direct destination must be contiguous")
            mapping[self._tkey(sample_ref.sample_id, int(sample_ref.metadata["generation"]), name)] = tensor
        previous = getattr(self._capture_read_local, "destinations", {})
        self._capture_read_local.destinations = mapping
        try:
            return self.get(sample_ref, device="cpu", names=list(destinations))
        finally:
            self._capture_read_local.destinations = previous

    def get(self, sample_ref: SampleRef, *, device="cpu", names=None):
        with self._lock:
            if self._closing or self._closed:
                raise RuntimeError("Mooncake feature store is closing or closed")
            self._active_get_calls += 1
        try:
            return self._get_with_capture_fence(sample_ref, device=device, names=names)
        finally:
            with self._lock:
                self._active_get_calls -= 1

    def _get_with_capture_fence(self, sample_ref: SampleRef, *, device="cpu", names=None):
        self.auth.check(self._credential)
        self._recover_capture_reads()
        intent = self._begin_capture_read(sample_ref)
        previous = getattr(self._capture_read_local, "intent", None)
        self._capture_read_local.intent = intent
        try:
            result = self._get_impl(sample_ref, device=device, names=names)
        except BaseException:
            if intent is not None:
                intent["active"] = False
                if not intent["unsafe_fetch"]:
                    intent["release_requested"] = True
                    self._release_capture_read(intent)
            raise
        finally:
            self._capture_read_local.intent = previous
        if intent is not None:
            with self._lock:
                intent["handle"] = result[1]
                self._capture_reads[result[1].lease_token] = intent
                intent["active"] = False
        return result

    def _get_impl(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        self.auth.check(self._credential)
        self._capture_from_ref(sample_ref)
        sid = sample_ref.sample_id
        ref_gen = sample_ref.metadata.get("generation")
        with self._lock:
            if ref_gen is not None and self._capture_ref_freed_locked(sid, sample_ref.metadata.get("capture_fence") or {"gen": ref_gen}):
                # logically freed here; the remote bytes may still linger under
                # Mooncake's read-lease, but this ref must not resolve (B5)
                raise KeyError(
                    f"sample {sid} generation {ref_gen} was released/aborted; "
                    f"refusing use-after-free"
                )
        wanted = names or list(sample_ref.feature_keys.keys())
        self._locate(sample_ref)
        out, gen = self._get_tensors(sample_ref, wanted, device)
        # Pooled reads already placed every feature (integer features on the
        # host); pageable reads land on the host and move here.
        if self._receive_pool is None and str(device) != "cpu":
            target = torch.device(device)
            out = {
                k: (v if v.device == target else v.to(target)) for k, v in out.items()
            }
        with self._lock:
            self._counter += 1
            # Consumer-side cache: a process that only get()s a sample (never
            # put() it) still needs gen + feature names so its release()/abort()
            # can free the per-tensor keys. setdefault keeps the producer's own
            # entries authoritative when producer and consumer are one instance.
            self._generation.setdefault(sid, gen)
            self._sample_names.setdefault(sid, list(sample_ref.feature_keys.keys()))
            handle = FeatureHandle(
                sample_id=sid,
                generation=gen,
                lease_token=f"{sid}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
            fence = self._capture_fences.get(sid)
            if fence is not None:
                self._capture_handle_tokens[handle.lease_token] = fence.token
        return out, handle

    def _locate(self, sample_ref: SampleRef) -> None:
        """Pass server-owned object locations to backends that need them.

        The arena backend (``mooncake_arena.ArenaObjectClient``) has no
        master to resolve keys; the capture server's addresses ride on the ref.
        """
        locate = getattr(self._store, "locate_ref_objects", None)
        gen = sample_ref.metadata.get("generation")
        if locate is None or gen is None:
            return
        locate(
            {
                name: self._tkey(sample_ref.sample_id, int(gen), name)
                for name in sample_ref.feature_keys
            },
            sample_ref.metadata,
        )

    def _get_tensors(
        self, ref: SampleRef, wanted: List[str], device="cpu"
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Read each feature straight into a spec-allocated tensor."""
        sid = ref.sample_id
        gen = ref.metadata.get("generation")
        if gen is None:
            raise KeyError(f"sample {sid} ref carries no generation; cannot locate")
        gen = int(gen)
        out: Dict[str, torch.Tensor] = {}
        for n in wanted:
            spec = ref.feature_specs.get(n)
            if spec is None:
                raise KeyError(f"sample {sid} ref has no spec for feature {n!r}")
            key = self._tkey(sid, gen, n)
            if sid in self._segment_manifests and n in self._segment_manifests[sid]['features']:
                if not self._store_exists(key):
                    raise KeyError('manifest ownership object is no longer readable')
                try:
                    destination = getattr(self._capture_read_local, 'destinations', {}).get(key)
                    if destination is None:
                        dtype = _TORCH_DTYPES[spec.dtype]
                        target = torch.device('cpu') if _stays_on_host(dtype) else torch.device(device)
                        destination = torch.empty(spec.shape, dtype=dtype, device=target)
                    self._get_segments(ref, n, destination)
                    out[n] = destination
                except BaseException:
                    intent = getattr(self._capture_read_local, 'intent', None)
                    if intent is not None:
                        intent['unsafe_fetch'] = True
                    raise
                continue
            if not self._store_exists(key):
                # freed (release/abort), superseded by a re-put, or never written
                raise KeyError(
                    f"sample {sid} gen {gen} feature {n!r} not available "
                    f"(freed, stale, or never written)"
                )
            # fresh alloc (or a pooled slot copied out) -> clone-on-fetch for free (B5)
            try:
                out[n] = self._fetch_tensor(key, spec, device)
            except BaseException:
                intent = getattr(self._capture_read_local, "intent", None)
                if intent is not None:
                    intent["unsafe_fetch"] = True
                raise
        return out, gen

    # -- lifetime ----------------------------------------------------------
    def _try_partial_put_free_locked(
        self, attempt: Tuple[str, int], *, force: bool = False
    ) -> bool:
        """Remove keys from an unpublished generation, retaining failed keys."""
        keys, attempts = self._partial_put_pending[attempt]
        remaining = [key for key in keys if not self._store_remove(key, force=force)]
        if remaining:
            self._partial_put_pending[attempt] = (remaining, attempts)
            return False
        del self._partial_put_pending[attempt]
        return True

    def _try_physical_free(
        self,
        sample_id: str,
        *,
        confirm_absent_on_failure: bool = True,
        force: bool = False,
    ) -> bool:
        """Remove all tensor objects. False on a retryable RPC failure.

        Order matters against Mooncake's lease semantics: an is_exist probe
        GRANTS a read lease, and a remove during any live lease fails (-706).
        So each key is removed FIRST. The optional exist probe runs only after a
        failed remove, purely to classify "already gone" as freed. Retry loops
        disable that probe because probing a still-live key would renew its
        lease and make every following remove fail again.
        """
        fence = self._capture_fences.get(sample_id)
        expected = getattr(self._capture_cleanup_local, "records", {}).get(sample_id)
        if expected is not None and fence is not expected:
            return False
        if fence is not None:
            if expected is not fence:
                return False
            if fence.remote_retired:
                return True  # Other store retired old attempt: discard LOCAL metadata only.
            if not fence.writer_drained or (not fence.transferred and not fence.cleanup_token):
                return False
        if fence is not None and fence.transferred and any(intent['fence'] is fence for intent in self._capture_read_intents.values()):
            return False
        gen = self._generation.get(sample_id)
        if gen is None:
            return True  # nothing tracked to remove (already freed)
        if sample_id in self._segment_manifests:
            return self._try_manifest_free(sample_id)
        ok = True
        for name in self._sample_names.get(sample_id, []):
            key = self._tkey(sample_id, gen, name)
            if self._store_remove(key, force=force):
                continue
            if confirm_absent_on_failure and not self._store_exists(key):
                continue  # already gone (freed remotely) counts as freed
            ok = False
        return ok

    def _free_bookkeeping_locked(self, sample_id: str) -> int:
        """Drop in-process tracking for a sample. Returns bytes accounted freed."""
        fence = self._capture_fences.get(sample_id)
        if fence is not None:
            if not self._capture_cleanup_allows(sample_id):
                raise RuntimeError("stale cleanup cannot mutate replacement bookkeeping")
            if not fence.writer_drained:
                raise RuntimeError("free bookkeeping before capture writer drain")
            fence.removed = True
            self._capture_ack_pending[sample_id] = fence
        self._segment_manifests.pop(sample_id, None)
        self._segment_ack_sent.discard(sample_id)
        self._segment_receipt_seen.discard(sample_id)
        generation = self._generation.get(sample_id)
        nbytes = self._sample_bytes.pop(sample_id, 0)
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        self._sample_names.pop(sample_id, None)
        self._release_pending.pop(sample_id, None)
        if generation is not None:
            self._external_provisional.pop((sample_id, generation), None)
        return nbytes

    def _still_leased_locked(self, sample_id: str, generation: Optional[int]) -> bool:
        # generation-aware: a stale older-generation lease does not pin the
        # current generation (matches LocalFeatureStore's invariant).
        return any(
            h.sample_id == sample_id and h.generation == generation
            for h in self._active_leases.values()
        )

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            intent = self._capture_reads.get(handle.lease_token)
            expected = intent["fence"] if intent is not None else self._capture_fences.get(handle.sample_id)
            if intent is None and handle.lease_token not in self._active_leases and expected is not None:
                return
        if intent is not None:
            intent["release_requested"] = True
            self._release_capture_read(intent)
            if self.retain_on_release:
                return
            with self._lock:
                if expected.transferred and any(other["fence"] is expected for other in self._capture_read_intents.values()):
                    return  # The final local reader owns the eventual deletion.
        with self._capture_cleanup_scope([handle.sample_id], expected_records={handle.sample_id: expected}):
            return self._release_impl(handle, reason=reason)

    def _release_impl(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            if not self._capture_cleanup_allows(handle.sample_id):
                return
            expected = self._capture_handle_tokens.pop(handle.lease_token, None)
            fence = self._capture_fences.get(handle.sample_id)
            if expected is not None and (fence is None or fence.token != expected):
                self._active_leases.pop(handle.lease_token, None)
                return

            self._active_leases.pop(handle.lease_token, None)
            if self.retain_on_release:
                return  # offline re-iterable set: keep for the next epoch
            sid = handle.sample_id
            cur = self._generation.get(sid)
            if cur is not None and handle.generation != cur:
                return  # stale lease -> no-op
            if self._still_leased_locked(sid, cur):
                return
            self._mark_freed_locked(sid, handle.generation)  # immediate logical free
            if self._try_physical_free(sid):
                self._free_bookkeeping_locked(sid)
            else:
                # remote free deferred (lease) / failed -> gc() retries
                self._release_pending.setdefault(sid, 0)

        self._flush_capture_acks()

    def abort_ref(self, ref: SampleRef, *, reason="aborted") -> None:
        wire = ref.metadata.get("capture_fence")
        if wire is not None:
            if wire.get("sample_id") != ref.sample_id or str(wire.get("store_id")) != str(self.store_id) or not wire.get("reservation_token"):
                raise ValueError("capture cleanup ref identity mismatch")
            self.abort(ref.sample_id, reason=reason, attempt_token=wire["reservation_token"])
            return
        # A legacy callback carries no authority over a later negotiated
        # attempt. Keep the generation comparison and delete under one lock.
        with self._lock:
            if ref.sample_id in self._capture_fences:
                return
            generation = ref.metadata.get("generation")
            if generation is None or self._generation.get(ref.sample_id) != int(generation):
                return
            self.abort(ref.sample_id, reason=reason)

    def abort(self, sample_id: str, *, reason: str = "aborted", attempt_token=None) -> None:
        with self._lock:
            current = self._capture_fences.get(sample_id)
            if attempt_token is not None and (current is None or current.token != attempt_token):
                return
        with self._capture_cleanup_scope([sample_id], expected_records={sample_id: current}):
            return self._abort_impl(sample_id, reason=reason, attempt_token=attempt_token)

    def _abort_impl(self, sample_id: str, *, reason: str = "aborted", attempt_token=None) -> None:
        with self._lock:
            if not self._capture_cleanup_allows(sample_id):
                return
            fence = self._capture_fences.get(sample_id)
            if attempt_token is not None and (fence is None or fence.token != attempt_token):
                return
        if fence is not None:
            self._resolve_capture_writer(sample_id, attempt_token)
        with self._lock:
            if fence is not None:
                if self._capture_fences.get(sample_id) is not fence:
                    return
                if any(h.sample_id == sample_id for h in self._active_leases.values()):
                    raise RuntimeError("cannot abort capture with active consumer lease")
                # Lost responses have no adopted generation, but still own exact keys.
                self._generation.setdefault(sample_id, int(fence.wire["gen"]))
                self._sample_names.setdefault(sample_id, list(fence.names))
                self._sample_bytes.setdefault(sample_id, 0)

            gen = self._generation.get(sample_id)
            if gen is not None:
                self._mark_freed_locked(sample_id, gen)  # immediate logical free
            if self._try_physical_free(sample_id):
                self._free_bookkeeping_locked(sample_id)
            else:
                self._release_pending.setdefault(sample_id, 0)

        self._flush_capture_acks()

    def drain_sample_removals(
        self,
        sample_ids: List[str],
        *,
        max_attempts: int = DEFAULT_SAMPLE_DRAIN_MAX_ATTEMPTS,
        retry_interval_s: float = DEFAULT_SAMPLE_DRAIN_RETRY_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, int]:
        """Force-remove only the named optimizer-durable samples.

        Other pending samples may belong to prefetched, not-yet-durable
        batches and must remain available for crash replay.
        """
        result = self._drain_removals(
            sample_ids=sample_ids,
            max_attempts=max_attempts,
            retry_interval_s=retry_interval_s,
            sleep=sleep,
        )

        self._flush_capture_acks()
        return result

    def retry_sample_removals(self, sample_ids: List[str]) -> Dict[str, Any]:
        with self._lock:
            targets = [sid for sid in sample_ids if sid in self._release_pending or sid in self._capture_ack_pending]
        with self._capture_cleanup_scope(targets):
            return self._retry_sample_removals_impl(sample_ids)

    def _retry_sample_removals_impl(self, sample_ids: List[str]) -> Dict[str, Any]:
        """Try selected removals once, without probing or sleeping.

        The optimizer path calls this only for samples made durable at an
        earlier boundary.  A failed remove remains in ``_release_pending`` for
        the next batched attempt; avoiding ``is_exist`` here is important
        because that probe renews Mooncake's read lease.
        """
        target_ids = set(sample_ids)
        removed = removed_bytes = 0
        with self._lock:
            pending = [
                sample_id
                for sample_id in self._release_pending
                if sample_id in target_ids
            ]
            for sample_id in pending:
                if not self._capture_cleanup_allows(sample_id):
                    continue
                physically_removed = self._try_physical_free(
                    sample_id,
                    force=True,
                    confirm_absent_on_failure=False,
                )
                if physically_removed:
                    sample_bytes = self._free_bookkeeping_locked(sample_id)
                    removed += 1
                    removed_bytes += sample_bytes
                    self._stats["force_freed"] += 1
                    self._stats["force_freed_bytes"] += sample_bytes
                else:
                    self._release_pending[sample_id] = min(
                        self.max_release_attempts,
                        self._release_pending.get(sample_id, 0) + 1,
                    )
            partial_pending = [
                attempt
                for attempt in self._partial_put_pending
                if attempt[0] in target_ids
            ]
            for attempt in partial_pending:
                if self._try_partial_put_free_locked(attempt, force=True):
                    removed += 1
                    self._stats["force_freed"] += 1
                else:
                    keys, count = self._partial_put_pending[attempt]
                    self._partial_put_pending[attempt] = (
                        keys,
                        min(self.max_release_attempts, count + 1),
                    )
            remaining = [
                sample_id
                for sample_id in self._release_pending
                if sample_id in target_ids
            ]
            remaining += [
                sample_id
                for sample_id, _ in self._partial_put_pending
                if sample_id in target_ids
            ]
        self._flush_capture_acks()
        return {
            "removed": removed,
            "removed_bytes": removed_bytes,
            "release_pending": len(remaining),
            "remaining_ids": remaining,
            "attempts": 1 if pending or partial_pending else 0,
        }

    def drain_pending_removals(
        self,
        *,
        max_attempts: int = DEFAULT_PENDING_DRAIN_MAX_ATTEMPTS,
        retry_interval_s: float = DEFAULT_PENDING_DRAIN_RETRY_INTERVAL_S,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, int]:
        """Retry deferred removes at lifecycle shutdown or fail loudly.

        ``gc()`` is a periodic best-effort pump.  This method is the stronger
        terminal contract used by online producer/consumer finalization: it is
        bounded, never discards the keys needed for another remove attempt, and
        raises with the remaining sample ids when the remote RPC cannot drain.
        ``sleep`` is injectable so protocol tests can advance a fake lease clock
        without wall-clock delays.
        """
        result = self._drain_removals(
            sample_ids=None,
            max_attempts=max_attempts,
            retry_interval_s=retry_interval_s,
            sleep=sleep,
        )

        self._flush_capture_acks()
        return result

    def _drain_removals(self, *, sample_ids, **kwargs):
        self._recover_capture_reads(require_drained=True)
        with self._lock:
            targets = [sid for sid in set(self._release_pending) | set(self._capture_ack_pending) if sample_ids is None or sid in sample_ids]
        with self._capture_cleanup_scope(targets):
            return self._drain_removals_impl(sample_ids=sample_ids, **kwargs)

    def _drain_removals_impl(
        self,
        *,
        sample_ids: Optional[List[str]],
        max_attempts: int,
        retry_interval_s: float,
        sleep: Callable[[float], None],
    ) -> Dict[str, int]:
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if retry_interval_s < 0:
            raise ValueError("retry_interval_s must be >= 0")
        target_ids = None if sample_ids is None else set(sample_ids)
        with self._lock:
            fenced = [sid for sid in self._release_pending if target_ids is None or sid in target_ids]
        for sid in fenced:
            self._resolve_capture_writer(sid)
        self._flush_capture_acks()
        removed = removed_bytes = 0
        last_errors: Dict[str, str] = {}
        attempts_run = 0
        for attempt in range(max_attempts):
            attempts_run = attempt + 1
            with self._lock:
                pending = [
                    sample_id
                    for sample_id in self._release_pending
                    if target_ids is None or sample_id in target_ids
                ]
                partial_pending = [
                    key
                    for key in self._partial_put_pending
                    if target_ids is None or key[0] in target_ids
                ]
                if not pending and not partial_pending:
                    return {
                        "removed": removed,
                        "removed_bytes": removed_bytes,
                        "release_pending": 0,
                        "attempts": attempt,
                    }
                final_attempt = attempt + 1 == max_attempts
                for sample_id in pending:
                    if not self._capture_cleanup_allows(sample_id):
                        continue
                    try:
                        physically_removed = self._try_physical_free(
                            sample_id,
                            # The application lease has already been released
                            # before a sample enters _release_pending.  Use the
                            # lifecycle-authority path in current Mooncake so
                            # its default multi-minute KV lease does not turn a
                            # clean trainer shutdown into a false failure.
                            force=True,
                            # Intermediate retries must not renew Mooncake's
                            # read lease. The final probe only classifies an
                            # already-absent key and has no following retry to
                            # poison.
                            confirm_absent_on_failure=final_attempt,
                        )
                    except Exception as exc:  # preserve state for the next retry
                        last_errors[sample_id] = f"{type(exc).__name__}: {exc}"
                        physically_removed = False
                    if physically_removed:
                        sample_bytes = self._free_bookkeeping_locked(sample_id)
                        removed_bytes += sample_bytes
                        removed += 1
                        self._stats["force_freed"] += 1
                        self._stats["force_freed_bytes"] += sample_bytes
                        last_errors.pop(sample_id, None)
                    else:
                        self._release_pending[sample_id] = min(
                            self.max_release_attempts,
                            self._release_pending.get(sample_id, 0) + 1,
                        )
                for key in partial_pending:
                    label = f"{key[0]}/g{key[1]}"
                    try:
                        physically_removed = self._try_partial_put_free_locked(
                            key, force=True
                        )
                    except Exception as exc:
                        last_errors[label] = f"{type(exc).__name__}: {exc}"
                        physically_removed = False
                    if physically_removed:
                        removed += 1
                        self._stats["force_freed"] += 1
                        last_errors.pop(label, None)
                    else:
                        keys, count = self._partial_put_pending[key]
                        self._partial_put_pending[key] = (
                            keys,
                            min(self.max_release_attempts, count + 1),
                        )
                remaining = [
                    sample_id
                    for sample_id in self._release_pending
                    if target_ids is None or sample_id in target_ids
                ]
                remaining += [
                    f"{sid}/g{generation}"
                    for sid, generation in self._partial_put_pending
                    if target_ids is None or sid in target_ids
                ]
            if not remaining:
                return {
                    "removed": removed,
                    "removed_bytes": removed_bytes,
                    "release_pending": 0,
                    "attempts": attempts_run,
                }
            if attempt + 1 < max_attempts and retry_interval_s:
                sleep(retry_interval_s)

        with self._lock:
            remaining = [
                sample_id
                for sample_id in self._release_pending
                if target_ids is None or sample_id in target_ids
            ]
            remaining += [
                f"{sid}/g{generation}"
                for sid, generation in self._partial_put_pending
                if target_ids is None or sid in target_ids
            ]
        preview = remaining[:16]
        detail = f"; last errors={last_errors}" if last_errors else ""
        raise RuntimeError(
            f"MooncakeFeatureStore {self.store_id} could not drain "
            f"{len(remaining)} pending removal(s) after {attempts_run} attempts: "
            f"{preview}{detail}"
        )

    def _capture_ref_freed_locked(self, sid, wire):
        if wire.get('object_key_layout') == 'attempt-v1':
            return (sid, wire.get('reservation_token')) in self._freed_capture_attempts
        return (sid, int(wire['gen'])) in self._freed

    def _mark_freed_locked(self, sid, generation):
        self._freed.add((sid, generation))
        fence = self._capture_fences.get(sid)
        if fence is not None and fence.wire.get('object_key_layout') == 'attempt-v1':
            self._freed_capture_attempts.add((sid, fence.token))

    def _published_capture_locked(self, sample_id):
        fence = self._capture_fences.get(sample_id)
        return fence is not None and fence.transferred

    def relinquish_published(self, ref):
        """Disown a possibly visible ref after the streaming publisher handoff.

        This drops producer metadata only. The durable ref and its designated
        consumer/optimizer authority now own deletion of the immutable objects.
        """
        wire = ref.metadata.get('capture_fence') or {}
        if wire.get('publication_ownership') != 'attempt-v1':
            return
        sid = ref.sample_id
        with self._lock:
            fence = self._capture_fences.get(sid)
            if fence is None:
                return
        with fence.lock, self._lock:
            if self._capture_fences.get(sid) is not fence or fence.token != wire.get('reservation_token'):
                return
            if not fence.transferred or any(h.sample_id == sid for h in self._active_leases.values()) or any(intent['fence'] is fence for intent in self._capture_read_intents.values()):
                raise RuntimeError('cannot relinquish capture with local reader ownership')
            self._mark_freed_locked(sid, int(wire['gen']))
            for mapping in (self._generation, self._sample_names, self._sample_bytes, self._put_time,
                            self._release_pending, self._capture_ack_pending, self._capture_fences,
                            self._segment_manifests):
                mapping.pop(sid, None)
            self._segment_ack_sent.discard(sid)
            self._segment_receipt_seen.discard(sid)
            self._external_provisional.pop((sid, int(wire['gen'])), None)

    def _validate_terminal_capture_ref(self, ref, wire):
        """Pure validation must precede even an already-cleaned fast return."""
        from specforge.inference.capture_protocol import WriterFence

        if (wire.get("protocol_version") != 1 or wire.get("sample_id") != ref.sample_id
                or str(wire.get("store_id")) != str(self.store_id)
                or int(wire.get("gen", -1)) != 1
                or ref.metadata.get("generation") is None
                or int(ref.metadata["generation"]) != int(wire["gen"])):
            raise ValueError("terminal capture ref sample/store/generation identity mismatch")
        if ref.feature_store_uri != f"mooncake://{self.store_id}/{ref.sample_id}":
            raise ValueError("terminal capture ref canonical URI identity mismatch")
        proof = WriterFence(dict(wire), tuple(ref.feature_keys))
        if not proof.accept_transfer(wire) or ref.feature_keys != wire.get("physical_keys"):
            raise ValueError("terminal capture ref certificate/physical keys mismatch")

    def cleanup_terminal_refs(self, refs, *, consumer_done=False, producer_closed=False):
        """Reclaim exact published attempts after both sides have finished.

        Normal handoff tombstones stay sealed. A separate metadata authority
        may adopt immutable refs only after successful consumer drain and the
        producer's joined-worker EOF. Failure/timeout alone proves neither.
        The authority borrows this native backend and must never close it.
        """
        refs = tuple(refs)
        errors = []
        cleaned = 0
        with self._terminal_cleanup_lock:
            for ref in refs:
                wire = ref.metadata.get("capture_fence") or {}
                immutable = (wire.get("object_key_layout") == "attempt-v1" or
                             wire.get("publication_ownership") == "attempt-v1")
                try:
                    if not immutable:
                        # Legacy cleanup retains its existing remote reader /
                        # writer permits, even during an abnormal shutdown.
                        self.abort_ref(ref, reason="online-attempt-finished")
                        cleaned += 1
                        continue
                    if consumer_done is not True or producer_closed is not True:
                        raise RuntimeError("terminal ownership unresolved: successful consumer_done and closed producer are required")
                    self._validate_terminal_capture_ref(ref, wire)
                    authority = self._terminal_cleanup_authority
                    if authority is None:
                        authority = MooncakeFeatureStore(store_id=self.store_id, store=self._store,
                            auth=self.auth, credential=self._credential,
                            max_release_attempts=self.max_release_attempts, node_pool_enabled=False)
                        # Borrow the native backend together with its typed put config.
                        authority._put_config = self._put_config
                        self._terminal_cleanup_authority = authority
                    token = wire.get("reservation_token")
                    with authority._lock:
                        already_cleaned = ((ref.sample_id, token) in authority._freed_capture_attempts
                                           and ref.sample_id not in authority._capture_fences)
                    if not already_cleaned:
                        authority.adopt(ref)  # Strict certificate + exact key validation.
                        authority.abort_ref(ref, reason="online-attempt-finished")
                        authority.drain_sample_removals([ref.sample_id])
                    cleaned += 1
                except Exception as exc:
                    errors.append(f"{ref.sample_id}/{wire.get('reservation_token')}: {type(exc).__name__}: {exc}")
            authority = self._terminal_cleanup_authority
            if authority is not None and consumer_done is True and producer_closed is True:
                try:
                    authority.drain_pending_removals()
                except Exception as exc:
                    errors.append(f"terminal exact-key removal drain: {type(exc).__name__}: {exc}")
        if errors:
            raise RuntimeError("terminal published-ref cleanup unresolved: " + repr(errors))
        return {"refs_cleaned": cleaned}

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        self._recover_capture_reads()
        current = self._clock() if now is None else now
        with self._lock:
            targets = set(self._release_pending) | set(self._capture_ack_pending)
            if self.max_hold_age_s is not None:
                targets.update(sid for sid, start in self._put_time.items()
                               if current-start > self.max_hold_age_s and not self._published_capture_locked(sid) and not self._still_leased_locked(sid, self._generation.get(sid)))
        with self._capture_cleanup_scope(targets):
            return self._gc_impl(now=current)

    def _gc_impl(self, *, now: Optional[float] = None) -> Dict[str, int]:
        now = self._clock() if now is None else now
        freed = freed_bytes = 0
        with self._lock:
            # max-hold sweep: force-free abandoned samples (spare still-leased)
            if self.max_hold_age_s is not None:
                stale = [
                    sid
                    for sid, t in list(self._put_time.items())
                    if now - t > self.max_hold_age_s
                    and not self._published_capture_locked(sid)
                    and not self._still_leased_locked(sid, self._generation.get(sid))
                ]
                for sid in stale:
                    if not self._capture_cleanup_allows(sid):
                        continue
                    if self._try_physical_free(sid, confirm_absent_on_failure=False):
                        freed_bytes += self._free_bookkeeping_locked(sid)
                        freed += 1
                    else:
                        self._release_pending.setdefault(sid, 0)
            # Reconcile release-pending without an exists probe: is_exist grants
            # a read lease that would make the next remove fail (-706).
            for sid in list(self._release_pending):
                if not self._capture_cleanup_allows(sid):
                    continue
                if self._release_pending[sid] >= self.max_release_attempts:
                    # Keep the physical key metadata and surface the pending
                    # sample. Lifecycle drain owns the final bounded retry and
                    # loud failure; silently dropping this bookkeeping would
                    # make a remote object leak invisible.
                    continue
                attempts = self._release_pending[sid] + 1
                if self._try_physical_free(sid, confirm_absent_on_failure=False):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    freed += 1
                else:
                    self._release_pending[sid] = attempts
            for key, (_, attempts) in list(self._partial_put_pending.items()):
                if attempts >= self.max_release_attempts:
                    continue
                if self._try_partial_put_free_locked(key):
                    freed += 1
                else:
                    remaining, _ = self._partial_put_pending[key]
                    self._partial_put_pending[key] = (remaining, attempts + 1)
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
        self._flush_capture_acks()
        return {
            "force_freed": freed,
            "force_freed_bytes": freed_bytes,
            "release_pending": len(self._release_pending)
            + len(self._partial_put_pending),
        }

    def health(self) -> Dict[str, Any]:
        with self._quarantine_lock:
            quarantined_buffers = len(self._quarantined_buffers)
            quarantined_bytes = self._quarantined_bytes
        with self._lock:
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            # NOTE: resident_bytes is an in-process accounting sum, not a live
            # Mooncake pool-usage query (the Python API exposes only per-key
            # get_size). A cross-node pool-usage signal is a follow-up.
            result = {
                "store_id": self.store_id,
                "backend": "mooncake",
                "resident_samples": len(self._generation),
                "provisional_external": len(self._external_provisional),
                "capture_pending_begin": len(self._capture_pending_begin),
                "capture_fences": len(self._capture_fences),
                "capture_ack_pending": len(self._capture_ack_pending),
                "capture_handle_attempts": len(self._capture_handle_tokens),
                "capture_read_intents": len(self._capture_read_intents),
                "capture_active_reads": len(self._capture_reads),
                "capture_failed_reads": sum(bool(x["unsafe_fetch"]) for x in self._capture_read_intents.values()),
                "capture_quarantined_registrations": len(self._capture_quarantined_registrations),
                "active_leases": len(self._active_leases),
                "resident_bytes": sum(self._sample_bytes.values()),
                "max_resident_bytes": self.max_resident_bytes,
                "auth_required": self.auth.required,
                "release_pending": len(self._release_pending)
                + len(self._partial_put_pending),
                "oldest_age_s": max(ages) if ages else 0.0,
                "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
                "force_freed_total": self._stats["force_freed"],
                "hard_pin": bool(getattr(self._put_config, "with_hard_pin", False)),
                "quarantined_buffers": quarantined_buffers,
                "quarantined_bytes": quarantined_bytes,
                "max_quarantined_bytes": self.max_quarantined_bytes,
                "h2d_quarantined_buffers": len(self._h2d_quarantined),
                "h2d_quarantined_bytes": sum({item['destination'].untyped_storage().data_ptr(): item['destination'].untyped_storage().nbytes() for item in self._h2d_quarantined}.values()),
                "segment_manifests": len(self._segment_manifests),
                "manifest_ack_unsafe": len(self._segment_ack_unsafe),
                **self._segment_counters,
                **(self._terminal_proofs.health() if self._terminal_proofs is not None and not self._terminal_proofs.closed else self._terminal_proof_health),
                **(self._node_pool.health() if self._node_pool is not None else {"node_pool_enabled": False}),
            }
            if self._receive_pool is not None:
                pool_health = self._receive_pool.health()
                result.update(pool_health)
                # Keep the backend-wide quarantine metrics introduced on main
                # meaningful for pooled receive modes as well.
                result["quarantined_buffers"] = pool_health["receive_pool_quarantined"]
                result["quarantined_bytes"] = pool_health[
                    "receive_pool_quarantined_bytes"
                ]
            return result


__all__ = ["MooncakeFeatureStore"]
