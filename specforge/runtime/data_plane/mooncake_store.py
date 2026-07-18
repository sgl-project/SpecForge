# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake-backed FeatureStore: the M6 *fast path* for disaggregated EAGLE3.

``SharedDirFeatureStore`` (``disaggregated.py``) locked down the disaggregation
*contract* over a shared POSIX directory. ``MooncakeFeatureStore`` swaps that
transport for the Mooncake distributed object store (RDMA zero-copy across
nodes) **behind the exact same FeatureStore API** — the control/training planes
see nothing new. The producer (rollout/ingest) ``put()``s feature tensors into
the Mooncake store on one node; the consumer (trainer) ``get()``s them on
another, peer-to-peer, with no shared filesystem. That is what makes this a
genuine network object store rather than a shared mount.

Lifetime roles are explicit. The producer-side owner keeps the generation index
and calls ``reclaim(ref)`` after the fan-out controller advances its global ACK
prefix. Independent readers use ``lifetime_owner=False``: their releases drop
only local lease bookkeeping and never remove the shared object. Offline
re-iterable owners continue to use ``retain_on_release``. A durable lifecycle
index records planned, resident, tombstoned, and cleaned generations so owner
restart can finish cleanup without trusting process-local bookkeeping.

Contract carried from the reference backend:

* **B5 — no use-after-free.** ``get()`` after ``release``/``abort`` raises
  ``KeyError``; a generation guard rejects a stale ref after a re-``put``;
  clone-on-fetch is the default.
* **B9 — auth in disaggregated mode** (shared-secret :class:`AuthPolicy`).

Lifetime: Mooncake's default eviction is approximate-LRU for a KV *cache*, which
would silently drop a committed-but-unacked feature when the trainer lags hours
(turning ``get()`` into a ``KeyError`` and violating the controller's
no-data-loss guarantee). We therefore **hard-pin** every object on ``put`` and
free it only by explicit ``remove()`` on consume/abort — SpecForge is the sole
lifetime authority, not Mooncake's LRU. Because ``remove()`` is a real (fallible)
RPC, ``release()`` parks a failed free in ``_release_pending`` and ``gc()``
retries up to ``max_release_attempts`` during steady state. Lifecycle shutdown
calls :meth:`drain_pending_removals`, a separate bounded retry that raises if
physical removal never succeeds; failed hard-pinned objects are never silently
dropped from bookkeeping.
In fan-out mode, trainer ``release(handle)`` is local-only and the owner calls
``reclaim(ref)`` after every subscriber acknowledges it. Required fan-out
reclaims remain tracked and raise when retries are exhausted.

Concurrency: ``release``/``abort``/``gc`` hold ``self._lock`` across the
``remove()`` RPC. The lock is what makes consume-once free race-free against a
concurrent ``get()`` (it prevents a re-lease between "decide to free" and the
remote delete), exactly as ``SharedDirFeatureStore`` holds its lock across
``os.remove``. Fan-out performs this RPC only once on the owner, after all readers
have released and acknowledged the ordered prefix.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import urlparse

import torch

from specforge.runtime.contracts import SCHEMA_VERSION, FeatureHandle, SampleRef
from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.feature_store import FeatureStore, spec_from_tensor
from specforge.runtime.data_plane.mooncake_lifecycle import (
    LifecycleRecord,
    SQLiteMooncakeLifecycleIndex,
)

logger = logging.getLogger(__name__)

# Defaults for MooncakeDistributedStore.setup(); override via ``setup_kwargs``.
_MOONCAKE_SETUP_DEFAULTS = {
    "global_segment_size": 1 << 30,  # 1 GiB per-node segment
    "local_buffer_size": 1 << 30,
    "protocol": "tcp",  # bring up on TCP; flip to "rdma" once NICs are verified
    "rdma_devices": "",
}

_MOONCAKE_MISSING_OBJECT = -704


class _PinConfig:
    """Fallback for Mooncake's ``ReplicateConfig`` when the package is absent
    (local unit tests with an injected store). Mirrors the fields the real
    config exposes so the injected backend can assert on them."""

    def __init__(
        self,
        replica_num: int = 1,
        with_hard_pin: bool = True,
        with_soft_pin: bool = False,
    ) -> None:
        self.replica_num = replica_num
        self.with_hard_pin = with_hard_pin
        self.with_soft_pin = with_soft_pin


def _build_replicate_config(replica_num: int, hard_pin: bool) -> Any:
    """Real ``ReplicateConfig`` if mooncake is importable, else a shim."""
    try:
        from mooncake.store import ReplicateConfig  # type: ignore
    except Exception:
        return _PinConfig(replica_num=replica_num, with_hard_pin=hard_pin)
    cfg = ReplicateConfig()
    cfg.replica_num = replica_num
    cfg.with_hard_pin = hard_pin
    return cfg


def _connect_store(setup_kwargs: Dict[str, Any]) -> Any:
    """Construct + ``setup()`` a real ``MooncakeDistributedStore``."""
    try:
        from mooncake.store import MooncakeDistributedStore  # type: ignore
    except Exception as e:  # pragma: no cover - exercised only without mooncake
        raise RuntimeError(
            "MooncakeFeatureStore requires the 'mooncake' package "
            "(pip install mooncake-transfer-engine). Pass store=<obj> to inject "
            "a backend for testing."
        ) from e
    store = MooncakeDistributedStore()
    rc = store.setup(**setup_kwargs)
    if rc is not None and int(rc) != 0:
        raise RuntimeError(
            f"Mooncake setup failed (status {rc}); kwargs={setup_kwargs}"
        )
    return store


# FeatureSpec.dtype (a string) -> torch dtype, for allocating zero-copy receive
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


def _alloc_from_spec(spec, *, pin_memory: bool = False) -> torch.Tensor:
    """A fresh contiguous tensor matching a FeatureSpec (the zero-copy dst)."""
    dtype = _TORCH_DTYPES.get(spec.dtype)
    if dtype is None:
        raise KeyError(f"unsupported feature dtype {spec.dtype!r} for zero-copy get")
    return torch.empty(
        tuple(int(d) for d in spec.shape), dtype=dtype, pin_memory=pin_memory
    )


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class MooncakeFeatureStore(FeatureStore):
    """A disaggregated :class:`FeatureStore` backed by the Mooncake store.

    **Zero-copy transport (default).** One hard-pinned Mooncake object per
    *tensor*, keyed ``{store_id}/{sample_id}/g{gen}/{name}``. ``put()`` writes each
    tensor straight from its storage with ``put_from(ptr)``; ``get()`` reads each
    straight into a tensor allocated from the ref's ``FeatureSpec`` with
    ``get_into(ptr)``. There is no ``torch.save``/``torch.load`` pickle round-trip
    on the hot path — shape/dtype travel on the ref, the bytes are the raw tensor
    buffer. The generation lives in the key (like ``SharedDirFeatureStore``'s
    filename generation), so a re-put supersedes the old key set and a stale ref's
    keys are gone -> ``get()`` raises (B5).

    Set ``zero_copy=False`` (or inject a backend without ``put_from``/``get_into``)
    to fall back to the single-object ``torch.save`` blob path.

    ``store`` may be injected (any object exposing the Mooncake method subset:
    ``is_exist/remove`` plus either ``put_from``/``get_into`` or ``put``/``get``)
    so the contract is unit-testable without a running master.
    """

    @property
    def get_returns_fresh_tensors(self) -> bool:
        # Raw reads allocate from FeatureSpec; pickle reads clone selected tensors.
        return True

    def __init__(
        self,
        *,
        store_id: Optional[str] = None,
        store: Optional[Any] = None,
        setup_kwargs: Optional[Dict[str, Any]] = None,
        auth: Optional[AuthPolicy] = None,
        credential: Optional[str] = None,
        lifecycle_db_path: Optional[str] = None,
        max_resident_bytes: Optional[int] = None,
        max_hold_age_s: Optional[float] = None,
        retain_on_release: bool = False,
        lifetime_owner: bool = True,
        max_release_attempts: int = 3,
        replica_num: int = 1,
        hard_pin: bool = True,
        zero_copy: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.auth = auth or AuthPolicy()
        self._credential = credential
        self.auth.check(credential)  # attach-time gate (B9)
        self.store_id = store_id or uuid.uuid4().hex[:8]
        if store is None:
            kw = dict(_MOONCAKE_SETUP_DEFAULTS)
            kw.update(setup_kwargs or {})
            store = _connect_store(kw)
        self._store = store
        self._put_config = _build_replicate_config(replica_num, hard_pin)
        # Zero-copy transport: one Mooncake object per *tensor*, written straight
        # from the tensor's storage via put_from(ptr) and read straight into a
        # spec-allocated tensor via get_into(ptr) -- no torch.save/torch.load
        # pickle round-trip. Falls back to the pickle path if the backend lacks
        # the raw-buffer API (older mooncake / a fake without it).
        self._zero_copy = (
            bool(zero_copy)
            and callable(getattr(store, "put_from", None))
            and callable(getattr(store, "get_into", None))
        )
        self.max_resident_bytes = max_resident_bytes
        self.max_hold_age_s = max_hold_age_s
        # Offline re-iterable mode: release() must NOT free (multi-epoch); mirrors
        # SharedDirFeatureStore / LocalFeatureStore file:// no-op release.
        self.retain_on_release = retain_on_release
        # Exactly one store instance owns remote object lifetime. The default is
        # the legacy consume-once behavior; fan-out trainers opt out explicitly.
        self.lifetime_owner = lifetime_owner
        self.max_release_attempts = max_release_attempts
        self._clock = clock
        # in-process liveness index (single-host; see module docstring)
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._sample_bytes: Dict[str, int] = {}
        # feature names per resident sample -> the per-tensor keys to remove on
        # free (zero-copy mode). Owner instances cache these on put/adopt/get;
        # non-owner readers intentionally do not retain a per-sample index.
        self._sample_names: Dict[str, List[str]] = {}
        # Server capture registers deterministic keys before issuing HTTP. If
        # the response is lost, no SampleRef exists to adopt/abort them. Keep a
        # shared (multi-adapter) provisional index so terminal producer cleanup
        # can reclaim those hard-pinned objects; a successful adopt clears it.
        self._external_provisional: Dict[Tuple[str, int], List[str]] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        # Samples whose remote remove() failed. gc() performs bounded
        # steady-state retries; lifecycle drain either removes them or raises.
        self._release_pending: Dict[str, int] = {}
        self._external_release_pending: Dict[Tuple[str, int], int] = {}
        self._external_release_records: Dict[Tuple[str, int], LifecycleRecord] = {}
        self._force_release_pending: set[str] = set()
        self._external_force_release_pending: set[Tuple[str, int]] = set()
        # Fan-out reclaim is correctness-critical: unlike legacy best-effort
        # cleanup, these entries must remain visible if retries are exhausted.
        self._required_reclaims: set = set()
        # (sample_id, generation) logically freed in THIS process. Stores without
        # a lifecycle index need this guard while Mooncake deletion is lease-
        # deferred. Production owners use the durable lifecycle state instead, so
        # this set remains empty rather than growing once per consumed sample.
        self._freed: set[Tuple[str, int]] = set()
        self._lock = threading.RLock()
        self._counter = 0
        self._gen_counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}
        self._lifecycle = (
            SQLiteMooncakeLifecycleIndex(lifecycle_db_path, store_id=self.store_id)
            if lifecycle_db_path is not None
            else None
        )
        if self._lifecycle is not None and self.lifetime_owner:
            with self._lock:
                self._sync_lifecycle_pending_locked()

    def _sync_lifecycle_pending_locked(self) -> None:
        """Import server/local planned writes that appeared after owner startup."""
        if self._lifecycle is None or not self.lifetime_owner:
            return
        records = self._lifecycle.pending()
        pending_identities = {
            (record.sample_id, record.generation) for record in records
        }
        for sample_id, generation in list(self._generation.items()):
            if (sample_id, generation) in pending_identities:
                continue
            self._generation.pop(sample_id, None)
            self._sample_names.pop(sample_id, None)
            self._sample_bytes.pop(sample_id, None)
            self._put_time.pop(sample_id, None)
            self._release_pending.pop(sample_id, None)
            self._force_release_pending.discard(sample_id)
            self._required_reclaims.discard(sample_id)
        latest: Dict[str, Any] = {}
        for record in records:
            prior = latest.get(record.sample_id)
            if prior is None or record.generation > prior.generation:
                latest[record.sample_id] = record
            self._gen_counter = max(self._gen_counter, record.generation)
        for sample_id, record in latest.items():
            current = self._generation.get(sample_id)
            if current is not None and current > record.generation:
                continue
            self._generation[sample_id] = record.generation
            self._sample_names[sample_id] = list(record.feature_names)
            self._sample_bytes[sample_id] = record.estimated_bytes
            self._put_time.setdefault(sample_id, self._clock())
            if record.state == "tombstoned":
                self._release_pending.setdefault(sample_id, 0)
                self._required_reclaims.add(sample_id)
        for record in records:
            if latest[record.sample_id].generation == record.generation:
                continue
            identity = (record.sample_id, record.generation)
            if record.state != "tombstoned":
                self._lifecycle.tombstone(
                    record.sample_id,
                    record.generation,
                    "superseded-generation",
                )
            self._external_release_pending.setdefault(identity, 0)

    def _external_records_locked(self) -> Dict[Tuple[str, int], LifecycleRecord]:
        records = dict(self._external_release_records)
        if self._lifecycle is not None:
            records.update(
                {
                    (record.sample_id, record.generation): record
                    for record in self._lifecycle.pending()
                }
            )
        return records

    def _require_lifetime_owner(self, operation: str) -> None:
        if not self.lifetime_owner:
            raise PermissionError(
                f"MooncakeFeatureStore {self.store_id} is a non-owner reader; "
                f"{operation} is reserved for the lifetime owner"
            )

    def _validate_ref_namespace(self, sample_ref: SampleRef) -> None:
        parsed = urlparse(sample_ref.feature_store_uri)
        if (
            parsed.scheme != "mooncake"
            or parsed.netloc != self.store_id
            or parsed.path != f"/{sample_ref.sample_id}"
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                f"ref {sample_ref.sample_id} points to "
                f"{sample_ref.feature_store_uri!r}, not Mooncake store "
                f"{self.store_id!r}"
            )

    # -- keys --------------------------------------------------------------
    def _key(self, sample_id: str) -> str:
        return f"{self.store_id}/{sample_id}"

    def _tkey(self, sample_id: str, gen: int, name: str) -> str:
        # generation lives in the key (like SharedDirFeatureStore's filename gen):
        # a re-put writes a new-gen key set and removes the old, so a stale ref's
        # keys are gone -> get() raises (B5), no payload-carried gen needed.
        return f"{self.store_id}/{sample_id}/g{gen}/{name}"

    # -- store wrappers (status-code aware) --------------------------------
    def _store_exists(self, key: str) -> bool:
        return int(self._store.is_exist(key)) == 1

    def _store_put(self, key: str, value: bytes) -> None:
        rc = self._store.put(key, value, self._put_config)
        if rc is not None and int(rc) != 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")

    def _store_put_tensor(self, key: str, t: torch.Tensor) -> None:
        """Zero-copy publish: DMA straight from the tensor's storage, hard-pinned.

        ``t`` must be contiguous + CPU (caller stages it). No torch.save: the
        bytes are the raw tensor buffer; shape/dtype travel on the ref's
        FeatureSpec, so get() needs no header. The source is registered with the
        transfer engine for the duration of the put -- RDMA transfers it by DMA
        and rejects an unregistered address (AddressNotRegistered); TCP ignores
        the registration.
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
        try:
            rc = self._store.get_into(key, out.data_ptr(), nb)
        finally:
            try:
                self._store.unregister_buffer(out.data_ptr())
            except Exception:  # pragma: no cover
                pass
        if rc is None or int(rc) < 0:
            raise KeyError(f"mooncake get_into failed (status {rc}) for {key}")
        # get_into returns the number of bytes read; a full read returns exactly
        # nb. A short read (0 <= rc < nb) would leave the tail of this freshly
        # torch.empty'd buffer as uninitialized garbage -- and unlike the pickle
        # path (torch.load reconstructs whole tensors) the raw-buffer path cannot
        # otherwise detect under-fill. Reject it rather than hand the trainer
        # silently-corrupt data (B5: never serve wrong bytes).
        if int(rc) != nb:
            raise KeyError(
                f"mooncake get_into short read for {key}: got {rc} of {nb} bytes"
            )

    def _store_remove(self, key: str, *, force: bool = False) -> bool:
        """Best-effort physical free. Returns True on confirmed removal."""
        try:
            try:
                rc = self._store.remove(key, force)
            except TypeError:
                # Older Mooncake bindings and injected test backends expose only
                # remove(key). They do not support force removal, but ordinary
                # cleanup must remain compatible with that API.
                rc = self._store.remove(key)
        except Exception:  # pragma: no cover - transient RPC failure
            return False
        return rc is None or int(rc) in (0, _MOONCAKE_MISSING_OBJECT)

    # -- write -------------------------------------------------------------
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        self._require_lifetime_owner("put")
        self.auth.check(self._credential)
        if not tensors:
            raise ValueError("put requires at least one tensor")
        staged = {k: v.detach().cpu().contiguous() for k, v in tensors.items()}
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        nbytes = sum(_nbytes(t) for t in staged.values())
        with self._lock:
            if sample_id in self._release_pending:
                raise RuntimeError(
                    f"cannot put {sample_id}: its prior generation is still "
                    "pending remote removal; run gc() first"
                )
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
            prior_names = self._sample_names.get(sample_id, [])
        if self._lifecycle is not None:
            self._lifecycle.record_planned(sample_id, gen, staged, nbytes)
        if self._zero_copy:
            # One hard-pinned object per tensor, DMA'd straight from its storage.
            # staged keeps the source tensors alive across the synchronous puts.
            attempted: List[str] = []
            try:
                for name, t in staged.items():
                    attempted.append(name)
                    self._store_put_tensor(self._tkey(sample_id, gen, name), t)
            except BaseException as error:
                leaked = [
                    name
                    for name in attempted
                    if not self._store_remove(
                        self._tkey(sample_id, gen, name), force=False
                    )
                ]
                identity = (sample_id, gen)
                if self._lifecycle is not None:
                    self._lifecycle.tombstone(sample_id, gen, "partial-put-failure")
                    if not leaked:
                        self._lifecycle.mark_cleaned(sample_id, gen)
                if leaked:
                    record = LifecycleRecord(
                        sample_id=sample_id,
                        generation=gen,
                        feature_names=tuple(leaked),
                        estimated_bytes=sum(_nbytes(staged[name]) for name in leaked),
                        state="tombstoned",
                    )
                    with self._lock:
                        self._external_release_records[identity] = record
                        self._external_release_pending.setdefault(identity, 0)
                    error.add_note(
                        "partial Mooncake put cleanup is pending for "
                        f"{sample_id!r} generation {gen}: {leaked}"
                    )
                raise
            # Overwrite-safe: drop the prior generation's tensor keys so a stale
            # ref's keys are gone (its get() then raises -> no use-after-free).
            if prior_gen is not None and prior_gen != gen:
                leaked = [
                    name
                    for name in prior_names
                    if not self._store_remove(
                        self._tkey(sample_id, prior_gen, name), force=False
                    )
                ]
                if leaked:
                    logger.warning(
                        "MooncakeFeatureStore re-put of %s gen %s: removing prior "
                        "generation %s tensors %s failed; hard-pinned objects may be "
                        "orphaned (and the stale ref stays readable until reclaimed)",
                        sample_id,
                        prior_gen,
                        prior_gen,
                        leaked,
                    )
                elif self._lifecycle is not None:
                    self._lifecycle.mark_cleaned(sample_id, prior_gen)
        else:
            buf = io.BytesIO()
            torch.save({"generation": gen, "tensors": staged}, buf)
            key = self._key(sample_id)
            # Overwrite-safe publish: a re-put bumps the generation. remove() first
            # so the hard-pinned prior blob is released rather than orphaned; if
            # that remove fails the old (pinned) blob may leak, so surface it.
            if not self._store_remove(key, force=False):
                logger.warning(
                    "MooncakeFeatureStore re-put of %s: removing the stale blob "
                    "failed; a hard-pinned object may be orphaned",
                    key,
                )
            elif prior_gen is not None and self._lifecycle is not None:
                self._lifecycle.mark_cleaned(sample_id, prior_gen)
            self._store_put(key, buf.getvalue())
        ref = SampleRef(
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
        if self._lifecycle is not None:
            self._lifecycle.record_resident(sample_id, gen, staged, nbytes)
        with self._lock:
            self._generation[sample_id] = gen
            self._put_time[sample_id] = self._clock()
            self._sample_bytes[sample_id] = nbytes
            self._sample_names[sample_id] = list(staged)
        return ref

    def adopt(self, sample_ref: SampleRef) -> None:
        """Register an externally-produced sample for lifecycle management.

        The server-capture transport writes tensors into this store's key
        namespace from ANOTHER process (the SGLang server's sink), so this
        instance has no put-side bookkeeping for them. ``adopt()`` records the
        ref's generation / feature names / size so ``release``/``abort``/``gc``
        can free the server-written objects exactly like locally-put ones.
        """
        self._require_lifetime_owner("adopt")
        self._validate_ref_namespace(sample_ref)
        gen = sample_ref.metadata.get("generation")
        if gen is None:
            raise ValueError(
                f"cannot adopt {sample_ref.sample_id}: ref carries no generation"
            )
        gen = int(gen)
        sample_bytes = int(sample_ref.estimated_bytes or 0)
        feature_names = list(sample_ref.feature_keys.keys())
        if self._lifecycle is not None and sample_bytes == 0:
            inventory = self._lifecycle.record(sample_ref.sample_id, gen)
            if inventory is not None:
                sample_bytes = inventory.estimated_bytes
        if sample_bytes == 0 and feature_names:
            sizes: List[int] = []
            for name in feature_names:
                try:
                    size = int(
                        self._store.get_size(
                            self._tkey(sample_ref.sample_id, gen, name)
                        )
                    )
                except (AttributeError, TypeError, ValueError):
                    sizes = []
                    break
                if size < 0:
                    sizes = []
                    break
                sizes.append(size)
            if sizes:
                sample_bytes = sum(sizes)
        if sample_bytes == 0 and self.max_resident_bytes is not None:
            raise ValueError(
                f"cannot adopt {sample_ref.sample_id}: payload size is unknown "
                "while max_resident_bytes is enforced"
            )
        with self._lock:
            if sample_ref.sample_id in self._release_pending:
                raise RuntimeError(
                    f"cannot adopt {sample_ref.sample_id}: its prior generation "
                    "is still pending remote removal; run gc() first"
                )
        if self._lifecycle is not None:
            self._lifecycle.record_resident(
                sample_ref.sample_id, gen, feature_names, sample_bytes
            )
        with self._lock:
            previous_bytes = self._sample_bytes.get(sample_ref.sample_id, 0)
            projected = sum(self._sample_bytes.values()) - previous_bytes + sample_bytes
            self._generation[sample_ref.sample_id] = gen
            self._sample_names[sample_ref.sample_id] = feature_names
            self._sample_bytes[sample_ref.sample_id] = sample_bytes
            self._put_time[sample_ref.sample_id] = self._clock()
            self._external_provisional.pop((sample_ref.sample_id, gen), None)
            over_budget = (
                self.max_resident_bytes is not None
                and projected > self.max_resident_bytes
            )
            if over_budget:
                self._abort_locked(
                    sample_ref.sample_id,
                    reason="adopt-over-budget",
                    required_reclaim=True,
                    force=True,
                )
        if over_budget:
            raise MemoryError(
                f"MooncakeFeatureStore {self.store_id} adopt would exceed "
                f"{self.max_resident_bytes} bytes; server-written sample was "
                "scheduled for required cleanup"
            )

    def track_external_attempt(
        self,
        sample_id: str,
        *,
        generation: int,
        feature_names: List[str],
    ) -> None:
        """Track server-owned keys before an HTTP response makes a ref adoptable."""
        self._require_lifetime_owner("track external capture")
        names = list(dict.fromkeys(str(name) for name in feature_names))
        if not names:
            raise ValueError("external capture attempt must name at least one feature")
        with self._lock:
            self._external_provisional[(str(sample_id), int(generation))] = names

    def discard_external_attempts(
        self, *, reason: str = "unadopted-external-capture"
    ) -> int:
        """Abort server writes that never produced an adoptable response.

        The index belongs to the shared store rather than an adapter, so a retry
        that succeeds through another capture server clears the provisional
        entry in :meth:`adopt` and cannot be deleted by the failed adapter's
        shutdown path. Physical-remove failures remain visible to
        :meth:`drain_pending_removals`.
        """
        self._require_lifetime_owner("discard external captures")
        with self._lock:
            attempts = list(self._external_provisional.items())

        errors: Dict[str, str] = {}
        discarded = 0
        for (sample_id, generation), names in attempts:
            with self._lock:
                attempt_key = (sample_id, generation)
                # A response can be adopted after the snapshot but before this
                # attempt is visited. adopt() removes the provisional entry
                # under the same lock, so never delete that now-live sample.
                if attempt_key not in self._external_provisional:
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
        if errors:
            raise RuntimeError(
                "could not discard all provisional external captures: " f"{errors}"
            )
        return discarded

    # -- read --------------------------------------------------------------
    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
        pin_memory: bool = False,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        self.auth.check(self._credential)
        self._validate_ref_namespace(sample_ref)
        sid = sample_ref.sample_id
        ref_gen = sample_ref.metadata.get("generation")
        if self._lifecycle is not None:
            if ref_gen is None:
                raise KeyError(f"sample {sid} ref carries no lifecycle generation")
            state = self._lifecycle.state(sid, int(ref_gen))
            if state != "resident":
                raise KeyError(
                    f"sample {sid} generation {ref_gen} lifecycle state is "
                    f"{state!r}; refusing read"
                )
        with self._lock:
            if ref_gen is not None and (sid, int(ref_gen)) in self._freed:
                # logically freed here; the remote bytes may still linger under
                # Mooncake's read-lease, but this ref must not resolve (B5)
                raise KeyError(
                    f"sample {sid} generation {ref_gen} was released/aborted; "
                    f"refusing use-after-free"
                )
        wanted = names or list(sample_ref.feature_keys.keys())
        if self._zero_copy:
            out, gen = self._get_zero_copy(sample_ref, wanted, pin_memory=pin_memory)
        else:
            out, gen = self._get_pickle(sample_ref, wanted)
            if pin_memory:
                out = {
                    name: tensor if tensor.is_pinned() else tensor.pin_memory()
                    for name, tensor in out.items()
                }
        if str(device) != "cpu":
            out = {k: v.to(device) for k, v in out.items()}
        if self._lifecycle is not None:
            state = self._lifecycle.state(sid, int(ref_gen))
            if state != "resident":
                raise KeyError(
                    f"sample {sid} generation {ref_gen} was tombstoned while "
                    "materializing; refusing read"
                )
        with self._lock:
            self._counter += 1
            if self.lifetime_owner:
                # A legacy consume-once instance that only get()s still needs
                # generation + names so release() can remove the remote keys.
                # Non-owner readers intentionally retain no per-sample index.
                self._generation.setdefault(sid, gen)
                self._sample_names.setdefault(sid, list(sample_ref.feature_keys.keys()))
            handle = FeatureHandle(
                sample_id=sid,
                generation=gen,
                lease_token=f"{sid}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
        return out, handle

    def _get_zero_copy(
        self, ref: SampleRef, wanted: List[str], *, pin_memory: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        """Read each feature straight into a spec-allocated tensor (no pickle)."""
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
            if not self._store_exists(key):
                # freed (release/abort), superseded by a re-put, or never written
                raise KeyError(
                    f"sample {sid} gen {gen} feature {n!r} not available "
                    f"(freed, stale, or never written)"
                )
            # Fresh storage makes the loader's clone-on-fetch redundant (B5).
            out[n] = _alloc_from_spec(spec, pin_memory=pin_memory)
            self._store_get_tensor(key, out[n])
        return out, gen

    def _get_pickle(
        self, ref: SampleRef, wanted: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
        sid = ref.sample_id
        key = self._key(sid)
        if not self._store_exists(key):
            raise KeyError(f"sample {sid} not available in store {self.store_id}")
        value = self._store.get(key)
        if not value:
            raise KeyError(f"sample {sid} not available in store {self.store_id}")
        # weights_only=True: these bytes arrive over the wire from a producer node,
        # so refuse arbitrary-pickle deserialization (the payload is only an int +
        # a dict of tensors, all of which the safe unpickler supports).
        payload = torch.load(io.BytesIO(value), weights_only=True)
        on_disk_gen = payload.get("generation")
        on_disk_gen = int(on_disk_gen) if on_disk_gen is not None else None
        ref_gen = ref.metadata.get("generation", on_disk_gen)
        if on_disk_gen is not None and ref_gen != on_disk_gen:
            raise KeyError(
                f"sample {sid} generation {ref_gen} is stale "
                f"(current {on_disk_gen}); refusing use-after-free"
            )
        raw = payload["tensors"]
        out: Dict[str, torch.Tensor] = {}
        for n in wanted:
            raw_key = ref.feature_keys.get(n, n)
            raw_key = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            if raw_key not in raw:
                raise KeyError(
                    f"sample {sid} missing key {raw_key!r} for feature {n!r}"
                )
            out[n] = raw[raw_key].clone()  # clone-on-fetch (B5)
        return out, (on_disk_gen or 0)

    # -- lifetime ----------------------------------------------------------
    def _try_physical_free(
        self,
        sample_id: str,
        *,
        force: bool = False,
        confirm_absent_on_failure: bool = True,
    ) -> bool:
        """Remove the remote object(s). False on a (retryable) RPC failure.

        Zero-copy: one object per tensor, so remove every per-tensor key of the
        sample's current generation. Pickle: a single object.

        ``force`` is reserved for callers with an external proof that no reader
        may still use the generation, such as the fan-out global durable ACK.
        Mooncake reports an already-missing key as -704, so no ``is_exist``
        probe is needed; that probe would grant a fresh read lease.
        A failed remove may be classified as already absent, but retry loops
        disable that probe because ``is_exist`` grants a fresh read lease.
        """
        self._require_lifetime_owner("remote removal")
        if not self._zero_copy:
            return self._store_remove(self._key(sample_id), force=force)
        gen = self._generation.get(sample_id)
        if gen is None:
            return True  # nothing tracked to remove (already freed)
        ok = True
        for name in self._sample_names.get(sample_id, []):
            key = self._tkey(sample_id, gen, name)
            if self._store_remove(key, force=force):
                continue
            if confirm_absent_on_failure and not self._store_exists(key):
                continue
            ok = False
        return ok

    def _try_physical_free_record(self, record, *, force: bool = False) -> bool:
        """Remove one exact lifecycle generation not represented by current maps."""
        if not self._zero_copy:
            return self._store_remove(self._key(record.sample_id), force=force)
        ok = True
        for name in record.feature_names:
            key = self._tkey(record.sample_id, record.generation, name)
            if self._store_remove(key, force=force):
                continue
            ok = False
        return ok

    def _sample_exists(self, sample_id: str) -> bool:
        """True if any object backing the sample's current generation is present."""
        if not self._zero_copy:
            return self._store_exists(self._key(sample_id))
        gen = self._generation.get(sample_id)
        if gen is None:
            return False
        return any(
            self._store_exists(self._tkey(sample_id, gen, n))
            for n in self._sample_names.get(sample_id, [])
        )

    def _free_bookkeeping_locked(self, sample_id: str) -> int:
        """Drop in-process tracking for a sample. Returns bytes accounted freed."""
        generation = self._generation.get(sample_id)
        if self._lifecycle is not None and generation is not None:
            self._lifecycle.mark_cleaned(sample_id, generation)
        nbytes = self._sample_bytes.pop(sample_id, 0)
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        self._sample_names.pop(sample_id, None)
        self._release_pending.pop(sample_id, None)
        self._force_release_pending.discard(sample_id)
        self._required_reclaims.discard(sample_id)
        if generation is not None:
            self._external_provisional.pop((sample_id, generation), None)
        return nbytes

    def _tombstone_locked(self, sample_id: str, reason: str) -> Optional[int]:
        generation = self._generation.get(sample_id)
        if generation is None:
            return None
        if self._lifecycle is not None:
            self._lifecycle.tombstone(sample_id, generation, reason)
        else:
            self._freed.add((sample_id, generation))
        return generation

    def _still_leased_locked(self, sample_id: str, generation: Optional[int]) -> bool:
        # generation-aware: a stale older-generation lease does not pin the
        # current generation (matches LocalFeatureStore's invariant).
        return any(
            h.sample_id == sample_id and h.generation == generation
            for h in self._active_leases.values()
        )

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        """End a materialization lease.

        On a non-owner reader this is strictly local: it drops the active handle
        and never removes shared Mooncake data. The fan-out coordinator must use
        :meth:`reclaim` after the global minimum acknowledgement advances.
        """
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            sid = handle.sample_id
            cur = self._generation.get(sid)
            if not self.lifetime_owner:
                if not self._still_leased_locked(sid, cur):
                    self._free_bookkeeping_locked(sid)
                return
            if self.retain_on_release:
                return  # offline re-iterable set: keep for the next epoch
            if cur is not None and handle.generation != cur:
                return  # stale lease -> no-op
            if self._still_leased_locked(sid, cur):
                return
            self._tombstone_locked(sid, reason)
            if self._try_physical_free(sid, force=False):
                self._free_bookkeeping_locked(sid)
            else:
                # remote free deferred (lease) / failed -> gc() retries
                self._release_pending.setdefault(sid, 0)

    def _abort_locked(
        self,
        sample_id: str,
        *,
        reason: str,
        required_reclaim: bool = False,
        force: bool = False,
    ) -> None:
        self._tombstone_locked(sample_id, reason)
        if self._try_physical_free(sample_id, force=force):
            self._free_bookkeeping_locked(sample_id)
        else:
            self._release_pending.setdefault(sample_id, 0)
            if force:
                self._force_release_pending.add(sample_id)
            if required_reclaim:
                self._required_reclaims.add(sample_id)

    def reclaim(
        self, sample_ref: SampleRef, *, reason: str = "globally-consumed"
    ) -> None:
        """Owner-only deletion of one exact, globally-consumed reference.

        ``release(handle)`` means one reader finished materializing tensors;
        ``reclaim(ref)`` means *all* fan-out subscribers acknowledged the ref and
        its remote objects may be deleted. Matching the ref generation before
        removal prevents a delayed acknowledgement from deleting a newer sample
        with the same ID. A lease/busy removal failure enters ``release_pending``
        and is retried by :meth:`gc`.
        """
        self._require_lifetime_owner("reclaim")
        self._validate_ref_namespace(sample_ref)
        ref_gen = sample_ref.metadata.get("generation")
        if ref_gen is None:
            raise ValueError(
                f"cannot reclaim {sample_ref.sample_id}: ref carries no generation"
            )
        ref_gen = int(ref_gen)
        with self._lock:
            current_gen = self._generation.get(sample_ref.sample_id)
            if current_gen is None:
                raise KeyError(
                    f"cannot reclaim untracked sample {sample_ref.sample_id}; "
                    "the lifetime owner must put() or adopt() the ref first"
                )
            if current_gen != ref_gen:
                raise KeyError(
                    f"refusing to reclaim stale sample {sample_ref.sample_id} "
                    f"generation {ref_gen}; current generation is {current_gen}"
                )
            self._abort_locked(
                sample_ref.sample_id,
                reason=reason,
                required_reclaim=True,
                force=True,
            )

    def abort(
        self, sample_id: str, *, reason: str = "aborted", force: bool = False
    ) -> None:
        """Owner-only terminal cleanup by sample ID.

        Error paths that already own/adopt a sample use this legacy API. Fan-out
        acknowledgement cleanup must use :meth:`reclaim` so generation matching
        is explicit.
        """
        self._require_lifetime_owner("abort")
        with self._lock:
            self._abort_locked(sample_id, reason=reason, force=force)

    def abort_all(self, *, reason: str = "owner-failure", force: bool = False) -> int:
        """Persist tombstones and attempt cleanup for every tracked sample."""
        self._require_lifetime_owner("abort_all")
        with self._lock:
            self._sync_lifecycle_pending_locked()
            records = self._lifecycle.pending() if self._lifecycle is not None else ()
            handled = {
                (sample_id, generation)
                for sample_id, generation in self._generation.items()
            }
            sample_ids = list(self._generation)
            for sample_id in sample_ids:
                self._abort_locked(
                    sample_id,
                    reason=reason,
                    required_reclaim=True,
                    force=force,
                )
            for record in records:
                identity = (record.sample_id, record.generation)
                if identity in handled:
                    continue
                if record.state != "tombstoned":
                    self._lifecycle.tombstone(
                        record.sample_id, record.generation, reason
                    )
                if self._try_physical_free_record(record, force=force):
                    self._lifecycle.mark_cleaned(record.sample_id, record.generation)
                    self._external_release_pending.pop(identity, None)
                    self._external_force_release_pending.discard(identity)
                    self._external_release_records.pop(identity, None)
                else:
                    self._external_release_pending.setdefault(identity, 0)
                    if force:
                        self._external_force_release_pending.add(identity)
            return len(records) if self._lifecycle is not None else len(sample_ids)

    def drain_pending_removals(
        self,
        *,
        max_attempts: int = 8,
        retry_interval_s: float = 0.25,
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
        if max_attempts < 1:
            raise ValueError("max_attempts must be >= 1")
        if retry_interval_s < 0:
            raise ValueError("retry_interval_s must be >= 0")
        removed = removed_bytes = 0
        last_errors: Dict[str, str] = {}
        attempts_run = 0
        for attempt in range(max_attempts):
            attempts_run = attempt + 1
            with self._lock:
                self._sync_lifecycle_pending_locked()
                pending = list(self._release_pending)
                external_pending = list(self._external_release_pending)
                if not pending and not external_pending:
                    return {
                        "removed": removed,
                        "removed_bytes": removed_bytes,
                        "release_pending": 0,
                        "attempts": attempt,
                    }
                final_attempt = attempt + 1 == max_attempts
                for sample_id in pending:
                    try:
                        physically_removed = self._try_physical_free(
                            sample_id,
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
                records = self._external_records_locked()
                for identity in external_pending:
                    record = records.get(identity)
                    if record is None:
                        self._external_release_pending.pop(identity, None)
                        self._external_force_release_pending.discard(identity)
                        self._external_release_records.pop(identity, None)
                        continue
                    label = f"{record.sample_id}:g{record.generation}"
                    try:
                        physically_removed = self._try_physical_free_record(
                            record,
                            force=identity in self._external_force_release_pending,
                        )
                    except Exception as exc:
                        last_errors[label] = f"{type(exc).__name__}: {exc}"
                        physically_removed = False
                    if physically_removed:
                        if self._lifecycle is not None:
                            lifecycle_record = self._lifecycle.record(*identity)
                            if lifecycle_record is not None:
                                self._lifecycle.mark_cleaned(*identity)
                        self._external_release_pending.pop(identity, None)
                        self._external_force_release_pending.discard(identity)
                        self._external_release_records.pop(identity, None)
                        removed += 1
                        removed_bytes += record.estimated_bytes
                        self._stats["force_freed"] += 1
                        self._stats["force_freed_bytes"] += record.estimated_bytes
                        last_errors.pop(label, None)
                    else:
                        self._external_release_pending[identity] = min(
                            self.max_release_attempts,
                            self._external_release_pending.get(identity, 0) + 1,
                        )
                remaining_count = len(self._release_pending) + len(
                    self._external_release_pending
                )
            if not remaining_count:
                return {
                    "removed": removed,
                    "removed_bytes": removed_bytes,
                    "release_pending": 0,
                    "attempts": attempts_run,
                }
            if attempt + 1 < max_attempts and retry_interval_s:
                sleep(retry_interval_s)

        with self._lock:
            remaining = list(self._release_pending)
            remaining.extend(
                f"{sample_id}:g{generation}"
                for sample_id, generation in self._external_release_pending
            )
        preview = remaining[:16]
        detail = f"; last errors={last_errors}" if last_errors else ""
        raise RuntimeError(
            f"MooncakeFeatureStore {self.store_id} could not drain "
            f"{len(remaining)} pending removal(s) after {attempts_run} attempts: "
            f"{preview}{detail}"
        )

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        if not self.lifetime_owner:
            return {
                "force_freed": 0,
                "force_freed_bytes": 0,
                "release_pending": 0,
            }
        now = self._clock() if now is None else now
        freed = freed_bytes = 0
        with self._lock:
            self._sync_lifecycle_pending_locked()
            # max-hold sweep: force-free abandoned samples (spare still-leased)
            if self.max_hold_age_s is not None:
                stale = [
                    sid
                    for sid, t in list(self._put_time.items())
                    if now - t > self.max_hold_age_s
                    and not self._still_leased_locked(sid, self._generation.get(sid))
                ]
                for sid in stale:
                    self._tombstone_locked(sid, "max-hold-age")
                    if self._try_physical_free(
                        sid, force=False, confirm_absent_on_failure=False
                    ):
                        freed_bytes += self._free_bookkeeping_locked(sid)
                        freed += 1
                    else:
                        self._release_pending.setdefault(sid, 0)
            # Reconcile release-pending without an existence probe: is_exist
            # grants a read lease, while remove(-704) already reports absence.
            for sid in list(self._release_pending):
                if self._release_pending[sid] >= self.max_release_attempts:
                    # Keep the physical key metadata and surface the pending
                    # sample. Lifecycle drain owns the final bounded retry and
                    # loud failure; silently dropping this bookkeeping would
                    # make a hard-pinned remote leak invisible.
                    continue
                attempts = self._release_pending[sid] + 1
                if self._try_physical_free(
                    sid,
                    force=sid in self._force_release_pending,
                    confirm_absent_on_failure=False,
                ):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    freed += 1
                else:
                    self._release_pending[sid] = attempts
            records = self._external_records_locked()
            for identity in list(self._external_release_pending):
                record = records.get(identity)
                if record is None:
                    self._external_release_pending.pop(identity, None)
                    self._external_force_release_pending.discard(identity)
                    self._external_release_records.pop(identity, None)
                    continue
                if (
                    self._external_release_pending[identity]
                    >= self.max_release_attempts
                ):
                    continue
                attempts = self._external_release_pending[identity] + 1
                if self._try_physical_free_record(
                    record,
                    force=identity in self._external_force_release_pending,
                ):
                    if self._lifecycle is not None:
                        self._lifecycle.mark_cleaned(*identity)
                    self._external_release_pending.pop(identity, None)
                    self._external_force_release_pending.discard(identity)
                    self._external_release_records.pop(identity, None)
                    freed += 1
                    freed_bytes += record.estimated_bytes
                else:
                    self._external_release_pending[identity] = min(
                        self.max_release_attempts, attempts
                    )
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
            report = {
                "force_freed": freed,
                "force_freed_bytes": freed_bytes,
                "release_pending": len(self._release_pending)
                + len(self._external_release_pending),
            }
        return report

    def health(self) -> Dict[str, Any]:
        with self._lock:
            self._sync_lifecycle_pending_locked()
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            lifecycle_pending = (
                self._lifecycle.pending()
                if self._lifecycle is not None and self.lifetime_owner
                else ()
            )
            # NOTE: resident_bytes is an in-process accounting sum, not a live
            # Mooncake pool-usage query (the Python API exposes only per-key
            # get_size). A cross-node pool-usage signal is a follow-up.
            return {
                "store_id": self.store_id,
                "backend": "mooncake",
                "lifetime_owner": self.lifetime_owner,
                "resident_samples": (
                    len(lifecycle_pending)
                    if self._lifecycle is not None and self.lifetime_owner
                    else len(self._generation)
                ),
                "provisional_external": len(self._external_provisional),
                "active_leases": len(self._active_leases),
                "resident_bytes": (
                    sum(record.estimated_bytes for record in lifecycle_pending)
                    if self._lifecycle is not None and self.lifetime_owner
                    else sum(self._sample_bytes.values())
                ),
                "max_resident_bytes": self.max_resident_bytes,
                "retain_on_release": self.retain_on_release,
                "auth_required": self.auth.required,
                "release_pending": len(self._release_pending)
                + len(self._external_release_pending),
                "required_reclaims_pending": len(self._required_reclaims)
                + len(self._external_release_pending),
                "local_tombstones": len(self._freed),
                "oldest_age_s": max(ages) if ages else 0.0,
                "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
                "force_freed_total": self._stats["force_freed"],
                "hard_pin": bool(getattr(self._put_config, "with_hard_pin", False)),
            }


__all__ = ["MooncakeFeatureStore"]
