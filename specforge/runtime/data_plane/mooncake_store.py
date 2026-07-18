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

Concurrency: ``release``/``abort``/``gc`` hold ``self._lock`` across the
``remove()`` RPC. The lock is what makes consume-once free race-free against a
concurrent ``get()`` (it prevents a re-lease between "decide to free" and the
remote delete), exactly as ``SharedDirFeatureStore`` holds its lock across
``os.remove``. For the offline single-consumer path this is fine; a high-fanout
online deployment that wants the ``remove()`` RPC off the critical section needs
a tombstone-then-free protocol — a follow-up tied to the shared metadata index.
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from specforge.runtime.contracts import SCHEMA_VERSION, FeatureHandle, SampleRef
from specforge.runtime.data_plane.disaggregated import AuthPolicy
from specforge.runtime.data_plane.feature_store import FeatureStore, spec_from_tensor

logger = logging.getLogger(__name__)

# Defaults for MooncakeDistributedStore.setup(); override via ``setup_kwargs``.
_MOONCAKE_SETUP_DEFAULTS = {
    "global_segment_size": 1 << 30,  # 1 GiB per-node segment
    "local_buffer_size": 1 << 30,
    "protocol": "tcp",  # bring up on TCP; flip to "rdma" once NICs are verified
    "rdma_devices": "",
}


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


class MooncakeFeatureStore(FeatureStore):
    """A disaggregated :class:`FeatureStore` backed by the Mooncake store.

    **Zero-copy transport.** One hard-pinned Mooncake object per
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
    """

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
        replica_num: int = 1,
        hard_pin: bool = True,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.auth = auth or AuthPolicy()
        self._credential = credential
        self.auth.check(credential)  # attach-time gate (B9)
        self.store_id = store_id or uuid.uuid4().hex[:8]
        if store is None:
            kw = dict(_MOONCAKE_SETUP_DEFAULTS)
            kw.update(setup_kwargs or {})
            store, replicate_config_type = _connect_store(kw)
            put_config = replicate_config_type()
        else:
            # Injected stores are a unit-test seam and do not require importing
            # the optional Mooncake package merely to construct its config type.
            put_config = _InjectedReplicateConfig()
        _require_store_api(store)
        self._store = store
        put_config.replica_num = replica_num
        put_config.with_hard_pin = hard_pin
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
        # can reclaim those hard-pinned objects; a successful adopt clears it.
        self._external_provisional: Dict[Tuple[str, int], List[str]] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        # Samples whose remote remove() failed. gc() performs bounded
        # steady-state retries; lifecycle drain either removes them or raises.
        self._release_pending: Dict[str, int] = {}
        # (sample_id, generation) logically freed in THIS process. Mooncake's
        # remove() is lease-deferred (an object keeps a short read-lease), so the
        # bytes can linger after release/abort; this makes the B5 "no
        # use-after-free" guarantee immediate — get() of a freed ref raises even
        # while physical reclamation is still pending. Grows with consume-once
        # frees within a run (empty in retain_on_release/offline mode); a durable
        # shared index would own this in the online multi-node follow-up.
        self._freed: set = set()
        self._lock = threading.RLock()
        self._counter = 0
        self._gen_counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}

    # -- keys --------------------------------------------------------------
    def _tkey(self, sample_id: str, gen: int, name: str) -> str:
        # generation lives in the key (like SharedDirFeatureStore's filename gen):
        # a re-put writes a new-gen key set and removes the old, so a stale ref's
        # keys are gone -> get() raises (B5), no payload-carried gen needed.
        return f"{self.store_id}/{sample_id}/g{gen}/{name}"

    # -- store wrappers (status-code aware) --------------------------------
    def _store_exists(self, key: str) -> bool:
        return int(self._store.is_exist(key)) == 1

    def _store_put_tensor(self, key: str, t: torch.Tensor) -> None:
        """Zero-copy publish: DMA straight from the tensor's storage, hard-pinned.

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
        # allocated buffer as uninitialized garbage. Reject it rather than hand
        # the trainer silently-corrupt data (B5: never serve wrong bytes).
        if int(rc) != nb:
            raise KeyError(
                f"mooncake get_into short read for {key}: got {rc} of {nb} bytes"
            )

    def _store_remove(self, key: str) -> bool:
        """Best-effort physical free. Returns True on confirmed removal."""
        try:
            rc = self._store.remove(key)
        except Exception:  # pragma: no cover - transient RPC failure
            return False
        return rc is None or int(rc) == 0

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
        # One hard-pinned object per tensor, DMA'd straight from its storage.
        # staged keeps the source tensors alive across the synchronous puts.
        for name, t in staged.items():
            self._store_put_tensor(self._tkey(sample_id, gen, name), t)
        # Overwrite-safe: drop the prior generation's tensor keys so a stale
        # ref's keys are gone (its get() then raises -> no use-after-free).
        if prior_gen is not None and prior_gen != gen:
            leaked = [
                name
                for name in prior_names
                if not self._store_remove(self._tkey(sample_id, prior_gen, name))
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
    ) -> None:
        """Track server-owned keys before an HTTP response makes a ref adoptable."""
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
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        self.auth.check(self._credential)
        sid = sample_ref.sample_id
        ref_gen = sample_ref.metadata.get("generation")
        with self._lock:
            if ref_gen is not None and (sid, int(ref_gen)) in self._freed:
                # logically freed here; the remote bytes may still linger under
                # Mooncake's read-lease, but this ref must not resolve (B5)
                raise KeyError(
                    f"sample {sid} generation {ref_gen} was released/aborted; "
                    f"refusing use-after-free"
                )
        wanted = names or list(sample_ref.feature_keys.keys())
        out, gen = self._get_tensors(sample_ref, wanted)
        if str(device) != "cpu":
            out = {k: v.to(device) for k, v in out.items()}
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
        return out, handle

    def _get_tensors(
        self, ref: SampleRef, wanted: List[str]
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
            if not self._store_exists(key):
                # freed (release/abort), superseded by a re-put, or never written
                raise KeyError(
                    f"sample {sid} gen {gen} feature {n!r} not available "
                    f"(freed, stale, or never written)"
                )
            out[n] = _alloc_from_spec(spec)  # fresh -> clone-on-fetch for free (B5)
            self._store_get_tensor(key, out[n])
        return out, gen

    # -- lifetime ----------------------------------------------------------
    def _try_physical_free(
        self,
        sample_id: str,
        *,
        confirm_absent_on_failure: bool = True,
    ) -> bool:
        """Remove all tensor objects. False on a retryable RPC failure.

        Order matters against Mooncake's lease semantics: an is_exist probe
        GRANTS a read lease, and a remove during any live lease fails (-706).
        So each key is removed FIRST. The optional exist probe runs only after a
        failed remove, purely to classify "already gone" as freed. Retry loops
        disable that probe because probing a still-live key would renew its
        lease and make every following remove fail again.
        """
        gen = self._generation.get(sample_id)
        if gen is None:
            return True  # nothing tracked to remove (already freed)
        ok = True
        for name in self._sample_names.get(sample_id, []):
            key = self._tkey(sample_id, gen, name)
            if self._store_remove(key):
                continue
            if confirm_absent_on_failure and not self._store_exists(key):
                continue  # already gone (freed remotely) counts as freed
            ok = False
        return ok

    def _free_bookkeeping_locked(self, sample_id: str) -> int:
        """Drop in-process tracking for a sample. Returns bytes accounted freed."""
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
            self._active_leases.pop(handle.lease_token, None)
            if self.retain_on_release:
                return  # offline re-iterable set: keep for the next epoch
            sid = handle.sample_id
            cur = self._generation.get(sid)
            if cur is not None and handle.generation != cur:
                return  # stale lease -> no-op
            if self._still_leased_locked(sid, cur):
                return
            self._freed.add((sid, handle.generation))  # immediate logical free
            if self._try_physical_free(sid):
                self._free_bookkeeping_locked(sid)
            else:
                # remote free deferred (lease) / failed -> gc() retries
                self._release_pending.setdefault(sid, 0)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        with self._lock:
            gen = self._generation.get(sample_id)
            if gen is not None:
                self._freed.add((sample_id, gen))  # immediate logical free
            if self._try_physical_free(sample_id):
                self._free_bookkeeping_locked(sample_id)
            else:
                self._release_pending.setdefault(sample_id, 0)

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
                pending = list(self._release_pending)
                if not pending:
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
                remaining = list(self._release_pending)
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
            remaining = list(self._release_pending)
        preview = remaining[:16]
        detail = f"; last errors={last_errors}" if last_errors else ""
        raise RuntimeError(
            f"MooncakeFeatureStore {self.store_id} could not drain "
            f"{len(remaining)} pending removal(s) after {attempts_run} attempts: "
            f"{preview}{detail}"
        )

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        now = self._clock() if now is None else now
        freed = freed_bytes = 0
        with self._lock:
            # max-hold sweep: force-free abandoned samples (spare still-leased)
            if self.max_hold_age_s is not None:
                stale = [
                    sid
                    for sid, t in list(self._put_time.items())
                    if now - t > self.max_hold_age_s
                    and not self._still_leased_locked(sid, self._generation.get(sid))
                ]
                for sid in stale:
                    if self._try_physical_free(sid, confirm_absent_on_failure=False):
                        freed_bytes += self._free_bookkeeping_locked(sid)
                        freed += 1
                    else:
                        self._release_pending.setdefault(sid, 0)
            # Reconcile release-pending without an exists probe: is_exist grants
            # a read lease that would make the next remove fail (-706).
            for sid in list(self._release_pending):
                if self._release_pending[sid] >= self.max_release_attempts:
                    # Keep the physical key metadata and surface the pending
                    # sample. Lifecycle drain owns the final bounded retry and
                    # loud failure; silently dropping this bookkeeping would
                    # make a hard-pinned remote leak invisible.
                    continue
                attempts = self._release_pending[sid] + 1
                if self._try_physical_free(sid, confirm_absent_on_failure=False):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    freed += 1
                else:
                    self._release_pending[sid] = attempts
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
        return {
            "force_freed": freed,
            "force_freed_bytes": freed_bytes,
            "release_pending": len(self._release_pending),
        }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            # NOTE: resident_bytes is an in-process accounting sum, not a live
            # Mooncake pool-usage query (the Python API exposes only per-key
            # get_size). A cross-node pool-usage signal is a follow-up.
            return {
                "store_id": self.store_id,
                "backend": "mooncake",
                "resident_samples": len(self._generation),
                "provisional_external": len(self._external_provisional),
                "active_leases": len(self._active_leases),
                "resident_bytes": sum(self._sample_bytes.values()),
                "max_resident_bytes": self.max_resident_bytes,
                "auth_required": self.auth.required,
                "release_pending": len(self._release_pending),
                "oldest_age_s": max(ages) if ages else 0.0,
                "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
                "force_freed_total": self._stats["force_freed"],
                "hard_pin": bool(getattr(self._put_config, "with_hard_pin", False)),
            }


__all__ = ["MooncakeFeatureStore"]
