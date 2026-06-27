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

Scope (PR-A, the offline path M6 ships): this backend is correct for a single
consumer process with ``retain_on_release`` for re-iterable epochs and
whole-store cleanup at run end. The per-sample generation/lease *index* is
in-process, mirroring ``SharedDirFeatureStore``'s documented single-host
limitation. A true online multi-node deployment lifts that index into a shared
metadata service — a separate follow-up, not this PR.

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
retries up to ``max_release_attempts`` before giving up — the
``LocalFeatureStore`` reclamation seam that ``SharedDirFeatureStore`` dropped.

Concurrency: ``release``/``abort``/``gc`` hold ``self._lock`` across the
``remove()`` RPC. The lock is what makes consume-once free race-free against a
concurrent ``get()`` (it prevents a re-lease between "decide to free" and the
remote delete), exactly as ``SharedDirFeatureStore`` holds its lock across
``os.remove``. For the offline single-consumer path this is fine; a high-fanout
online deployment that wants the ``remove()`` RPC off the critical section needs
a tombstone-then-free protocol — a follow-up tied to the shared metadata index.
"""

from __future__ import annotations

import io
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


class _PinConfig:
    """Fallback for Mooncake's ``ReplicateConfig`` when the package is absent
    (local unit tests with an injected store). Mirrors the fields the real
    config exposes so the injected backend can assert on them."""

    def __init__(
        self, replica_num: int = 1, with_hard_pin: bool = True, with_soft_pin: bool = False
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
        raise RuntimeError(f"Mooncake setup failed (status {rc}); kwargs={setup_kwargs}")
    return store


class MooncakeFeatureStore(FeatureStore):
    """A disaggregated :class:`FeatureStore` backed by the Mooncake store.

    One Mooncake object per sample (``{store_id}/{sample_id}``) holds a
    ``torch.save``'d ``{"generation": int, "tensors": dict}`` blob, hard-pinned so
    Mooncake's LRU never reclaims a live feature. ``store`` may be injected (any
    object exposing ``is_exist/put/get/remove`` with the Mooncake signatures) so
    the contract is unit-testable without a running master.
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
            store = _connect_store(kw)
        self._store = store
        self._put_config = _build_replicate_config(replica_num, hard_pin)
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
        self._active_leases: Dict[str, FeatureHandle] = {}
        # samples whose remote remove() failed; retried/force-freed by gc()
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
    def _key(self, sample_id: str) -> str:
        return f"{self.store_id}/{sample_id}"

    # -- store wrappers (status-code aware) --------------------------------
    def _store_exists(self, key: str) -> bool:
        return int(self._store.is_exist(key)) == 1

    def _store_put(self, key: str, value: bytes) -> None:
        rc = self._store.put(key, value, self._put_config)
        if rc is not None and int(rc) != 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")

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
        staged = {k: v.detach().cpu() for k, v in tensors.items()}
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        nbytes = sum(t.numel() * t.element_size() for t in staged.values())
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
        buf = io.BytesIO()
        torch.save({"generation": gen, "tensors": staged}, buf)
        key = self._key(sample_id)
        # Overwrite-safe publish: a re-put bumps the generation. remove() first so
        # the hard-pinned prior blob is released rather than orphaned; if that
        # remove fails the old (pinned) blob may leak, so surface it loudly.
        if self._store_exists(key) and not self._store_remove(key):
            logger.warning(
                "MooncakeFeatureStore re-put of %s: removing the stale blob failed; "
                "a hard-pinned object may be orphaned",
                key,
            )
        self._store_put(key, buf.getvalue())
        with self._lock:
            self._generation[sample_id] = gen
            self._put_time[sample_id] = self._clock()
            self._sample_bytes[sample_id] = nbytes
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
        key = self._key(sid)
        if not self._store_exists(key):
            # freed by release/abort, or never written -> never hand back stale
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
        ref_gen = sample_ref.metadata.get("generation", on_disk_gen)
        if on_disk_gen is not None and ref_gen != on_disk_gen:
            # re-put after this ref was minted -> stale handle
            raise KeyError(
                f"sample {sid} generation {ref_gen} is stale "
                f"(current {on_disk_gen}); refusing use-after-free"
            )
        raw = payload["tensors"]
        wanted = names or list(sample_ref.feature_keys.keys())
        out: Dict[str, torch.Tensor] = {}
        for n in wanted:
            raw_key = sample_ref.feature_keys.get(n, n)
            raw_key = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            if raw_key not in raw:
                raise KeyError(f"sample {sid} missing key {raw_key!r} for feature {n!r}")
            # clone-on-fetch (B5): returned tensor is independent of the transport
            out[n] = raw[raw_key].clone()
        if str(device) != "cpu":
            out = {k: v.to(device) for k, v in out.items()}
        with self._lock:
            self._counter += 1
            handle = FeatureHandle(
                sample_id=sid,
                generation=on_disk_gen or 0,
                lease_token=f"{sid}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
        return out, handle

    # -- lifetime ----------------------------------------------------------
    def _try_physical_free(self, sample_id: str) -> bool:
        """Remove the remote object. False on a (retryable) RPC failure."""
        return self._store_remove(self._key(sample_id))

    def _free_bookkeeping_locked(self, sample_id: str) -> int:
        """Drop in-process tracking for a sample. Returns bytes accounted freed."""
        nbytes = self._sample_bytes.pop(sample_id, 0)
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        self._release_pending.pop(sample_id, None)
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
                    if self._try_physical_free(sid):
                        freed_bytes += self._free_bookkeeping_locked(sid)
                        freed += 1
                    else:
                        self._release_pending.setdefault(sid, 0)
            # reconcile release-pending: retry the fallible remote free
            for sid in list(self._release_pending):
                if not self._store_exists(self._key(sid)):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    continue
                attempts = self._release_pending[sid] + 1
                if self._try_physical_free(sid):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    freed += 1
                elif attempts >= self.max_release_attempts:
                    # give up retrying the remote remove; stop tracking it. The
                    # remote object may leak — surfaced via force_freed stats.
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
