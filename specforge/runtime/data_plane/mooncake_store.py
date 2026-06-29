# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake-backed FeatureStore for disaggregated EAGLE3.

The backend stores feature tensors in Mooncake behind the unchanged
``FeatureStore`` API. It uses Mooncake's raw-buffer API when available and falls
back to the original pickle blob path for older or injected stores.

SpecForge owns object lifetime: puts are hard-pinned, and release/abort/gc call
``remove()`` explicitly. The generation/lease index is still in-process; a shared
metadata index is required for full online multi-node tombstones.
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

_MOONCAKE_SETUP_DEFAULTS = {
    "global_segment_size": 1 << 30,  # 1 GiB per-node segment
    "local_buffer_size": 1 << 30,
    "protocol": "tcp",  # bring up on TCP; flip to "rdma" once NICs are verified
    "rdma_devices": "",
}


class _PinConfig:
    """Fallback for tests when Mooncake is not importable."""

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
    dtype = _TORCH_DTYPES.get(spec.dtype)
    if dtype is None:
        raise KeyError(f"unsupported feature dtype {spec.dtype!r} for zero-copy get")
    return torch.empty(tuple(int(d) for d in spec.shape), dtype=dtype)


def _nbytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


class MooncakeFeatureStore(FeatureStore):
    """A disaggregated :class:`FeatureStore` backed by the Mooncake store.

    In zero-copy mode, each tensor is a hard-pinned object keyed by
    ``{store_id}/{sample_id}/g{gen}/{name}``; shape and dtype travel in the
    ``SampleRef``. ``zero_copy=False`` or a backend without ``put_from``/``get_into``
    uses the pickle blob fallback.
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
        self._zero_copy = (
            bool(zero_copy)
            and callable(getattr(store, "put_from", None))
            and callable(getattr(store, "get_into", None))
        )
        self.max_resident_bytes = max_resident_bytes
        self.max_hold_age_s = max_hold_age_s
        self.retain_on_release = retain_on_release
        self.max_release_attempts = max_release_attempts
        self._clock = clock
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._sample_bytes: Dict[str, int] = {}
        self._sample_names: Dict[str, List[str]] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._release_pending: Dict[str, int] = {}
        # Same-process tombstones cover Mooncake's lease-deferred remove().
        self._freed: set = set()
        self._lock = threading.RLock()
        self._counter = 0
        self._gen_counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}

    # -- keys --------------------------------------------------------------
    def _key(self, sample_id: str) -> str:
        return f"{self.store_id}/{sample_id}"

    def _tkey(self, sample_id: str, gen: int, name: str) -> str:
        return f"{self.store_id}/{sample_id}/g{gen}/{name}"

    # -- store wrappers (status-code aware) --------------------------------
    def _store_exists(self, key: str) -> bool:
        return int(self._store.is_exist(key)) == 1

    def _store_put(self, key: str, value: bytes) -> None:
        rc = self._store.put(key, value, self._put_config)
        if rc is not None and int(rc) != 0:
            raise RuntimeError(f"mooncake put failed (status {rc}) for {key}")

    def _store_put_tensor(self, key: str, t: torch.Tensor) -> None:
        """Publish one contiguous CPU tensor via Mooncake's raw-buffer API."""
        nb = _nbytes(t)
        # RDMA requires registered source/destination buffers; TCP tolerates this.
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
        """Fetch one object into a pre-allocated tensor."""
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
        if self._zero_copy:
            for name, t in staged.items():
                self._store_put_tensor(self._tkey(sample_id, gen, name), t)
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
        else:
            buf = io.BytesIO()
            torch.save({"generation": gen, "tensors": staged}, buf)
            key = self._key(sample_id)
            if self._store_exists(key) and not self._store_remove(key):
                logger.warning(
                    "MooncakeFeatureStore re-put of %s: removing the stale blob "
                    "failed; a hard-pinned object may be orphaned",
                    key,
                )
            self._store_put(key, buf.getvalue())
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
                "generation": gen,
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
                raise KeyError(
                    f"sample {sid} generation {ref_gen} was released/aborted; "
                    f"refusing use-after-free"
                )
        wanted = names or list(sample_ref.feature_keys.keys())
        if self._zero_copy:
            out, gen = self._get_zero_copy(sample_ref, wanted)
        else:
            out, gen = self._get_pickle(sample_ref, wanted)
        if str(device) != "cpu":
            out = {k: v.to(device) for k, v in out.items()}
        with self._lock:
            self._counter += 1
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
        self, ref: SampleRef, wanted: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], int]:
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
                raise KeyError(
                    f"sample {sid} gen {gen} feature {n!r} not available "
                    f"(freed, stale, or never written)"
                )
            out[n] = _alloc_from_spec(spec)
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
        # Payloads come from producer nodes; keep pickle deserialization restricted.
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
            out[n] = raw[raw_key].clone()
        return out, (on_disk_gen or 0)

    # -- lifetime ----------------------------------------------------------
    def _try_physical_free(self, sample_id: str) -> bool:
        """Remove the remote object(s). False on a (retryable) RPC failure.

        Zero-copy uses one object per tensor; pickle mode uses one object per sample.
        """
        if not self._zero_copy:
            return self._store_remove(self._key(sample_id))
        gen = self._generation.get(sample_id)
        if gen is None:
            return True  # nothing tracked to remove (already freed)
        ok = True
        for name in self._sample_names.get(sample_id, []):
            if not self._store_remove(self._tkey(sample_id, gen, name)):
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
        nbytes = self._sample_bytes.pop(sample_id, 0)
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        self._sample_names.pop(sample_id, None)
        self._release_pending.pop(sample_id, None)
        return nbytes

    def _still_leased_locked(self, sample_id: str, generation: Optional[int]) -> bool:
        return any(
            h.sample_id == sample_id and h.generation == generation
            for h in self._active_leases.values()
        )

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            if self.retain_on_release:
                return
            sid = handle.sample_id
            cur = self._generation.get(sid)
            if cur is not None and handle.generation != cur:
                return
            if self._still_leased_locked(sid, cur):
                return
            self._freed.add((sid, handle.generation))
            if self._try_physical_free(sid):
                self._free_bookkeeping_locked(sid)
            else:
                self._release_pending.setdefault(sid, 0)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        with self._lock:
            gen = self._generation.get(sample_id)
            if gen is not None:
                self._freed.add((sample_id, gen))
            if self._try_physical_free(sample_id):
                self._free_bookkeeping_locked(sample_id)
            else:
                self._release_pending.setdefault(sample_id, 0)

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        now = self._clock() if now is None else now
        freed = freed_bytes = 0
        with self._lock:
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
            for sid in list(self._release_pending):
                if not self._sample_exists(sid):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    continue
                attempts = self._release_pending[sid] + 1
                if self._try_physical_free(sid):
                    freed_bytes += self._free_bookkeeping_locked(sid)
                    freed += 1
                elif attempts >= self.max_release_attempts:
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
