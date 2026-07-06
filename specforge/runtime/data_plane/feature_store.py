# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FeatureStore: the data plane's large-tensor storage and transfer boundary.

``FeatureStore`` is the abstract contract; ``LocalFeatureStore`` is the local
implementation. The local backend keeps features in memory on the hot path,
with an *optional* disk/mmap debug dump that doubles as the capture/replay tap.
It also supports a read-only "existing file" mode so the
``OfflineManifestReader`` can reference precomputed ``.ckpt`` files without
copying them.

Other backends (shared memory, Mooncake/RDMA) slot in behind the same API; the
lease/generation/clone-on-fetch primitives are carried here so the in-memory
backend pays nothing for them but the contract is already exercised.

This core carries three correctness fixes:

* **generation-in-URI**: ``mem://`` refs carry their generation in the URI, and
  ``get()`` rejects a ref whose generation no longer matches the resident
  sample. This closes the at-least-once redelivery hole where a stale ref could
  silently alias a freshly re-put sample.
* **atomic lease registration**: for ``mem://``, the resident read and the lease
  registration happen under one lock, so a concurrent ``abort`` can never slip
  between "I read the tensors" and "I registered my borrow".
* **best-effort dump**: an optional debug dump failure no longer aborts an
  otherwise successful in-memory publish (mem is authoritative, disk is a tap).

Memory is bounded by three cooperating mechanisms (M5; see ``DESIGN.md``):

* **Consume-once free** — ``release()`` frees a ``mem://`` sample on its last
  lease drop (the steady-state bound).
* **Backpressure** — ``max_resident_bytes`` makes "consumer fell behind" a loud
  ``MemoryError`` on ``put`` instead of a silent OOM (the controller pauses
  rollout first via ``health()``; the cap is the backstop).
* **GC / max-hold** — ``gc()`` reclaims abandoned samples backpressure cannot
  free; see :meth:`gc`.
"""

from __future__ import annotations

import abc
import dataclasses
import gzip
import io
import itertools
import logging
import os
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, quote, urlparse

import torch

from specforge.runtime.contracts import (
    SCHEMA_VERSION,
    FeatureHandle,
    FeatureSpec,
    SampleRef,
)

logger = logging.getLogger(__name__)

_DTYPE_BYTES = {  # best-effort; falls back to element_size() for real tensors
    "float32": 4,
    "float16": 2,
    "bfloat16": 2,
    "int64": 8,
    "int32": 4,
    "int16": 2,
    "uint8": 1,
    "bool": 1,
}

_GENERATION_QUERY_KEY = "generation"


def _dtype_str(t: torch.Tensor) -> str:
    return str(t.dtype).replace("torch.", "")


def spec_from_tensor(name: str, t: torch.Tensor, **kw: Any) -> FeatureSpec:
    return FeatureSpec(name=name, shape=tuple(t.shape), dtype=_dtype_str(t), **kw)


def _tensors_bytes(tensors: Dict[str, torch.Tensor]) -> int:
    return sum(t.numel() * t.element_size() for t in tensors.values())


def _make_mem_uri(store_id: str, sample_id: str, generation: int) -> str:
    return (
        f"mem://{store_id}/{quote(sample_id, safe='')}"
        f"?{_GENERATION_QUERY_KEY}={generation}"
    )


def _mem_uri_generation(uri: str) -> Optional[int]:
    """Extract the generation a ``mem://`` ref was minted for, if present."""
    values = parse_qs(urlparse(uri).query).get(_GENERATION_QUERY_KEY)
    return int(values[0]) if values else None


class FeatureStore(abc.ABC):
    """Stores and serves large feature tensors. Carries no scheduling state."""

    @abc.abstractmethod
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef: ...

    @abc.abstractmethod
    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]: ...

    @abc.abstractmethod
    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None: ...

    @abc.abstractmethod
    def abort(self, sample_id: str, *, reason: str) -> None: ...

    def estimate_bytes(self, specs: Dict[str, FeatureSpec]) -> int:
        total = 0
        for spec in specs.values():
            n = 1
            for d in spec.shape:
                n *= int(d)
            total += n * _DTYPE_BYTES.get(spec.dtype, 4)
        return total

    @abc.abstractmethod
    def health(self) -> Dict[str, Any]: ...

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        """Force-free abandoned/past-max-age features and reconcile cleanup.

        Default backend has nothing to sweep (no max-hold configured); override
        in backends with an independent reclamation policy. Returns a summary of
        what was reclaimed so callers/monitors can log it.
        """
        return {"force_freed": 0, "force_freed_bytes": 0, "release_pending": 0}


def load_feature_file(path: str) -> Dict[str, torch.Tensor]:
    """Load a SpecForge offline feature file (mirrors OfflineEagle3Dataset)."""
    if path.endswith(".gz"):
        with gzip.open(path, "rb") as f:
            return torch.load(io.BytesIO(f.read()), weights_only=False)
    return torch.load(path, weights_only=False, mmap=True)


class LocalFeatureStore(FeatureStore):
    """In-memory feature store with optional disk dump and read-only file mode.

    Two ref flavours are served transparently so the loader/trainer path is
    identical online vs offline:

    * ``mem://<store_id>/<sample_id>?generation=<n>`` — produced by :meth:`put`
      (online rollout).
    * ``file://<abs_path>`` — produced by ``OfflineManifestReader``; :meth:`get`
      lazily loads the named keys out of the existing file.
    """

    def __init__(
        self,
        store_id: Optional[str] = None,
        *,
        dump_dir: Optional[str] = None,
        clone_on_get: bool = False,
        max_resident_bytes: Optional[int] = None,
        max_hold_age_s: Optional[float] = None,
        max_release_attempts: int = 3,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.store_id = store_id or uuid.uuid4().hex[:8]
        self.dump_dir = dump_dir
        # When True the store itself clones on get(); normally the *loader* owns
        # the clone policy (clone-on-fetch default lives there), so this is off.
        self.clone_on_get = clone_on_get
        # Optional cap on mem:// residency. None = unbounded. When set, put()
        # raises loudly once exceeded — a defined failure instead of silent OOM.
        self.max_resident_bytes = max_resident_bytes
        # gc() max-hold age (distinct from the SampleRefQueue lease timeout): an
        # unleased sample older than this is force-freed by gc(). None = never.
        self.max_hold_age_s = max_hold_age_s
        # Bounded retry window for release-pending before a final force-free.
        self.max_release_attempts = max_release_attempts
        self._clock = clock
        self._mem: Dict[str, Dict[str, torch.Tensor]] = {}
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        # release-pending: sample_id -> retry attempts. A fallible backend parks
        # frees here for gc() to retry; empty for the local (synchronous) backend.
        self._release_pending: Dict[str, int] = {}
        self._lock = threading.RLock()
        self._counter = itertools.count()
        # Global monotonic generation: a re-put never reuses a prior generation,
        # so a stale handle can never alias freshly stored data. This lets
        # release() drop the _generation entry too (bounding metadata growth).
        self._gen_counter = itertools.count(1)
        # Cumulative counters for observability (never reset).
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)

    def _resident_bytes_locked(self) -> int:
        return sum(_tensors_bytes(feats) for feats in self._mem.values())

    def _free_sample_locked(self, sample_id: str) -> int:
        """Drop a sample's tensors + bookkeeping. Returns bytes freed."""
        feats = self._mem.pop(sample_id, None)
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        self._release_pending.pop(sample_id, None)
        return _tensors_bytes(feats) if feats else 0

    def _still_leased_locked(self, sample_id: str, generation: Optional[int]) -> bool:
        """True if a lease on the CURRENT generation still holds ``sample_id``.

        Counting only current-generation leases (not any lease) is what lets a
        sample re-put while an older generation is still leased get freed when its
        own last current-gen lease drops; a stale older-gen lease must not pin the
        current generation. See
        ``test_release_after_reput_while_leased_does_not_leak``.
        """
        return any(
            h.sample_id == sample_id and h.generation == generation
            for h in self._active_leases.values()
        )

    # -- write -------------------------------------------------------------
    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        if not tensors:
            raise ValueError("put requires at least one tensor")
        # Atomic from the controller's view: materialize fully, *then* return a
        # ref. The over-budget check raises *before* the entry is committed, so
        # there is nothing to roll back on that path.
        staged = {k: v for k, v in tensors.items()}
        specs = {k: spec_from_tensor(k, v) for k, v in staged.items()}
        # Stamp the target feature's representation + vocab-map version onto its
        # spec so the trainer-side mapping is version-gated. For pruned_logits the
        # token-to-draft (t2d) map is applied at rollout, so the version must travel
        # with the feature.
        target_repr = metadata.get("target_repr")
        target_name = metadata.get("target_feature_name", "target")
        if target_repr and target_name in specs:
            vmv = metadata.get("vocab_map_version")
            specs[target_name] = dataclasses.replace(
                specs[target_name],
                target_repr=target_repr,
                target_meta={"vocab_map_version": vmv} if vmv else {},
            )
        num_tokens = int(metadata.get("num_tokens", 0))
        staged_bytes = _tensors_bytes(staged)
        with self._lock:
            if self.max_resident_bytes is not None:
                projected = self._resident_bytes_locked() + staged_bytes
                if projected > self.max_resident_bytes:
                    raise MemoryError(
                        f"feature store {self.store_id} over budget: "
                        f"{projected} > {self.max_resident_bytes} bytes "
                        f"(resident_samples={len(self._mem)}); consumer is behind"
                    )
            gen = next(self._gen_counter)
            self._generation[sample_id] = gen
            self._mem[sample_id] = staged
            self._put_time[sample_id] = self._clock()
        # Best-effort capture/replay tap. mem is authoritative; a dump failure
        # must not undo a successful in-memory publish, so it is logged, not
        # raised.
        if self.dump_dir:
            try:
                self._dump(sample_id, staged)
            except Exception:  # pragma: no cover - disk is a debug side channel
                logger.warning(
                    "feature dump failed for sample %s in store %s; "
                    "mem publish kept",
                    sample_id,
                    self.store_id,
                    exc_info=True,
                )
        return SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", "unknown")),
            source_task_id=metadata.get("source_task_id"),
            feature_store_uri=_make_mem_uri(self.store_id, sample_id, gen),
            feature_keys={k: f"{sample_id}/{k}" for k in staged},
            feature_specs=specs,
            strategy=metadata.get("strategy", "eagle3"),
            schema_version=int(metadata.get("schema_version", SCHEMA_VERSION)),
            target_model_version=str(metadata.get("target_model_version", "unknown")),
            draft_weight_version=metadata.get("draft_weight_version"),
            tokenizer_version=str(metadata.get("tokenizer_version", "unknown")),
            num_tokens=num_tokens,
            estimated_bytes=staged_bytes,
            metadata={k: v for k, v in metadata.items() if k not in ("num_tokens",)},
        )

    def _dump(self, sample_id: str, tensors: Dict[str, torch.Tensor]) -> None:
        path = os.path.join(self.dump_dir, f"{sample_id}.ckpt")
        tmp = path + ".tmp"
        torch.save({k: v.detach().cpu() for k, v in tensors.items()}, tmp)
        os.replace(tmp, path)  # atomic publish

    # -- read --------------------------------------------------------------
    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cpu",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        uri = sample_ref.feature_store_uri
        wanted = names or list(sample_ref.feature_keys.keys())
        if uri.startswith("file://"):
            tensors = self._get_from_file(uri[len("file://") :], sample_ref, wanted)
            handle = self._register_file_lease(sample_ref)
        else:
            tensors, handle = self._get_from_mem(sample_ref, wanted)
        # Materialization (device move / clone) can fail or OOM. It happens
        # *after* the lease exists, so on failure we drop the lease before
        # propagating — no leaked borrow.
        try:
            if str(device) != "cpu":
                tensors = {k: v.to(device) for k, v in tensors.items()}
            if self.clone_on_get:
                tensors = {k: v.clone() for k, v in tensors.items()}
        except Exception:
            self.release(handle, reason="materialization_failed")
            raise
        return tensors, handle

    def _get_from_mem(
        self, ref: SampleRef, wanted: List[str]
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        # The resident read and the lease registration share one lock so a
        # concurrent gc()/abort() cannot free the sample between the existence
        # check and the lease being recorded (otherwise max-hold gc could free a
        # just-read sample whose lease was not yet visible).
        expected_generation = _mem_uri_generation(ref.feature_store_uri)
        with self._lock:
            if ref.sample_id not in self._mem:
                raise KeyError(f"sample {ref.sample_id} not in store {self.store_id}")
            gen = self._generation.get(ref.sample_id, 0)
            if expected_generation is not None and gen != expected_generation:
                raise KeyError(
                    f"sample {ref.sample_id} generation {expected_generation} "
                    f"is not resident in store {self.store_id} (current={gen})"
                )
            stored = self._mem[ref.sample_id]
            missing = [n for n in wanted if n not in stored]
            if missing:
                raise KeyError(f"sample {ref.sample_id} missing features {missing}")
            handle = FeatureHandle(
                sample_id=ref.sample_id,
                generation=gen,
                lease_token=f"{ref.sample_id}:{gen}:{next(self._counter)}",
            )
            self._active_leases[handle.lease_token] = handle
            out = {n: stored[n] for n in wanted}
        return out, handle

    def _register_file_lease(self, ref: SampleRef) -> FeatureHandle:
        handle = FeatureHandle(
            sample_id=ref.sample_id,
            generation=0,
            lease_token=f"{ref.sample_id}:0:{next(self._counter)}",
        )
        with self._lock:
            self._active_leases[handle.lease_token] = handle
        return handle

    def _get_from_file(
        self, path: str, ref: SampleRef, wanted: List[str]
    ) -> Dict[str, torch.Tensor]:
        raw = load_feature_file(path)
        out = {}
        for n in wanted:
            # feature_keys may remap a logical name -> a raw file key.
            raw_key = ref.feature_keys.get(n, n)
            raw_key = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            if raw_key not in raw:
                raise KeyError(f"{path} missing key {raw_key!r} for feature {n!r}")
            out[n] = raw[raw_key]
        return out

    # -- lifetime ----------------------------------------------------------
    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        # mem:// samples are consume-once: free the *current* resident generation
        # once the last lease ON THAT generation drops. Freeing is keyed on the
        # current generation via ``_still_leased_locked(sid, cur)`` — not on the
        # released handle's own generation — so a sample re-put while an older
        # lease is still outstanding (gen N held, gen N+1 leased+released, then the
        # stale gen-N handle released) is still freed: the gen-N+1 lease was the
        # last on the current generation. The reverse hazard — a stale release
        # freeing a freshly re-put-but-never-consumed sample — is closed by the
        # ``handle.generation != cur`` guard: to possess a handle for the current
        # generation you must have got() it, so a never-leased re-put can only be
        # targeted by an older-generation handle, which this guard turns into a
        # no-op. file:// samples never enter _mem, so this is a harmless no-op for
        # them. Idempotent + stale-safe.
        sid = handle.sample_id
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            cur = self._generation.get(sid)
            if cur is not None and handle.generation != cur:
                return  # stale handle (sample was re-put) -> no-op
            # Count only leases on the CURRENT generation. A sample re-put while
            # an older generation is still leased gets a fresh generation, so the
            # stale older-gen lease must not keep the current generation pinned —
            # otherwise the last current-gen release would leak it.
            if self._still_leased_locked(sid, cur):
                return  # another (current-gen) lease still holds it
            # Last current-gen lease dropped -> free. A fallible backend
            # (Mooncake) parks it release-pending for gc() to retry; local frees
            # synchronously.
            if not self._try_physical_free(sid):
                self._release_pending.setdefault(sid, 0)

    def _try_physical_free(self, sample_id: str) -> bool:
        """Physically free a sample. Returns True on success.

        Override in a backend whose free is async/fallible. The local backend
        owns its RAM so this always succeeds. Caller holds ``self._lock``.
        """
        self._free_sample_locked(sample_id)
        return True

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        """Evict a sample immediately (e.g. failed put, terminal sample drop)."""
        with self._lock:
            self._free_sample_locked(sample_id)

    # -- garbage collection / reclamation ----------------------------------
    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        """Independent reclamation sweep. Idempotent; safe to call on a timer.

        Does the two reclamations backpressure cannot: force-free abandoned
        samples past ``max_hold_age_s``, and retry release-pending frees a
        fallible backend left behind.
        """
        now = self._clock() if now is None else now
        with self._lock:
            freed, freed_bytes = self._sweep_max_hold_locked(now)
            f2, b2 = self._reconcile_release_pending_locked()
            freed, freed_bytes = freed + f2, freed_bytes + b2
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
            return {
                "force_freed": freed,
                "force_freed_bytes": freed_bytes,
                "release_pending": len(self._release_pending),
            }

    def _sweep_max_hold_locked(self, now: float) -> Tuple[int, int]:
        """Force-free unleased samples older than ``max_hold_age_s``.

        Still-leased samples are spared: force-freeing one is a use-after-free
        for the holder (B5). "Still leased" is generation-aware — only a lease on
        the sample's current generation spares it.
        """
        if self.max_hold_age_s is None:
            return 0, 0
        stale = [
            sid
            for sid, t in self._put_time.items()
            if now - t > self.max_hold_age_s
            and not self._still_leased_locked(sid, self._generation.get(sid))
        ]
        return len(stale), sum(self._free_sample_locked(sid) for sid in stale)

    def _reconcile_release_pending_locked(self) -> Tuple[int, int]:
        """Retry frees a fallible backend deferred; force-free after the window.

        No-op for the local (synchronous) backend; only does work when
        ``_try_physical_free`` is overridden to fail (e.g. Mooncake).
        """
        freed = freed_bytes = 0
        for sid in list(self._release_pending):
            if sid not in self._mem:
                self._release_pending.pop(sid, None)
                continue
            attempts = self._release_pending[sid] + 1
            # Bytes the sample is holding now; counted if this retry frees it.
            # _try_physical_free drops the tensors, so capture the size first.
            sample_bytes = _tensors_bytes(self._mem.get(sid, {}))
            if self._try_physical_free(sid):
                self._release_pending.pop(sid, None)
                freed += 1
                freed_bytes += sample_bytes
            elif attempts >= self.max_release_attempts:
                freed_bytes += self._free_sample_locked(sid)  # final force-free
                freed += 1
            else:
                self._release_pending[sid] = attempts
        return freed, freed_bytes

    def health(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            return {
                "store_id": self.store_id,
                "resident_samples": len(self._mem),
                "active_leases": len(self._active_leases),
                "resident_bytes": self._resident_bytes_locked(),
                "max_resident_bytes": self.max_resident_bytes,
                "release_pending": len(self._release_pending),
                "oldest_age_s": max(ages) if ages else 0.0,
                "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
                "force_freed_total": self._stats["force_freed"],
            }


__all__ = ["FeatureStore", "LocalFeatureStore", "load_feature_file", "spec_from_tensor"]
