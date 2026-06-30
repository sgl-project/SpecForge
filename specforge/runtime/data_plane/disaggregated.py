# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Disaggregated FeatureStore over a shared directory (M6 seam).

Producer (rollout) and consumer (trainer) run as separate processes that share
only a directory. The control plane still carries only ``SampleRef`` metadata;
``get()`` resolves a sample from the ref + filesystem alone, with no shared
in-process state. A real ``MooncakeFeatureStore`` later swaps the shared-dir
transport for RDMA behind this same API.

Scope: this is the CPU-testable *reference* backend that pins the contract, not
the fast path. The read/data path is genuinely cross-process, but the
generation/lease index is in-process — so the B5 guarantees below hold for the
single-host (single-producer) case. True multi-node liveness needs that index
lifted into a shared/durable metadata store (a later milestone).

Contract this backend locks down:

* **B5 — no use-after-free.** Each generation is a distinct file
  (``{sample_id}.g{gen}.ckpt``) published by a single atomic rename, and a
  re-``put`` removes superseded generations — so a stale ref's file is gone and
  its ``get()`` raises ``KeyError`` rather than aliasing fresh data. ``release()``
  is generation-aware (frees only the generation its lease held); clone-on-fetch
  is the default.
* **B9 — auth in disaggregated mode.** A missing/mismatched shared credential is
  a ``PermissionError`` at attach time and on the data path.
"""

from __future__ import annotations

import os
import re
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from specforge.runtime.contracts import SCHEMA_VERSION, FeatureHandle, SampleRef
from specforge.runtime.data_plane.feature_store import (
    FeatureStore,
    load_feature_file,
    spec_from_tensor,
)


class AuthPolicy:
    """Shared-secret auth (B9). ``token=None`` means auth disabled (colocated)."""

    def __init__(self, token: Optional[str] = None) -> None:
        self.token = token

    @property
    def required(self) -> bool:
        return self.token is not None

    def check(self, presented: Optional[str]) -> None:
        if self.required and presented != self.token:
            raise PermissionError(
                "disaggregated feature store: auth required and token missing/mismatched"
            )


class SharedDirFeatureStore(FeatureStore):
    """A disaggregated ``FeatureStore`` backed by a shared directory."""

    def __init__(
        self,
        root: str,
        store_id: Optional[str] = None,
        *,
        auth: Optional[AuthPolicy] = None,
        credential: Optional[str] = None,
        max_hold_age_s: Optional[float] = None,
        retain_on_release: bool = False,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.auth = auth or AuthPolicy()
        self._credential = credential
        self.auth.check(credential)  # attach-time gate
        self.store_id = store_id or uuid.uuid4().hex[:8]
        self.root = os.path.join(root, self.store_id)
        os.makedirs(self.root, exist_ok=True)
        self.max_hold_age_s = max_hold_age_s
        # Read-only re-iterable mode: an offline-imported feature set is consumed
        # across multiple epochs, so release() must NOT physically free it (mirrors
        # LocalFeatureStore's file:// no-op release). Cleanup is whole-store at run
        # end; consume-once free (retain_on_release=False) is for online rollout.
        self.retain_on_release = retain_on_release
        self._clock = clock
        # in-process liveness index (generation / put-time / active leases)
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}

    # -- paths -------------------------------------------------------------
    # Generation is encoded in the FILENAME ({sid}.g{gen}.ckpt) so a generation
    # is published with ONE atomic rename — a reader either sees a full
    # generation file or none, never new data under an old generation's number.
    _FNAME_RE = re.compile(r"^(?P<sid>.+)\.g(?P<gen>\d+)\.ckpt$")

    def _data_path(self, sample_id: str, gen: int) -> str:
        return os.path.join(self.root, f"{sample_id}.g{gen}.ckpt")

    def _disk_gens(self, sample_id: str) -> List[int]:
        gens: List[int] = []
        try:
            for name in os.listdir(self.root):
                m = self._FNAME_RE.match(name)
                if m and m.group("sid") == sample_id:
                    gens.append(int(m.group("gen")))
        except FileNotFoundError:
            pass
        return sorted(gens)

    def _current_gen(self, sample_id: str) -> Optional[int]:
        gens = self._disk_gens(sample_id)
        return gens[-1] if gens else None

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
        with self._lock:
            # next generation, derived from disk so a re-put across instances is
            # monotonic; the new file is published with a single atomic rename.
            gen = (self._current_gen(sample_id) or 0) + 1
            data_tmp = self._data_path(sample_id, gen) + f".{uuid.uuid4().hex}.tmp"
            torch.save(staged, data_tmp)
            os.replace(data_tmp, self._data_path(sample_id, gen))
            # remove superseded generations so a stale ref's file is gone (its
            # get() then raises -> no use-after-free on re-put).
            for old in self._disk_gens(sample_id):
                if old != gen:
                    try:
                        os.remove(self._data_path(sample_id, old))
                    except FileNotFoundError:
                        pass
            self._generation[sample_id] = gen
            self._put_time[sample_id] = self._clock()
        nbytes = sum(t.numel() * t.element_size() for t in staged.values())
        return SampleRef(
            sample_id=sample_id,
            run_id=str(metadata.get("run_id", "unknown")),
            source_task_id=metadata.get("source_task_id"),
            feature_store_uri=f"disagg://{self.store_id}/{sample_id}",
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
        gen = sample_ref.metadata.get("generation")
        if gen is None:
            gen = self._current_gen(sid)
        data_path = self._data_path(sid, gen) if gen is not None else None
        # A missing file means: freed (release/abort), superseded by a re-put, or
        # never written. In every case refuse to hand back data (B5: no
        # use-after-free, no stale generation).
        if data_path is None or not os.path.exists(data_path):
            raise KeyError(
                f"sample {sid} generation {gen} not available in store {self.store_id} "
                f"(freed, stale, or never written)"
            )
        raw = load_feature_file(data_path)  # gen and data come from one file
        wanted = names or list(sample_ref.feature_keys.keys())
        out = {}
        for n in wanted:
            raw_key = sample_ref.feature_keys.get(n, n)
            raw_key = raw_key.split("/")[-1] if "/" in raw_key else raw_key
            if raw_key not in raw:
                raise KeyError(f"{data_path} missing key {raw_key!r} for feature {n!r}")
            # clone-on-fetch (B5): the returned tensor is independent of the store
            out[n] = raw[raw_key].clone()
        if str(device) != "cpu":
            out = {k: v.to(device) for k, v in out.items()}
        with self._lock:
            self._counter += 1
            handle = FeatureHandle(
                sample_id=sid,
                generation=int(gen),
                lease_token=f"{sid}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
        return out, handle

    # -- lifetime ----------------------------------------------------------
    def _free_gen_locked(self, sample_id: str, gen: int) -> int:
        path = self._data_path(sample_id, gen)
        nbytes = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        if self._generation.get(sample_id) == gen:
            self._generation.pop(sample_id, None)
            self._put_time.pop(sample_id, None)
        return nbytes

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        # Free this generation's file once the last lease ON THAT generation
        # drops. A stale handle (its generation already superseded + removed by a
        # re-put) frees a file that is already gone -> harmless no-op; it can
        # never delete the freshly re-put current generation (different filename).
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            if self.retain_on_release:
                return  # offline re-iterable set: keep the file for the next epoch
            sid, gen = handle.sample_id, handle.generation
            if any(
                h.sample_id == sid and h.generation == gen
                for h in self._active_leases.values()
            ):
                return  # a lease on this generation still holds it
            self._free_gen_locked(sid, gen)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        with self._lock:
            for gen in self._disk_gens(sample_id):
                self._free_gen_locked(sample_id, gen)
            self._generation.pop(sample_id, None)
            self._put_time.pop(sample_id, None)

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
        # Max-hold force-free uses this instance's _put_time (single-host). A
        # true cross-node sweeper reads the durable index / file mtime; that is
        # the documented disaggregated follow-up.
        now = self._clock() if now is None else now
        freed = freed_bytes = 0
        with self._lock:
            if self.max_hold_age_s is not None:
                stale = [
                    sid
                    for sid, t in list(self._put_time.items())
                    if now - t > self.max_hold_age_s
                    and not any(
                        h.sample_id == sid for h in self._active_leases.values()
                    )
                ]
                for sid in stale:
                    gen = self._generation.get(sid)
                    if gen is not None:
                        freed_bytes += self._free_gen_locked(sid, gen)
                        freed += 1
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
        return {
            "force_freed": freed,
            "force_freed_bytes": freed_bytes,
            "release_pending": 0,
        }

    def health(self) -> Dict[str, Any]:
        # Residency is read from DISK (cross-process truth) and computed OUTSIDE
        # the lock so a directory stat never serializes the put/get/release path.
        sids_on_disk = set()
        resident_bytes = 0
        try:
            for name in os.listdir(self.root):
                m = self._FNAME_RE.match(name)
                if m:
                    sids_on_disk.add(m.group("sid"))
                    try:
                        resident_bytes += os.path.getsize(os.path.join(self.root, name))
                    except FileNotFoundError:
                        pass
        except FileNotFoundError:
            pass
        with self._lock:
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            active_leases = len(self._active_leases)
            force_freed = self._stats["force_freed"]
        return {
            "store_id": self.store_id,
            "backend": "shared_dir",
            "root": self.root,
            "resident_samples": len(sids_on_disk),
            "active_leases": active_leases,
            "resident_bytes": resident_bytes,
            "auth_required": self.auth.required,
            "oldest_age_s": max(ages) if ages else 0.0,
            "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
            "force_freed_total": force_freed,
        }


__all__ = ["AuthPolicy", "SharedDirFeatureStore"]
