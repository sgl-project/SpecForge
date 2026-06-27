# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Disaggregated FeatureStore: producer and consumer in different processes.

``SharedDirFeatureStore`` is the M6 disaggregation seam. Producer (rollout) and
consumer (trainer) run as separate processes that share only a directory — the
control plane still carries nothing but ``SampleRef`` metadata (the URI points at
the shared store), exactly as in the colocated case. ``get()`` resolves a sample
from the shared filesystem *and the ref alone*, with no shared in-process state,
which is what makes it a true cross-process boundary.

This is the framework, not the fast path. A real ``MooncakeFeatureStore`` swaps
the shared-dir transport for RDMA zero-copy **behind this same API**; everything
the control/training planes see is unchanged. What this reference backend exists
to lock down *now* is the contract the fast backend must honor:

* **B5 — no use-after-free.** ``get()`` after ``release``/``abort`` raises
  loudly (``KeyError``) instead of returning stale data, and a generation guard
  rejects a stale ref after a re-``put``. Clone-on-fetch is the default, so a
  consumer never holds a pointer a free could invalidate.
* **B9 — auth required in disaggregated mode.** A process must present the
  shared credential to attach to and use the store; a missing/mismatched token
  is a ``PermissionError``, not a silent open door.

Cross-node note: the per-sample generation/lease *index* here is in-process for
the single-host case. A true multi-node deployment moves that index to the
durable metadata store (so liveness survives a process restart); the on-disk
payload + URI contract is already cross-process.
"""

from __future__ import annotations

import json
import os
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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.auth = auth or AuthPolicy()
        self._credential = credential
        self.auth.check(credential)  # attach-time gate
        self.store_id = store_id or uuid.uuid4().hex[:8]
        self.root = os.path.join(root, self.store_id)
        os.makedirs(self.root, exist_ok=True)
        self.max_hold_age_s = max_hold_age_s
        self._clock = clock
        # in-process liveness index (generation / put-time / active leases)
        self._generation: Dict[str, int] = {}
        self._put_time: Dict[str, float] = {}
        self._active_leases: Dict[str, FeatureHandle] = {}
        self._lock = threading.RLock()
        self._counter = 0
        self._gen_counter = 0
        self._stats = {"force_freed": 0, "force_freed_bytes": 0}

    # -- paths -------------------------------------------------------------
    def _data_path(self, sample_id: str) -> str:
        return os.path.join(self.root, f"{sample_id}.ckpt")

    def _gen_path(self, sample_id: str) -> str:
        return os.path.join(self.root, f"{sample_id}.gen")

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
            self._gen_counter += 1
            gen = self._gen_counter
        # atomic publish: write data + generation sidecar, rename into place
        data_tmp = self._data_path(sample_id) + f".{uuid.uuid4().hex}.tmp"
        torch.save(staged, data_tmp)
        os.replace(data_tmp, self._data_path(sample_id))
        with open(self._gen_path(sample_id) + ".tmp", "w") as f:
            json.dump({"generation": gen}, f)
        os.replace(self._gen_path(sample_id) + ".tmp", self._gen_path(sample_id))
        with self._lock:
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
        data_path = self._data_path(sid)
        if not os.path.exists(data_path):
            # freed by release/abort, or never written -> never hand back stale
            raise KeyError(f"sample {sid} not available in store {self.store_id}")
        on_disk_gen = self._read_gen(sid)
        ref_gen = sample_ref.metadata.get("generation", on_disk_gen)
        if on_disk_gen is not None and ref_gen != on_disk_gen:
            # the sample was re-put after this ref was minted -> stale handle
            raise KeyError(
                f"sample {sid} generation {ref_gen} is stale "
                f"(current {on_disk_gen}); refusing use-after-free"
            )
        raw = load_feature_file(data_path)
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
                generation=on_disk_gen or 0,
                lease_token=f"{sid}:{self._counter}",
            )
            self._active_leases[handle.lease_token] = handle
        return out, handle

    def _read_gen(self, sample_id: str) -> Optional[int]:
        try:
            with open(self._gen_path(sample_id)) as f:
                return int(json.load(f)["generation"])
        except (FileNotFoundError, ValueError, KeyError):
            return None

    # -- lifetime ----------------------------------------------------------
    def _free_locked(self, sample_id: str) -> int:
        nbytes = 0
        data_path = self._data_path(sample_id)
        if os.path.exists(data_path):
            nbytes = os.path.getsize(data_path)
        for p in (data_path, self._gen_path(sample_id)):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
        self._generation.pop(sample_id, None)
        self._put_time.pop(sample_id, None)
        return nbytes

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            self._active_leases.pop(handle.lease_token, None)
            cur = self._generation.get(handle.sample_id)
            if cur is not None and handle.generation != cur:
                return  # stale -> no-op
            still_leased = any(
                h.sample_id == handle.sample_id for h in self._active_leases.values()
            )
            if not still_leased:
                self._free_locked(handle.sample_id)

    def abort(self, sample_id: str, *, reason: str = "aborted") -> None:
        with self._lock:
            self._free_locked(sample_id)

    def gc(self, *, now: Optional[float] = None) -> Dict[str, int]:
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
                    freed_bytes += self._free_locked(sid)
                    freed += 1
            self._stats["force_freed"] += freed
            self._stats["force_freed_bytes"] += freed_bytes
        return {
            "force_freed": freed,
            "force_freed_bytes": freed_bytes,
            "release_pending": 0,
        }

    def health(self) -> Dict[str, Any]:
        with self._lock:
            now = self._clock()
            ages = [now - t for t in self._put_time.values()]
            resident = sum(
                os.path.getsize(self._data_path(s))
                for s in self._generation
                if os.path.exists(self._data_path(s))
            )
            return {
                "store_id": self.store_id,
                "backend": "shared_dir",
                "root": self.root,
                "resident_samples": len(self._generation),
                "active_leases": len(self._active_leases),
                "resident_bytes": resident,
                "auth_required": self.auth.required,
                "oldest_age_s": max(ages) if ages else 0.0,
                "avg_age_s": (sum(ages) / len(ages)) if ages else 0.0,
                "force_freed_total": self._stats["force_freed"],
            }


__all__ = ["AuthPolicy", "SharedDirFeatureStore"]
