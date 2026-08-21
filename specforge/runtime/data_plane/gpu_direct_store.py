# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
# Licensed under the Apache License, Version 2.0
"""Mooncake GPU-direct materialization for SGLang server captures.

The tensor payload moves from the capture server's CUDA allocation directly
into trainer CUDA allocations.  TCP is restricted to release acknowledgements.
"""

from __future__ import annotations

import concurrent.futures
import json
import os
import socket
import threading
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch

from specforge.runtime.contracts import FeatureHandle, SampleRef
from specforge.runtime.data_plane.feature_store import FeatureStore

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


def _nbytes(tensor: torch.Tensor) -> int:
    return tensor.numel() * tensor.element_size()


def _control_request(endpoint: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    host, port_text = endpoint.rsplit(":", 1)
    with socket.create_connection((host, int(port_text)), timeout=30.0) as conn:
        conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        conn.sendall(
            json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        )
        response = conn.makefile("rb").readline(1 << 20)
    if not response:
        raise RuntimeError(f"empty GPU-direct control response from {endpoint}")
    decoded = json.loads(response.decode("utf-8"))
    if not decoded.get("ok"):
        raise RuntimeError(str(decoded.get("error", "GPU-direct control failure")))
    return decoded


class MooncakeGpuDirectFeatureStore(FeatureStore):
    """Consumer-side FeatureStore for Mooncake NVLink and RDMA descriptors."""

    def __init__(
        self,
        *,
        store_id: str,
        local_hostname: str,
        transport: str,
        rdma_devices: str = "",
        device: "torch.device | str | int" = "cuda",
        retain_on_release: bool = False,
        enable_transfers: bool = True,
    ) -> None:
        normalized = "nvlink" if transport == "mnnvl" else transport
        if normalized not in {"nvlink", "nvlink_intra", "rdma"}:
            raise ValueError(
                "GPU-direct Mooncake transport must be nvlink, nvlink_intra, or rdma"
            )
        self.store_id = str(store_id)
        self.local_hostname = str(local_hostname)
        self.transport = normalized
        self.rdma_devices = str(rdma_devices)
        self.retain_on_release = retain_on_release
        self.enable_transfers = enable_transfers
        self.device = (
            self._resolve_device(device) if enable_transfers else torch.device("cpu")
        )
        self.materialize_device = self.device
        self.clone_on_fetch = False
        self._engine = None
        self._session_id: Optional[str] = None
        self._lock = threading.RLock()
        self._active_leases: Dict[str, SampleRef] = {}
        self._known_refs: Dict[str, SampleRef] = {}
        self._external_attempts: set[Tuple[str, int]] = set()
        self._release_pending: Dict[str, SampleRef] = {}
        self._freed: set[Tuple[str, int]] = set()
        self._stats = {
            "gets": 0,
            "releases": 0,
            "aborts": 0,
            "gpu_direct_bytes": 0,
            "host_payload_bytes": 0,
            "release_batch_requests": 0,
            "release_batch_items": 0,
            "release_batch_fallbacks": 0,
        }
        if enable_transfers:
            self._ensure_engine()

    @staticmethod
    def _resolve_device(device: "torch.device | str | int") -> torch.device:
        if not torch.cuda.is_available():
            raise RuntimeError("Mooncake GPU-direct transport requires CUDA")
        resolved = (
            torch.device("cuda", device)
            if isinstance(device, int)
            else torch.device(device)
        )
        if resolved.type != "cuda":
            raise ValueError("Mooncake GPU-direct destination must be CUDA")
        if resolved.index is None:
            resolved = torch.device("cuda", torch.cuda.current_device())
        return resolved

    def _ensure_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from mooncake import engine as mooncake_engine
            except ImportError as exc:
                raise ImportError(
                    "mooncake-transfer-engine is required for GPU-direct capture"
                ) from exc
            if self.transport == "nvlink":
                os.environ["MC_FORCE_MNNVL"] = "true"
                os.environ.pop("MC_INTRA_NVLINK", None)
                os.environ.pop("MC_FORCE_HCA", None)
                if not bool(getattr(mooncake_engine, "SUPPORT_MNNVL", False)):
                    raise RuntimeError(
                        "installed Mooncake wheel has SUPPORT_MNNVL=False"
                    )
            elif self.transport == "nvlink_intra":
                os.environ.pop("MC_FORCE_MNNVL", None)
                os.environ["MC_INTRA_NVLINK"] = "true"
                os.environ.pop("MC_FORCE_HCA", None)
                if not bool(
                    getattr(mooncake_engine, "SUPPORT_INTRA_NVLINK", False)
                ):
                    raise RuntimeError(
                        "installed Mooncake wheel has SUPPORT_INTRA_NVLINK=False"
                    )
            else:
                os.environ.pop("MC_FORCE_MNNVL", None)
                os.environ.pop("MC_INTRA_NVLINK", None)
                os.environ["MC_FORCE_HCA"] = "true"
            engine = mooncake_engine.TransferEngine()
            status = engine.initialize(
                self.local_hostname,
                "P2PHANDSHAKE",
                self.transport,
                self.rdma_devices if self.transport == "rdma" else "",
            )
            if int(status) != 0:
                raise RuntimeError(
                    f"Mooncake {self.transport} initialization failed: {status}"
                )
            self._engine = engine
            self._session_id = (
                f"{self.local_hostname}:{int(engine.get_rpc_port())}"
            )
            return engine

    @staticmethod
    def _descriptor(ref: SampleRef) -> Dict[str, Any]:
        descriptor = ref.metadata.get("mooncake_gpu_direct")
        if not isinstance(descriptor, dict):
            raise KeyError("SampleRef carries no mooncake_gpu_direct descriptor")
        return descriptor

    def put(
        self,
        tensors: Dict[str, torch.Tensor],
        *,
        sample_id: str,
        metadata: Dict[str, Any],
    ) -> SampleRef:
        raise RuntimeError(
            "MooncakeGpuDirectFeatureStore receives server-owned captures; "
            "publish through SGLang spec_capture"
        )

    def adopt(self, sample_ref: SampleRef) -> None:
        descriptor = self._descriptor(sample_ref)
        ref_transport = str(descriptor["transport"])
        if ref_transport != self.transport:
            raise ValueError(
                f"capture transport {ref_transport!r} differs from store "
                f"transport {self.transport!r}"
            )
        generation = int(sample_ref.metadata["generation"])
        with self._lock:
            self._known_refs[sample_ref.sample_id] = sample_ref
            self._external_attempts.discard((sample_ref.sample_id, generation))

    def track_external_attempt(
        self,
        sample_id: str,
        *,
        generation: int,
        feature_names: List[str],
    ) -> None:
        del feature_names
        with self._lock:
            self._external_attempts.add((str(sample_id), int(generation)))

    def discard_external_attempts(
        self, *, reason: str = "unadopted-external-capture"
    ) -> int:
        del reason
        with self._lock:
            count = len(self._external_attempts)
            self._external_attempts.clear()
        return count

    def _allocate_outputs(
        self,
        sample_ref: SampleRef,
        names: List[str],
        device: torch.device,
    ) -> Tuple[Dict[str, torch.Tensor], List[int], List[int], List[int]]:
        descriptor = self._descriptor(sample_ref)
        features = descriptor.get("features")
        if not isinstance(features, dict):
            raise KeyError("GPU-direct descriptor carries no feature buffers")
        outputs: Dict[str, torch.Tensor] = {}
        local_addresses: List[int] = []
        remote_addresses: List[int] = []
        lengths: List[int] = []
        for name in names:
            spec = sample_ref.feature_specs.get(name)
            remote = features.get(name)
            if spec is None or not isinstance(remote, dict):
                raise KeyError(f"missing GPU-direct descriptor for {name!r}")
            dtype = _TORCH_DTYPES.get(spec.dtype)
            if dtype is None:
                raise TypeError(f"unsupported GPU-direct dtype {spec.dtype!r}")
            output = torch.empty(tuple(spec.shape), dtype=dtype, device=device)
            expected = _nbytes(output)
            described = int(remote["nbytes"])
            if expected != described:
                raise ValueError(
                    f"feature {name!r} descriptor has {described} bytes; "
                    f"FeatureSpec requires {expected}"
                )
            outputs[name] = output
            local_addresses.append(output.data_ptr())
            remote_addresses.append(int(remote["address"]))
            lengths.append(expected)
        return outputs, local_addresses, remote_addresses, lengths

    def get(
        self,
        sample_ref: SampleRef,
        *,
        device: "torch.device | str" = "cuda",
        names: Optional[List[str]] = None,
    ) -> Tuple[Dict[str, torch.Tensor], FeatureHandle]:
        destination = torch.device(device)
        if destination.type != "cuda":
            raise ValueError("GPU-direct feature materialization requires CUDA")
        if destination.index is None:
            destination = self.device
        generation = int(sample_ref.metadata["generation"])
        with self._lock:
            if (sample_ref.sample_id, generation) in self._freed:
                raise KeyError(
                    f"sample {sample_ref.sample_id} generation {generation} was freed"
                )
        descriptor = self._descriptor(sample_ref)
        if str(descriptor["transport"]) != self.transport:
            raise ValueError("SampleRef transport does not match this store")
        wanted = names or list(sample_ref.feature_keys)
        with torch.cuda.device(destination):
            outputs, local, remote, lengths = self._allocate_outputs(
                sample_ref, wanted, destination
            )
            engine = self._ensure_engine()
            registered: List[int] = []
            try:
                if self.transport == "rdma":
                    for address, length in zip(local, lengths):
                        status = engine.register_memory(address, length)
                        if status is not None and int(status) != 0:
                            raise RuntimeError(
                                f"Mooncake CUDA registration failed: {status}"
                            )
                        registered.append(address)
                batch_read = getattr(engine, "batch_transfer_sync_read", None)
                # Mooncake documents an accuracy caveat for batch reads over
                # multi-node NVLink.  NVL72 therefore uses individual reads.
                if self.transport != "nvlink" and callable(batch_read):
                    status = batch_read(
                        str(descriptor["session_id"]), local, remote, lengths
                    )
                    if int(status) != 0:
                        raise RuntimeError(
                            f"Mooncake batch GPU read failed: {status}"
                        )
                else:
                    for local_address, remote_address, length in zip(
                        local, remote, lengths
                    ):
                        status = engine.transfer_sync_read(
                            str(descriptor["session_id"]),
                            local_address,
                            remote_address,
                            length,
                        )
                        if int(status) != 0:
                            raise RuntimeError(
                                f"Mooncake GPU read failed: {status}"
                            )
            finally:
                for address in registered:
                    engine.unregister_memory(address)

        handle = FeatureHandle(
            sample_id=sample_ref.sample_id,
            generation=generation,
            lease_token=uuid.uuid4().hex,
        )
        with self._lock:
            self._known_refs[sample_ref.sample_id] = sample_ref
            self._active_leases[handle.lease_token] = sample_ref
            self._stats["gets"] += 1
            self._stats["gpu_direct_bytes"] += sum(lengths)
        return outputs, handle

    @staticmethod
    def _release_remote(ref: SampleRef, *, op: str, reason: str) -> None:
        descriptor = MooncakeGpuDirectFeatureStore._descriptor(ref)
        _control_request(
            str(descriptor["control_endpoint"]),
            {
                "op": op,
                "token": str(descriptor["control_token"]),
                "sample_id": ref.sample_id,
                "generation": int(ref.metadata["generation"]),
                "reason": reason,
            },
        )

    @staticmethod
    def _release_remote_batch(
        refs: List[SampleRef], *, op: str, reason: str
    ) -> None:
        if not refs:
            return
        descriptors = [
            MooncakeGpuDirectFeatureStore._descriptor(ref) for ref in refs
        ]
        endpoint = str(descriptors[0]["control_endpoint"])
        token = str(descriptors[0]["control_token"])
        for descriptor in descriptors[1:]:
            if str(descriptor["control_endpoint"]) != endpoint:
                raise ValueError("GPU-direct release batch spans control endpoints")
            if str(descriptor["control_token"]) != token:
                raise ValueError("GPU-direct release batch spans control tokens")
        _control_request(
            endpoint,
            {
                "op": f"{op}_batch",
                "token": token,
                "items": [
                    {
                        "sample_id": ref.sample_id,
                        "generation": int(ref.metadata["generation"]),
                    }
                    for ref in refs
                ],
                "reason": reason,
            },
        )

    def _finish_release_many(
        self, refs: List[SampleRef], *, op: str, reason: str
    ) -> int:
        unique = {ref.sample_id: ref for ref in refs}
        groups: Dict[Tuple[str, str], List[SampleRef]] = {}
        for ref in unique.values():
            descriptor = self._descriptor(ref)
            key = (
                str(descriptor["control_endpoint"]),
                str(descriptor["control_token"]),
            )
            groups.setdefault(key, []).append(ref)

        def release_group(group: List[SampleRef]):
            if len(group) == 1:
                ref = group[0]
                try:
                    self._release_remote(ref, op=op, reason=reason)
                except Exception:
                    return {ref.sample_id: False}, False
                return {ref.sample_id: True}, False
            try:
                self._release_remote_batch(group, op=op, reason=reason)
            except Exception as exc:
                if "unsupported control operation" not in str(exc):
                    return {ref.sample_id: False for ref in group}, False
                outcomes = {}
                for ref in group:
                    try:
                        self._release_remote(ref, op=op, reason=reason)
                    except Exception:
                        outcomes[ref.sample_id] = False
                    else:
                        outcomes[ref.sample_id] = True
                return outcomes, True
            return {ref.sample_id: True for ref in group}, False

        outcomes: Dict[str, bool] = {}
        fallbacks = 0
        grouped = list(groups.values())
        if len(grouped) == 1:
            group_outcomes, fallback = release_group(grouped[0])
            outcomes.update(group_outcomes)
            fallbacks += int(fallback)
        elif grouped:
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=len(grouped),
                thread_name_prefix="gpudirect-release",
            ) as executor:
                futures = [executor.submit(release_group, group) for group in grouped]
                for future in futures:
                    group_outcomes, fallback = future.result()
                    outcomes.update(group_outcomes)
                    fallbacks += int(fallback)

        released = 0
        with self._lock:
            self._stats["release_batch_requests"] += sum(
                len(group) > 1 for group in grouped
            )
            self._stats["release_batch_items"] += sum(
                len(group) for group in grouped if len(group) > 1
            )
            self._stats["release_batch_fallbacks"] += fallbacks
            for sample_id, ref in unique.items():
                if not outcomes.get(sample_id, False):
                    self._release_pending[sample_id] = ref
                    continue
                generation = int(ref.metadata["generation"])
                self._release_pending.pop(sample_id, None)
                self._known_refs.pop(sample_id, None)
                self._freed.add((sample_id, generation))
                released += 1
        return released

    def _finish_release(self, ref: SampleRef, *, op: str, reason: str) -> bool:
        return bool(self._finish_release_many([ref], op=op, reason=reason))

    def release(self, handle: FeatureHandle, *, reason: str = "consumed") -> None:
        with self._lock:
            ref = self._active_leases.pop(handle.lease_token, None)
        if ref is None:
            return
        if self.retain_on_release:
            return
        if self._finish_release(ref, op="release", reason=reason):
            with self._lock:
                self._stats["releases"] += 1

    def abort(self, sample_id: str, *, reason: str) -> None:
        with self._lock:
            ref = self._known_refs.get(sample_id)
        if ref is None:
            return
        if self._finish_release(ref, op="abort", reason=reason):
            with self._lock:
                self._stats["aborts"] += 1

    def abort_many(self, sample_ids: List[str], *, reason: str) -> int:
        target = set(sample_ids)
        with self._lock:
            refs = [
                ref
                for sample_id, ref in self._known_refs.items()
                if sample_id in target
            ]
        removed = self._finish_release_many(refs, op="abort", reason=reason)
        with self._lock:
            self._stats["aborts"] += removed
        return removed

    def retry_sample_removals(self, sample_ids: List[str]) -> Dict[str, Any]:
        target = set(sample_ids)
        with self._lock:
            pending = [
                ref
                for sample_id, ref in self._release_pending.items()
                if sample_id in target
            ]
        removed = self._finish_release_many(
            pending, op="abort", reason="release-retry"
        )
        with self._lock:
            remaining = [sid for sid in self._release_pending if sid in target]
        return {
            "removed": removed,
            "removed_bytes": 0,
            "release_pending": len(remaining),
            "remaining_ids": remaining,
            "attempts": 1 if pending else 0,
        }

    def drain_sample_removals(
        self,
        sample_ids: List[str],
        *,
        max_attempts: int = 8,
        retry_interval_s: float = 0.25,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, int]:
        return self._drain(
            sample_ids=sample_ids,
            max_attempts=max_attempts,
            retry_interval_s=retry_interval_s,
            sleep=sleep,
        )

    def drain_pending_removals(
        self,
        *,
        max_attempts: int = 40,
        retry_interval_s: float = 0.5,
        sleep: Callable[[float], None] = time.sleep,
    ) -> Dict[str, int]:
        return self._drain(
            sample_ids=None,
            max_attempts=max_attempts,
            retry_interval_s=retry_interval_s,
            sleep=sleep,
        )

    def _drain(
        self,
        *,
        sample_ids: Optional[List[str]],
        max_attempts: int,
        retry_interval_s: float,
        sleep: Callable[[float], None],
    ) -> Dict[str, int]:
        target = None if sample_ids is None else set(sample_ids)
        removed = 0
        for attempt in range(max_attempts):
            with self._lock:
                pending = [
                    ref
                    for sample_id, ref in self._release_pending.items()
                    if target is None or sample_id in target
                ]
            if not pending:
                return {
                    "removed": removed,
                    "removed_bytes": 0,
                    "release_pending": 0,
                    "attempts": attempt,
                }
            removed += self._finish_release_many(
                pending, op="abort", reason="lifecycle-drain"
            )
            if attempt + 1 < max_attempts and retry_interval_s:
                sleep(retry_interval_s)
        with self._lock:
            remaining = [
                sample_id
                for sample_id in self._release_pending
                if target is None or sample_id in target
            ]
        raise RuntimeError(
            f"MooncakeGpuDirectFeatureStore could not release {remaining[:16]}"
        )

    def health(self) -> Dict[str, Any]:
        with self._lock:
            return {
                "backend": "mooncake_gpu_direct",
                "transport": self.transport,
                "device": str(self.device),
                "session_id": self._session_id,
                "known_samples": len(self._known_refs),
                "active_leases": len(self._active_leases),
                "release_pending": len(self._release_pending),
                **self._stats,
            }


__all__ = ["MooncakeGpuDirectFeatureStore"]
