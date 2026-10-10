# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake writer for captured features, in MooncakeFeatureStore's layout.

Each tensor becomes one hard-pinned object at ``{store_id}/{sample_id}/g{gen}/
{name}`` holding raw bytes; shape and dtype travel in the returned result, so
tensors never touch the HTTP response. One background thread publishes each
scheduler batch with a single ``batch_put_from`` while the next prefill runs.

Mooncake connection uses the standard ``MOONCAKE_*`` environment variables.
CUDA capture publishes device tensors on RDMA by default;
``SGLANG_SPEC_CAPTURE_GPU_PUT=0`` selects host publication and ``1`` requires
device publication (and RDMA).
"""

from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

logger = logging.getLogger(__name__)

# torch dtype -> the FeatureSpec dtype string SpecForge's zero-copy get() maps
# back to a torch dtype. Keep in sync with MooncakeFeatureStore._TORCH_DTYPES.
DTYPE_STR = {
    torch.float32: "float32",
    torch.float64: "float64",
    torch.float16: "float16",
    torch.bfloat16: "bfloat16",
    torch.int64: "int64",
    torch.int32: "int32",
    torch.int16: "int16",
    torch.int8: "int8",
    torch.uint8: "uint8",
    torch.bool: "bool",
}
STR_DTYPE = {v: k for k, v in DTYPE_STR.items()}

ARTIFACT_AUX = "aux"
ARTIFACT_LAST_HIDDEN = "last_hidden"

# One captured sample: (request spec, aux rows or None, last-hidden rows or None).
Sample = Tuple[Dict[str, Any], Optional[torch.Tensor], Optional[torch.Tensor]]


def gpu_put_enabled() -> bool:
    protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")
    configured = os.environ.get("SGLANG_SPEC_CAPTURE_GPU_PUT")
    if configured is None:
        return (
            protocol == "rdma"
            and torch.version.cuda is not None
            and torch.cuda.is_available()
        )
    enabled = configured == "1"
    if enabled and protocol != "rdma":
        raise ValueError(
            "SGLANG_SPEC_CAPTURE_GPU_PUT=1 requires MOONCAKE_PROTOCOL=rdma"
        )
    return enabled


def object_key(store_id: str, sample_id: str, gen: int, name: str) -> str:
    return f"{store_id}/{sample_id}/g{gen}/{name}"


class SpecCaptureSink:
    """Writes captured per-request tensors into Mooncake in SpecForge layout."""

    def __init__(self, aux_layer_ids: Optional[Sequence[int]] = None) -> None:
        self.aux_layer_ids = list(aux_layer_ids) if aux_layer_ids else None
        self._store = None
        self._put_config = None
        self._lock = threading.Lock()
        # One writer is enough: Mooncake stripes a batched transfer itself.
        # The thread only decouples that transfer from the scheduler.
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="spec-capture-batch-put"
        )

    # -- connection ---------------------------------------------------------
    def _connect(self):
        if self._store is not None:
            return self._store
        with self._lock:
            if self._store is not None:
                return self._store
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            store = MooncakeDistributedStore()
            global_segment_size = int(
                os.environ.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)
            )
            local_buffer_size = int(
                os.environ.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)
            )
            protocol = os.environ.get("MOONCAKE_PROTOCOL", "tcp")
            # Ascend Mooncake rejects the wildcard location ("location:* is not
            # supported"); skip it in setup() and mount with location="cpu".
            ascend_host = bool(os.environ.get("ASCEND_RT_VISIBLE_DEVICES"))
            segment_to_mount = global_segment_size if ascend_host else 0
            if ascend_host:
                global_segment_size = 0
                local_buffer_size = 0
            rc = store.setup(
                local_hostname=os.environ.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
                metadata_server=os.environ.get(
                    "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
                ),
                global_segment_size=global_segment_size,
                local_buffer_size=local_buffer_size,
                protocol=protocol,
                rdma_devices=os.environ.get("MOONCAKE_RDMA_DEVICES", ""),
                master_server_addr=os.environ.get(
                    "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
                ),
            )
            if rc is not None and int(rc) != 0:
                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")
            if segment_to_mount:
                mount = getattr(store, "allocate_and_mount_segment", None)
                if mount is None:
                    raise RuntimeError(
                        "Mooncake build on this Ascend host cannot register a "
                        "wildcard segment and has no allocate_and_mount_segment; "
                        "upgrade mooncake-transfer-engine"
                    )
                result = mount(segment_to_mount, protocol, "cpu")
                mrc = result.get("ret", -1) if isinstance(result, dict) else result
                if mrc is not None and int(mrc) != 0:
                    raise RuntimeError(
                        f"spec-capture mooncake mount segment failed (status {mrc})"
                    )
            # Hard-pin every object: SpecForge, not Mooncake's LRU, owns the
            # lifetime, so a committed feature is never evicted before use.
            cfg = ReplicateConfig()
            cfg.replica_num = 1
            # Older ROCm Mooncake builds only expose `with_soft_pin`.
            if hasattr(cfg, "with_hard_pin"):
                cfg.with_hard_pin = True
            elif hasattr(cfg, "with_soft_pin"):
                cfg.with_soft_pin = True
            self._put_config = cfg
            self._store = store
            logger.info("spec-capture mooncake sink connected")
            return store

    def _remove_many_quiet(self, keys: List[str]) -> None:
        if not keys:
            return
        store = self._connect()
        batch_remove = getattr(store, "batch_remove", None)
        if batch_remove is not None:
            try:
                batch_remove(keys)
                return
            except Exception:
                pass
        for key in keys:
            try:
                store.remove(key)
            except Exception:
                pass

    # -- batch entry points ---------------------------------------------------
    def submit_samples(
        self,
        samples: List[Sample],
        *,
        ready_event: Optional[torch.cuda.Event] = None,
    ) -> Future:
        """Queue one scheduler batch without blocking the scheduler thread.

        ``ready_event`` orders device tensors after the work that produced them.
        """
        return self._executor.submit(self._put_ready_samples, samples, ready_event)

    def _put_ready_samples(self, samples, ready_event):
        if ready_event is not None:
            ready_event.synchronize()
        return self.put_samples(samples)

    def put_samples(self, samples: List[Sample]) -> List[Dict[str, Any]]:
        """Publish samples with one batched Mooncake call; return their results.

        The results are returned only after every object is written, so a
        reference built from them never points at an incomplete sample.
        """
        if not samples:
            return []

        store = self._connect()
        timing_enabled = os.environ.get("SGLANG_SPEC_CAPTURE_TIMING", "0") == "1"
        gpu_put = gpu_put_enabled()
        started = time.perf_counter()
        keys: List[str] = []
        tensors: List[torch.Tensor] = []
        sizes: List[int] = []
        replace_keys: List[str] = []
        results: List[Dict[str, Any]] = []

        def stage(result_feats, *, store_id, sample_id, gen, replace, name, tensor):
            if gpu_put and tensor.is_cuda:
                tensor = tensor.detach().contiguous()
            else:
                tensor = tensor.detach().to("cpu").contiguous()
            key = object_key(store_id, sample_id, gen, name)
            keys.append(key)
            tensors.append(tensor)
            sizes.append(tensor.element_size() * tensor.numel())
            if replace:
                replace_keys.append(key)
            result_feats[name] = {
                "shape": list(tensor.shape),
                "dtype": DTYPE_STR.get(
                    tensor.dtype, str(tensor.dtype).replace("torch.", "")
                ),
            }

        for spec, aux, last_hidden in samples:
            store_id = str(spec["store_id"])
            sample_id = str(spec["sample_id"])
            gen = int(spec.get("gen", 1))
            replace = bool(spec.get("replace", False))
            features: Dict[str, str] = dict(spec.get("features") or {})
            result_feats: Dict[str, Dict[str, Any]] = {}
            common = dict(store_id=store_id, sample_id=sample_id, gen=gen)

            for artifact, tensor in (
                (ARTIFACT_AUX, aux),
                (ARTIFACT_LAST_HIDDEN, last_hidden),
            ):
                name = features.get(artifact)
                if name is None:
                    continue
                if tensor is None:
                    raise RuntimeError(
                        f"spec_capture requested {artifact!r} but the server did "
                        "not capture it; launch with --aux-hidden-state-capture"
                    )
                stage(
                    result_feats,
                    replace=replace,
                    name=name,
                    tensor=tensor.unsqueeze(0),
                    **common,
                )
            for item in spec.get("passthrough") or []:
                dtype = STR_DTYPE.get(str(item.get("dtype", "int64")))
                if dtype is None:
                    raise RuntimeError(
                        f"spec_capture passthrough {item.get('name')!r}: "
                        f"unsupported dtype {item.get('dtype')!r}"
                    )
                tensor = torch.tensor(item["data"], dtype=dtype).reshape(
                    [int(d) for d in item["shape"]]
                )
                stage(
                    result_feats,
                    replace=replace,
                    name=str(item["name"]),
                    tensor=tensor,
                    **common,
                )
            results.append(
                {
                    "sample_id": sample_id,
                    "store_id": store_id,
                    "gen": gen,
                    "aux_layer_ids": self.aux_layer_ids,
                    "features": result_feats,
                }
            )

        for device in {t.device for t in tensors if t.is_cuda}:
            # Complete the contiguous copies this thread made.
            torch.cuda.current_stream(device).synchronize()
        materialize_ms = (time.perf_counter() - started) * 1000.0
        self._remove_many_quiet(replace_keys)
        registered: List[int] = []
        seen_storage = set()
        register_started = time.perf_counter()
        try:
            for tensor, nbytes in zip(tensors, sizes):
                storage = tensor.untyped_storage()
                ptr = storage.data_ptr()
                if ptr in seen_storage or nbytes == 0:
                    continue
                seen_storage.add(ptr)
                try:
                    # Per-request slices can share one batch allocation.
                    rc = store.register_buffer(ptr, storage.nbytes())
                except Exception:
                    if tensor.is_cuda:
                        raise
                else:
                    if rc is None or int(rc) == 0:
                        registered.append(ptr)
                    elif tensor.is_cuda:
                        raise RuntimeError(
                            f"GPU capture buffer registration failed ({rc}); "
                            "use RDMA-registerable CUDA memory and disable "
                            "PyTorch expandable_segments for this server"
                        )
            register_ms = (time.perf_counter() - register_started) * 1000.0
            put_started = time.perf_counter()
            batch_put = getattr(store, "batch_put_from", None)
            if batch_put is None:
                statuses = [
                    store.put_from(key, tensor.data_ptr(), nbytes, self._put_config)
                    for key, tensor, nbytes in zip(keys, tensors, sizes)
                ]
            else:
                statuses = batch_put(
                    keys,
                    [tensor.data_ptr() for tensor in tensors],
                    sizes,
                    self._put_config,
                )
            put_ms = (time.perf_counter() - put_started) * 1000.0
        except Exception:
            self._remove_many_quiet(keys)
            raise
        finally:
            for ptr in registered:
                try:
                    store.unregister_buffer(ptr)
                except Exception:
                    pass

        if statuses is None:
            statuses = [0] * len(keys)
        if len(statuses) != len(keys):
            self._remove_many_quiet(keys)
            raise RuntimeError(
                f"spec-capture batch_put_from returned {len(statuses)} statuses "
                f"for {len(keys)} keys"
            )
        failed = [
            (key, status)
            for key, status in zip(keys, statuses)
            if status is not None and int(status) < 0
        ]
        if failed:
            self._remove_many_quiet(keys)
            raise RuntimeError(
                f"spec-capture batch_put_from failed for {len(failed)}/{len(keys)} "
                f"keys; first={failed[0]}"
            )

        if timing_enabled:
            logger.info(
                "[spec-capture-timing] batch_sink samples=%d objects=%d bytes=%d "
                "materialize_ms=%.3f register_ms=%.3f put_ms=%.3f total_ms=%.3f",
                len(samples),
                len(keys),
                sum(sizes),
                materialize_ms,
                register_ms,
                put_ms,
                (time.perf_counter() - started) * 1000.0,
            )
        return results
