"""Mooncake writer for TPU spec capture, in MooncakeFeatureStore's layout.

Keys are {store_id}/{sample_id}/g{gen}/{name}, one hard-pinned raw-bytes object
per tensor; shapes and dtypes travel in the returned result dict.
"""

from __future__ import annotations

import logging
import os
import threading
from typing import Any

import torch

logger = logging.getLogger(__name__)

_DTYPE_STR = {
    torch.float32: "float32",
    torch.bfloat16: "bfloat16",
    torch.float16: "float16",
    torch.int64: "int64",
    torch.int32: "int32",
}
_STR_DTYPE = {v: k for k, v in _DTYPE_STR.items()}


class CaptureSink:
    def __init__(self, aux_layer_ids: list[int]) -> None:
        self.aux_layer_ids = aux_layer_ids
        self._store: Any = None
        self._put_config: Any = None
        self._lock = threading.Lock()

    def _connect(self) -> Any:
        with self._lock:
            if self._store is not None:
                return self._store
            from mooncake.store import MooncakeDistributedStore, ReplicateConfig

            store = MooncakeDistributedStore()
            env = os.environ
            rc = store.setup(
                local_hostname=env.get("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
                metadata_server=env.get(
                    "MOONCAKE_METADATA_SERVER", "http://localhost:8080/metadata"
                ),
                global_segment_size=int(
                    env.get("MOONCAKE_GLOBAL_SEGMENT_SIZE", 1 << 30)
                ),
                local_buffer_size=int(env.get("MOONCAKE_LOCAL_BUFFER_SIZE", 1 << 30)),
                protocol=env.get("MOONCAKE_PROTOCOL", "tcp"),
                rdma_devices=env.get("MOONCAKE_RDMA_DEVICES", ""),
                master_server_addr=env.get(
                    "MOONCAKE_MASTER_SERVER_ADDR", "localhost:50051"
                ),
            )
            if rc is not None and int(rc) != 0:
                raise RuntimeError(f"spec-capture mooncake setup failed (status {rc})")
            config = ReplicateConfig()
            config.replica_num = 1
            # SpecForge, not Mooncake's LRU, owns object lifetime.
            config.with_hard_pin = True
            self._put_config = config
            self._store = store
            logger.info("spec-capture mooncake sink connected")
            return store

    def put_samples(
        self, samples: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]]
    ) -> list[dict[str, Any]]:
        """Publish (spec, aux [L, n*H], last_hidden [L, H]) CPU tensors."""
        store = self._connect()
        keys: list[str] = []
        tensors: list[torch.Tensor] = []
        replace_keys: list[str] = []
        results: list[dict[str, Any]] = []
        for spec, aux, last_hidden in samples:
            store_id, sample_id = str(spec["store_id"]), str(spec["sample_id"])
            gen = int(spec.get("gen", 1))
            named = dict(spec.get("features") or {})
            staged: dict[str, torch.Tensor] = {}
            if "aux" in named:
                staged[named["aux"]] = aux.unsqueeze(0)
            if "last_hidden" in named:
                staged[named["last_hidden"]] = last_hidden.unsqueeze(0)
            for item in spec.get("passthrough") or []:
                dtype = _STR_DTYPE[str(item.get("dtype", "int64"))]
                staged[str(item["name"])] = torch.tensor(
                    item["data"], dtype=dtype
                ).reshape([int(d) for d in item["shape"]])
            features = {}
            for name, tensor in staged.items():
                key = f"{store_id}/{sample_id}/g{gen}/{name}"
                keys.append(key)
                tensors.append(tensor.contiguous())
                if spec.get("replace"):
                    replace_keys.append(key)
                features[name] = {
                    "shape": list(tensor.shape),
                    "dtype": _DTYPE_STR[tensor.dtype],
                }
            results.append(
                {
                    "sample_id": sample_id,
                    "store_id": store_id,
                    "gen": gen,
                    "aux_layer_ids": self.aux_layer_ids,
                    "features": features,
                }
            )

        for key in replace_keys:
            store.remove(key)
        sizes = [t.element_size() * t.numel() for t in tensors]
        registered = []
        try:
            for tensor, size in zip(tensors, sizes):
                if store.register_buffer(tensor.data_ptr(), size) == 0:
                    registered.append(tensor.data_ptr())
            statuses = store.batch_put_from(
                keys, [t.data_ptr() for t in tensors], sizes, self._put_config
            )
        finally:
            for ptr in registered:
                store.unregister_buffer(ptr)
        failed = [(k, s) for k, s in zip(keys, statuses) if s is not None and s < 0]
        if failed:
            for key in keys:
                store.remove(key)
            raise RuntimeError(
                f"spec-capture batch_put_from failed for {len(failed)}/{len(keys)} "
                f"keys; first={failed[0]}"
            )
        return results
