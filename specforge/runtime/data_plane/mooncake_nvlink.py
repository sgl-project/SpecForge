# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake object API over multi-node NVLink for server-owned captures.

With ``MOONCAKE_PROTOCOL=nvlink`` the patched capture server keeps every
feature object in a fabric-memory arena on its writer GPU, because Mooncake's
NVLink transport can only export such memory and the Mooncake store cannot
place objects there. The server returns each object's device address, which
:class:`SGLangServerCaptureAdapter` stores in ``SampleRef.metadata``.

:class:`NvlinkObjectClient` implements the part of the Mooncake store API that
:class:`MooncakeFeatureStore` uses, so the store's lifecycle (generations,
leases, removal retries) is unchanged: ``get_into`` reads straight from the
server GPU with ``TransferEngine.transfer_sync_read`` and ``remove`` frees the
object through the server's control endpoint. Objects are located per ref via
:meth:`locate_ref_objects`, which the feature store calls on ``adopt``/``get``.
"""

from __future__ import annotations

import os
import threading
from typing import Any, Dict, Set, Tuple

from specforge.runtime.data_plane.mooncake_store import MOONCAKE_OBJECT_NOT_FOUND

#: ``SampleRef.metadata`` entry holding ``{"session", "control", "addresses"}``.
REF_METADATA_KEY = "mooncake_nvlink"
_FREE_TIMEOUT_S = 30.0


class NvlinkObjectClient:
    """Mooncake store API backed by MNNVL reads and server-side frees."""

    def __init__(self, *, local_hostname: str) -> None:
        self._local_hostname = local_hostname
        self._engine = None
        self._http = None
        # key -> (transfer-engine session, device address, control endpoint)
        self._objects: Dict[str, Tuple[str, int, str]] = {}
        self._controls: Set[str] = set()
        self._lock = threading.Lock()
        self._http_lock = threading.Lock()

    def locate_ref_objects(
        self, keys: Dict[str, str], metadata: Dict[str, Any]
    ) -> None:
        """Record where a ref's objects live; ``keys`` maps feature name -> key."""
        info = metadata.get(REF_METADATA_KEY)
        if info is None:
            return
        addresses = info["addresses"]
        with self._lock:
            self._controls.add(info["control"])
            for name, key in keys.items():
                if name in addresses:
                    self._objects[key] = (
                        info["session"],
                        int(addresses[name]),
                        info["control"],
                    )

    # -- the Mooncake store API subset MooncakeFeatureStore uses ---------------
    def is_exist(self, key: str) -> int:
        with self._lock:
            return int(key in self._objects)

    def get_into(self, key: str, ptr: int, nbytes: int) -> int:
        with self._lock:
            entry = self._objects.get(key)
        if entry is None:
            return MOONCAKE_OBJECT_NOT_FOUND
        if nbytes == 0:
            return 0
        session, address, _ = entry
        rc = int(
            self._transfer_engine().transfer_sync_read(session, ptr, address, nbytes)
        )
        return nbytes if rc == 0 else min(rc, -1)

    def remove(self, key: str, force: bool = False) -> int:
        with self._lock:
            entry = self._objects.pop(key, None)
            # A key no response located (a lost capture response) may live on
            # any server this client has seen.
            controls = [entry[2]] if entry is not None else sorted(self._controls)
        try:
            for control in controls:
                self._post(f"{control}/free", {"keys": [key]})
        except Exception:
            if entry is not None:
                with self._lock:
                    self._objects.setdefault(key, entry)
            return -1
        return 0

    def put_from(self, key: str, ptr: int, nbytes: int, config: Any = None) -> int:
        raise RuntimeError(
            "MOONCAKE_PROTOCOL=nvlink objects are written by the capture server"
        )

    def register_buffer(self, ptr: int, nbytes: int) -> int:
        return 0  # NVLink reads into ordinary device memory

    def unregister_buffer(self, ptr: int) -> int:
        return 0

    def close(self) -> int:
        return 0

    # -- transports ------------------------------------------------------------
    def _transfer_engine(self):
        with self._lock:
            if self._engine is None:
                from mooncake import engine as mooncake_engine

                if not getattr(mooncake_engine, "SUPPORT_MNNVL", False):
                    raise RuntimeError(
                        "MOONCAKE_PROTOCOL=nvlink needs a Mooncake build with "
                        "USE_MNNVL (the aarch64 CUDA wheels)"
                    )
                if os.environ.get("MC_FORCE_MNNVL") is None:
                    raise RuntimeError(
                        "MOONCAKE_PROTOCOL=nvlink needs MC_FORCE_MNNVL so Mooncake "
                        "selects NVLink over the RDMA NICs"
                    )
                engine = mooncake_engine.TransferEngine()
                rc = engine.initialize(
                    self._local_hostname, "P2PHANDSHAKE", "nvlink", ""
                )
                if int(rc) != 0:
                    raise RuntimeError(f"Mooncake NVLink transfer engine failed ({rc})")
                self._engine = engine
            return self._engine

    def _post(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        import requests

        with self._http_lock:
            if self._http is None:
                self._http = requests.Session()
            response = self._http.post(url, json=payload, timeout=_FREE_TIMEOUT_S)
        response.raise_for_status()
        return response.json()
