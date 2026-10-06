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
object by writing its key to the server's control connection. Frees are one-way
messages the server applies before it next allocates, so the durable ack never
waits on the capture server. Objects are located per ref via
:meth:`locate_ref_objects`, which the feature store calls on ``adopt``/``get``.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import Any, Dict, List, Set, Tuple

from specforge.runtime.data_plane.mooncake_store import MOONCAKE_OBJECT_NOT_FOUND

#: ``SampleRef.metadata`` entry holding ``{"session", "control", "addresses"}``.
REF_METADATA_KEY = "mooncake_nvlink"
_CONNECT_TIMEOUT_S = 30.0


class NvlinkObjectClient:
    """Mooncake store API backed by MNNVL reads and server-side frees."""

    def __init__(self, *, local_hostname: str) -> None:
        self._local_hostname = local_hostname
        self._engine = None
        # key -> (transfer-engine session, device address, control endpoint)
        self._objects: Dict[str, Tuple[str, int, str]] = {}
        self._controls: Set[str] = set()
        self._connections: Dict[str, socket.socket] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()

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
                self._send_free(control, [key])
        except OSError:
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

    def _send_free(self, control: str, keys: List[str]) -> None:
        line = (json.dumps(keys) + "\n").encode()
        with self._send_lock:
            connection = self._connections.get(control)
            if connection is None:
                host, port = control.rsplit(":", 1)
                connection = socket.create_connection(
                    (host, int(port)), timeout=_CONNECT_TIMEOUT_S
                )
                self._connections[control] = connection
            try:
                connection.sendall(line)
            except OSError:
                # Reconnect on the next free; the caller retries this one.
                self._connections.pop(control).close()
                raise
