# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Mooncake object API for captures kept in a capture server's arena.

A capture server can keep feature objects in an arena it owns instead of the
Mooncake store. Currently only online NVLink capture (``MOONCAKE_PROTOCOL=
nvlink``) uses the arena: Mooncake's NVLink transport can only export fabric
memory, and the store cannot place objects there. The server returns each
object's address, which :class:`SGLangServerCaptureAdapter` stores in
``SampleRef.metadata``.

:class:`ArenaObjectClient` implements the part of the Mooncake store API that
:class:`MooncakeFeatureStore` uses, so the store's lifecycle (generations,
leases, removal retries) is unchanged: ``get_into`` reads straight from the
server with ``TransferEngine.transfer_sync_read`` and ``remove`` frees the
object by writing its key to the server's control connection. Frees are one-way
messages the server applies before it next allocates, so the durable ack never
waits on the capture server. Objects are located per ref via
:meth:`locate_ref_objects`, which the feature store calls on ``adopt``/``get``.
Before submitting capture, ``/server_info`` resolves the cleanup endpoints.
Every attempted endpoint travels with the final ref so a trainer also frees
copies left by a lost response followed by a retry on another server.
"""

from __future__ import annotations

import json
import os
import socket
import threading
from typing import Any, Dict, List, Set, Tuple

from specforge.runtime.data_plane.mooncake_store import MOONCAKE_OBJECT_NOT_FOUND

#: Ref metadata: session, read addresses, and all attempted control endpoints.
REF_METADATA_KEY = "mooncake_arena"
_CONNECT_TIMEOUT_S = 30.0


class ArenaObjectClient:
    """Mooncake store API backed by reads from capture-server arenas."""

    def __init__(self, *, local_hostname: str) -> None:
        self._local_hostname = local_hostname
        self._engine = None
        # key -> (transfer-engine session, device address, control endpoint)
        self._objects: Dict[str, Tuple[str, int, str]] = {}
        # Every arena that may hold this key, including failed capture attempts.
        self._object_controls: Dict[str, Set[str]] = {}
        self._server_controls: Dict[str, Set[str]] = {}
        self._controls: Set[str] = set()
        self._connections: Dict[str, socket.socket] = {}
        self._lock = threading.Lock()
        self._send_lock = threading.Lock()

    def track_capture_attempt(self, keys: List[str], server_url: str) -> None:
        """Resolve cleanup destinations before a request can write any objects."""
        with self._lock:
            controls = self._server_controls.get(server_url)
        if controls is None:
            import requests

            response = requests.get(
                f"{server_url.rstrip('/')}/server_info", timeout=_CONNECT_TIMEOUT_S
            )
            response.raise_for_status()
            controls = {
                state["spec_capture_control"]
                for state in response.json().get("internal_states", [])
                if state.get("spec_capture_control")
            }
            if not controls:
                raise RuntimeError(
                    f"{server_url} does not advertise spec_capture_control; "
                    "apply the current SGLang capture patch before NVLink capture"
                )
        with self._lock:
            self._server_controls[server_url] = controls
            self._controls.update(controls)
            for key in keys:
                self._object_controls.setdefault(key, set()).update(controls)

    def locate_ref_objects(
        self, keys: Dict[str, str], metadata: Dict[str, Any]
    ) -> None:
        """Record where a ref's objects live; ``keys`` maps feature name -> key."""
        info = metadata.get(REF_METADATA_KEY)
        if info is None:
            return
        addresses = info["addresses"]
        with self._lock:
            controls = set(info.get("controls", [])) | {info["control"]}
            for key in keys.values():
                controls.update(self._object_controls.get(key, ()))
            # Preserve attempt locations across the producer -> trainer boundary.
            info["controls"] = sorted(controls)
            self._controls.update(controls)
            for name, key in keys.items():
                self._object_controls.setdefault(key, set()).update(controls)
                if name in addresses:
                    self._objects[key] = (
                        info["session"],
                        int(addresses[name]),
                        info["control"],
                    )

    # -- the Mooncake store API subset MooncakeFeatureStore uses ---------------
    def is_exist(self, key: str) -> int:
        # A pending attempt may exist remotely even without a read address.
        # Never let the feature store treat its failed removal as absence.
        with self._lock:
            return int(key in self._objects or key in self._object_controls)

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
            entry = self._objects.get(key)
            controls = set(self._object_controls.get(key, ()))
            if entry is not None:
                controls.add(entry[2])
            if not controls:
                controls.update(self._controls)
            # Unknown is not confirmed absent. Keep it pending even if no
            # destination is known, and preserve every destination on failure.
            self._object_controls[key] = controls
        if not controls:
            return -1
        try:
            for control in sorted(controls):
                self._send_free(control, [key])
        except OSError:
            return -1
        with self._lock:
            self._objects.pop(key, None)
            self._object_controls.pop(key, None)
        return 0

    def put_from(self, key: str, ptr: int, nbytes: int, config: Any = None) -> int:
        raise RuntimeError("arena objects are written by the capture server")

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
                # Currently only online NVLink capture uses the arena.
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
