"""NCCL-based tensor transport for GPU-to-GPU transfer.

Replaces the unreliable CUDA IPC path with NCCL point-to-point send/recv.
NCCL provides built-in synchronization semantics and works reliably across
all GPU topologies (NVLink, PCIe, etc.) without the memory coherence issues
that plague CUDA IPC on multi-GPU TP configurations.

Architecture
------------
  - Server (target model, rank=0 in the NCCL data-transfer group)
  - Client (training GPU, rank=1 in the NCCL data-transfer group)

The NCCL group is established once via TCP rendezvous and reused for all
subsequent requests.  HTTP remains the control plane (input_ids up, metadata
down); NCCL is the data plane for large tensors.

Environment variables
---------------------
  SPECFORGE_ENABLE_NCCL : str
      "1" (default) to enable NCCL transport.  "0" to disable, falling back
      to POSIX SHM.
  SPECFORGE_NCCL_PORT : str
      TCP port for NCCL rendezvous (default: HTTP port + 100).
"""

import json
import logging
import os
import threading
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from ._tensor_wire import _DTYPE_TABLE, _DTYPE_TO_CODE

logger = logging.getLogger(__name__)


# dtypes that NCCL does NOT support for P2P send/recv.
# These are transmitted as uint8 views (raw bytes).
_NCCL_UNSUPPORTED_DTYPES = {torch.int16, torch.int8, torch.bool}

# Element size for NCCL-unsupported dtypes (avoids allocating a temp tensor).
_ELEMENT_SIZE = {torch.int16: 2, torch.int8: 1, torch.bool: 1}


class NCCLTransport:
    """Manages a dedicated NCCL process group for data transfer between
    the target model server (rank 0) and the training client (rank 1).

    Parameters
    ----------
    nccl_port : int
        TCP port for the rendezvous store.
    host : str
        Hostname/IP for the rendezvous store (server's address).
    is_server : bool
        True for the server side (rank 0), False for client (rank 1).
    """

    def __init__(self, nccl_port: int, host: str, is_server: bool):
        self._nccl_port = nccl_port
        self._host = host
        self._is_server = is_server
        self._rank = 0 if is_server else 1
        self._pg: Optional[dist.ProcessGroup] = None
        self._initialized = False
        self._init_lock = threading.Lock()
        # Unique group name per port to avoid collisions across tests/instances
        self._group_name = f"specforge_nccl_data_transfer_{nccl_port}"

    @property
    def is_initialized(self) -> bool:
        return self._initialized

    def initialize(self, timeout_seconds: int = 120) -> bool:
        """Establish the NCCL process group via TCP rendezvous.

        This blocks until both server and client have connected.

        We create the TCPStore explicitly (bypassing torchelastic's agent
        store logic which makes rank 0 a client too) and pass it directly
        to init_custom_process_group via the ``store`` parameter.

        Returns True on success, False on failure.
        """
        with self._init_lock:
            if self._initialized:
                return True

            try:
                from sglang.srt.utils.common import init_custom_process_group
                from torch.distributed import TCPStore

                timeout = torch.distributed.distributed_c10d.timedelta(
                    seconds=timeout_seconds
                )

                logger.info(
                    "NCCL transport initializing: rank=%d, host=%s, port=%d, timeout=%ds",
                    self._rank,
                    self._host,
                    self._nccl_port,
                    timeout_seconds,
                )

                # Create TCPStore explicitly: rank 0 (server) is the master
                # that listens; rank 1 (client) connects.
                # This bypasses torchelastic's _torchelastic_use_agent_store()
                # which would make ALL ranks connect (breaking standalone use).
                is_master = (self._rank == 0)
                store = TCPStore(
                    host_name=self._host,
                    port=self._nccl_port,
                    world_size=2,
                    is_master=is_master,
                    timeout=timeout,
                    multi_tenant=True,
                )

                self._pg = init_custom_process_group(
                    backend="nccl",
                    store=store,
                    world_size=2,
                    rank=self._rank,
                    group_name=self._group_name,
                    timeout=timeout,
                )

                self._initialized = True
                logger.info(
                    "NCCL transport initialized successfully (rank=%d)", self._rank
                )
                return True

            except Exception as exc:
                logger.error(
                    "NCCL transport initialization failed (rank=%d): %s",
                    self._rank,
                    exc,
                )
                self._pg = None
                self._initialized = False
                return False

    def send_tensors(
        self, tensor_dict: Dict[str, Optional[torch.Tensor]], keys_order: List[str]
    ) -> None:
        """Send tensors to the peer (client) via NCCL send.

        Only tensors that are not None and are on CUDA are sent.
        The caller must ensure keys_order matches what the receiver expects.

        Unsupported dtypes (int16, int8, bool) are viewed as uint8 for transfer.
        """
        assert self._initialized and self._pg is not None, (
            "NCCL transport not initialized"
        )
        assert self._is_server, "Only the server can send tensors"

        for key in keys_order:
            tensor = tensor_dict.get(key)
            if tensor is None:
                continue
            if not tensor.is_cuda:
                tensor = tensor.cuda()
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            # NCCL doesn't support int16/int8/bool — view as raw uint8 bytes
            if tensor.dtype in _NCCL_UNSUPPORTED_DTYPES:
                tensor = tensor.view(torch.uint8)
            dist.send(tensor, dst=1, group=self._pg)

    def recv_tensors(
        self, metadata: Dict[str, dict], keys_order: List[str]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """Receive tensors from the peer (server) via NCCL recv.

        Parameters
        ----------
        metadata : dict
            Mapping of key -> {"dtype_code": int, "shape": list} for each
            tensor to receive.  Keys with value None are skipped.
        keys_order : list
            Order in which tensors are received (must match server send order).

        Returns
        -------
        dict of tensors, already on the current CUDA device.
        """
        assert self._initialized and self._pg is not None, (
            "NCCL transport not initialized"
        )
        assert not self._is_server, "Only the client can receive tensors"

        result: Dict[str, Optional[torch.Tensor]] = {}

        for key in keys_order:
            meta = metadata.get(key)
            if meta is None:
                result[key] = None
                continue

            dtype = _DTYPE_TABLE[meta["dtype_code"]]
            shape = tuple(meta["shape"])

            if dtype in _NCCL_UNSUPPORTED_DTYPES:
                # Receive as uint8 then view back to original dtype
                numel = 1
                for s in shape:
                    numel *= s
                nbytes = numel * _ELEMENT_SIZE[dtype]
                buf = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
                dist.recv(buf, src=0, group=self._pg)
                result[key] = buf.view(dtype).reshape(shape)
            else:
                buf = torch.empty(shape, dtype=dtype, device="cuda")
                dist.recv(buf, src=0, group=self._pg)
                result[key] = buf

        return result

    def destroy(self) -> None:
        """Mark the NCCL process group as destroyed.

        Note: We intentionally skip dist.destroy_process_group() because it
        requires both sides to participate (blocking), which causes hangs when
        client finishes before server. The NCCL communicator is cleaned up
        automatically when the process exits.

        We also unregister the PG from torch.distributed's internal tracking
        to prevent PyTorch's atexit hooks from calling destroy on it (which
        would also block).
        """
        if self._pg is not None:
            try:
                from torch.distributed.distributed_c10d import _world
                _world.pg_group_ranks.pop(self._pg, None)
            except Exception:
                pass
        self._pg = None
        self._initialized = False


def encode_nccl_metadata(
    tensor_dict: Dict[str, Optional[torch.Tensor]], keys_order: List[str],
    cpu_scalars: Optional[Dict[str, list]] = None,
) -> bytes:
    """Encode tensor metadata (dtype, shape) as JSON for HTTP response.

    This is sent over HTTP so the client knows what to expect from NCCL recv.
    Only metadata — no tensor data bytes.

    Parameters
    ----------
    cpu_scalars : dict, optional
        Small CPU tensors serialized as Python lists (included in JSON).
    """
    metadata = {}
    for key in keys_order:
        tensor = tensor_dict.get(key)
        if tensor is None:
            metadata[key] = None
        else:
            metadata[key] = {
                "dtype_code": _DTYPE_TO_CODE[tensor.dtype],
                "shape": list(tensor.shape),
            }

    payload = {
        "keys_order": keys_order,
        "metadata": metadata,
    }
    if cpu_scalars:
        payload["cpu_scalars"] = cpu_scalars
    return json.dumps(payload).encode("utf-8")


def decode_nccl_metadata(raw: bytes) -> Tuple[List[str], Dict[str, Optional[dict]], Dict[str, list]]:
    """Decode NCCL metadata from HTTP response body.

    Returns (keys_order, metadata_dict, cpu_scalars) where metadata_dict maps
    key to {"dtype_code": int, "shape": list} or None, and cpu_scalars maps
    key to a Python list (small values sent inline in JSON).
    """
    payload = json.loads(raw.decode("utf-8"))
    keys_order = payload["keys_order"]
    metadata = payload["metadata"]
    cpu_scalars = payload.get("cpu_scalars", {})
    return keys_order, metadata, cpu_scalars
