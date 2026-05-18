"""Shared-memory tensor transport replacing torch.save/load for remote backends.

Instead of serialising large tensors via pickle we copy CPU tensors into
POSIX shared memory and pass only metadata (name, shape, dtype) over HTTP.
The peer reconstructs tensors from the shared memory blocks, then unlinks
them.  Only activated when server and client are on the same machine
(matching NVIDIA base-image environments where /dev/shm is available).

V2: Uses compact binary metadata (no pickle) and ctypes memmove for
zero-copy server → shared-memory writes.
"""

import ctypes
import logging
import struct
from multiprocessing.shared_memory import SharedMemory
from typing import Dict, Optional

import torch

from ._tensor_wire import _DTYPE_TABLE, _DTYPE_TO_CODE

logger = logging.getLogger(__name__)

# HTTP header used by the client to signal "I can use shared memory".
SHM_HEADER = "X-SpecForge-Shm-Enabled"

# Binary metadata format (per entry):
#   [2B]  key_len (uint16)
#   [K B] key (UTF-8)
#   [1B]  flags (0=tensor, 1=none)
# If tensor:
#   [1B]  dtype_code
#   [1B]  ndim
#   [ndim×8B] shape (int64 each)
#   [8B]  nbytes (uint64)
#   [2B]  shm_name_len (uint16)
#   [S B] shm_name (UTF-8)

_MAGIC = 0x53484D32  # "SHM2"


def pack_response(tensor_dict: dict, use_shm: bool) -> bytes:
    """Serialise *tensor_dict* for HTTP response.

    With *use_shm* tensors go into POSIX shared memory and only compact
    binary metadata is transmitted over HTTP.
    """
    if not use_shm:
        # Fallback: wire format (imported here to avoid circular)
        from . import _tensor_wire as _wire
        return _wire.encode_to_buffer(tensor_dict)

    buf = bytearray()
    buf += struct.pack("<I", _MAGIC)

    for key, tensor in tensor_dict.items():
        key_bytes = key.encode("utf-8")
        buf += struct.pack("<H", len(key_bytes))
        buf += key_bytes

        if tensor is None or not isinstance(tensor, torch.Tensor):
            buf += struct.pack("<B", 1)  # flag=none
            continue

        buf += struct.pack("<B", 0)  # flag=tensor

        dt = tensor.dtype
        code = _DTYPE_TO_CODE.get(dt)
        if code is None:
            raise TypeError(f"Unsupported dtype for SHM: {dt}")
        buf += struct.pack("<B", code)

        ndim = tensor.ndim
        buf += struct.pack("<B", ndim)
        for s in tensor.shape:
            buf += struct.pack("<q", s)

        if not tensor.is_contiguous():
            tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        buf += struct.pack("<Q", nbytes)

        # Create shared memory and copy tensor data directly via memmove
        shm = SharedMemory(create=True, size=max(nbytes, 1))
        ctypes.memmove(
            ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))),
            ctypes.c_void_p(tensor.data_ptr()),
            nbytes,
        )
        shm_name = shm.name
        shm.close()

        shm_name_bytes = shm_name.encode("utf-8")
        buf += struct.pack("<H", len(shm_name_bytes))
        buf += shm_name_bytes

    return bytes(buf)


def unpack_response(raw: bytes, shm_enabled: bool, map_location="cpu") -> dict:
    """Deserialise a response produced by :func:`pack_response`."""
    if not shm_enabled:
        from . import _tensor_wire as _wire
        return _wire.decode(raw, map_location=map_location)

    mv = memoryview(raw)
    pos = 0

    magic = struct.unpack_from("<I", mv, pos)[0]
    pos += 4
    if magic != _MAGIC:
        raise ValueError(f"Bad SHM magic: 0x{magic:08x} (expected 0x{_MAGIC:08x})")

    result: Dict[str, Optional[torch.Tensor]] = {}

    while pos < len(mv):
        key_len = struct.unpack_from("<H", mv, pos)[0]
        pos += 2
        key = bytes(mv[pos:pos + key_len]).decode("utf-8")
        pos += key_len

        flags = struct.unpack_from("<B", mv, pos)[0]
        pos += 1

        if flags == 1:
            result[key] = None
            continue

        code = struct.unpack_from("<B", mv, pos)[0]
        pos += 1
        dt = _DTYPE_TABLE.get(code)
        if dt is None:
            raise ValueError(f"Unknown dtype code: {code}")

        ndim = struct.unpack_from("<B", mv, pos)[0]
        pos += 1
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from("<q", mv, pos)[0])
            pos += 8
        shape = tuple(shape)

        nbytes = struct.unpack_from("<Q", mv, pos)[0]
        pos += 8

        shm_name_len = struct.unpack_from("<H", mv, pos)[0]
        pos += 2
        shm_name = bytes(mv[pos:pos + shm_name_len]).decode("utf-8")
        pos += shm_name_len

        # Open shared memory and copy to tensor
        shm = SharedMemory(name=shm_name)
        # Create tensor directly from SHM buffer via ctypes memmove
        tensor = torch.empty(shape, dtype=dt)
        ctypes.memmove(
            ctypes.c_void_p(tensor.data_ptr()),
            ctypes.c_void_p(ctypes.addressof(ctypes.c_char.from_buffer(shm.buf))),
            nbytes,
        )
        shm.close()
        shm.unlink()

        if map_location != "cpu" and map_location is not None:
            tensor = tensor.to(map_location)
        result[key] = tensor

    return result
