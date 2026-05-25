"""Custom binary tensor serialisation for SpecForge remote training.

Compact wire format that encodes dtype, shape, and raw contiguous bytes,
suitable for high-throughput tensor transport over HTTP or shared memory.

Format (little-endian unless noted)
------------------------------------
  [4B]  magic         0x53504647  ("SPFG")
Per entry:
  [4B]  key_len       (uint32)
  [key_len B] key      UTF-8
  [1B]  flags         bit0 = is_none
If not none:
  [1B]  dtype_code    (see _DTYPE_TABLE)
  [1B]  ndim
  [ndim × 8B] shape   (int64)
  [8B]  nbytes        (uint64)
  [nbytes B] data      raw contiguous tensor bytes

The format is self-delimiting: a consumer can skip entries whose keys it
does not recognise.
"""

import ctypes
import struct
from typing import Dict, Optional

import torch

MAGIC = 0x53504647  # "SPFG"

# dtype_code → (torch.dtype, element_size)
_DTYPE_TABLE: Dict[int, torch.dtype] = {
    0: torch.float32,
    1: torch.float64,
    2: torch.float16,
    3: torch.bfloat16,
    4: torch.int64,
    5: torch.int32,
    6: torch.int16,
    7: torch.int8,
    8: torch.uint8,
    9: torch.bool,
}
_DTYPE_TO_CODE = {dt: c for c, dt in _DTYPE_TABLE.items()}

# struct formats (little-endian)
_HEADER_FMT = struct.Struct("<I")         # magic
_ENTRY_KEYLEN_FMT = struct.Struct("<I")   # key_len
_FLAG_FMT = struct.Struct("<B")           # flags
_DTYPE_FMT = struct.Struct("<B")          # dtype_code
_NDIM_FMT = struct.Struct("<B")           # ndim
_SHAPE_FMT = struct.Struct("<q")          # single int64
_NBYTES_FMT = struct.Struct("<Q")         # uint64

_FLAG_NONE = 0x01


def encode(tensor_dict: Dict[str, Optional[torch.Tensor]]) -> bytearray:
    """Encode a dict of tensors into the wire format.

    ``None`` values are preserved (the consumer receives ``None`` for that key).
    Tensors are serialised in their current dtype and device – the caller is
    responsible for moving them to CPU / pinning memory as needed.
    """
    # Pre-calculate total size for efficient allocation
    total_size = _HEADER_FMT.size
    entries = []
    for key, tensor in tensor_dict.items():
        key_bytes = key.encode("utf-8")
        entry_size = _ENTRY_KEYLEN_FMT.size + len(key_bytes) + _FLAG_FMT.size
        if tensor is None:
            entries.append((key_bytes, None, 0))
        else:
            if not tensor.is_contiguous():
                tensor = tensor.contiguous()
            if tensor.is_cuda:
                raise ValueError("wire encoding expects CPU tensors, got CUDA tensor")
            dt = tensor.dtype
            code = _DTYPE_TO_CODE.get(dt)
            if code is None:
                raise TypeError(f"Unsupported dtype for wire encoding: {dt}")
            nbytes = tensor.numel() * tensor.element_size()
            entry_size += _DTYPE_FMT.size + _NDIM_FMT.size + tensor.ndim * _SHAPE_FMT.size + _NBYTES_FMT.size + nbytes
            entries.append((key_bytes, tensor, nbytes))
        total_size += entry_size

    buf = bytearray(total_size)
    pos = 0

    _HEADER_FMT.pack_into(buf, pos, MAGIC)
    pos += _HEADER_FMT.size

    for key_bytes, tensor, nbytes in entries:
        _ENTRY_KEYLEN_FMT.pack_into(buf, pos, len(key_bytes))
        pos += _ENTRY_KEYLEN_FMT.size
        buf[pos:pos + len(key_bytes)] = key_bytes
        pos += len(key_bytes)

        if tensor is None:
            _FLAG_FMT.pack_into(buf, pos, _FLAG_NONE)
            pos += _FLAG_FMT.size
            continue

        _FLAG_FMT.pack_into(buf, pos, 0)
        pos += _FLAG_FMT.size

        dt = tensor.dtype
        code = _DTYPE_TO_CODE[dt]
        _DTYPE_FMT.pack_into(buf, pos, code)
        pos += _DTYPE_FMT.size

        ndim = tensor.ndim
        _NDIM_FMT.pack_into(buf, pos, ndim)
        pos += _NDIM_FMT.size
        for s in tensor.shape:
            _SHAPE_FMT.pack_into(buf, pos, s)
            pos += _SHAPE_FMT.size

        _NBYTES_FMT.pack_into(buf, pos, nbytes)
        pos += _NBYTES_FMT.size

        # Fast bulk copy via ctypes memmove (avoids byte-by-byte Python loop)
        ctypes.memmove(
            (ctypes.c_char * nbytes).from_buffer(buf, pos),
            tensor.data_ptr(),
            nbytes,
        )
        pos += nbytes

    return buf


def decode(
    raw: bytes,
    map_location: str = "cpu",
    device: Optional[torch.device] = None,
) -> Dict[str, Optional[torch.Tensor]]:
    """Decode a wire-format blob back into a dict of tensors.

    Parameters
    ----------
    raw : bytes
        The wire-format payload.
    map_location : str
        If ``"cpu"`` the tensor storage is created on CPU and the returned
        tensor shares that storage (no copy).
    device : torch.device, optional
        Explicit target device.  Overrides *map_location* when given.
    """
    if device is not None:
        target_device = device
        map_location = str(device)
    elif map_location == "cpu":
        target_device = torch.device("cpu")
    else:
        target_device = torch.device(map_location)

    mv = memoryview(raw)
    pos = 0

    magic = _HEADER_FMT.unpack_from(mv, pos)[0]
    pos += _HEADER_FMT.size
    if magic != MAGIC:
        raise ValueError(
            f"Bad wire format magic: 0x{magic:08x} (expected 0x{MAGIC:08x})"
        )

    result: Dict[str, Optional[torch.Tensor]] = {}

    while pos < len(mv):
        key_len = _ENTRY_KEYLEN_FMT.unpack_from(mv, pos)[0]
        pos += _ENTRY_KEYLEN_FMT.size
        key = bytes(mv[pos : pos + key_len]).decode("utf-8")
        pos += key_len

        flags = _FLAG_FMT.unpack_from(mv, pos)[0]
        pos += _FLAG_FMT.size

        if flags & _FLAG_NONE:
            result[key] = None
            continue

        code = _DTYPE_FMT.unpack_from(mv, pos)[0]
        pos += _DTYPE_FMT.size
        dt = _DTYPE_TABLE.get(code)
        if dt is None:
            raise ValueError(f"Unknown dtype code: {code}")

        ndim = _NDIM_FMT.unpack_from(mv, pos)[0]
        pos += _NDIM_FMT.size

        shape = []
        for _ in range(ndim):
            shape.append(_SHAPE_FMT.unpack_from(mv, pos)[0])
            pos += _SHAPE_FMT.size
        shape = tuple(shape)

        nbytes = _NBYTES_FMT.unpack_from(mv, pos)[0]
        pos += _NBYTES_FMT.size

        # Build tensor from raw bytes — use torch.frombuffer for all dtypes
        # (including bf16) to avoid extra numpy copy.
        storage = torch.frombuffer(bytearray(mv[pos : pos + nbytes]), dtype=torch.uint8)
        tensor = storage.view(dt).reshape(shape)
        if target_device.type != "cpu":
            tensor = tensor.to(target_device)

        result[key] = tensor
        pos += nbytes

    return result


def encode_to_buffer(tensor_dict: Dict[str, Optional[torch.Tensor]]) -> bytes:
    """Convenience: encode and return immutable ``bytes``."""
    return bytes(encode(tensor_dict))
