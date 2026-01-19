"""
Zero-copy GPU buffer management for RDMA transfers.

This module provides pre-allocated GPU buffer pools that are registered with
Mooncake Store for zero-copy RDMA transfers, avoiding serialization overhead.

Memory Layout with Alignment:
    [Header (64 bytes, padded)] [hidden_states (aligned)] [target (aligned)] ...
    
Each tensor section is aligned to TENSOR_ALIGNMENT bytes for optimal RDMA performance.
"""

import logging
import struct
import threading
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch

logger = logging.getLogger(__name__)

CACHE_LINE_SIZE = 64
PAGE_SIZE = 4096
TENSOR_ALIGNMENT = 64

HEADER_FORMAT = "!6Q4s4sQ"
HEADER_SIZE = 64


def align_up(size: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    """Round up size to the next alignment boundary."""
    return (size + alignment - 1) & ~(alignment - 1)


def align_offset(offset: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    """Calculate padding needed to align offset."""
    remainder = offset % alignment
    return (alignment - remainder) % alignment


@dataclass
class TensorMetadata:
    """Metadata describing a tensor's layout in the buffer."""

    offset: int
    size: int
    shape: Tuple[int, ...]
    dtype: torch.dtype

    @staticmethod
    def dtype_to_bytes(dtype: torch.dtype) -> bytes:
        """Convert dtype to 4-byte identifier."""
        dtype_map = {
            torch.float32: b"fp32",
            torch.float16: b"fp16",
            torch.bfloat16: b"bf16",
            torch.int64: b"i64_",
            torch.int32: b"i32_",
            torch.bool: b"bool",
        }
        return dtype_map.get(dtype, b"unk_")

    @staticmethod
    def bytes_to_dtype(b: bytes) -> torch.dtype:
        """Convert 4-byte identifier to dtype."""
        dtype_map = {
            b"fp32": torch.float32,
            b"fp16": torch.float16,
            b"bf16": torch.bfloat16,
            b"i64_": torch.int64,
            b"i32_": torch.int32,
            b"bool": torch.bool,
        }
        return dtype_map.get(b, torch.float32)


@dataclass
class Eagle3BufferLayout:
    """
    Layout specification for Eagle3 output in a contiguous buffer.
    
    Buffer format:
    [Header (48 bytes)] [hidden_states] [target] [loss_mask] [input_ids] [attention_mask] [last_hidden_states?]
    
    Header format (packed):
    - hidden_states_size (8 bytes)
    - target_size (8 bytes)
    - loss_mask_size (8 bytes)
    - input_ids_size (8 bytes)
    - attention_mask_size (8 bytes)
    - last_hidden_states_size (8 bytes) - 0 if not present
    - hidden_states_dtype (4 bytes)
    - target_dtype (4 bytes)
    """

    batch_size: int
    seq_len: int
    hidden_dim: int
    vocab_size: int
    hidden_states_dtype: torch.dtype = torch.bfloat16
    target_dtype: torch.dtype = torch.bfloat16
    include_last_hidden_states: bool = False

    @property
    def hidden_states_size(self) -> int:
        elem_size = torch.tensor([], dtype=self.hidden_states_dtype).element_size()
        return self.batch_size * self.seq_len * self.hidden_dim * 3 * elem_size

    @property
    def target_size(self) -> int:
        elem_size = torch.tensor([], dtype=self.target_dtype).element_size()
        return self.batch_size * self.seq_len * self.vocab_size * elem_size

    @property
    def loss_mask_size(self) -> int:
        return self.batch_size * self.seq_len * torch.tensor([], dtype=torch.bool).element_size()

    @property
    def input_ids_size(self) -> int:
        return self.batch_size * self.seq_len * torch.tensor([], dtype=torch.int64).element_size()

    @property
    def attention_mask_size(self) -> int:
        return self.batch_size * self.seq_len * torch.tensor([], dtype=torch.int64).element_size()

    @property
    def last_hidden_states_size(self) -> int:
        if not self.include_last_hidden_states:
            return 0
        elem_size = torch.tensor([], dtype=self.hidden_states_dtype).element_size()
        return self.batch_size * self.seq_len * self.hidden_dim * elem_size

    @property
    def total_size(self) -> int:
        return (
            HEADER_SIZE
            + self.hidden_states_size
            + self.target_size
            + self.loss_mask_size
            + self.input_ids_size
            + self.attention_mask_size
            + self.last_hidden_states_size
        )


class GPUBuffer:
    """
    A pre-allocated GPU buffer that can be registered with Mooncake for RDMA.
    
    The buffer is allocated as a contiguous block of GPU memory that can be
    used for zero-copy transfers.
    """

    def __init__(
        self,
        size: int,
        device: torch.device,
        pin_memory: bool = True,
    ):
        self.size = size
        self.device = device
        self.pin_memory = pin_memory

        self._buffer = torch.empty(size, dtype=torch.uint8, device=device)
        self._registered = False
        self._lock = threading.Lock()

    @property
    def data_ptr(self) -> int:
        """Get the raw pointer to the buffer memory."""
        return self._buffer.data_ptr()

    @property
    def buffer(self) -> torch.Tensor:
        """Get the underlying buffer tensor."""
        return self._buffer

    def register_with_mooncake(self, store) -> bool:
        """Register this buffer with Mooncake for RDMA transfers."""
        if self._registered:
            return True

        try:
            if hasattr(store, "registerLocalMemory"):
                store.registerLocalMemory(self.data_ptr, self.size)
                self._registered = True
                logger.debug(f"Registered GPU buffer at {self.data_ptr:#x}, size={self.size}")
                return True
        except Exception as e:
            logger.warning(f"Failed to register GPU buffer: {e}")

        return False

    def unregister_from_mooncake(self, store) -> None:
        """Unregister this buffer from Mooncake."""
        if not self._registered:
            return

        try:
            if hasattr(store, "unregisterLocalMemory"):
                store.unregisterLocalMemory(self.data_ptr, self.size)
        except Exception as e:
            logger.warning(f"Failed to unregister GPU buffer: {e}")

        self._registered = False

    def write_eagle3_output(
        self,
        hidden_states: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> int:
        """
        Write Eagle3 output tensors into this buffer.
        
        Returns the total bytes written.
        """
        with self._lock:
            hidden_states = hidden_states.contiguous()
            target = target.contiguous()
            loss_mask = loss_mask.contiguous()
            input_ids = input_ids.contiguous()
            attention_mask = attention_mask.contiguous()
            if last_hidden_states is not None:
                last_hidden_states = last_hidden_states.contiguous()

            hs_size = hidden_states.numel() * hidden_states.element_size()
            tgt_size = target.numel() * target.element_size()
            lm_size = loss_mask.numel() * loss_mask.element_size()
            ids_size = input_ids.numel() * input_ids.element_size()
            am_size = attention_mask.numel() * attention_mask.element_size()
            lhs_size = (
                last_hidden_states.numel() * last_hidden_states.element_size()
                if last_hidden_states is not None
                else 0
            )

            total_data_size = hs_size + tgt_size + lm_size + ids_size + am_size + lhs_size
            total_size = HEADER_SIZE + total_data_size

            if total_size > self.size:
                raise ValueError(
                    f"Buffer too small: need {total_size}, have {self.size}"
                )

            header = struct.pack(
                HEADER_FORMAT,
                hs_size,
                tgt_size,
                lm_size,
                ids_size,
                am_size,
                lhs_size,
                TensorMetadata.dtype_to_bytes(hidden_states.dtype),
                TensorMetadata.dtype_to_bytes(target.dtype),
            )

            header_tensor = torch.frombuffer(
                bytearray(header), dtype=torch.uint8
            ).to(self.device)
            self._buffer[:HEADER_SIZE].copy_(header_tensor)

            offset = HEADER_SIZE
            offset = self._copy_tensor_to_buffer(hidden_states, offset)
            offset = self._copy_tensor_to_buffer(target, offset)
            offset = self._copy_tensor_to_buffer(loss_mask, offset)
            offset = self._copy_tensor_to_buffer(input_ids, offset)
            offset = self._copy_tensor_to_buffer(attention_mask, offset)
            if last_hidden_states is not None:
                offset = self._copy_tensor_to_buffer(last_hidden_states, offset)

            return total_size

    def _copy_tensor_to_buffer(self, tensor: torch.Tensor, offset: int) -> int:
        """Copy a tensor into the buffer at the given offset, return new offset."""
        tensor_bytes = tensor.numel() * tensor.element_size()
        tensor_view = tensor.view(torch.uint8).view(-1)
        self._buffer[offset : offset + tensor_bytes].copy_(tensor_view)
        return offset + tensor_bytes

    def read_eagle3_output(
        self,
        shapes: Dict[str, Tuple[int, ...]],
    ) -> "Eagle3TargetOutput":
        """
        Read Eagle3 output from this buffer.
        
        Args:
            shapes: Dictionary with keys 'hidden_states', 'target', 'loss_mask',
                   'input_ids', 'attention_mask', and optionally 'last_hidden_states',
                   mapping to their respective shapes.
        """
        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

        with self._lock:
            header_bytes = self._buffer[:HEADER_SIZE].cpu().numpy().tobytes()
            (
                hs_size,
                tgt_size,
                lm_size,
                ids_size,
                am_size,
                lhs_size,
                hs_dtype_bytes,
                tgt_dtype_bytes,
            ) = struct.unpack(HEADER_FORMAT, header_bytes)

            hs_dtype = TensorMetadata.bytes_to_dtype(hs_dtype_bytes)
            tgt_dtype = TensorMetadata.bytes_to_dtype(tgt_dtype_bytes)

            offset = HEADER_SIZE

            hidden_states = self._read_tensor_from_buffer(
                offset, hs_size, shapes["hidden_states"], hs_dtype
            )
            offset += hs_size

            target = self._read_tensor_from_buffer(
                offset, tgt_size, shapes["target"], tgt_dtype
            )
            offset += tgt_size

            loss_mask = self._read_tensor_from_buffer(
                offset, lm_size, shapes["loss_mask"], torch.bool
            )
            offset += lm_size

            input_ids = self._read_tensor_from_buffer(
                offset, ids_size, shapes["input_ids"], torch.int64
            )
            offset += ids_size

            attention_mask = self._read_tensor_from_buffer(
                offset, am_size, shapes["attention_mask"], torch.int64
            )
            offset += am_size

            last_hidden_states = None
            if lhs_size > 0 and "last_hidden_states" in shapes:
                last_hidden_states = self._read_tensor_from_buffer(
                    offset, lhs_size, shapes["last_hidden_states"], hs_dtype
                )

            return Eagle3TargetOutput(
                hidden_states=hidden_states,
                target=target,
                loss_mask=loss_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                last_hidden_states=last_hidden_states,
            )

    def _read_tensor_from_buffer(
        self,
        offset: int,
        size: int,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Read a tensor from the buffer at the given offset."""
        buffer_view = self._buffer[offset : offset + size]
        tensor = buffer_view.view(dtype).view(shape).clone()
        return tensor

    def get_slice(self, start: int, end: int) -> torch.Tensor:
        """Get a slice of the buffer as a tensor view."""
        return self._buffer[start:end]

    def as_bytes(self, size: int) -> bytes:
        """Get the first `size` bytes of the buffer as a bytes object."""
        return self._buffer[:size].cpu().numpy().tobytes()

    def from_bytes(self, data: bytes) -> None:
        """Copy bytes into the buffer."""
        data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        self._buffer[: len(data)].copy_(data_tensor.to(self.device))


class GPUBufferPool:
    """
    Pool of pre-allocated GPU buffers for zero-copy RDMA transfers.
    
    Manages a set of buffers that can be acquired and released for use
    in receiving data from Mooncake Store.
    """

    def __init__(
        self,
        buffer_size: int,
        pool_size: int,
        device: torch.device,
    ):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self.device = device

        self._buffers: List[GPUBuffer] = []
        self._available: List[GPUBuffer] = []
        self._lock = threading.Lock()
        self._condition = threading.Condition(self._lock)
        self._initialized = False

    def initialize(self) -> None:
        """Initialize the buffer pool."""
        if self._initialized:
            return

        with self._lock:
            for _ in range(self.pool_size):
                buffer = GPUBuffer(self.buffer_size, self.device)
                self._buffers.append(buffer)
                self._available.append(buffer)

            self._initialized = True
            logger.info(
                f"Initialized GPU buffer pool: {self.pool_size} buffers, "
                f"{self.buffer_size / (1024**2):.1f}MB each on {self.device}"
            )

    def register_all_with_mooncake(self, store) -> int:
        """Register all buffers with Mooncake. Returns count of registered buffers."""
        if not self._initialized:
            self.initialize()

        count = 0
        for buffer in self._buffers:
            if buffer.register_with_mooncake(store):
                count += 1
        return count

    def acquire(self, timeout: Optional[float] = None) -> Optional[GPUBuffer]:
        """
        Acquire a buffer from the pool.
        
        Blocks until a buffer is available or timeout expires.
        """
        if not self._initialized:
            self.initialize()

        with self._condition:
            while not self._available:
                if not self._condition.wait(timeout=timeout):
                    return None

            return self._available.pop()

    def release(self, buffer: GPUBuffer) -> None:
        """Release a buffer back to the pool."""
        with self._condition:
            self._available.append(buffer)
            self._condition.notify()

    def shutdown(self) -> None:
        """Shutdown the pool and free all buffers."""
        with self._lock:
            self._available.clear()
            self._buffers.clear()
            self._initialized = False


class Eagle3ZeroCopyWriter:
    """
    Helper class for inference workers to write outputs using zero-copy.
    
    This handles packing Eagle3 output into a contiguous buffer format
    that can be efficiently transferred via Mooncake RDMA.
    """

    def __init__(self, device: torch.device, max_buffer_size: int = 2 * 1024**3):
        self.device = device
        self.max_buffer_size = max_buffer_size
        self._buffer: Optional[GPUBuffer] = None

    def get_buffer(self, required_size: int) -> GPUBuffer:
        """Get or create a buffer large enough for the data."""
        if self._buffer is None or self._buffer.size < required_size:
            actual_size = min(max(required_size, 256 * 1024**2), self.max_buffer_size)
            self._buffer = GPUBuffer(actual_size, self.device)
        return self._buffer

    def pack_eagle3_output(
        self,
        hidden_states: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[bytes, Dict[str, Tuple[int, ...]]]:
        """
        Pack Eagle3 output into a bytes object for Mooncake storage.
        
        Memory layout with alignment:
            [Header (64 bytes)] [pad] [hidden_states] [pad] [target] [pad] ...
        
        Each tensor is aligned to TENSOR_ALIGNMENT bytes for optimal RDMA.
        
        Returns:
            Tuple of (packed_bytes, shapes_dict) where shapes_dict contains
            the shapes needed to reconstruct tensors.
        """
        hidden_states = hidden_states.contiguous()
        target = target.contiguous()
        loss_mask = loss_mask.contiguous()
        input_ids = input_ids.contiguous()
        attention_mask = attention_mask.contiguous()
        if last_hidden_states is not None:
            last_hidden_states = last_hidden_states.contiguous()

        hs_bytes = hidden_states.view(torch.uint8).view(-1).cpu().numpy().tobytes()
        tgt_bytes = target.view(torch.uint8).view(-1).cpu().numpy().tobytes()
        lm_bytes = loss_mask.view(torch.uint8).view(-1).cpu().numpy().tobytes()
        ids_bytes = input_ids.view(torch.uint8).view(-1).cpu().numpy().tobytes()
        am_bytes = attention_mask.view(torch.uint8).view(-1).cpu().numpy().tobytes()
        lhs_bytes = (
            last_hidden_states.view(torch.uint8).view(-1).cpu().numpy().tobytes()
            if last_hidden_states is not None
            else b""
        )

        offset = HEADER_SIZE
        hs_offset = align_up(offset)
        offset = hs_offset + len(hs_bytes)

        tgt_offset = align_up(offset)
        offset = tgt_offset + len(tgt_bytes)

        lm_offset = align_up(offset)
        offset = lm_offset + len(lm_bytes)

        ids_offset = align_up(offset)
        offset = ids_offset + len(ids_bytes)

        am_offset = align_up(offset)
        offset = am_offset + len(am_bytes)

        lhs_offset = align_up(offset) if lhs_bytes else 0
        if lhs_bytes:
            offset = lhs_offset + len(lhs_bytes)

        total_size = align_up(offset, PAGE_SIZE)

        header = struct.pack(
            HEADER_FORMAT,
            len(hs_bytes),
            len(tgt_bytes),
            len(lm_bytes),
            len(ids_bytes),
            len(am_bytes),
            len(lhs_bytes),
            TensorMetadata.dtype_to_bytes(hidden_states.dtype),
            TensorMetadata.dtype_to_bytes(target.dtype),
            total_size,
        )
        header = header.ljust(HEADER_SIZE, b"\x00")

        packed = bytearray(total_size)
        packed[:len(header)] = header
        packed[hs_offset : hs_offset + len(hs_bytes)] = hs_bytes
        packed[tgt_offset : tgt_offset + len(tgt_bytes)] = tgt_bytes
        packed[lm_offset : lm_offset + len(lm_bytes)] = lm_bytes
        packed[ids_offset : ids_offset + len(ids_bytes)] = ids_bytes
        packed[am_offset : am_offset + len(am_bytes)] = am_bytes
        if lhs_bytes:
            packed[lhs_offset : lhs_offset + len(lhs_bytes)] = lhs_bytes

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "target": tuple(target.shape),
            "loss_mask": tuple(loss_mask.shape),
            "input_ids": tuple(input_ids.shape),
            "attention_mask": tuple(attention_mask.shape),
        }
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)

        return bytes(packed), shapes


class Eagle3ZeroCopyReader:
    """
    Helper class for training nodes to read outputs using zero-copy.
    
    This manages a buffer pool and handles unpacking Eagle3 output from
    Mooncake Store directly into GPU memory.
    """

    def __init__(
        self,
        device: torch.device,
        buffer_size: int = 2 * 1024**3,
        pool_size: int = 4,
    ):
        self.device = device
        self.buffer_pool = GPUBufferPool(buffer_size, pool_size, device)
        self._registered = False

    def initialize(self, mooncake_store=None) -> None:
        """Initialize the reader and optionally register buffers with Mooncake."""
        self.buffer_pool.initialize()
        self._mooncake_store = mooncake_store
        if mooncake_store is not None:
            for buffer in self.buffer_pool._buffers:
                if mooncake_store.register_gpu_buffer(buffer.buffer, buffer.size):
                    self._registered = True
            if self._registered:
                logger.info(
                    f"Registered {len(self.buffer_pool._buffers)} GPU buffers for RDMA"
                )

    def unpack_rdma_to_gpu(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        mooncake_store=None,
    ) -> "Eagle3TargetOutput":
        """
        Transfer data directly from Mooncake to GPU via RDMA and unpack.
        
        This is the zero-copy path: CPU (remote) → RDMA → GPU (local).
        
        Args:
            key: Mooncake key to retrieve
            shapes: Dictionary of tensor shapes
            mooncake_store: MooncakeHiddenStateStore instance
            
        Returns:
            Eagle3TargetOutput with tensors on GPU
        """
        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

        store = mooncake_store or self._mooncake_store
        if store is None:
            raise ValueError("No mooncake_store provided")

        buffer = self.buffer_pool.acquire(timeout=30.0)
        if buffer is None:
            raise RuntimeError("Failed to acquire GPU buffer from pool")

        try:
            nbytes = store.get_into_gpu_buffer(key, buffer.buffer)
            logger.debug(f"RDMA transferred {nbytes} bytes to GPU")

            header_bytes = buffer.buffer[:HEADER_SIZE].cpu().numpy().tobytes()
            header_data = header_bytes[:struct.calcsize("!6Q4s4sQ")]
            (
                hs_size,
                tgt_size,
                lm_size,
                ids_size,
                am_size,
                lhs_size,
                hs_dtype_bytes,
                tgt_dtype_bytes,
                total_size,
            ) = struct.unpack("!6Q4s4sQ", header_data)

            hs_dtype = TensorMetadata.bytes_to_dtype(hs_dtype_bytes)
            tgt_dtype = TensorMetadata.bytes_to_dtype(tgt_dtype_bytes)

            offset = HEADER_SIZE
            hs_offset = align_up(offset)
            offset = hs_offset + hs_size

            tgt_offset = align_up(offset)
            offset = tgt_offset + tgt_size

            lm_offset = align_up(offset)
            offset = lm_offset + lm_size

            ids_offset = align_up(offset)
            offset = ids_offset + ids_size

            am_offset = align_up(offset)
            offset = am_offset + am_size

            lhs_offset = align_up(offset) if lhs_size > 0 else 0

            hidden_states = (
                buffer.buffer[hs_offset : hs_offset + hs_size]
                .view(hs_dtype)
                .view(shapes["hidden_states"])
                .clone()
            )

            target = (
                buffer.buffer[tgt_offset : tgt_offset + tgt_size]
                .view(tgt_dtype)
                .view(shapes["target"])
                .clone()
            )

            loss_mask = (
                buffer.buffer[lm_offset : lm_offset + lm_size]
                .view(torch.bool)
                .view(shapes["loss_mask"])
                .clone()
            )

            input_ids = (
                buffer.buffer[ids_offset : ids_offset + ids_size]
                .view(torch.int64)
                .view(shapes["input_ids"])
                .clone()
            )

            attention_mask = (
                buffer.buffer[am_offset : am_offset + am_size]
                .view(torch.int64)
                .view(shapes["attention_mask"])
                .clone()
            )

            last_hidden_states = None
            if lhs_size > 0 and "last_hidden_states" in shapes:
                last_hidden_states = (
                    buffer.buffer[lhs_offset : lhs_offset + lhs_size]
                    .view(hs_dtype)
                    .view(shapes["last_hidden_states"])
                    .clone()
                )

            return Eagle3TargetOutput(
                hidden_states=hidden_states,
                target=target,
                loss_mask=loss_mask,
                input_ids=input_ids,
                attention_mask=attention_mask,
                last_hidden_states=last_hidden_states,
            )

        finally:
            self.buffer_pool.release(buffer)

    def unpack_to_gpu(
        self,
        data: bytes,
        shapes: Dict[str, Tuple[int, ...]],
    ) -> "Eagle3TargetOutput":
        """
        Unpack bytes directly into GPU tensors.
        
        Handles aligned memory layout for optimal RDMA transfer.
        """
        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

        if len(data) < HEADER_SIZE:
            raise ValueError(f"Data too small: {len(data)} < {HEADER_SIZE}")

        header_data = data[:struct.calcsize("!6Q4s4sQ")]
        (
            hs_size,
            tgt_size,
            lm_size,
            ids_size,
            am_size,
            lhs_size,
            hs_dtype_bytes,
            tgt_dtype_bytes,
            total_size,
        ) = struct.unpack("!6Q4s4sQ", header_data)

        hs_dtype = TensorMetadata.bytes_to_dtype(hs_dtype_bytes)
        tgt_dtype = TensorMetadata.bytes_to_dtype(tgt_dtype_bytes)

        offset = HEADER_SIZE
        hs_offset = align_up(offset)
        offset = hs_offset + hs_size

        tgt_offset = align_up(offset)
        offset = tgt_offset + tgt_size

        lm_offset = align_up(offset)
        offset = lm_offset + lm_size

        ids_offset = align_up(offset)
        offset = ids_offset + ids_size

        am_offset = align_up(offset)
        offset = am_offset + am_size

        lhs_offset = align_up(offset) if lhs_size > 0 else 0

        hidden_states = self._bytes_to_tensor(
            data[hs_offset : hs_offset + hs_size], shapes["hidden_states"], hs_dtype
        )

        target = self._bytes_to_tensor(
            data[tgt_offset : tgt_offset + tgt_size], shapes["target"], tgt_dtype
        )

        loss_mask = self._bytes_to_tensor(
            data[lm_offset : lm_offset + lm_size], shapes["loss_mask"], torch.bool
        )

        input_ids = self._bytes_to_tensor(
            data[ids_offset : ids_offset + ids_size], shapes["input_ids"], torch.int64
        )

        attention_mask = self._bytes_to_tensor(
            data[am_offset : am_offset + am_size], shapes["attention_mask"], torch.int64
        )

        last_hidden_states = None
        if lhs_size > 0 and "last_hidden_states" in shapes:
            last_hidden_states = self._bytes_to_tensor(
                data[lhs_offset : lhs_offset + lhs_size],
                shapes["last_hidden_states"],
                hs_dtype,
            )

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )

    def _bytes_to_tensor(
        self,
        data: bytes,
        shape: Tuple[int, ...],
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """Convert bytes to a GPU tensor."""
        tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
        tensor = tensor.to(self.device, non_blocking=True)
        tensor = tensor.view(dtype).view(shape)
        return tensor.clone()

    def shutdown(self) -> None:
        """Shutdown the reader and free resources."""
        self.buffer_pool.shutdown()


def estimate_buffer_size(
    batch_size: int,
    seq_len: int,
    hidden_dim: int,
    vocab_size: int,
    dtype: torch.dtype = torch.bfloat16,
    include_last_hidden_states: bool = False,
) -> int:
    """
    Estimate the buffer size needed for one Eagle3 output with alignment.
    
    This can be used to configure the buffer pool size.
    
    Args:
        batch_size: Batch size
        seq_len: Sequence length
        hidden_dim: Hidden dimension of the model
        vocab_size: Vocabulary size
        dtype: Data type for hidden states and logits
        include_last_hidden_states: Whether to include last hidden states
    
    Returns:
        Estimated buffer size in bytes (page-aligned)
    """
    elem_size = torch.tensor([], dtype=dtype).element_size()

    hidden_states_size = batch_size * seq_len * hidden_dim * 3 * elem_size
    target_size = batch_size * seq_len * vocab_size * elem_size
    loss_mask_size = batch_size * seq_len
    input_ids_size = batch_size * seq_len * 8
    attention_mask_size = batch_size * seq_len * 8
    last_hidden_size = batch_size * seq_len * hidden_dim * elem_size if include_last_hidden_states else 0

    offset = HEADER_SIZE
    offset = align_up(offset) + hidden_states_size
    offset = align_up(offset) + target_size
    offset = align_up(offset) + loss_mask_size
    offset = align_up(offset) + input_ids_size
    offset = align_up(offset) + attention_mask_size
    if last_hidden_size > 0:
        offset = align_up(offset) + last_hidden_size

    return align_up(offset, PAGE_SIZE)
