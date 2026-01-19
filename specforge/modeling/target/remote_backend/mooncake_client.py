"""
Mooncake Store client wrapper for hidden state storage.

This module provides a high-level interface to Mooncake Store for storing
and retrieving hidden states data in a distributed training setup.

Supports both serialized (pickle-based) and zero-copy (raw bytes) modes.
"""

import logging
import os
import random
import struct
import threading
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Generic, List, Optional, Tuple, TypeVar

import torch

from mooncake.store import MooncakeDistributedStore

from .host_mem import MooncakeHostTensorAllocator

if TYPE_CHECKING:
    from .messages import Eagle3OutputData
    from ..eagle3_target_model import Eagle3TargetOutput

logger = logging.getLogger(__name__)

HEADER_FORMAT = "!6Q4s4sQ"
HEADER_SIZE = 64
TENSOR_ALIGNMENT = 64
PAGE_SIZE = 4096


def _align_up(size: int, alignment: int = TENSOR_ALIGNMENT) -> int:
    return (size + alignment - 1) & ~(alignment - 1)


def _dtype_to_bytes(dtype: torch.dtype) -> bytes:
    dtype_map = {
        torch.float32: b"fp32",
        torch.float16: b"fp16",
        torch.bfloat16: b"bf16",
        torch.int64: b"i64_",
        torch.int32: b"i32_",
        torch.bool: b"bool",
    }
    return dtype_map.get(dtype, b"unk_")


class HostBuffer:
    """
    Pre-allocated host buffer using MooncakeHostTensorAllocator.
    
    The buffer is allocated in RDMA-registered host memory, enabling
    zero-copy transfers when used with Mooncake Store.
    """

    def __init__(self, size: int):
        self.size = size
        self._allocator = MooncakeHostTensorAllocator()
        self._buffer = self._allocator.allocate((size,), dtype=torch.uint8)
        self._lock = threading.Lock()

    @property
    def data_ptr(self) -> int:
        return self._allocator.ptr

    @property
    def buffer(self) -> torch.Tensor:
        return self._buffer

    def write_eagle3_output(
        self,
        hidden_states: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[int, Dict[str, Tuple[int, ...]]]:
        """
        Write Eagle3 output tensors into this host buffer.
        
        Returns (total_bytes_written, shapes_dict).
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

            offset = HEADER_SIZE
            hs_offset = _align_up(offset)
            offset = hs_offset + hs_size

            tgt_offset = _align_up(offset)
            offset = tgt_offset + tgt_size

            lm_offset = _align_up(offset)
            offset = lm_offset + lm_size

            ids_offset = _align_up(offset)
            offset = ids_offset + ids_size

            am_offset = _align_up(offset)
            offset = am_offset + am_size

            lhs_offset = _align_up(offset) if lhs_size > 0 else 0
            if lhs_size > 0:
                offset = lhs_offset + lhs_size

            total_size = _align_up(offset, PAGE_SIZE)

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
                _dtype_to_bytes(hidden_states.dtype),
                _dtype_to_bytes(target.dtype),
                total_size,
            )
            header = header.ljust(HEADER_SIZE, b"\x00")

            header_tensor = torch.frombuffer(bytearray(header), dtype=torch.uint8)
            self._buffer[:HEADER_SIZE].copy_(header_tensor)

            self._copy_tensor_to_buffer(hidden_states, hs_offset)
            self._copy_tensor_to_buffer(target, tgt_offset)
            self._copy_tensor_to_buffer(loss_mask, lm_offset)
            self._copy_tensor_to_buffer(input_ids, ids_offset)
            self._copy_tensor_to_buffer(attention_mask, am_offset)
            if last_hidden_states is not None:
                self._copy_tensor_to_buffer(last_hidden_states, lhs_offset)

            shapes = {
                "hidden_states": tuple(hidden_states.shape),
                "target": tuple(target.shape),
                "loss_mask": tuple(loss_mask.shape),
                "input_ids": tuple(input_ids.shape),
                "attention_mask": tuple(attention_mask.shape),
            }
            if last_hidden_states is not None:
                shapes["last_hidden_states"] = tuple(last_hidden_states.shape)

            return total_size, shapes

    def _copy_tensor_to_buffer(self, tensor: torch.Tensor, offset: int) -> None:
        tensor_bytes = tensor.numel() * tensor.element_size()
        tensor_view = tensor.view(torch.uint8).view(-1)
        if tensor.is_cuda:
            self._buffer[offset : offset + tensor_bytes].copy_(tensor_view.cpu())
        else:
            self._buffer[offset : offset + tensor_bytes].copy_(tensor_view)

    def get_slice(self, size: int) -> torch.Tensor:
        return self._buffer[:size]

    def as_bytes(self, size: int) -> bytes:
        return self._buffer[:size].numpy().tobytes()


class Eagle3HostBufferWriter:
    """
    Helper class for inference workers to write outputs using host buffers.
    
    Uses MooncakeHostTensorAllocator to allocate RDMA-registered host memory,
    then copies GPU tensors directly into the buffer without serialization.
    """

    def __init__(self, max_buffer_size: int = 4 * 1024**3):
        self.max_buffer_size = max_buffer_size
        self._buffer: Optional[HostBuffer] = None

    def get_buffer(self, required_size: int) -> HostBuffer:
        if self._buffer is None or self._buffer.size < required_size:
            actual_size = min(max(required_size, 256 * 1024**2), self.max_buffer_size)
            self._buffer = HostBuffer(actual_size)
        return self._buffer

    def pack_eagle3_output(
        self,
        hidden_states: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Tuple[HostBuffer, int, Dict[str, Tuple[int, ...]]]:
        """
        Pack Eagle3 output into a host buffer.
        
        Returns (host_buffer, total_size, shapes_dict).
        The buffer's data_ptr can be used directly with Mooncake's put operation.
        """
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

        estimated_size = (
            HEADER_SIZE + hs_size + tgt_size + lm_size + ids_size + am_size + lhs_size
            + 6 * TENSOR_ALIGNMENT + PAGE_SIZE
        )

        buffer = self.get_buffer(estimated_size)
        total_size, shapes = buffer.write_eagle3_output(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )

        return buffer, total_size, shapes


T = TypeVar("T")


@dataclass
class MooncakeConfig:
    """
    Configuration for Mooncake Store client.
    
    Memory parameters:
        global_segment_size: Memory this client CONTRIBUTES to the distributed pool.
            Other clients can store objects in this space. Inference workers should
            set this large enough to hold multiple outputs (~2-4GB per batch
            for large models).
            
        local_buffer_size: Memory for RECEIVING data via Get() operations.
            This is the buffer for reading objects from other clients. Trainers
            should set this large enough to receive hidden states (~512MB-2GB).
    
    Note: The metadata_server defaults to the Mooncake Master's built-in HTTP
    metadata server (same host as master, port 8080).
    """

    local_hostname: str = "localhost"
    metadata_server: str = "http://localhost:8090/metadata"
    master_server_address: str = "localhost:50051"
    global_segment_size: int = 4 * 1024 * 1024 * 1024
    local_buffer_size: int = 512 * 1024 * 1024
    protocol: str = "tcp"
    device_name: str = ""
    replica_num: int = 1
    enable_soft_pin: bool = False

    @classmethod
    def from_env(cls) -> "MooncakeConfig":
        """Create config from environment variables."""
        master_host = os.getenv("MOONCAKE_MASTER_HOST", "localhost")
        master_port = os.getenv("MOONCAKE_MASTER_PORT", "50051")
        metadata_port = os.getenv("MOONCAKE_METADATA_PORT", "8080")

        return cls(
            local_hostname=os.getenv("MOONCAKE_LOCAL_HOSTNAME", "localhost"),
            metadata_server=os.getenv(
                "MOONCAKE_METADATA_SERVER", f"http://{master_host}:{metadata_port}/metadata"
            ),
            master_server_address=os.getenv(
                "MOONCAKE_MASTER_SERVER", f"{master_host}:{master_port}"
            ),
            global_segment_size=int(
                os.getenv("MOONCAKE_GLOBAL_SEGMENT_SIZE", str(4 * 1024 * 1024 * 1024))
            ),
            local_buffer_size=int(
                os.getenv("MOONCAKE_LOCAL_BUFFER_SIZE", str(512 * 1024 * 1024))
            ),
            protocol=os.getenv("MOONCAKE_PROTOCOL", "tcp"),
            device_name=os.getenv("MOONCAKE_DEVICE_NAME", ""),
        )

    @classmethod
    def from_master_address(
        cls,
        master_host: str,
        master_port: int = 50051,
        metadata_port: int = 8080,
        **kwargs,
    ) -> "MooncakeConfig":
        """
        Create config from master address.
        
        Assumes the master is running with built-in HTTP metadata server enabled.
        """
        return cls(
            metadata_server=f"http://{master_host}:{metadata_port}/metadata",
            master_server_address=f"{master_host}:{master_port}",
            **kwargs,
        )

    @staticmethod
    def parse_size(size_str: str) -> int:
        """Parse size string like '4GB' or '512MB' to bytes."""
        size_str = size_str.upper().strip()
        multipliers = {
            "TB": 1024 * 1024 * 1024 * 1024,
            "GB": 1024 * 1024 * 1024,
            "MB": 1024 * 1024,
            "KB": 1024,
            "B": 1,
        }
        for suffix, multiplier in multipliers.items():
            if size_str.endswith(suffix):
                return int(float(size_str[: -len(suffix)]) * multiplier)
        return int(size_str)


class MooncakeHiddenStateStore(ABC):
    """
    Base class for Mooncake Store wrapper to store hidden states from target model.
    
    This class provides:
    - Store setup and teardown
    - Raw bytes storage/retrieval
    - GPU buffer registration for RDMA direct transfers
    
    Subclasses should implement application-specific methods for storing
    and retrieving structured data (e.g., Eagle3 outputs).
    
    For CPU→GPU RDMA transfers:
    1. Call register_gpu_buffer() with your GPU tensor's data_ptr
    2. Use get_into_gpu_buffer() to transfer directly to GPU
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._registered_buffers: Dict[int, int] = {}

    def setup(self) -> None:
        """Initialize the Mooncake Store client."""
        if self._initialized:
            return

        self._store = MooncakeDistributedStore()
        self._store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )

        self._initialized = True
        logger.info(
            f"Mooncake Store client initialized (protocol={self.config.protocol})"
        )

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            self.setup()

    def register_gpu_buffer_tensor(self, gpu_tensor: torch.Tensor) -> bool:
        """
        Register a GPU tensor's memory with Mooncake for RDMA transfers.
        
        This enables CPU→GPU RDMA: data on remote CPU can be transferred
        directly to this GPU buffer via RDMA, bypassing local CPU.
        
        Args:
            gpu_tensor: A contiguous GPU tensor to register
            
        Returns:
            True if registration succeeded
        """
        self._ensure_initialized()

        if not gpu_tensor.is_cuda:
            logger.warning("Cannot register non-CUDA tensor for GPU RDMA")
            return False

        if not gpu_tensor.is_contiguous():
            logger.warning("GPU tensor must be contiguous for RDMA registration")
            return False

        ptr = gpu_tensor.data_ptr()
        size = gpu_tensor.numel() * gpu_tensor.element_size()

        return self.register_gpu_buffer(ptr, size)

    def register_gpu_buffer(self, buffer_ptr: int, size: int) -> bool:
        """
        Register a GPU buffer for RDMA transfers.
        
        Returns True if successful.
        """
        self._ensure_initialized()

        if buffer_ptr in self._registered_buffers:
            return True

        try:
            if hasattr(self._store, "registerLocalMemory"):
                self._store.registerLocalMemory(buffer_ptr, size)
                self._registered_buffers[buffer_ptr] = size
                logger.debug(f"Registered GPU buffer at {buffer_ptr:#x}, size={size}")
                return True
        except Exception as e:
            logger.warning(f"Failed to register GPU buffer: {e}")

        return False

    def unregister_gpu_buffer_tensor(self, gpu_tensor: torch.Tensor) -> None:
        """Unregister a previously registered GPU buffer."""
        ptr = gpu_tensor.data_ptr()
        size = self._registered_buffers.pop(ptr, None)

        if size is not None and hasattr(self._store, "unregisterLocalMemory"):
            try:
                self._store.unregister_local_memory(ptr, size)
            except Exception as e:
                logger.warning(f"Failed to unregister GPU buffer: {e}")

    def get_into_gpu_buffer(
        self,
        key: str,
        gpu_buffer: torch.Tensor,
        offset: int = 0,
    ) -> int:
        """
        Transfer data directly into a registered GPU buffer via RDMA.
        
        This is the zero-copy path: CPU (remote) → RDMA → GPU (local).
        The gpu_buffer must have been registered with register_gpu_buffer().
        
        Args:
            key: Key of the data to retrieve
            gpu_buffer: Pre-registered GPU buffer (must be large enough)
            offset: Offset in the buffer to write to
            
        Returns:
            Number of bytes transferred
        """
        self._ensure_initialized()

        if not gpu_buffer.is_cuda:
            raise ValueError("Buffer must be a CUDA tensor")

        ptr = gpu_buffer.data_ptr()
        if ptr not in self._registered_buffers:
            if not self.register_gpu_buffer_tensor(gpu_buffer):
                raise RuntimeError("Failed to register GPU buffer for RDMA")

        try:
            if hasattr(self._store, "get_into_buffer"):
                return self._store.get_into_buffer(key, ptr + offset)
            elif hasattr(self._store, "get"):
                data = self._store.get(key)
                if data is None:
                    raise KeyError(f"Key not found: {key}")
                data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                gpu_buffer.view(torch.uint8)[offset : offset + len(data)].copy_(
                    data_tensor.cuda()
                )
                return len(data)
        except Exception as e:
            logger.error(f"Failed to get data into GPU buffer: {e}")
            raise

        return 0

    def put_raw(self, key: str, data: bytes) -> None:
        """Store raw bytes in Mooncake Store."""
        self._ensure_initialized()
        self._store.put(key, data)
        logger.debug(f"Stored raw data with key: {key}, size: {len(data)}")

    def put_from_host_buffer(self, key: str, host_buffer: HostBuffer, size: int) -> None:
        """
        Store data from a pre-allocated host buffer using zero-copy.
        
        The host buffer should be allocated using MooncakeHostTensorAllocator,
        which ensures the memory is RDMA-compatible. This method also registers
        the buffer with MooncakeDistributedStore for zero-copy operations.
        """
        self._ensure_initialized()

        if hasattr(self._store, "put_from"):
            ptr = host_buffer.data_ptr
            if ptr not in self._registered_buffers:
                if hasattr(self._store, "register_buffer"):
                    result = self._store.register_buffer(ptr, host_buffer.size)
                    if result != 0:
                        logger.warning(f"Failed to register host buffer: {result}")
                    else:
                        self._registered_buffers[ptr] = host_buffer.size
                        logger.debug(f"Registered host buffer at {ptr:#x}, size={host_buffer.size}")

            result = self._store.put_from(key, ptr, size)
            if result != 0:
                raise RuntimeError(f"Mooncake put_from failed with code: {result}")
            logger.debug(f"Stored data from host buffer (zero-copy) with key: {key}, size: {size}")
        else:
            data = host_buffer.as_bytes(size)
            self._store.put(key, data)
            logger.debug(f"Stored data from host buffer with key: {key}, size: {size}")

    def get_raw(self, key: str) -> bytes:
        """Retrieve raw bytes from Mooncake Store."""
        self._ensure_initialized()

        data = self._store.get(key)
        if data is None:
            raise KeyError(f"Key not found in Mooncake Store: {key}")

        return data

    def remove(self, key: str) -> None:
        """Remove data from Mooncake Store."""
        self._ensure_initialized()
        self._store.remove(key)
        logger.debug(f"Removed task result with key: {key}")

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        self._ensure_initialized()

        try:
            data = self._store.get(key)
            return data is not None
        except Exception:
            return False

    def close(self) -> None:
        """Close the Mooncake Store client."""
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False


class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.
    
    Provides methods to store and retrieve Eagle3OutputData and Eagle3TargetOutput.
    """

    def put_eagle3_output(
        self,
        key: str,
        hidden_states: torch.Tensor,
        target: torch.Tensor,
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Store Eagle3 output tensors in Mooncake Store.
        
        Args:
            key: Unique key for this data (typically task_id)
            hidden_states: Concatenated auxiliary hidden states
            target: Target logits
            loss_mask: Loss mask tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            last_hidden_states: Optional last hidden states
        """
        from .messages import Eagle3OutputData

        self._ensure_initialized()

        output_data = Eagle3OutputData.from_tensors(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )
        serialized = output_data.serialize()
        self._store.put(key, serialized)
        logger.debug(f"Stored Eagle3 output with key: {key}, size: {len(serialized)}")

    def put_eagle3_output_data(self, key: str, output_data) -> None:
        """Store pre-serialized Eagle3OutputData."""
        self._ensure_initialized()

        serialized = output_data.serialize()
        self._store.put(key, serialized)
        logger.debug(f"Stored Eagle3 output with key: {key}, size: {len(serialized)}")

    def get_eagle3_output(self, key: str, device: str = "cuda") -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 output from Mooncake Store.
        
        Args:
            key: Key used when storing the data
            device: Device to move tensors to
            
        Returns:
            Eagle3TargetOutput with tensors on the specified device
        """
        from .messages import Eagle3OutputData

        self._ensure_initialized()

        data = self._store.get(key)
        if data is None:
            raise KeyError(f"Key not found in Mooncake Store: {key}")

        output_data = Eagle3OutputData.deserialize(data)
        return output_data.to_eagle3_output(device=device)

    def get_eagle3_output_data(self, key: str) -> "Eagle3OutputData":
        """Retrieve Eagle3OutputData without converting to tensors."""
        from .messages import Eagle3OutputData

        self._ensure_initialized()

        data = self._store.get(key)
        if data is None:
            raise KeyError(f"Key not found in Mooncake Store: {key}")

        return Eagle3OutputData.deserialize(data)

    def get_eagle3_output_zero_copy(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 output using zero-copy unpacking.
        
        Args:
            key: Key used when storing the data
            shapes: Dictionary of tensor shapes
            device: Device to place tensors on
            
        Returns:
            Eagle3TargetOutput with tensors on the specified device
        """
        self._ensure_initialized()

        data = self._store.get(key)
        if data is None:
            raise KeyError(f"Key not found in Mooncake Store: {key}")

        from .zero_copy import Eagle3ZeroCopyReader

        reader = Eagle3ZeroCopyReader(device, buffer_size=len(data) + 1024, pool_size=1)
        return reader.unpack_to_gpu(data, shapes)

    def put_eagle3_output_host_buffer(
        self,
        key: str,
        host_buffer: HostBuffer,
        size: int,
    ) -> None:
        """
        Store Eagle3 output from a pre-allocated host buffer.
        
        Use with Eagle3HostBufferWriter.pack_eagle3_output() which returns
        (host_buffer, size, shapes).
        """
        self.put_from_host_buffer(key, host_buffer, size)


StoreT = TypeVar("StoreT", bound=MooncakeHiddenStateStore)


class MooncakeHiddenStateStorePool(Generic[StoreT]):
    """
    Pool of Mooncake Store clients for concurrent access.
    
    Useful when multiple threads need to access Mooncake Store simultaneously.
    """

    def __init__(
        self,
        config: MooncakeConfig,
        pool_size: int = 4,
        store_class: type[StoreT] = EagleMooncakeStore,
    ):
        self.config = config
        self.pool_size = pool_size
        self._store_class = store_class
        self._stores: List[StoreT] = []
        self._initialized = False

    def setup(self) -> None:
        """Initialize the pool of Mooncake Store clients."""
        if self._initialized:
            return

        for i in range(self.pool_size):
            store = self._store_class(self.config)
            store.setup()
            self._stores.append(store)

        self._initialized = True
        logger.info(f"Mooncake Store pool initialized with {self.pool_size} clients")

    def get_store(self) -> StoreT:
        """Get a store from the pool (random selection for load balancing)."""
        if not self._initialized:
            self.setup()
        return random.choice(self._stores)

    def close(self) -> None:
        """Close all stores in the pool."""
        for store in self._stores:
            store.close()
        self._stores.clear()
        self._initialized = False
