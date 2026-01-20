"""
Mooncake Store client wrapper for hidden state storage.

This module provides a high-level interface to Mooncake Store for storing
and retrieving hidden states data in a distributed training setup.

Uses MooncakeHostMemAllocator for RDMA-registered host buffers and put_from
for zero-copy transfers.
"""

import ctypes
import logging
import os
import random
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Generic, List, Optional, Tuple, TypeVar

import torch

from mooncake.store import MooncakeDistributedStore, MooncakeHostMemAllocator

if TYPE_CHECKING:
    from .messages import Eagle3OutputData
    from ..eagle3_target_model import Eagle3TargetOutput

logger = logging.getLogger(__name__)


class HostBuffer:
    """
    RDMA-registered host buffer using MooncakeHostMemAllocator.
    
    Provides a reusable buffer for zero-copy transfers to Mooncake Store.
    """

    def __init__(self, size: int):
        self.size = size
        self._allocator = MooncakeHostMemAllocator()
        self._ptr = self._allocator.alloc(size)
        if self._ptr == 0:
            raise RuntimeError(f"Failed to allocate {size} bytes from MooncakeHostMemAllocator")

    @property
    def ptr(self) -> int:
        return self._ptr

    def copy_from_tensor(self, tensor: torch.Tensor, offset: int = 0) -> int:
        """
        Copy tensor data into this host buffer at the given offset.
        
        Returns the number of bytes copied.
        """
        tensor = tensor.contiguous()
        nbytes = tensor.numel() * tensor.element_size()
        
        if offset + nbytes > self.size:
            raise ValueError(f"Buffer overflow: need {offset + nbytes}, have {self.size}")

        c_type = ctypes.c_byte * nbytes
        c_array = c_type.from_address(self._ptr + offset)
        host_view = torch.frombuffer(c_array, dtype=torch.uint8, count=nbytes)
        
        if tensor.is_cuda:
            host_view.copy_(tensor.view(torch.uint8).view(-1).cpu())
        else:
            host_view.copy_(tensor.view(torch.uint8).view(-1))
        
        return nbytes

    def free(self) -> None:
        if self._ptr != 0:
            self._allocator.free(self._ptr)
            self._ptr = 0

    def __del__(self):
        self.free()


class HostBufferPool:
    """
    Pool of RDMA-registered host buffers for tensor storage.
    """

    def __init__(self, buffer_size: int = 4 * 1024**3, pool_size: int = 2):
        self.buffer_size = buffer_size
        self.pool_size = pool_size
        self._buffers: List[HostBuffer] = []
        self._current_idx = 0

    def initialize(self) -> None:
        """Pre-allocate all buffers."""
        for _ in range(self.pool_size):
            self._buffers.append(HostBuffer(self.buffer_size))
        logger.info(f"Initialized host buffer pool: {self.pool_size} x {self.buffer_size / (1024**3):.1f}GB")

    def get_buffer(self) -> HostBuffer:
        """Get the next buffer (round-robin)."""
        if not self._buffers:
            self.initialize()
        buf = self._buffers[self._current_idx]
        self._current_idx = (self._current_idx + 1) % len(self._buffers)
        return buf

    def shutdown(self) -> None:
        for buf in self._buffers:
            buf.free()
        self._buffers.clear()


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
    host_buffer_size: int = 4 * 1024 * 1024 * 1024
    host_buffer_pool_size: int = 2

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
    
    Uses RDMA-registered host buffers (MooncakeHostMemAllocator) and put_from
    for zero-copy transfers.
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None

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

        self._host_buffer_pool = HostBufferPool(
            buffer_size=self.config.host_buffer_size,
            pool_size=self.config.host_buffer_pool_size,
        )
        self._host_buffer_pool.initialize()

        for buf in self._host_buffer_pool._buffers:
            self._register_buffer(buf.ptr, buf.size)

        self._initialized = True
        logger.info(
            f"Mooncake Store client initialized (protocol={self.config.protocol})"
        )

    def _ensure_initialized(self) -> None:
        """Ensure the store is initialized."""
        if not self._initialized:
            self.setup()

    def _register_buffer(self, buffer_ptr: int, size: int) -> bool:
        """Register a buffer for RDMA transfers."""
        if buffer_ptr in self._registered_buffers:
            return True

        try:
            if hasattr(self._store, "register_buffer"):
                result = self._store.register_buffer(buffer_ptr, size)
                if result == 0:
                    self._registered_buffers[buffer_ptr] = size
                    logger.debug(f"Registered buffer at {buffer_ptr:#x}, size={size}")
                    return True
                else:
                    logger.warning(f"register_buffer returned error code: {result}")
                    return False
        except Exception as e:
            logger.warning(f"Failed to register buffer: {e}")

        return False

    def put_from(self, key: str, buffer_ptr: int, size: int) -> int:
        """
        Store data from a pre-registered buffer (zero-copy).
        
        Args:
            key: Unique key for this data
            buffer_ptr: Pointer to RDMA-registered buffer
            size: Number of bytes to store
            
        Returns:
            Status code (0 = success, negative = error)
        """
        self._ensure_initialized()

        try:
            if hasattr(self._store, "put_from"):
                return self._store.put_from(key, buffer_ptr, size)
            else:
                c_type = ctypes.c_byte * size
                c_array = c_type.from_address(buffer_ptr)
                data = bytes(c_array)
                self._store.put(key, data)
                return 0
        except Exception as e:
            logger.error(f"Failed put_from for key {key}: {e}")
            raise

    def put_tensor(self, key: str, tensor: torch.Tensor) -> int:
        """
        Store a PyTorch tensor using RDMA-registered host buffer.
        
        Copies tensor to host buffer, then uses put_from for zero-copy transfer.
        
        Args:
            key: Unique key for this tensor
            tensor: PyTorch tensor to store
            
        Returns:
            Status code (0 = success, negative = error)
        """
        self._ensure_initialized()

        try:
            host_buf = self._host_buffer_pool.get_buffer()
            nbytes = host_buf.copy_from_tensor(tensor, offset=0)
            return self.put_from(key, host_buf.ptr, nbytes)
        except Exception as e:
            logger.error(f"Failed put_tensor for key {key}: {e}")
            raise

    def batch_put_tensor(
        self,
        keys: List[str],
        tensors: List[torch.Tensor],
    ) -> List[int]:
        """
        Store multiple PyTorch tensors using RDMA-registered host buffer.
        
        Packs all tensors into a host buffer, then stores each one.
        
        Args:
            keys: List of unique keys
            tensors: List of PyTorch tensors to store
            
        Returns:
            List of status codes (0 = success, negative = error)
        """
        self._ensure_initialized()

        try:
            host_buf = self._host_buffer_pool.get_buffer()
            results = []
            offset = 0
            
            offsets_and_sizes = []
            for tensor in tensors:
                nbytes = host_buf.copy_from_tensor(tensor, offset=offset)
                offsets_and_sizes.append((offset, nbytes))
                offset += nbytes

            for key, (off, size) in zip(keys, offsets_and_sizes):
                result = self.put_from(key, host_buf.ptr + off, size)
                results.append(result)

            return results
        except Exception as e:
            logger.error(f"Failed batch_put_tensor: {e}")
            raise

    def get_tensor_into(
        self,
        key: str,
        tensor: torch.Tensor,
    ) -> int:
        """
        Retrieve a tensor directly into a pre-allocated tensor.
        
        Args:
            key: Key of the tensor to retrieve
            tensor: Pre-allocated destination tensor
            
        Returns:
            Number of bytes read (positive = success, negative = error)
        """
        self._ensure_initialized()

        try:
            ptr = tensor.data_ptr()
            size = tensor.numel() * tensor.element_size()
            if ptr not in self._registered_buffers:
                self._register_buffer(ptr, size)
            if hasattr(self._store, "get_into"):
                return self._store.get_into(key, ptr, size)
            else:
                data = self._store.get(key)
                if data is None:
                    raise KeyError(f"Key not found: {key}")
                data_tensor = torch.frombuffer(bytearray(data), dtype=torch.uint8)
                tensor.view(torch.uint8).view(-1).copy_(data_tensor.to(tensor.device))
                return len(data)
        except Exception as e:
            logger.error(f"Failed get_tensor_into for key {key}: {e}")
            raise

    def batch_get_tensor_into(
        self,
        keys: List[str],
        tensors: List[torch.Tensor],
    ) -> List[int]:
        """
        Retrieve multiple tensors directly into pre-allocated tensors.
        
        Args:
            keys: List of keys to retrieve
            tensors: List of pre-allocated destination tensors
            
        Returns:
            List of bytes read for each operation (positive = success, negative = error)
        """
        self._ensure_initialized()

        try:
            buffer_ptrs = []
            sizes = []
            for tensor in tensors:
                ptr = tensor.data_ptr()
                size = tensor.numel() * tensor.element_size()
                if ptr not in self._registered_buffers:
                    self._register_buffer(ptr, size)
                buffer_ptrs.append(ptr)
                sizes.append(size)

            if hasattr(self._store, "batch_get_into"):
                return self._store.batch_get_into(keys, buffer_ptrs, sizes)
            else:
                results = []
                for key, tensor in zip(keys, tensors):
                    result = self.get_tensor_into(key, tensor)
                    results.append(result)
                return results
        except Exception as e:
            logger.error(f"Failed batch_get_tensor_into: {e}")
            raise

    def remove(self, key: str) -> None:
        """Remove data from Mooncake Store."""
        self._ensure_initialized()
        self._store.remove(key)
        logger.debug(f"Removed data with key: {key}")

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
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False


class EagleMooncakeStore(MooncakeHiddenStateStore):
    """
    Mooncake Store wrapper specialized for Eagle3 hidden states.
    
    Uses RDMA-registered host buffers and put_from for zero-copy transfers.
    Each Eagle3 output is stored as multiple tensors with key suffixes:
    - {key}_hs: hidden_states
    - {key}_tgt: target
    - {key}_lm: loss_mask
    - {key}_ids: input_ids
    - {key}_am: attention_mask
    - {key}_lhs: last_hidden_states (if present)
    """

    TENSOR_SUFFIXES = ["_hs", "_tgt", "_lm", "_ids", "_am", "_lhs"]

    def put_eagle3_tensors(
        self,
        key: str,
        hidden_states: torch.Tensor,
        target: Optional[torch.Tensor],
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        """
        Store Eagle3 output tensors using RDMA-registered host buffer.
        
        Args:
            key: Base key (typically task_id)
            hidden_states: Concatenated auxiliary hidden states
            target: Target logits
            loss_mask: Loss mask tensor
            input_ids: Input token IDs
            attention_mask: Attention mask
            last_hidden_states: Optional last hidden states
            
        Returns:
            Dictionary of tensor shapes for reconstruction
        """
        self._ensure_initialized()

        keys = [
            f"{key}_hs",
            f"{key}_lm",
            f"{key}_ids",
            f"{key}_am",
        ]
        tensors = [hidden_states, loss_mask, input_ids, attention_mask]

        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)

        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        results = self.batch_put_tensor(keys, tensors)

        for k, r in zip(keys, results):
            if r != 0:
                raise RuntimeError(f"batch_put_tensor failed for {k} with code: {r}")

        shapes = {
            "hidden_states": tuple(hidden_states.shape),
            "loss_mask": tuple(loss_mask.shape),
            "input_ids": tuple(input_ids.shape),
            "attention_mask": tuple(attention_mask.shape),
        }
        if target is not None:
            shapes["target"] = tuple(target.shape)
        if last_hidden_states is not None:
            shapes["last_hidden_states"] = tuple(last_hidden_states.shape)

        logger.debug(f"Stored Eagle3 tensors with base key: {key}")
        return shapes

    def get_eagle3_tensors_into(
        self,
        key: str,
        shapes: Dict[str, Tuple[int, ...]],
        dtypes: Dict[str, torch.dtype],
        device: torch.device,
    ) -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 tensors directly into pre-allocated GPU tensors.
        
        Args:
            key: Base key used when storing
            shapes: Dictionary of tensor shapes
            dtypes: Dictionary of tensor dtypes
            device: Device to allocate tensors on
            
        Returns:
            Eagle3TargetOutput with tensors on the specified device
        """
        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

        self._ensure_initialized()

        hidden_states = torch.empty(shapes["hidden_states"], dtype=dtypes.get("hidden_states", torch.bfloat16), device=device)
        loss_mask = torch.empty(shapes["loss_mask"], dtype=torch.bool, device=device)
        input_ids = torch.empty(shapes["input_ids"], dtype=torch.int64, device=device)
        attention_mask = torch.empty(shapes["attention_mask"], dtype=torch.int64, device=device)

        keys = [
            f"{key}_hs",
            f"{key}_lm",
            f"{key}_ids",
            f"{key}_am",
        ]
        tensors = [hidden_states, loss_mask, input_ids, attention_mask]

        target = None
        if "target" in shapes:
            target = torch.empty(shapes["target"], dtype=dtypes.get("target", torch.bfloat16), device=device)
            keys.append(f"{key}_tgt")
            tensors.append(target)

        last_hidden_states = None
        if "last_hidden_states" in shapes:
            last_hidden_states = torch.empty(
                shapes["last_hidden_states"],
                dtype=dtypes.get("hidden_states", torch.bfloat16),
                device=device,
            )
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        results = self.batch_get_tensor_into(keys, tensors)

        for k, r in zip(keys, results):
            if r < 0:
                raise RuntimeError(f"batch_get_tensor_into failed for {k} with code: {r}")

        logger.debug(f"Retrieved Eagle3 tensors with base key: {key}")

        return Eagle3TargetOutput(
            hidden_states=hidden_states,
            target=target,
            loss_mask=loss_mask,
            input_ids=input_ids,
            attention_mask=attention_mask,
            last_hidden_states=last_hidden_states,
        )

    def remove_eagle3_tensors(
        self,
        key: str,
        has_last_hidden_states: bool = False,
        has_target: bool = False,
    ) -> None:
        """
        Remove all tensors associated with an Eagle3 output.
        
        Args:
            key: Base key used when storing
            has_last_hidden_states: Whether last_hidden_states was stored
            has_target: Whether target (logits) was stored
        """
        self._ensure_initialized()

        keys = [f"{key}_hs", f"{key}_lm", f"{key}_ids", f"{key}_am"]
        if has_target:
            keys.append(f"{key}_tgt")
        if has_last_hidden_states:
            keys.append(f"{key}_lhs")

        for k in keys:
            try:
                self._store.remove(k)
            except Exception as e:
                logger.warning(f"Failed to remove key {k}: {e}")

        logger.debug(f"Removed Eagle3 tensors with base key: {key}")

    def get_eagle3_output(self, key: str, device: str = "cuda") -> "Eagle3TargetOutput":
        """
        Retrieve Eagle3 output from Mooncake Store (legacy serialization path).
        
        This is kept for backwards compatibility with older workers that use
        pickle serialization instead of the tensor API.
        
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
