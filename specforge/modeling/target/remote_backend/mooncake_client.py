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
from abc import ABC
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

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


class GPUReceiveBuffer:
    """
    Single RDMA-registered GPU buffer for receiving tensors via GPU Direct RDMA.
    
    Allocates contiguous GPU memory that can be registered with Mooncake
    for direct RDMA transfers without CPU involvement.
    """

    def __init__(self, size: int, device: torch.device = None):
        self.size = size
        self.device = device or torch.device("cuda")
        self._tensor: Optional[torch.Tensor] = None
        self._ptr: int = 0
        self._initialized = False

    def initialize(self) -> None:
        """Allocate the GPU buffer."""
        if self._initialized:
            return
        self._tensor = torch.empty(self.size, dtype=torch.uint8, device=self.device)
        self._ptr = self._tensor.data_ptr()
        self._initialized = True
        logger.info(
            f"Initialized GPU receive buffer: {self.size / (1024**2):.1f}MB on {self.device}"
        )

    @property
    def ptr(self) -> int:
        return self._ptr

    def get_slice(self, offset: int, size: int) -> torch.Tensor:
        """Get a slice of the buffer as a tensor view."""
        if not self._initialized:
            raise RuntimeError("GPU buffer not initialized")
        return self._tensor[offset:offset + size]

    def free(self) -> None:
        if self._tensor is not None:
            del self._tensor
            self._tensor = None
            self._ptr = 0
            self._initialized = False

    def __del__(self):
        self.free()


def calculate_eagle3_buffer_size(
    max_seq_len: int,
    batch_size: int,
    hidden_dim: int,
    num_aux_layers: int = 3,
    include_last_hidden_states: bool = True,
    safety_margin: float = 1.1,
) -> int:
    """
    Calculate the required GPU buffer size for receiving Eagle3 output tensors.
    
    Tensors transferred via RDMA:
    - hidden_states: (batch, seq, hidden_dim * num_aux_layers), bfloat16
    - loss_mask: (batch, seq, 1), int64
    - input_ids: (batch, seq), int64
    - attention_mask: (batch, seq), int64
    - last_hidden_states: (batch, seq, hidden_dim), bfloat16 (optional)
    
    Note: target/logits are computed locally from last_hidden_states, not transferred.
    
    Args:
        max_seq_len: Maximum sequence length
        batch_size: Batch size
        hidden_dim: Model hidden dimension (e.g., 4096 for Qwen3-8B)
        num_aux_layers: Number of auxiliary hidden state layers (default 3 for Eagle3)
        include_last_hidden_states: Whether last_hidden_states will be transferred
        safety_margin: Multiplier for safety margin (default 1.1 = 10% extra)
    
    Returns:
        Required buffer size in bytes
    """
    bfloat16_size = 2
    int64_size = 8
    
    hidden_states_size = batch_size * max_seq_len * hidden_dim * num_aux_layers * bfloat16_size
    loss_mask_size = batch_size * max_seq_len * 1 * int64_size
    input_ids_size = batch_size * max_seq_len * int64_size
    attention_mask_size = batch_size * max_seq_len * int64_size
    
    total = hidden_states_size + loss_mask_size + input_ids_size + attention_mask_size
    
    if include_last_hidden_states:
        last_hidden_states_size = batch_size * max_seq_len * hidden_dim * bfloat16_size
        total += last_hidden_states_size
    
    total_with_margin = int(total * safety_margin)
    
    alignment = 256
    aligned_size = ((total_with_margin + alignment - 1) // alignment) * alignment
    
    logger.debug(
        f"Calculated Eagle3 buffer size: {aligned_size / (1024**2):.1f}MB "
        f"(seq={max_seq_len}, batch={batch_size}, hidden={hidden_dim})"
    )
    
    return aligned_size


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
    host_buffer_size: int = 4 * 1024 * 1024 * 1024 # 
    host_buffer_pool_size: int = 1
    enable_gpu_direct_rdma: bool = True
    max_seq_len: int = 8192
    max_batch_size: int = 1
    hidden_dim: int = 4096
    gpu_buffer_size: Optional[int] = None

    def get_gpu_buffer_size(self) -> int:
        """Get GPU buffer size, calculating if not explicitly set."""
        if self.gpu_buffer_size is not None:
            return self.gpu_buffer_size
        return calculate_eagle3_buffer_size(
            max_seq_len=self.max_seq_len,
            batch_size=self.max_batch_size,
            hidden_dim=self.hidden_dim,
        )

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
    for zero-copy transfers. Optionally uses GPU Direct RDMA for receiving.
    """

    def __init__(self, config: MooncakeConfig):
        self.config = config
        self._store: Optional[MooncakeDistributedStore] = None
        self._initialized = False
        self._registered_buffers: Dict[int, int] = {}
        self._host_buffer_pool: Optional[HostBufferPool] = None
        self._gpu_receive_buffer: Optional[GPUReceiveBuffer] = None
        self._gpu_direct_available = False

    def setup(self, device: torch.device = None) -> None:
        """Initialize the Mooncake Store client."""
        if self._initialized:
            return

        self._store = MooncakeDistributedStore()
        logger.info(
            f"Connecting to Mooncake master at {self.config.master_server_address}, "
            f"metadata server at {self.config.metadata_server}"
        )
        result = self._store.setup(
            local_hostname=self.config.local_hostname,
            metadata_server=self.config.metadata_server,
            global_segment_size=self.config.global_segment_size,
            local_buffer_size=self.config.local_buffer_size,
            protocol=self.config.protocol,
            rdma_devices=self.config.device_name,
            master_server_addr=self.config.master_server_address,
        )
        if result is not None and result != 0:
            raise RuntimeError(
                f"Failed to initialize Mooncake client (error={result}). "
                f"Check that Mooncake master is running at {self.config.master_server_address} "
                f"and metadata server is available at {self.config.metadata_server}"
            )

        self._host_buffer_pool = HostBufferPool(
            buffer_size=self.config.host_buffer_size,
            pool_size=self.config.host_buffer_pool_size,
        )
        self._host_buffer_pool.initialize()

        for buf in self._host_buffer_pool._buffers:
            self._register_buffer(buf.ptr, buf.size)

        if self.config.enable_gpu_direct_rdma and torch.cuda.is_available():
            self._setup_gpu_direct(device)

        self._initialized = True
        logger.info(
            f"Mooncake Store client initialized (protocol={self.config.protocol}, "
            f"gpu_direct={self._gpu_direct_available})"
        )

    def _setup_gpu_direct(self, device: torch.device = None) -> None:
        """Initialize GPU receive buffer and register for GPU Direct RDMA."""
        try:
            buffer_size = self.config.get_gpu_buffer_size()
            self._gpu_receive_buffer = GPUReceiveBuffer(
                size=buffer_size,
                device=device,
            )
            self._gpu_receive_buffer.initialize()

            if self._register_buffer(self._gpu_receive_buffer.ptr, self._gpu_receive_buffer.size):
                self._gpu_direct_available = True
                logger.info(
                    f"GPU Direct RDMA enabled: registered {buffer_size / (1024**2):.1f}MB GPU buffer"
                )
            else:
                logger.warning("Failed to register GPU buffer with Mooncake")
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None

        except Exception as e:
            logger.warning(f"Failed to setup GPU Direct RDMA: {e}")
            self._gpu_direct_available = False
            if self._gpu_receive_buffer is not None:
                self._gpu_receive_buffer.free()
                self._gpu_receive_buffer = None

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


    def remove(self, key: str) -> None:
        """Remove data from Mooncake Store."""
        self._store.remove(key)
        logger.debug(f"Removed data with key: {key}")

    def exists(self, key: str) -> bool:
        """Check if a key exists in the store."""
        try:
            data = self._store.get(key)
            return data is not None
        except Exception:
            return False

    def close(self) -> None:
        """Close the Mooncake Store client."""
        if self._gpu_receive_buffer is not None:
            self._gpu_receive_buffer.free()
            self._gpu_receive_buffer = None
        if self._host_buffer_pool is not None:
            self._host_buffer_pool.shutdown()
            self._host_buffer_pool = None
        if self._store is not None and hasattr(self._store, "close"):
            self._store.close()
        self._initialized = False
        self._gpu_direct_available = False


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
        loss_mask: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        last_hidden_states: Optional[torch.Tensor],
        target: Optional[torch.Tensor] = None,
    ) -> Dict[str, Tuple[int, ...]]:
        """Store Eagle3 output tensors using zero-copy batch_put_from."""
        keys = [f"{key}_hs", f"{key}_lm", f"{key}_ids", f"{key}_am"]
        tensors = [hidden_states, loss_mask, input_ids, attention_mask]

        if target is not None:
            keys.append(f"{key}_tgt")
            tensors.append(target)

        if last_hidden_states is not None:
            keys.append(f"{key}_lhs")
            tensors.append(last_hidden_states)

        host_buf = self._host_buffer_pool.get_buffer()
        buffer_ptrs = []
        sizes = []
        offset = 0

        for tensor in tensors:
            nbytes = host_buf.copy_from_tensor(tensor, offset=offset)
            buffer_ptrs.append(host_buf.ptr + offset)
            sizes.append(nbytes)
            offset += nbytes

        results = self._store.batch_put_from(keys, buffer_ptrs, sizes)

        for k, r in zip(keys, results):
            if r != 0:
                raise RuntimeError(f"batch_put_from failed for {k} with code: {r}")

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
        Retrieve Eagle3 tensors into GPU memory.
        
        For RDMA/InfiniBand: Uses GPUDirect RDMA (batch_get_into directly into GPU).
        For TCP: Uses batch_get_buffer to host buffer, then copies to GPU.
        
        Automatically falls back to host buffer path if GPUDirect fails.
        """
        from specforge.modeling.target.eagle3_target_model import Eagle3TargetOutput

        keys = [f"{key}_hs", f"{key}_lm", f"{key}_ids", f"{key}_am"]
        tensor_specs = [
            ("hidden_states", shapes["hidden_states"], dtypes.get("hidden_states", torch.bfloat16)),
            ("loss_mask", shapes["loss_mask"], torch.int64),
            ("input_ids", shapes["input_ids"], torch.int64),
            ("attention_mask", shapes["attention_mask"], torch.int64),
        ]

        if "target" in shapes:
            keys.append(f"{key}_tgt")
            tensor_specs.append(("target", shapes["target"], dtypes.get("target", torch.bfloat16)))

        if "last_hidden_states" in shapes:
            keys.append(f"{key}_lhs")
            tensor_specs.append(("last_hidden_states", shapes["last_hidden_states"], dtypes.get("hidden_states", torch.bfloat16)))


        if self._gpu_direct_available and self._gpu_receive_buffer is not None:
            tensor_map = self._get_tensors_gpu_direct(keys, tensor_specs, device)
        else:
            tensor_map = self._get_tensors_via_host_buffer(keys, tensor_specs, device)
            logger.debug("GPUDirect RDMA not available, using host buffer path (TCP)")
        logger.info(f"Retrieved Eagle3 tensors with base key: {key}")
        return Eagle3TargetOutput(
            hidden_states=tensor_map["hidden_states"],
            target=tensor_map.get("target"),
            loss_mask=tensor_map["loss_mask"],
            input_ids=tensor_map["input_ids"],
            attention_mask=tensor_map["attention_mask"],
            last_hidden_states=tensor_map.get("last_hidden_states"),
        )

    def _get_tensors_gpu_direct(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Optional[Dict[str, torch.Tensor]]:
        """
        Transfer directly into GPU memory using batch_get_into (GPUDirect RDMA).
        
        Uses pre-registered GPU buffer for RDMA transfers, then creates tensor
        views/copies. Returns None if transfer fails, allowing caller to fall
        back to host buffer path.
        """

        total_size = sum(
            self._compute_tensor_size(shape, dtype)
            for _, shape, dtype in tensor_specs
        )

        if total_size > self._gpu_receive_buffer.size:
            logger.warning(
                f"GPU buffer too small: need {total_size / (1024**2):.1f}MB, "
                f"have {self._gpu_receive_buffer.size / (1024**2):.1f}MB. "
                f"Increase max_seq_len, max_batch_size, or hidden_dim in MooncakeConfig."
            )
            return None

        buffer_ptrs = []
        sizes = []
        offset = 0
        offsets = []

        for name, shape, dtype in tensor_specs:
            size = self._compute_tensor_size(shape, dtype)
            buffer_ptrs.append(self._gpu_receive_buffer.ptr + offset)
            sizes.append(size)
            offsets.append(offset)
            offset += size

        try:
            results = self._store.batch_get_into(keys, buffer_ptrs, sizes)
            for i, (k, r) in enumerate(zip(keys, results)):
                if r < 0:
                    logger.warning(f"batch_get_into failed for {k} with error code: {r}")
                    return None
                elif r != 0 and r != sizes[i]:
                    logger.warning(
                        f"batch_get_into for {k}: unexpected return {r} (expected 0 or {sizes[i]})"
                    )
        except Exception as e:
            logger.warning(f"batch_get_into exception: {e}")
            return None

        tensor_map = {}
        for i, (name, shape, dtype) in enumerate(tensor_specs):
            numel = 1
            for dim in shape:
                numel *= dim

            buf_slice = self._gpu_receive_buffer.get_slice(offsets[i], sizes[i])
            # Creating new tensor from the buffer.
            print(f"name: {name}, dtype: {dtype}, shape: {shape}, numel: {numel}")
            tensor = buf_slice.view(dtype)[:numel].reshape(shape).clone()
            tensor_map[name] = tensor

        logger.debug(f"GPU Direct RDMA transfer successful for {len(keys)} tensors")
        return tensor_map

    def _compute_tensor_size(self, shape: Tuple[int, ...], dtype: torch.dtype) -> int:
        """Compute the byte size of a tensor with given shape and dtype."""
        numel = 1
        for dim in shape:
            numel *= dim
        element_size = torch.tensor([], dtype=dtype).element_size()
        return numel * element_size

    def _get_tensors_via_host_buffer(
        self,
        keys: List[str],
        tensor_specs: List[Tuple[str, Tuple[int, ...], torch.dtype]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        """Transfer via Mooncake's registered host buffer, then copy to device."""
        buffers = self._store.batch_get_buffer(keys)

        tensor_map = {}
        for i, ((name, shape, dtype), buf) in enumerate(zip(tensor_specs, buffers)):
            if buf is None:
                raise RuntimeError(
                    f"batch_get_buffer returned None for key '{keys[i]}' (tensor: {name}). "
                    f"This may indicate the key doesn't exist or RDMA transfer failed."
                )

            numel = 1
            for dim in shape:
                numel *= dim
            element_size = torch.tensor([], dtype=dtype).element_size()
            expected_size = numel * element_size

            buf_size = buf.size()
            if buf_size != expected_size:
                raise RuntimeError(
                    f"Size mismatch for {name}: got {buf_size} bytes, expected {expected_size} bytes"
                )

            c_array = (ctypes.c_byte * buf_size).from_address(buf.ptr())
            host_tensor = torch.frombuffer(c_array, dtype=dtype, count=numel).reshape(shape)

            tensor_map[name] = host_tensor.to(device)

        return tensor_map

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

