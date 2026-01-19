import ctypes
import torch
from mooncake.store import MooncakeHostMemAllocator


class MooncakeHostTensorAllocator:
    def __init__(self):
        super().__init__()
        self.allocator = MooncakeHostMemAllocator()
        self.ptr = None

    def allocate(
        self, dims: tuple, dtype: torch.dtype, device: str = "cpu"
    ) -> torch.Tensor:
        """
        Allocates memory using MooncakeHostMemAllocator and wraps it in a PyTorch tensor.
        """
        self.dims = dims
        self.dtype = dtype
        size = 1
        for d in dims:
            size *= d
        size *= torch.tensor([], dtype=self.dtype).element_size()
        ptr_int = self.allocator.alloc(size)
        self.ptr = ptr_int
        c_type = ctypes.c_byte * size
        c_array = c_type.from_address(ptr_int)

        tensor = torch.frombuffer(c_array, dtype=torch.uint8, count=size)

        if dtype != torch.uint8:
            element_size = torch.tensor([], dtype=dtype).element_size()
            assert size % element_size == 0, "Size must be divisible by element size"
            tensor = tensor.view(dtype)

        return tensor.view(dims)


if __name__ == "__main__":
    allocator = MooncakeHostTensorAllocator()
    
    hidden_states = allocator.allocate((16, 10000, 1024), dtype=torch.bfloat16)
    logits = allocator.allocate((16, 1000, 30000), dtype=torch.float32)
    
    print(f"hidden_states shape: {hidden_states.shape}, dtype: {hidden_states.dtype}")
    print(f"logits shape: {logits.shape}, dtype: {logits.dtype}")
    print(f"Memory allocated via Mooncake at ptr: {allocator.ptr}")

