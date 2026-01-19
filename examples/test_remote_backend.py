import torch
from mooncake.store import MooncakeDistributedStore

# Initialize store with RDMA protocol for maximum performance
store = MooncakeDistributedStore()
store.setup(
    "localhost",           # Your node's address
    "http://localhost:8090/metadata",    # HTTP metadata server
    512*1024*1024,          # 512MB segment size
    128*1024*1024,          # 128MB local buffer
    "rdma",                             # Use TCP (RDMA for high performance)
    "mlx5_0",                            # Leave empty; Mooncake auto-picks RDMA devices when needed
    "localhost:50051"        # Master service
)

# Create data to store
original_data = torch.randn(1000, 1000).to(torch.float32)
buffer_ptr = original_data.data_ptr()
size = original_data.nbytes

# Step 1: Register the buffer
result = store.register_buffer(buffer_ptr, size)
if result != 0:
    raise RuntimeError(f"Failed to register buffer: {result}")

# Step 2: Zero-copy store
result = store.put_from("large_tensor", buffer_ptr, size)
if result == 0:
    print(f"Successfully stored {size} bytes with zero-copy")
else:
    raise RuntimeError(f"Store failed with code: {result}")

# Step 3: Pre-allocate buffer for retrieval
retrieved_data = torch.empty((1000, 1000), dtype=torch.float32)
recv_buffer_ptr = retrieved_data.data_ptr()
recv_size = retrieved_data.nbytes

# Step 4: Register receive buffer
result = store.register_buffer(recv_buffer_ptr, recv_size)
if result != 0:
    raise RuntimeError(f"Failed to register receive buffer: {result}")

# Step 5: Zero-copy retrieval
bytes_read = store.get_into("large_tensor", recv_buffer_ptr, recv_size)
if bytes_read > 0:
    print(f"Successfully retrieved {bytes_read} bytes with zero-copy")
    # Verify the data
    print(f"Data matches: {torch.allclose(original_data, retrieved_data)}")
else:
    raise RuntimeError(f"Retrieval failed with code: {bytes_read}")

# Step 6: Clean up - unregister both buffers
store.unregister_buffer(buffer_ptr)
store.unregister_buffer(recv_buffer_ptr)
store.close()