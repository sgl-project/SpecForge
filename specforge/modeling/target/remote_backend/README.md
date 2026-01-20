# Remote Backend for Distributed Eagle3 Training

This module provides a distributed inference backend that separates training and inference onto different nodes, enabling efficient Eagle3 training at scale.

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────────┐
│                              Training Nodes (DP)                                │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  RemoteEagle3TargetModel                                                │   │
│  │  ┌─────────────────┐  ┌─────────────────────┐  ┌────────────────────┐   │   │
│  │  │  TaskProducer   │  │NotificationSubscriber│  │Eagle3ZeroCopyReader│  │   │
│  │  │  (ZMQ PUSH)     │  │ (ZMQ SUB)           │  │ (GPUBufferPool)    │   │   │
│  │  └────────┬────────┘  └──────────▲──────────┘  └─────────▲──────────┘   │   │
│  │           │connect              │connect                 │              │   │
│  │  ┌────────┴───────────────────────────────────────────────────────────┐│   │
│  │  │                        EagleMooncakeStore                          ││   │
│  │  └────────────────────────────────────────────────────────────────────┘│   │
│  └───────────│──────────────────────│───────────────────────│──────────────┘   │
└──────────────│──────────────────────│───────────────────────│──────────────────┘
               │                      │                       │
               │ Tasks                │ Notifications         │ Hidden States
               │                      │                       │ (RDMA/TCP)
               ▼                      │                       │
┌──────────────────────────────────┐  │  ┌────────────────────┴──────────────────┐
│      (Optional) Brokers          │  │  │           Mooncake Store              │
│  ┌────────────────────────────┐  │  │  │  ┌────────────────────────────────┐   │
│  │  TaskQueueBroker           │  │  │  │  │ Distributed Hidden State Store |   │
│  │  PULL ──────────► PUSH     │  │  │  │  │  (Host Memory / RDMA Segments) │   │
│  ├────────────────────────────┤  │  │  │  └────────────────────────────────┘   │
│  │  NotificationBroker        │  │  │  └────────────────────▲──────────────────┘
│  │  XSUB ──────────► XPUB     │  │  │                       │
│  └────────────────────────────┘  │  │                       │
└──────────────┬───────────────────┘  │                       │
               │                      │                       │
               ▼                      │                       │
┌─────────────────────────────────────┴───────────────────────│──────────────────┐
│                           Inference Workers (TP)                               │
│  ┌─────────────────────────────────────────────────────────────────────────┐   │
│  │  InferenceWorker                                                        │   │
│  │  ┌─────────────────┐  ┌─────────────────────┐  ┌────────────────────┐   │   │
│  │  │  TaskConsumer   │  │NotificationPublisher│  │Eagle3HostBuffer    │   │   │
│  │  │  (ZMQ PULL)     │  │ (ZMQ PUB)           │  │Writer (Pack→Host)  │   │   │
│  │  │  bind           │  │ bind                │  └────────────────────┘   │   │
│  │  └─────────────────┘  └─────────────────────┘                           │   │
│  │  ┌────────────────────────────────────────────────────────────────────┐ │   │
│  │  │                        EagleMooncakeStore                          │ │   │
│  │  └────────────────────────────────────────────────────────────────────┘ │   │
│  │  ┌─────────────────────────────────────────────────────────────────┐   │   │
│  │  │  SGLangEagle3TargetModel (SGLang Runtime)                       │   │   │
│  │  │  - Runs forward pass with hidden state extraction               │   │   │
│  │  │  - Returns aux_hidden_states, logits                            │   │   │
│  │  └─────────────────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────────────────────┘
```

## Components

### Training Side (`remote_target_model.py`)

| Component | Description |
|-----------|-------------|
| `RemoteEagle3TargetModel` | Implements `Eagle3TargetModel` interface by delegating to remote workers. Supports both sync (`generate_eagle3_data`) and async (`submit_task`/`get_result`) modes. |
| `TaskProducer` | ZeroMQ PUSH socket that connects to worker's PULL socket. |
| `NotificationSubscriber` | ZeroMQ SUB socket with background listener thread for receiving completion notifications filtered by task_id. |
| `Eagle3ZeroCopyReader` | Manages `GPUBufferPool` and unpacks results from Mooncake directly into GPU memory via RDMA. |
| `EagleMooncakeStore` | Client wrapper for retrieving hidden states from Mooncake Store. |

### Inference Side (`inference_worker.py`)

| Component | Description |
|-----------|-------------|
| `InferenceWorker` | Main worker service. Binds to addresses, pulls tasks, runs inference via SGLang, stores results in Mooncake. Supports TP via torchrun (rank 0 handles ZMQ, all ranks participate in inference). |
| `InferenceWorkerManager` | Manages multiple workers in a single process (for testing or single-node multi-GPU setups). |
| `TaskConsumer` | ZeroMQ PULL socket that binds to address (workers bind, trainers connect). |
| `NotificationPublisher` | ZeroMQ PUB socket that binds to address for broadcasting task completion with task_id as topic. |
| `Eagle3HostBufferWriter` | Packs Eagle3 output into RDMA-registered host memory buffer (`HostBuffer`) for zero-copy storage. |
| `EagleMooncakeStore` | Client wrapper for storing hidden states in Mooncake Store. |
| `SGLangEagle3TargetModel` | The actual inference backend using SGLang Runtime. |

### Communication (`task_queue.py`, `messages.py`)

| Component | Description |
|-----------|-------------|
| `InferenceTask` | Message containing input_ids, attention_mask, loss_mask, aux_hidden_states_layers (serialized tensors). |
| `TaskNotification` | Completion message with task_id, status, mooncake_key, tensor_shapes, data_size, error_message. |
| `TaskQueueBroker` | Optional broker for task distribution. Training PUSH → Broker PULL/PUSH → Worker PULL. |
| `NotificationBroker` | Optional broker for notifications. Worker PUB → Broker XSUB/XPUB → Training SUB. |
| `QueueConfig` | Configuration for task queue and notification addresses, timeouts, HWM. |

### Storage (`mooncake_client.py`, `zero_copy.py`)

| Component | Description |
|-----------|-------------|
| `EagleMooncakeStore` | Specialized store for Eagle3 outputs. Extends `MooncakeHiddenStateStore` with put/get methods for Eagle3TargetOutput. |
| `MooncakeHiddenStateStore` | Base client wrapper for Mooncake Store. Handles raw bytes, GPU buffer registration for RDMA. |
| `HostBuffer` | Pre-allocated host buffer using `MooncakeHostTensorAllocator` for RDMA-compatible memory. Used by inference workers to pack outputs. |
| `Eagle3HostBufferWriter` | Helper for inference workers to write outputs into host buffers without serialization overhead. |
| `GPUBuffer` | Pre-allocated GPU buffer that can be registered with Mooncake for RDMA transfers. Used by training nodes. |
| `GPUBufferPool` | Pool of GPU buffers for concurrent zero-copy reads on training side. |
| `Eagle3ZeroCopyReader` | Helper for training nodes to read outputs via RDMA directly into GPU buffers. |
| `MooncakeConfig` | Configuration for Mooncake Store connection (master address, segment sizes, protocol). |

## Data Flow

1. **Task Submission**: Training node creates `InferenceTask` with serialized tensors, subscribes to task_id notifications, and pushes to the worker's task queue via `TaskProducer`.

2. **Task Processing**: Inference worker's `TaskConsumer` pulls task (rank 0 broadcasts to other TP ranks), deserializes tensors, runs forward pass via `SGLangEagle3TargetModel.extend()`.

3. **Result Storage**: Worker packs `Eagle3TargetOutput` (aux_hidden_states, target logits, loss_mask, input_ids, attention_mask) into `HostBuffer` via `Eagle3HostBufferWriter`, then stores in Mooncake via `put_from_host_buffer()` using task_id as key.

4. **Completion Notification**: Worker publishes `TaskNotification` with task_id as ZMQ topic, including status, mooncake_key, tensor_shapes, and data_size.

5. **Result Retrieval**: Training node's `NotificationSubscriber` receives notification, then either:
   - **Zero-copy path**: `Eagle3ZeroCopyReader.unpack_rdma_to_gpu()` transfers data directly to GPU buffer via RDMA
   - **Fallback path**: `EagleMooncakeStore.get_eagle3_output()` deserializes via pickle

## Zero-Copy Transfer

The system supports two data transfer modes:

### Serialized Mode (Default Fallback)
- Uses pickle to serialize/deserialize tensors via `Eagle3OutputData`
- Data flow: GPU → CPU → Pickle → Mooncake → Unpickle → CPU → GPU

### Zero-Copy Mode (Recommended)
- Packs tensors into aligned contiguous host buffer (`HostBuffer` via `MooncakeHostTensorAllocator`)
- Memory layout: `[Header 64B] [pad] [hidden_states] [pad] [target] [pad] [loss_mask] [pad] [input_ids] [pad] [attention_mask] [pad] [last_hidden_states?]`
- Each tensor aligned to 64 bytes for optimal RDMA
- Header contains tensor sizes (6 × uint64), dtypes (2 × 4B), and total_size

**Data flow:**
```
Inference Worker:
  GPU tensors → Eagle3HostBufferWriter.pack_eagle3_output()
             → HostBuffer (RDMA-registered host memory)
             → EagleMooncakeStore.put_from_host_buffer()

Training Node:
  EagleMooncakeStore.get_into_gpu_buffer() → GPUBuffer (RDMA to GPU)
             → Eagle3ZeroCopyReader.unpack_rdma_to_gpu()
             → Eagle3TargetOutput tensors on GPU
```

- With RDMA: Remote Host Memory → RDMA → Local GPU (bypasses local CPU)
- Without RDMA: Remote Host Memory → TCP → Local GPU buffer → Clone

## Usage

### Running Inference Workers

```bash
python -m specforge.modeling.target.remote_backend \
    --model-path Qwen/Qwen3-8B \
    --tp-size 4 \
    --task-queue-addr tcp://broker:5555 \
    --notify-addr tcp://broker:5556 \
    --mooncake-master-addr mooncake-master:50051 \
    --use-zero-copy true
```

### Training Side Integration

```python
from specforge.modeling.target.remote_backend import (
    RemoteEagle3TargetModel,
    RemoteBackendConfig,
    MooncakeConfig,
)

config = RemoteBackendConfig(
    task_queue_addr="tcp://broker:5555",
    notify_addr="tcp://broker:5556",
    use_zero_copy=True,
    mooncake_config=MooncakeConfig(
        master_server_address="mooncake-master:50051",
    ),
)

model = RemoteEagle3TargetModel(config)
model.connect()

# Option 1: Synchronous (simple, but blocks)
output = model.generate_eagle3_data(
    input_ids=batch["input_ids"],
    attention_mask=batch["attention_mask"],
    loss_mask=batch["loss_mask"],
)

# Option 2: Asynchronous with prefetching (recommended for training)
# Submit tasks ahead of time to keep inference pipeline saturated
prefetch_depth = 4
pending_task_ids = []
pending_data = []

for i, batch in enumerate(dataloader):
    # Submit new task
    task_id = model.submit_task(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        loss_mask=batch["loss_mask"],
    )
    pending_task_ids.append(task_id)
    pending_data.append(batch)
    
    # Once we have enough in flight, start retrieving results
    if len(pending_task_ids) > prefetch_depth:
        oldest_task_id = pending_task_ids.pop(0)
        oldest_data = pending_data.pop(0)
        eagle3_data = model.get_result(oldest_task_id)
        # Use oldest_data and eagle3_data for training step

# Drain remaining tasks at end of epoch
while pending_task_ids:
    task_id = pending_task_ids.pop(0)
    data = pending_data.pop(0)
    eagle3_data = model.get_result(task_id)
    # Process final batches
```

When using `train_eagle3.py` with `--target-model-backend remote`, prefetching is 
automatically handled via the `RemotePrefetchIterator`. Set `--prefetch-depth N` 
to control how many batches are submitted ahead of time (default: 4).

## Configuration

### RemoteBackendConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `task_queue_addr` | `tcp://localhost:5555` | ZeroMQ task queue address |
| `notify_addr` | `tcp://localhost:5556` | ZeroMQ notification address |
| `task_timeout` | 300.0 | Timeout in seconds for task completion |
| `zero_copy_pool_size` | 4 | Number of GPU buffers in pool |
| `dp_rank` | 0 | Data parallel rank (for task ID uniqueness) |

### MooncakeConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `local_hostname` | `localhost` | Local hostname for Mooncake registration |
| `metadata_server` | `http://localhost:8090/metadata` | Mooncake metadata server URL |
| `master_server_address` | `localhost:50051` | Mooncake master gRPC address |
| `global_segment_size` | 4GB | Memory contributed to distributed pool |
| `local_buffer_size` | 512MB | Buffer for receiving data via Get() |
| `protocol` | `tcp` | Transfer protocol (`tcp` or `rdma`) |
| `device_name` | `""` | RDMA device name (empty for TCP) |

### InferenceWorkerConfig

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model_path` | (required) | Path to the model |
| `tp_size` | 1 | Tensor parallelism size |
| `dtype` | `bfloat16` | Model dtype |
| `mem_fraction_static` | 0.8 | GPU memory fraction for KV cache |
| `attention_backend` | `flashinfer` | Attention backend |
| `disable_cuda_graph` | False | Disable CUDA graph optimization |

## Network Topology

### Direct Connection (Default)
```
                    Tasks (ZMQ)                        Notifications (ZMQ)
Training ──PUSH connect──► Worker PULL bind    Worker PUB bind ──► Training SUB connect

                    Hidden States (Mooncake)
Worker Host Memory ────────────────────────────► Training GPU Buffer
                        (RDMA or TCP)
```

In direct mode:
- Workers **bind** to addresses (TaskConsumer, NotificationPublisher)
- Training nodes **connect** to worker addresses (TaskProducer, NotificationSubscriber)

### Broker-Based (Scalable)
```
Training ──PUSH──► TaskQueueBroker (PULL/PUSH) ──► Worker PULL connect
Worker PUB connect ──► NotificationBroker (XSUB/XPUB) ──► Training SUB
```

When using brokers:
- Brokers bind to frontend and backend addresses
- Both training and workers connect to broker addresses

The broker-based topology is recommended for:
- Dynamic scaling of workers
- Centralized connection management
- Multi-node deployments where direct connectivity is complex
# 1. Send a message, receive the tensor, run inference

