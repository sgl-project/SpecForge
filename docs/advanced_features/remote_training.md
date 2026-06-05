# 🌐 Train-Inference Disaggregation (Remote Training)

## 📍 Overview

SpecForge supports **train-inference disaggregation**, which deploys the target model as an independent inference server while the draft model trains on a separate GPU. This architecture eliminates GPU memory contention and enables pipeline parallelism between target inference and draft training.

In the traditional co-located setup, the target model (e.g., Qwen3-30B-A3B-FP8) and draft model training share the same GPU, causing:
- **Memory contention**: 30B model FP8 weights + KV Cache consume significant GPU memory, squeezing training memory
- **Compute serialization**: Target inference and draft training cannot overlap, wasting GPU cycles
- **Scaling limitations**: Inference and training resources cannot scale independently

The disaggregated architecture solves these by separating the target model into a standalone inference service:

```
┌──────────────────────────┐          ┌──────────────────────────┐
│   Training Client        │          │   Target Model Server    │
│   Draft Model + Optimizer│          │   SGLang Target Model    │
│                          │  HTTP    │                          │
│  requests / metadata     │ ───────► │  forward / setup APIs    │
│                          │ ◄─────── │                          │
│  large CUDA tensors      │ ═NCCL══► │  hidden states / target_p│
└──────────────────────────┘          └──────────────────────────┘
```

- **HTTP control plane**: Request scheduling, metadata transport, health checks, configuration sync
- **NCCL data plane**: GPU-to-GPU large tensor transfer (supports intra-node NVLink / inter-node RDMA)
- **Wire format fallback**: Compact binary format when NCCL is unavailable

## 🏗️ Architecture

### Server

`launch_target_server.py` is launched via `torchrun` with support for TP>1 multi-rank:

- **Rank 0**: Runs the HTTP server, processes requests, and performs NCCL send
- **Rank 1+**: Participates in TP forward, syncs requests via `broadcast_object_list`

Request processing flow:
1. Receive HTTP POST (input_ids, attention_mask, and other metadata)
2. Broadcast request to all TP ranks
3. Execute target model forward (SGLang backend)
4. Compute `target_p` (softmax + optional top-k compression)
5. Return hidden_states / target_p to client via NCCL send

### Client

`remote_target_client.py` serves as the target model backend in the training script:

- On first request, initializes the NCCL data channel via POST `/init_nccl`
- Retrieves model configuration (hidden_size, vocab_size, etc.) via POST `/setup`
- Each training step sends a request via POST `/generate` and receives results via NCCL recv
- Supports TP>1 training: only rank 0 sends requests, results are broadcast to other ranks
- After the first successful connection, a background heartbeat is started; on `close()`, a best-effort `/disconnect` is sent

### Client Lifecycle and Automatic Exit

The target server tracks client activity and automatically shuts down after the client exits, preventing leftover GPU-occupying server processes after training completes:

- After the client's first successful request or successful NCCL initialization, a background heartbeat thread is started, sending POST `/heartbeat` every 15 seconds by default
- When the client exits normally, it sends a best-effort POST `/disconnect`; upon receiving it, the server immediately triggers shutdown
- When the client exits abnormally, the server watchdog triggers shutdown after `--client-heartbeat-timeout` is exceeded (default 60 seconds)
- The server only counts actual client API calls as active requests; `GET /health` and unrelated POSTs do not renew the watchdog timer
- `--client-heartbeat-timeout 0` disables the server-side timeout watchdog, but `/disconnect` will still trigger automatic shutdown

Since NCCL transport does not support safe disconnect and reconnect within the same server process, it is recommended to treat each target server process as a resource for a single training session: it automatically exits after training completes or the client disconnects, and a new instance is started for the next training run.

### NCCL Transport

`_nccl_transport.py` implements a dedicated NCCL transport layer:

- Server = rank 0, Client = rank 1 forming a 2-process NCCL group
- TCP rendezvous via `torch.distributed.TCPStore`
- Supports intra-node (NVLink) and inter-node (RDMA/RoCE) transfers
- `SPECFORGE_NCCL_PORT` controls the rendezvous port (default: HTTP port + 100)

### Prefetch Pipeline

The prefetch mechanism overlaps target model inference with draft model training:

```
Timeline (depth=2, 2 servers round-robin):
─────────────────────────────────────────────────────────────
Server A: [req1]────────[req3]────────[req5]────────
Server B:     [req2]────────[req4]────────[req6]────
Client:   [train1][train2][train3][train4][train5]──
                  ↑ req1 ready    ↑ req3 ready
```

- `fill_prefetch_queue()` maintains up to `depth` in-flight requests
- Multiple servers are dispatched via `itertools.cycle` round-robin
- At the start of each training step, `future.result()` retrieves the completed prefetch result
- When server latency < training step time, the server is fully overlapped

## 🚀 Usage

### Launch Target Server

```bash
# Single-GPU server (recommended)
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/Qwen3-30B-A3B-FP8 \
  --mode eagle3 \
  --port 8001 \
  --tp-size 1 \
  --mem-fraction-static 0.4 \
  --trust-remote-code \
  --attention-backend flashinfer
```

After startup, the server prints `listening on 0.0.0.0:8001`. Verify readiness with `curl http://<host>:8001/health`.

### Launch Training (Single Server)

```bash
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nproc_per_node=1 --master-port 29600 \
  scripts/train_eagle3.py \
  --target-model-path /path/to/Qwen3-30B-A3B-FP8 \
  --target-model-backend remote \
  --remote-url http://<server-host>:8001 \
  --target-prefetch-depth 1 \
  --train-data-path /path/to/data.jsonl \
  --max-length 10240 \
  --batch-size 1 \
  --tp-size 1 \
  --trust-remote-code \
  --is-preformatted \
  --output-dir /path/to/output
```

### Launch Training (Dual Server, Recommended)

Start one server on each machine and specify multiple URLs on the training side:

```bash
# Machine A: launch server
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/model --mode eagle3 --port 8001 --tp-size 1 \
  --mem-fraction-static 0.4 --trust-remote-code --attention-backend flashinfer

# Machine B: launch server
CUDA_VISIBLE_DEVICES=0 python -m torch.distributed.run --nproc_per_node=1 --master-port 29500 \
  scripts/launch_target_server.py \
  --model-path /path/to/model --mode eagle3 --port 8001 --tp-size 1 \
  --mem-fraction-static 0.4 --trust-remote-code --attention-backend flashinfer

# Training client (any machine)
CUDA_VISIBLE_DEVICES=4 python -m torch.distributed.run --nproc_per_node=1 --master-port 29600 \
  scripts/train_eagle3.py \
  --target-model-path /path/to/model \
  --target-model-backend remote \
  --remote-urls "http://machineA:8001,http://machineB:8001" \
  --target-prefetch-depth 2 \
  --train-data-path /path/to/data.jsonl \
  --max-length 10240 --batch-size 1 --tp-size 1 \
  --trust-remote-code --is-preformatted \
  --output-dir /path/to/output
```

### DFlash Mode

Usage is identical — replace `--mode eagle3` with `--mode dflash` and use `train_dflash.py`:

```bash
# Server
scripts/launch_target_server.py --mode dflash --port 8002 ...

# Training
scripts/train_dflash.py --target-model-backend remote --remote-url http://host:8002 ...
```

### Cross-Machine RDMA Configuration

For cross-machine deployment, NCCL automatically uses RDMA when available. Optimize with environment variables:

```bash
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=bond0        # network interface
export NCCL_IB_HCA=mlx5_bond_0         # IB HCA device
export NCCL_IB_GID_INDEX=3             # RoCE GID index
```

## ⚙️ Configuration Reference

### Training Client Arguments

| Argument | Description |
|----------|-------------|
| `--target-model-backend remote` | Enable remote backend |
| `--remote-url` | Single server URL |
| `--remote-urls` | Multiple server URLs (comma-separated) |
| `--target-prefetch-depth` | Prefetch queue depth (recommended = number of servers) |
| `--remote-timeout` | HTTP request timeout in seconds (default: 120) |

### Server Arguments

| Argument | Description |
|----------|-------------|
| `--mem-fraction-static` | SGLang KV cache memory fraction (TP=1: 0.4, TP=2: 0.35) |
| `--attention-backend` | Attention backend (recommended: flashinfer) |
| `--nccl-port` | NCCL rendezvous port (default: HTTP port + 100) |
| `--host` | Bind address (must be 0.0.0.0 for cross-machine) |
| `--client-heartbeat-timeout` | Auto-exit timeout after no active client requests (seconds, default 60; 0 disables the watchdog) |

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `SPECFORGE_ENABLE_NCCL` | `1` | Enable NCCL transport (`0` falls back to wire format) |
| `SPECFORGE_NCCL_PORT` | HTTP port + 100 | NCCL TCP rendezvous port |
| `SPECFORGE_TOPK` | `0` | Server-side target_p top-k compression (`0` = full distribution) |
| `SPECFORGE_TARGET_DTYPE` | `fp32` | target_p computation precision |
| `SPECFORGE_GPU_ID` | auto | Specify GPU device ID |
| `SPECFORGE_HEARTBEAT_INTERVAL` | `15` | Client heartbeat send interval (seconds; `<=0` means the heartbeat thread is not started) |

## 📊 Benchmark Results

### Test Environment

| Item | Specification |
|------|---------------|
| Server Machine | 8x NVIDIA Hopper 80GB |
| Client Machine | 8x NVIDIA Hopper 80GB |
| Interconnect | RoCE v2 RDMA, 388 Gb/s |
| Model | Qwen3-30B-A3B-Instruct-2507-FP8 |
| Data | 103998 samples, preformatted, max_length=12288 |
| Steps per experiment | 100 |

### Baseline (Co-located, SGLang Backend)

| Experiment | Model | TP | Loss | Avg Iter (s) |
|------------|-------|----|------|--------------|
| B1 | DFlash | 1 | 6.4594 | 0.263 |
| B2 | DFlash | 2 | 6.4624 | 0.215 |
| B3 | EAGLE3 | 1 | 0.1999 | 0.533 |
| B4 | EAGLE3 | 2 | 0.1588 | 0.610 |

### Single Server Disaggregation (Cross-Machine)

#### DFlash

| TP | Depth | Loss | Avg Iter (s) | Speedup vs Baseline |
|----|-------|------|--------------|---------------------|
| 1 | 0 | 6.4594 | 0.256 | 1.03x |
| 1 | 1 | 6.4594 | 0.188 | 1.39x |
| 1 | 2 | 6.4594 | 0.189 | 1.39x |
| 2 | 0 | 6.4624 | 0.202 | 1.06x |
| 2 | 1 | 6.4624 | 0.132 | 1.63x |
| 2 | 2 | 6.4624 | 0.132 | 1.63x |

#### EAGLE3

| TP | Depth | Loss | Avg Iter (s) | Speedup vs Baseline |
|----|-------|------|--------------|---------------------|
| 1 | 0 | 0.1999 | 0.521 | 1.02x |
| 1 | 1 | 0.1999 | 0.342 | 1.56x |
| 1 | 2 | 0.1999 | 0.323 | 1.65x |
| 2 | 0 | 0.2006 | 0.456 | 1.34x |
| 2 | 1 | 0.2006 | 0.325 | 1.88x |
| 2 | 2 | 0.2006 | 0.326 | 1.87x |

### Dual Server Disaggregation (Cross-Machine)

#### DFlash

| TP | Depth | Loss | Avg Iter (s) | Speedup vs Baseline |
|----|-------|------|--------------|---------------------|
| 1 | 0 | 6.4594 | 0.255 | 1.03x |
| 1 | 1 | 6.4594 | 0.184 | 1.43x |
| 1 | 2 | 6.4594 | 0.111 | **2.37x** |
| 2 | 0 | 6.4624 | 0.205 | 1.05x |
| 2 | 1 | 6.4624 | 0.134 | 1.61x |
| 2 | 2 | 6.4624 | 0.094 | **2.28x** |

#### EAGLE3

| TP | Depth | Loss | Avg Iter (s) | Speedup vs Baseline |
|----|-------|------|--------------|---------------------|
| 1 | 0 | 0.1999 | 0.472 | 1.13x |
| 1 | 1 | 0.1999 | 0.283 | 1.88x |
| 1 | 2 | 0.1999 | 0.273 | 1.95x |
| 2 | 0 | 0.2006 | 0.454 | 1.34x |
| 2 | 1 | 0.2006 | 0.291 | 2.09x |
| 2 | 2 | 0.2006 | 0.288 | **2.12x** |

### Key Findings

1. **Disaggregation alone (depth=0)** provides 3-6% speedup by eliminating GPU resource contention between target and draft models.

2. **Prefetch depth=1** delivers an additional 36-53% speedup over depth=0 by overlapping target forward with draft training.

3. **Dual server with depth=2** achieves the highest speedup for DFlash (**2.37x**) — two servers alternate prefetch requests, fully hiding target forward latency. For EAGLE3, the benefit is limited since draft forward time is the bottleneck.

4. **Single server depth=2 vs depth=1**: No difference for single-server setups (only one server, cannot parallelize prefetch).

5. **EAGLE3 TP=2 observation**: In baseline, TP=2 (0.610s) is slower than TP=1 (0.533s) due to SGLang backend TP communication overhead. After disaggregation, TP=2 becomes faster because draft model training is unaffected by TP communication.

### Accuracy Verification

- **Prefetch depth has no impact on accuracy** — pure pipeline optimization that does not alter the computation path.
- **Cross-machine RDMA introduces no precision loss** — direct GPU tensor transfer without serialization/deserialization.
- **DFlash**: baseline loss matches remote loss exactly (6.4594 / 6.4624 for TP=1 / TP=2).
- **Single-server vs dual-server accuracy is identical** across all configurations.

## 🎯 Recommended Configurations

| Scenario | Recommended Config | Expected Speedup | Iter Time |
|----------|-------------------|------------------|-----------|
| DFlash, single server | TP=2, depth=1 | **1.63x** | 0.132s |
| DFlash, dual server | TP=2, depth=2 | **2.28x** | 0.094s |
| EAGLE3, single server | TP=2, depth=1 | **1.88x** | 0.325s |
| EAGLE3, dual server | TP=2, depth=1 | **2.09x** | 0.291s |

**Selection guidance**:
- **1 extra GPU machine available**: Single-server disaggregation + depth=1 yields 1.4-1.9x speedup
- **2 extra GPU machines available**: DFlash benefits from depth=2 (2.3x); EAGLE3 only needs depth=1 (2.1x)
- EAGLE3 dual-server depth=2 vs depth=1 differs by only ~1%, not worth occupying an additional server

## 📁 File Reference

| File | Responsibility |
|------|----------------|
| `scripts/launch_target_server.py` | Server launcher (HTTP + NCCL + TP multi-rank) |
| `scripts/train_eagle3.py` | EAGLE3 training script (supports remote backend + prefetch) |
| `scripts/train_dflash.py` | DFlash training script |
| `specforge/modeling/target/remote_target_client.py` | Client (HTTP + NCCL recv + TP broadcast + prefetch) |
| `specforge/modeling/target/remote_target_server.py` | Server (forward + target_p + NCCL send) |
| `specforge/modeling/target/_nccl_transport.py` | NCCL transport layer (rendezvous + send/recv + teardown) |
| `specforge/modeling/target/_tensor_wire.py` | Binary wire format fallback |
| `specforge/args.py` | `RemoteBackendArgs` / `SGLangBackendArgs` parameter definitions |
