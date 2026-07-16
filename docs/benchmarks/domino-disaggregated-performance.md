# Domino disaggregated performance findings

This page preserves the measured tuning results from the Qwen3-8B Domino
online-disaggregated investigation on one 8×H200 node: one patched SGLang
capture server, seven FSDP draft trainers, and an approximately 1.1B-parameter
Domino/DFlash-family draft. Treat these numbers as historical evidence for that
hardware, model, dataset, and software revision, not as a current-release
performance claim. Rerun the benchmark before extrapolating to another setup.

All throughput values below are total samples divided by total elapsed time over
at least 390 seconds. Short, approximately 20-second windows were not stable
because supply was bursty and varied by roughly 1.8×.

## Headline

Configuration-only tuning took the 1-server + DP7 setup from **28.8 to 50.1
samples/s (+74%)**. The capture server reported 98% GPU utilization and 76% SM
utilization; trainer GPUs reported 77–100% GPU utilization and 76–85% SM
utilization. The model and default behavior were unchanged; every setting below
is opt-in.

## Best measured settings

The canonical entry remains `specforge train -c YAML`. Repeat the same URL in
the typed list to create multiple producer workers against one capture server;
use distinct URLs to fan out across distinct servers:

```bash
export CLONE_ON_FETCH=0
export LOADER_PREFETCH=2
export MOONCAKE_PROTOCOL=rdma
export MOONCAKE_RDMA_DEVICES=mlx5_0,mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7

# Start the external capture server with --mem-fraction-static 0.5, then run:
specforge train -c examples/configs/qwen3-8b-domino-disaggregated.yaml \
  'deployment.disaggregated.server_urls=["http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000","http://127.0.0.1:30000"]' \
  deployment.trainer.nproc_per_node=7 \
  training.batch_size=2 \
  training.accumulation_steps=8
```

GPU visibility and the external server's memory fraction remain deployment
settings. See the [disaggregated training guide](../basic_usage/disaggregated_training.md#multi-server-capture)
for the distinction between multiple capture servers and multiple workers
driving one server.

## Self-contained one-server + DP7 topology

The deleted self-contained shell's process topology is retained as a typed,
managed-local recipe. One command owns Mooncake, a capture server on GPU 0, and
seven trainer ranks on GPUs 1–7:

```bash
specforge train -c \
  examples/configs/qwen3-8b-domino-1server-dp7-disaggregated.yaml
```

The recipe records batch size 2, accumulation 8, and capture-server memory
fraction 0.5. It intentionally creates one producer worker for its one owned
server. The 50.1 samples/s result above used eight workers against one external
server, so use the repeated-URL command in **Best measured settings** when
reproducing that exact historical peak. `CLONE_ON_FETCH=0`,
`LOADER_PREFETCH=2`, and hardware-specific RDMA settings remain opt-in runtime
environment controls.

## What each setting does

| Setting | Effect | Mechanism |
| --- | --- | --- |
| Repeat one URL eight times in `deployment.disaggregated.server_urls` | Breaks the single-producer ceiling | Eight rollout workers take disjoint leases and issue blocking HTTP prefill calls concurrently to one server. |
| `training.accumulation_steps=8` | 460 to approximately 280 ms/microstep | FSDP `no_sync` amortizes reduce-scatter across eight microsteps; communication fell to about 1 ms/microstep. |
| `CLONE_ON_FETCH=0` | Approximately 15 ms/batch lower | Mooncake `get()` already allocates a fresh tensor, so the defensive clone is redundant on this path. |
| `LOADER_PREFETCH=2` | Removes fetch latency from the measured train step | A background thread materializes up to two batches before the trainer consumes them. |
| External SGLang `--mem-fraction-static 0.5` | Server memory fell from 126 GB to about 78 GB | The default 0.85 reservation held KV-cache memory that this capture-only server did not need. |

The server-side capture patch also kept hidden-state slices on GPU and used one
`torch.cat` plus one D2H copy per request instead of a per-prefill-batch
unpinned copy. In this run, capture D2H fell from 5–8 ms/sample to approximately
3.8 ms/sample.

## The pipeline is a supply/demand seesaw

At 50 samples/s the system was supply-bound: one capture server only just fed
seven trainers, loader waits were approximately 40 ms/batch, and producer
in-flight depth stayed low. The GPU split was therefore a real trade-off:

| Split | Throughput | Regime |
| --- | --- | --- |
| **1 server + 7 trainers** | **50.1/s** | Supply-bound; best measured split |
| 2 servers + 6 trainers, batch 2 | 39.0/s | Demand-bound; only six trainers |
| 2 servers + 6 trainers, batch 8 | 39.5/s | Demand-bound; larger batch did not help |

Each trainer GPU consumed approximately 7 samples/s, while one server supplied
approximately 52–57 samples/s. Seven trainers, about 50 samples/s of demand,
were almost exactly balanced by one server. Trading away a trainer for a second
server lost more demand than it added supply on this eight-GPU allocation.

## Bigger batches did not help

| Batch | Accumulation | Throughput | Trainer memory | Trainer utilization |
| --- | --- | --- | --- | --- |
| **2** | **8** | **50.1/s** | 45 GB (about 31%) | 77–100% |
| 4 | 4 | 47.5/s | 65 GB (about 45%) | 64–99% |
| 8 | 2 | 47.2/s | 113 GB (about 79%) | 100% |

Memory climbed with batch size, but throughput fell. The larger batch made each
fetch roughly four times heavier (`get_ms` rose from about 20 to 90) while
per-sample compute stayed flat. Low trainer memory in the best configuration
was efficient rather than evidence of a stall: the trainer was compute-bound
at approximately 44% measured MFU.

## Why per-sample trainer demand is low

The benchmark's synchronized step measurements at batch size two were:

| `num_anchors` | Forward | Backward | Optimizer | Data wait |
| --- | --- | --- | --- | --- |
| 256, recipe default | approximately 88 ms | approximately 150 ms | 1.3 ms | approximately 40 ms (14%) |
| 64 | approximately 43 ms | approximately 58 ms | 1.3 ms | approximately 145 ms (38%), supply-starved |

Domino/DFlash training expands every sample to `num_anchors × block_size`: 256
× 16 = 4096 draft positions independent of input sequence length. Each position
passes through the five-layer draft and a full 151,936-vocabulary head. A fit of
these measurements gives approximately 53 ms fixed work plus 0.73 ms per
anchor, so the anchor expansion accounts for roughly 78% of the 240 ms
microstep at 256 anchors.

Reducing anchors is therefore not a free throughput optimization. At 256
anchors the run delivered approximately 2,130 anchor updates/s, versus about
1,280 at 64 anchors. Fewer anchors raised sequences/s but reduced training
signal/s by roughly 40%, while putting more pressure on the already
supply-bound capture server. Treat `num_anchors` as a quality and data-efficiency
choice validated against acceptance length; the checked-in recipe keeps 256.

## MFU and FLOPs

The isolated trainer microbenchmark used `bench_domino_mfu.py`, BF16, real
Qwen3-8B Domino shapes, `num_anchors=256`, sequence length 768, CUDA event
timing, and `torch.utils.flop_counter`:

| Batch | Step | Per sample | FLOP/sample, forward + backward | Achieved | MFU | Peak memory |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 211 ms | 105.6 ms | approximately 45 T | 430 TFLOP/s | **43.5%** | 25 GB |
| 4 | 416 ms | 103.9 ms | approximately 45 T | 437 TFLOP/s | 44.1% | 46 GB |
| 8 | 836 ms | 104.6 ms | approximately 45 T | 434 TFLOP/s | 43.9% | 87 GB |

The approximately 1.1B draft was compute-bound, not stalled. Per-sample time
was flat across batch sizes, and memory grew by roughly 10 GB/sample. One sample
cost approximately 45 TFLOP for forward plus backward—about nine times a normal
1.1B forward—because of the 256-anchor expansion and two full-vocabulary paths.

Target prefill was estimated at a similar 43% MFU. At 50 samples/s the server
logged approximately 27,000 tokens/s over 1,080 batches, with an average prompt
of about 550 tokens. For the 8B target, `2 × parameters × tokens/s` is about 430
TFLOP/s. This estimate excludes attention FLOPs and the five auxiliary capture
layers.

## Remaining ceilings

1. Supply at approximately 50 samples/s was prefill-compute-bound, not
   sink-bound. With at least eight producer workers, target prefill ran nearly
   continuously at about 27,000 tokens/s. The Mooncake sink sustained roughly
   44–57 samples/s and was the next ceiling, not the measured one. Potential
   supply improvements include higher prefill MFU, larger prefill batches, and
   lighter auxiliary-state extraction/D2H; a second inference GPU was worse on
   this fixed eight-GPU split.
2. Trainer demand at approximately 50 samples/s was also compute-bound near 44%
   MFU. Raising trainer MFU would reduce trainer GPU cost, but could not improve
   end-to-end throughput until capture supply increased. Reducing
   `num_anchors` would change training signal and is not an equivalent systems
   optimization.

## Profiling controls

Production runs expose the typed `profiling` configuration (`enabled`,
`start_step`, `num_steps`, and `record_shapes`) for bounded PyTorch traces on
trainer ranks. The component timers used to collect this benchmark were
temporary measurement instrumentation and are not runtime environment APIs.
`FSDP_SHARDING=NO_SHARD` remains available for controlled sharding comparisons.
