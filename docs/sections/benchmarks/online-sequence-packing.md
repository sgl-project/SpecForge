# Online sequence packing: real Qwen3-4B pipeline benchmark

This benchmark measures actual pretrained-target capture and concurrent draft
training, extending the [consumer-compute measurements](dflash-sequence-packing.md).
It is based on upstream `53398a8f01ae47175bee8459c5b5cca3848c8a7e` plus the local
`codex/sequence-packing` changes, tested on 2026-10-02.

A subsequent [256-step full-model comparison](full-model-sequence-packing.md)
uses 1,024 real conversations, four runs per mode, periodic diagnostics and
checkpoints, and verifies every trainable parameter's optimizer participation.
It measures 1.044×/1.052× training-step improvements and 1.037×/1.033× full
completion improvements for DFlash/DFlash2.

## Measured results

On this workload, packing improved online pipeline throughput by **4.1% for
DFlash** and **4.5% for DFlash2**. Each time below is the median of two measured
runs processing the same 128 conversations in 32 optimizer steps.

| Model | Padded pipeline | Packed pipeline | Throughput speedup | Useful input tokens/s, padded → packed |
| --- | ---: | ---: | ---: | ---: |
| DFlash | 15.3356 s | 14.7329 s | **1.0409× (+4.09%)** | 10,811 → 11,253 |
| DFlash2 | 14.3867 s | 13.7686 s | **1.0449× (+4.49%)** | 11,524 → 12,041 |

Both packed runs were faster than both padded runs for each model. DFlash
ranges were 15.2822–15.3889 s padded and 14.6678–14.7980 s packed; DFlash2
ranges were 14.3754–14.3980 s padded and 13.7193–13.8178 s packed. These are
short repeated measurements, not a confidence interval or a whole-epoch result.

Including the final checkpoint changes the comparison:

| Model | Padded completion with checkpoint | Packed completion with checkpoint | Ratio |
| --- | ---: | ---: | ---: |
| DFlash | 25.2150 s | 24.9084 s | 1.0123× |
| DFlash2 | 25.1072 s | 24.0008 s | 1.0461× |

Each checkpoint took approximately 9.87–10.88 s. Storage timing varied between
arms: about 0.488 s of DFlash2's 1.106 s total-completion gap came from checkpoint
I/O. The pipeline measurement is therefore the cleaner estimate of packing's
effect. Checkpoint frequency will affect the realized full-run improvement.

All eight measured runs passed sample-order, HTTP-payload, initialization,
optimizer-grouping, durable-acknowledgement, and checkpoint-counter checks.
Every measured run had **zero new Dynamo compilations** and captured 30 of its
32 batches after its first optimizer acknowledgement, confirming concurrent
online feature production. Final losses were finite and repeatable within each
arm; this short run does not establish long-run convergence or serving quality.

The earlier 1.39×/1.47× consumer-compute results used synthetic lengths with
64% context padding and 42.5% invalid proposal slots. This real corpus has
33.28% context padding and only 13.08% removable proposal slots, and the online
timer includes target capture and transport. The synthetic speedups should not
be used as an estimate of end-to-end training gains.

## Workload and timing boundaries

- Two NVIDIA H200 GPUs on one host: a patched SGLang 0.5.18 Qwen3-4B target on
  GPU1 and a single-rank FSDP consumer on GPU0. The draft is freshly initialized;
  the target, frozen embeddings, and frozen LM head use the actual pretrained
  weights from `/cluster-storage/models/Qwen3-4B`.
- The repository's `configs/qwen3-4b-dflash.json`: five draft layers, hidden size
  2560, intermediate size 9728, 32 attention heads / 8 KV heads, head dimension
  128, vocabulary 151936, block size 16, and target capture layers
  `[1, 9, 17, 25, 33]`. DFlash2 uses the same base dimensions with its convolution
  and selector modules enabled.
- BF16, batch four, accumulation one, 512 anchors per document, objective chunks
  of 128 blocks, learning rate 0.0001, gradient clipping 0.5, teacher metrics
  enabled, and log interval 50. With 32 steps per arm, no periodic detailed-metric
  step runs. Both arms use identical settings apart from sequence packing.
- 128 real ShareGPT conversations, deterministically shuffled with seed 1729
  and prepared with SpecForge's Qwen parser and the target tokenizer. The
  original assistant supervision masks are retained. Maximum length is 2048;
  there are 165,788 input tokens and 131,810 supervised tokens.
- Lengths range from 35 to 2048, mean 1295.22 and median 1536. Batch-four context
  padding is **33.28%**. With the original per-batch anchor cap, proposal slots
  fall from 65,388 to 56,834, removing **13.08%** invalid slots. This is a natural
  corpus slice, not the previous synthetic 64%-padding length pattern.
- Mooncake uses TCP and host tensors, with pinned receive buffers. One canonical
  producer worker captures batches of four, concurrently with the canonical
  online consumer, with an eight-reference high watermark. One active capture
  batch may overshoot that watermark. Target capture and feature supply are
  rerun for every arm; no feature cache is replayed.

The primary timer starts at the first capture HTTP dispatch with an empty
channel, and ends after the last optimizer update, synchronous durable sample
acknowledgement, and CUDA synchronization. It includes pipeline fill/drain,
target prefill/capture, feature transport/fetch, data waiting, collation, model
forward/backward, optimizer work, and acknowledgement. It excludes data
preparation, model/server startup, and JIT warmup.

Each mode first performs an untimed replay of the complete ordered corpus.
Measured runs then use A → B → B → A order, where A is padded and B is packed.
Every run starts from the same seeded draft initialization and resets optimizer
state. The target adapter's fresh request namespaces force full prefill, and
the benchmark server also disables radix caching. Dynamo counters are saved
for every run to verify that warmed timing did not include new graph compilation.

Canonical final checkpoints are saved and validated in every run. Their cost,
and total completion time including checkpoint/cleanup, are reported separately
from the primary pipeline timer. Checkpoint files are deleted after validation
so repeated benchmarking does not retain many copies of the same initial run.
This is a warmed online training pipeline comparison, not complete cold CLI
startup, multi-node scaling, RDMA, convergence, or speculative-serving performance.

## Reproduction and evidence

The benchmark requires an already running capture-enabled SGLang server and
Mooncake master. The prompts are prepared once from the cached public
`anon8231489123/ShareGPT_Vicuna_unfiltered` dataset snapshot
`192ab2185289094fc556ec8ce5ce1e8e587154ca`; preparation is outside the timer.

```bash
CUDA_VISIBLE_DEVICES=0 \
MOONCAKE_MASTER_SERVER_ADDR=127.0.0.1:50212 \
MOONCAKE_METADATA_SERVER=http://127.0.0.1:8112/metadata \
MOONCAKE_LOCAL_HOSTNAME=127.0.0.1 \
MOONCAKE_PROTOCOL=tcp \
TORCH_LOGS=recompiles \
PYTHONPATH=/scratch/sglang-sequence-packing-online-20261002:. \
python scripts/benchmark_online_sequence_packing.py \
  --server-url http://127.0.0.1:31012 \
  --target-model /cluster-storage/models/Qwen3-4B \
  --draft-config configs/qwen3-4b-dflash.json \
  --prompts-path /scratch/specforge-packing-e2e-20261002/sharegpt-prompts.jsonl \
  --algorithm both --warmup-steps 32 --steps 32 --repeats 1 \
  --work-dir /scratch/specforge-packing-e2e-20261002/full_v1 \
  --output /scratch/specforge-packing-e2e-20261002/full_v1.json
```

The [committed 32-step benchmark summary](sequence-packing-results/online-32step.json)
preserves every run's timing samples across eight measured runs and four warmups,
plus settings, prompt/request hashes and lifecycle checks. Detailed per-capture
events remain in the local raw report. See the [evidence inventory](sequence-packing-results/README.md)
for the longer full-model comparison and provenance. The original raw report's
SHA256 is
`d0284b6ab87ad26bba3c6624c4d4b203cb4f18fef08609476729d650b55f9be4`.

The detailed prompt manifest, preprocessing/service-start scripts, exact source
snapshots, service logs and cleanup receipts remain local under
`artifacts/sequence-packing/e2e/`; they are not committed with this report.
Original conversation text was not copied into the local evidence directory.

The benchmark verifies actual HTTP payload hashes, publication order, consumed
sample order and optimizer grouping, all durable acknowledgements, finite final
loss, producer completion, and the final checkpoint's step and sample counters.
The benchmark-specific CPU tests and package architecture checks passed
21 tests with eight subtests before the GPU run.
