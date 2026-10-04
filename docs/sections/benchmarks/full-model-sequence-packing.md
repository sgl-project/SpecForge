# Full-size Qwen3-4B online training: longer sequence-packing comparison

This extends the [32-step online measurement](online-sequence-packing.md) to
256 optimizer steps over 1,024 real ShareGPT conversations, with normal periodic
training diagnostics and checkpoints. It uses base
`53398a8f01ae47175bee8459c5b5cca3848c8a7e` plus the local `codex/sequence-packing`
changes. The previous online measurement already used full model dimensions;
this experiment increases run length, repeat count, and model-training evidence.

## Results

The complete-model training step improved **1.044× for DFlash** and **1.052×
for DFlash2**. End-to-end completion including both checkpoints improved
**1.037× and 1.033×**, respectively. The values below are medians of four runs
per mode; each run processes 1,024 conversations in 256 optimizer steps.

| Model | Padded training step | Packed training step | Training-step ratio | Padded completion | Packed completion | Completion ratio |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash | 452.79 ms | 433.64 ms | **1.0442×** | 145.526 s | 140.378 s | **1.0367×** |
| DFlash2 | 426.98 ms | 405.76 ms | **1.0523×** | 137.320 s | 132.905 s | **1.0332×** |

Training-step values are the existing host-wall diagnostics over the first
250 steps, including the detailed metric steps. Completion covers all 256
steps, live target capture/transport/waiting, acknowledgements, and checkpoints
at steps 128 and 256. Both exclude startup and separate full-corpus warmup.

| Model | Padded completion range | Packed completion range | Matched-pair completion ratios |
| --- | ---: | ---: | ---: |
| DFlash | 144.905–145.755 s | 138.331–140.510 s | 1.0328–1.0533× |
| DFlash2 | 136.638–137.591 s | 132.296–133.598 s | 1.0258–1.0377× |

All chronological A/B pairs favored packing; the ranges do not overlap.
These are observed ranges from four runs per mode, not confidence intervals.
Checkpoint totals were approximately 18.46–19.79 s per run. Consumer fetch
waits also varied: 9.52–10.09 s padded versus 8.04–9.89 s packed for DFlash,
and 7.60–8.33 s versus 8.40–9.15 s for DFlash2. The full-run ratio therefore
includes normal pipeline and storage variation, not solely GPU computation.

The pipeline timer ending at the final durable acknowledgement, including the
intermediate checkpoint but excluding the final one, measured
135.774→130.651 s (**1.0392×**) for DFlash and 127.556→122.898 s (**1.0379×**)
for DFlash2. This boundary differs from the previous 32-step experiment, which
had no intermediate checkpoint.

All 16 measured runs and four warmups passed the independent full-model audit.
Every measured run had zero new compiler-counter activity. All 58 DFlash and
81 DFlash2 trainable tensors received exactly 256 AdamW updates, all five
draft layers had observed sampled weight changes, and all trainable elements
were finite. Final losses repeated exactly within each mode; they were
7.263832/7.263966 padded/packed for DFlash and 7.968346/7.968497 for DFlash2.
These short training comparisons do not establish equal final model quality
or time to convergence.

## Complete model and workload

- Actual Qwen3-4B target checkpoint: **4,022,468,096 saved parameters**, all 36
  decoder layers, hidden size 2560, vocabulary 151936. All layer indices were
  verified from the checkpoint tensor headers. The target runs full prefill;
  it is frozen, as required by speculative draft training.
- Complete repository `configs/qwen3-4b-dflash.json` draft: five layers,
  intermediate size 9728, 32 attention heads, eight KV heads, head dimension
  128, block size 16, 512 sampled anchors per document. DFlash2 uses the same
  dimensions with its convolution and selector modules enabled.
- Runtime parameter inventories confirm **537,427,200 trainable parameters in
  58 tensors for DFlash**, and **558,918,912 in 81 tensors for DFlash2**. The
  frozen target embedding/head share 388,956,160 parameters, counted once.
- Fresh, identically seeded draft initialization for every arm. All trainable
  draft parameters use the production backward and BF16 optimizer with FP32
  AdamW master parameters. Frozen target embeddings and the LM head are loaded
  from the real target checkpoint. Objective chunk size 128 processes every
  sampled block and the entire vocabulary.
- Two H200 GPUs on one host: target on GPU1, consumer on GPU0. Torch
  2.13.0+cu130, Transformers 5.12.1, isolated SGLang 0.5.18 with the repository's
  capture patch. The reserved host has eight GPUs; the experiment uses two.
- BF16, batch four, accumulation one, learning rate 0.0001, optimizer warmup
  ratio zero, gradient clipping 0.5, teacher metrics enabled. Detailed metrics
  run every 50 steps; checkpoints run at steps 128 and 256.
- 1,024 real ShareGPT examples, deterministic seed 1729, production Qwen parser
  and actual tokenizer, original assistant supervision masks, maximum length
  2048. There are **1,309,869 input tokens** and **1,026,019 supervised tokens**.
  Batch-four context padding is **32.55%**. Invalid proposal removal reduces
  slots from 523,408 to 454,930 (**13.08%**).
- Each mode receives a full untimed 256-step warmup before measurement. Two
  ABBA blocks provide four measured runs per mode per architecture. Every arm
  starts with identical model/optimizer initialization, prompt order and masks.

## Timing and verification

The producer and consumer use canonical SpecForge online builders and run
concurrently, with real target captures for every arm. Features travel through
Mooncake TCP host buffers. The producer has one worker, capture batch four and
an eight-reference high watermark. Radix caching is disabled, and fresh request
namespaces force full prefill. Data preparation, process/model startup and JIT
warmup are outside the main timer.

The full completion timer starts at the first actual capture dispatch with an
empty channel and ends after `trainer.fit()`, including the intermediate and
final checkpoints and trainer cleanup. The pipeline timer ends at the final
optimizer update and durable sample acknowledgement with CUDA synchronization;
it includes the step-128 checkpoint, but not the final step-256 checkpoint.
These boundaries must not be compared directly with a compute-only benchmark.

Existing trainer diagnostics also report training-thread wall time spent inside
`TrainerCore.train_step` for the five 50-step logging windows. This includes
strategy preparation and host-to-device transfers, forward,
objectives/diagnostics, backward and optimizer work; it excludes data
waiting, acknowledgements and checkpoint calls. It is a host-observed training
metric, not a CUDA-event kernel profile, and covers the first 250 of 256 steps.

Every run records exact parameter inventory, optimizer identity coverage,
per-parameter AdamW step counters, a complete finiteness scan of trainable
weights, and sampled before/after weight values for each layer. Tied target
weights are deduplicated. The warmup alone observes first-step gradient presence;
measured steps have no gradient-observation hooks. Sampling and scans occur
outside timing. Sample hashes demonstrate observed updates, not a full-tensor
numerical comparison or an assertion that every element must change.

The benchmark also checks identical prompt/request hashes, model initialization,
publication and consumption order, optimizer sample grouping, durable
acknowledgements and final checkpoint counters. Saved Dynamo counter deltas
identify any compilation during measured runs. Finite losses and updated layers
do not establish long-run convergence or serving quality.

This experiment uses the complete model and real training path, with one
consumer rank. It does not exercise the YAML CLI orchestration, evaluation,
resume, multi-node scaling or final speculative-serving quality. The logger
collects metrics in memory rather than publishing to an external dashboard.

## Why the full-model gain is modest

Packing compacts context rows and valid proposal blocks through the draft
backbone. `_forward_draft_blocks` then scatters hidden states back to the
original batch/anchor/block layout before the objective. The normal LM-head
path projects that restored layout across the full vocabulary and applies the
loss mask after cross-entropy. DFlash2's existing fused head skips the clean
anchor slot of every block, but still processes the restored invalid blocks.
Neither objective path compacts all invalid proposal rows in this change.

These are source facts in `specforge/algorithms/common/dflash_family_model.py`
and `specforge/core/dflash_head_triton.py`. Together with the corpus's 13.08%
removable proposal slots and full 151,936-token vocabulary, they explain why
removing context padding does not remove an equivalent fraction of all training
work. This is a structural explanation, not a measured kernel-time breakdown.
Packing the objective could be a separate optimization; it is not implemented
or benchmarked by this experiment.

## Reproduction and artifacts

The benchmark requires a capture-enabled SGLang server, Mooncake master and the
preprocessed public corpus described above. From the repository root, run:

```bash
python scripts/benchmark_online_sequence_packing.py \
  --server-url http://127.0.0.1:31012 \
  --target-model /cluster-storage/models/Qwen3-4B \
  --draft-config configs/qwen3-4b-dflash.json \
  --prompts-path /scratch/specforge-packing-full-model-20261002/sharegpt-prompts-1024.jsonl \
  --algorithm both --steps 256 --warmup-steps 256 --repeats 2 \
  --log-interval 50 --save-interval 128 \
  --work-dir /scratch/specforge-packing-full-model-20261002/long_v2 \
  --output /scratch/specforge-packing-full-model-20261002/long_v2.json
```

Committed evidence includes the [per-run timing summary](sequence-packing-results/full-model-analysis.json),
[independent full-model audit](sequence-packing-results/full-model-audit.json),
and [target tensor inventory](sequence-packing-results/target-parameter-inventory.json).
The summaries preserve every measured run's timing values, checks, parameter
counts and optimizer-update evidence. The [evidence inventory](sequence-packing-results/README.md)
describes their provenance and the raw configuration's unused CLI defaults.

The 9.65 MB raw `long_v2.json`, its log, exact source archive and hashes, detailed
data manifest and preprocessing provenance, runtime/service launch records and
cleanup receipts remain local under `artifacts/sequence-packing/full-model/`.
These local files, including `run_benchmark.py`, are not committed with this
report. The raw result SHA256 is
`f8895e415d1baaef7ddaa74a7f8e322d83d13e008d2f7606b0378fb0b4861179`.
The 636-file measured source archive was verified against the remote snapshot;
the final results documentation was written afterward. Raw conversation text
stays on the devbox, and checkpoints were validated before deletion between
runs to bound disk use. All benchmark-owned services were stopped successfully;
the existing devbox reservation was retained.
