# Native TorchTitan Trainer: H200 training measurements

The native TorchTitan path was 2.6–4.3% slower than the existing FSDP1 trainer
in this controlled two-GPU workload, while using about 0.5–0.6 GiB less peak
allocated memory. Switching the training engine alone did not improve speed.
Compilation and a different parallel layout are separate experiments below.
With model compilation enabled, three DFlash2 DP2 trials measured a median of
215.86 ms per window. That configuration produced a speed benefit in this workload.

This measures the actual TorchTitan `Trainer.train()` and inherited
`train_step`: native optimizer, accumulation, backward, clipping and scheduling.
The baseline executes SpecForge's existing `TrainerCore`, `FSDPTrainingBackend`
and `BF16Optimizer`. It requires the [native engine in PR #920](https://github.com/sgl-project/SpecForge/pull/920);
this benchmark PR does not provide that engine.

## Workload and measurement

- Same two H200 GPUs on one node; PyTorch 2.14.0, CUDA 13.0, TorchTitan 0.3.0,
  Transformers 5.12.1 and Triton 3.8.0.
- Controlled Qwen3-4B geometry: 5 draft layers, hidden size 2560, intermediate
  size 9728, vocabulary 151936 and block size 16. Each algorithm uses its actual
  architecture/objective, including DFlash2 convolution/selector and DSpark's
  Markov/confidence heads. This is a custom matched recipe, not a released
  DFlash2 checkpoint. `--recipe stock` preserves the repository recipes;
  stock DFlash2 has a different vocabulary and stock DSpark uses block size 7.
- DP2, `SHARD_GRAD_OP`, FlexAttention, batch 1 per rank, accumulation 2,
  sequence length 4096, 512 anchors and objective chunks of 128 blocks.
  Each optimizer window covers 16,384 input context tokens, not accepted tokens.
- BF16 computation and FP32 Adam moments in both engines. Native Titan keeps
  FP32 parameters and gradient reductions; the existing baseline keeps BF16
  parameters/reductions plus FP32 master weights. This is not a comparison
  where every precision policy is identical.
- Default fused kernels enabled; decoder activation checkpointing and CUDA
  graphs disabled. Algorithm-default objective chunking remains active.
  The DP2 matrix disables Titan's model/block compile option; FlexAttention's
  ordinary internal compilation remains part of its implementation.
- Identical initial draft values, frozen teacher tables, features and sampled
  anchors within each same-DP backend pair, verified by exact fingerprints.
  Hidden states are cached on GPU; token IDs and masks are pinned CPU tensors.
  CPU anchor sampling and small input transfers occur inside the timed window.
- Three fresh processes per backend/algorithm, alternating A/B, B/A, A/B;
  10 warmup and 20 measured optimizer windows per process. CUDA work completes
  before the clock stops; elapsed time is reduced with MAX across ranks.
  Peak allocation resets after warmup and includes resident features, frozen
  tables and optimizer state. Construction, hashing and checkpoint/export I/O
  are excluded.

The windows include each engine's actual training plumbing. In particular,
FSDP's `TrainerCore` reduces scalar/ratio diagnostics at optimizer boundaries;
Titan consumes the objective while detailed diagnostics are disabled on steady
steps. The timings therefore cannot isolate the cost of sharding collectives.

The features and teacher tables are synthetic. Two cached batches per DP rank
are reused; falling losses do not establish convergence on real data. These
measurements exclude teacher inference, feature production/transport, model
quality, acceptance and speculative serving speed.

## DP2 results

Times are the median of three per-process medians, with the range of those
three medians in parentheses. Positive latency change means native Titan is
slower. All 18 trials completed with finite losses; the raw JSON retains every
warmup/steady time, loss, memory value and comparison fingerprint.

| Algorithm | Trainable draft | FSDP1 ms/window | Native Titan ms/window | Native latency change |
| --- | ---: | ---: | ---: | ---: |
| DFlash v1 | 537.4M | 274.37 (273.89–275.25) | 284.83 (284.59–285.67) | +3.81% |
| DFlash2 | 632.4M | 255.86 (255.22–256.11) | 266.77 (266.51–266.96) | +4.26% |
| DSpark | 615.2M | 463.31 (463.07–464.00) | 475.27 (474.61–475.29) | +2.58% |

Peak memory is the median of three per-process peak allocations.

| Algorithm | FSDP1 peak GiB | Native Titan peak GiB |
| --- | ---: | ---: |
| DFlash v1 | 15.892 | 15.372 |
| DFlash2 | 18.354 | 17.756 |
| DSpark | 22.095 | 21.506 |

These results apply to this single-node, two-GPU configuration. They do not
establish larger-draft, multi-node or real-data scaling.

## DFlash2 compilation and TP2 exploration

Compiled DP2 has three fresh-process trials; TP2 and compiled TP2 have one each.
Every trial has 10 warmup and 20 steady windows, using the final native engine
commit
[`36eb793`](https://github.com/yushengsu-thu/SpecForge/commit/36eb793a4fba7fabb441821e9cb7b3f8279bca97).
The TP2 rows have less replication than the DP2 rows. DP2 times/memory and
first-window values are medians across their three trials; TP2 values come from
a single trial. The compiled DP2 steady-time median range is 215.48–215.96 ms.

| Native configuration | Steady ms/window | Peak GiB | First optimizer window, seconds |
| --- | ---: | ---: | ---: |
| DP2, uncompiled | 266.77 | 17.756 | 1.95 |
| DP2, compiled | 215.86 | 16.235 | 2.71 |
| TP2, uncompiled | 467.37 | 15.667 | 7.66 |
| TP2, compiled | 336.33 | 14.948 | 16.77 |

The first-window values are observed startup windows, not guaranteed cold-cache
compile latency: the node shares Inductor/Triton disk caches. The compiled DP2
first-window values were 15.92, 2.60 and 2.71 seconds. The complete warmup vectors
are retained. Compiled TP2 recorded one initial `requires_grad`
guard recompilation per rank, with no per-step mode-identity recompilation.

TP2 uses DP1 and accumulation 4 to preserve global batch 4 and the same 16,384
context tokens per window. Changing the DP degree changes the per-rank feature
cache and anchor RNG streams: DP2 versus TP2 is an equal-size synthetic workload
comparison, not an identical-example numerical comparison. DP2 compiled versus
DP2 uncompiled does preserve the same input/anchor fingerprints. The compile
experiment tests Titan's model compile option; a compiled FSDP1 baseline was
not measured.

For this draft and batch size, TP2 increased latency. Compiling TP2 reduced its
latency by 28.0%, but the result remained slower than uncompiled DP2. The compiled
DP2 trials reduced median latency by 19.1% relative to native uncompiled DP2 and
by 15.6% relative to the FSDP1 baseline; these percentages describe this workload only.

## Reproduction and evidence

Install the isolated native runtime described by [the core setup guide](https://github.com/yushengsu-thu/SpecForge/blob/36eb793a4fba7fabb441821e9cb7b3f8279bca97/docs/sections/basic_usage/torchtitan.md).
Run this command from the benchmark checkout, pointing `--specforge-root` at
the native engine checkout. Repeat with `fsdp`/`torchtitan`, all three algorithms
and unique output paths; launch a fresh process for each independent trial.

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_backends.py \
  --specforge-root /path/to/native-SpecForge \
  --backend torchtitan --algorithm dflash2 --sharding SHARD_GRAD_OP \
  --attention flex_attention --seq-length 4096 --num-anchors 512 \
  --objective-chunk-blocks 128 --batch-size 1 --accumulation-steps 2 \
  --warmup-steps 10 --steps 20 --output /tmp/native-dflash2-dp2-r0.json
```

Add `--compile` for DP2 compilation. For TP2, add `--tp-size 2` and change
`--accumulation-steps` to 4; optionally add `--compile`. Only the native engine
supports these flags in this harness. `training_benchmark_recipes.py` provides
input construction only; there is one executable benchmark driver.

[DP2 aggregate](https://github.com/yushengsu-thu/SpecForge/blob/5c1b1d02a8c4fdd630cae1bd72eac4dcf39fe4ba/artifacts/torchtitan-native-benchmark/h200-dp2.json),
[18 raw DP2 trials](https://github.com/yushengsu-thu/SpecForge/tree/5c1b1d02a8c4fdd630cae1bd72eac4dcf39fe4ba/artifacts/torchtitan-native-benchmark/dp2),
[compiled DP2 aggregate](https://github.com/yushengsu-thu/SpecForge/blob/5c1b1d02a8c4fdd630cae1bd72eac4dcf39fe4ba/artifacts/torchtitan-native-benchmark/h200-dp2-compile.json),
[exploratory trials](https://github.com/yushengsu-thu/SpecForge/tree/5c1b1d02a8c4fdd630cae1bd72eac4dcf39fe4ba/artifacts/torchtitan-native-benchmark/exploratory)
and [source provenance](https://github.com/yushengsu-thu/SpecForge/blob/5c1b1d02a8c4fdd630cae1bd72eac4dcf39fe4ba/artifacts/torchtitan-native-benchmark/source-provenance.json)
retain the comparison contracts, runtime fingerprints and source differences.
The measured DP2 snapshot preceded the final TP-only wait hooks, logger filter,
URI decoding and timeout fixes. Its objective module differs only in formatting;
forward/backward/optimizer behavior on the measured DP2 path is unchanged.
The provenance file includes an exact source reconstruction patch; the archive
also retains the exact measured driver and helper as text files with matching
SHA256 hashes. The native benchmark driver was renamed and recipe helpers
extracted after the DP2 matrix;
its measured operations/timer boundaries and recipe function ASTs are unchanged.

Separate [native engine validation](https://github.com/yushengsu-thu/SpecForge/blob/870e9a4/artifacts/torchtitan-native/README.md)
records numerical references, DP/TP/CP/PP checkpoint continuation, compiled CLI
training and PP×TP checks. This harness additionally ran tiny same-input A/B
checks with transparent tiny-fixture kernels for all three algorithms: maximum
loss difference across four windows was 0 for DFlash v1, 2.82e-5 for DFlash2 and
5.24e-5 for DSpark. Precision paths
are different, so these checks do not claim cross-engine bitwise updates.

The earlier FSDP2-helper experiment is superseded by this native Trainer
experiment; its results remain in the [immutable prototype archive](https://github.com/yushengsu-thu/SpecForge/tree/f381af478657365c6575950b4b7eb5f9dab97bc1).
