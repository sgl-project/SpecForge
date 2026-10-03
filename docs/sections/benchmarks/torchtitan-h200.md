# Native TorchTitan Trainer: H200 training measurements

In the final BF16 extension comparison, GraphTrainer full reduced DSpark
training latency by 26.94%, versus 3.45% for DFlash v1 and 0.38% for DFlash2.
Native CUDA graphs reduced latency by 1.57–2.42%. DFlash2's GraphTrainer loss
trajectory differed materially from its reference, so its timing result does
not establish a matched-quality improvement.

The initial native TorchTitan versus FSDP1 comparison and the later graph
comparison use separate source snapshots and matched baselines. Their timings
are not pooled.

In the initial experiment, the native TorchTitan path was 2.6–4.3% slower than
the existing FSDP1 trainer in this controlled two-GPU workload, while using
about 0.5–0.6 GiB less peak allocated memory. Enabling compilation for DFlash2
then measured 215.86 ms per window across three trials. Those historical
measurements remain below; the extension comparison also compiles the draft
feature projection and therefore remeasures its baseline.

This measures the actual TorchTitan `Trainer.train()` and inherited
`train_step`: native optimizer, accumulation, backward, clipping and scheduling.
The initial FSDP1 baseline executes SpecForge's existing `TrainerCore`, `FSDPTrainingBackend`
and `BF16Optimizer`. It requires the [native engine in PR #920](https://github.com/sgl-project/SpecForge/pull/920);
this benchmark PR does not provide that engine.

## Graph and CUDA graph extension comparison

The extension matrix compares three BF16 configurations on the same two H200
GPUs, with one fresh two-rank launch per trial and three trials per configuration and
algorithm:

| Configuration | Execution path | Compilation and capture |
| --- | --- | --- |
| Compiled native | TorchTitan `Trainer`, FSDP2 | Draft decoder blocks and feature projection compiled; CUDA graphs disabled |
| Native CUDA graph | TorchTitan `Trainer`, FSDP2 | Same model compilation, plus native forward/backward CUDA graph capture |
| GraphTrainer full | TorchTitan `GraphTrainer`, SimpleFSDP | Joint forward/backward tracing, graph memory/collective passes, full Inductor compilation and graph capture |

All use DP2, `SHARD_GRAD_OP`, batch 1 per rank, accumulation 2, sequence length
4096, 512 anchors, objective chunks of 128 blocks and the controlled model
geometry described below. They use BF16 computation, FP32 parameter storage,
FP32 gradient reductions and FP32 Adam moments. Frozen teacher tables remain
replicated. Fused head/convolution environment flags are enabled, but the
executed DFlash2 convolution differs: compiled native training takes its ATen
decomposition under the `torch.compiler.is_compiling()` guard; GraphTrainer
traces the wrapped custom Triton convolution. Native CUDA graphs replay the
compiled native path. This is not a comparison with identical convolution
kernels. No separate decoder activation-checkpointing option is enabled.

SimpleFSDP expresses its collectives in the traced graph; GraphTrainer's default
selective activation checkpointing (SAC) pass controls retention and
recomputation even with the separate decoder checkpointing option disabled.
Both paths receive `reshard_after_forward="never"`; the graph pass retains the
unsharded parameter outputs for this policy. Their resulting execution and
memory schedules can still differ. These are comparisons of complete training
configurations, not measurements that isolate one compiler pass or collective.

Each trial contains 10 warmup and 20 steady optimizer windows. Case order is
forward, reversed, then forward across the three repetitions. Initial draft
values, frozen tables, features and the full sampled-anchor stream must have
identical fingerprints within each algorithm. Steady latency is the median of
three per-trial medians; the range spans those three medians. Positive
latency change means slower than the newly measured compiled native baseline.
The earlier 215.86 ms DFlash2 result is not used as this comparison's baseline.

All 27 trials completed with finite recorded losses. Input contracts matched
exactly within each algorithm, and every trial used the same recorded source,
driver and runtime hashes. The [final matrix archive](https://github.com/yushengsu-thu/SpecForge/tree/0eb8fcf04684881c9c5ce6bbf426df5fb54c903a/artifacts/torchtitan-native-benchmark/extensions-p2) retains the raw
windows, launch commands, aggregate and source reconstruction metadata.

| Algorithm | Configuration | Steady ms/window (range) | Change vs compiled native |
| --- | --- | ---: | ---: |
| DFlash v1 | Compiled native | 230.62 (230.00–230.69) | Reference |
| DFlash v1 | Native CUDA graph | 225.20 (224.59–225.67) | -2.35% |
| DFlash v1 | GraphTrainer full | 222.66 (222.37–222.83) | -3.45% |
| DFlash2 | Compiled native | 213.69 (213.68–214.27) | Reference |
| DFlash2 | Native CUDA graph | 208.52 (207.87–208.71) | -2.42% |
| DFlash2 | GraphTrainer full | 212.87 (212.67–213.04) | -0.38% |
| DSpark | Compiled native | 419.72 (419.17–419.86) | Reference |
| DSpark | Native CUDA graph | 413.14 (413.10–414.77) | -1.57% |
| DSpark | GraphTrainer full | 306.63 (306.06–306.86) | -26.94% |

Memory and construction values are medians across the three trials.

| Algorithm | Configuration | Peak allocated GiB | Peak reserved GiB | Construction s | First window s (range) |
| --- | --- | ---: | ---: | ---: | ---: |
| DFlash v1 | Compiled native | 13.069 | 14.350 | 35.10 | 2.32 (2.28–11.35) |
| DFlash v1 | Native CUDA graph | 6.089 | 14.420 | 35.52 | 2.32 (2.28–2.37) |
| DFlash v1 | GraphTrainer full | 7.102 | 16.877 | 35.77 | 6.84 (6.74–37.55) |
| DFlash2 | Compiled native | 16.235 | 17.783 | 36.68 | 2.67 (2.65–2.69) |
| DFlash2 | Native CUDA graph | 6.806 | 17.748 | 36.66 | 2.80 (2.78–2.80) |
| DFlash2 | GraphTrainer full | 7.996 | 18.457 | 37.18 | 12.77 (12.69–12.83) |
| DSpark | Compiled native | 19.199 | 20.721 | 36.94 | 2.44 (2.44–2.44) |
| DSpark | Native CUDA graph | 6.673 | 21.096 | 37.04 | 2.63 (2.60–2.71) |
| DSpark | GraphTrainer full | 7.832 | 24.721 | 40.24 | 36.06 (35.84–44.01) |

GraphTrainer full reduced DSpark median latency by 26.94% in this workload,
while DFlash v1 improved by 3.45% and DFlash2 by 0.38%. It also reserved more
memory than the compiled native baseline for all three algorithms. Native
CUDA graphs reduced median latency by 1.57–2.42%.

### Numerical boundary of the timing comparison

Native CUDA graph runs recorded identical DP-reduced loss values to compiled
native training at every one of the 30 windows in all nine paired trials.
GraphTrainer full followed different loss trajectories:

| Algorithm | Largest absolute loss difference vs compiled native | Window of that difference | Relative difference at that window |
| --- | ---: | ---: | ---: |
| DFlash v1 | 0.009926 | 14 | 0.369% |
| DFlash2 | 0.516971 | 16 | 11.688% |
| DSpark | 0.005044 | 10 | 0.119% |

Differences are the maximum across all 30 warmup/steady windows and all three
trials. DFlash2's first-window difference was only 0.0000534, but increased to
0.516971 later in training. Matching initial forward losses therefore does
not establish a matching optimizer trajectory. Full graph compilation and the
DFlash2 convolution path differ, but these measurements do not isolate the
cause of the divergence. No matched-quality speedup, real-data convergence or
serving-acceptance claim is made for GraphTrainer full.

Construction time is recorded separately and includes CPU model initialization,
input creation, hashing and native trainer construction. The first optimizer
window and the full warmup vector include lazy tracing, compilation and capture
work. Fresh processes share the node's Inductor/Triton disk caches, so these
are observed startup costs, not guaranteed cold-cache compiler times. Neither
construction nor warmup contributes to the steady latency table.

Both peak allocated and peak reserved CUDA memory are reported. Statistics are
reset after warmup; captured graph pools can retain reservations for buffers
that are not simultaneously live during replay. A lower steady allocated peak
alone does not establish a corresponding reduction in GPU capacity required.
Resident cached features, teacher tables and optimizer state are included;
external driver/NCCL allocations are not measured by the PyTorch allocator.

The matrix measures synthetic cached-feature training only. Online capture,
feature transport, periodic evaluation, checkpoint/export I/O and speculative
serving are outside the timed windows. Finite losses, successful graph replay
and checkpoint continuation establish execution coverage; they do not establish
matched optimizer updates, real-data convergence or model quality.

### Earlier extension probes and deferred FP8

Earlier one-trial DFlash2 probes are diagnostic history, not members of the
three-trial matrix. [The exploratory archive](https://github.com/yushengsu-thu/SpecForge/tree/0eb8fcf04684881c9c5ce6bbf426df5fb54c903a/artifacts/torchtitan-native-benchmark/extensions-exploratory)
retains raw reports, source snapshots and the failed FP8 numerical gate. The
`p0` snapshot used 5 warmup and 10 steady windows:
compiled native measured 213.92 ms/window, native CUDA graphs 206.91 ms/window
and experimental TorchAO rowwise FP8 223.15 ms/window. FP8 was 4.3% slower than
its same-snapshot BF16 baseline. The `p1` snapshot used 10 warmup and 20 steady
windows: GraphTrainer regional measured 295.05 ms/window, while full Inductor
measured 212.38 ms/window. No repeated-trial or cross-snapshot speed claim is
made from these probes; profiler-instrumented runs are excluded from timings.

The FP8 converter and its configuration were deferred from the production
extension. The experiment converted 36 suitable DFlash2 draft linear layers,
including the feature projection; the frozen vocabulary head, convolution,
selector and other auxiliary objectives remained on their original precision
paths. Separate small-fixture DP2/TP2 tests exercised all three algorithms and
both rowwise recipes. Runtime and native DCP roundtrips worked, but the strict
numerical gate failed: the first Adam update differed from the matched BF16
reference by roughly 29–36% in relative L2 norm. Later full-parameter error is
a different measure and does not negate that failed update check. These results
support deferring FP8 here, not a general conclusion that FP8 cannot work for
these algorithms. No FP8 feature or reproduction option is exposed by this PR.

## Initial FSDP1/native experiment: workload and measurement

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

## Initial DP2 results

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

## Initial DFlash2 compilation and TP2 exploration

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

For the extension comparison, run the same DP2 command with `--compile` for
the compiled native baseline, add `--cuda-graphs` for native graph capture,
or add `--titan-engine graph --graph-inductor full --cuda-graphs` for the
GraphTrainer configuration. Use the extension engine source and its matching
benchmark driver for all three cases; the [extension setup guide](https://github.com/yushengsu-thu/SpecForge/blob/c70a5f413220e6ce68247afb528664986502300f/docs/sections/basic_usage/torchtitan.md)
describes that runtime. Repeat all cases for `dflash`, `dflash2`
and `dspark`, three fresh processes each, alternating the case order. The
original source-linked artifacts below remain the authority for reproducing
the historical experiment.

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
