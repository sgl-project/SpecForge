# Native TorchTitan Trainer: H200 training measurements

The current matched-input matrix measures native TorchTitan with compilation
and CUDA graphs at 224.94 / 204.63 / 413.00 ms per optimizer window for
DFlash v1 / DFlash2 / DSpark. Relative to the original FSDP1 training path on
the repository's Torch 2.13 baseline, with common benchmark buffer initialization,
this is 16.78% / 18.86% / 9.90% lower latency in this controlled workload. Upgrading only the FSDP environment to
Torch 2.14 changed
latency by +0.27% to +0.33%.

These are **implementation timings**, not evidence of matched training quality.
FSDP1 and native TorchTitan have different parameter/gradient precision policies
and different loss trajectories. Full GraphTrainer is reported separately as a
diagnostic because every paired trial failed the strict loss gate. FP8 is not
used or exposed by this benchmark.

This matrix supersedes the historical `c70a5f4` graph measurements and the
initial uncompiled native comparison retained in the appendices. The intervening
p4 FSDP comparison is also withdrawn because it rounded RoPE
buffers differently between engines. All current cases use core
[`5a35ff35`](https://github.com/yushengsu-thu/SpecForge/commit/5a35ff35b0ba939242f294907892a2d8955a52b9)
and the same input contracts. Historical timings use other snapshots and
aggregation methods and are not pooled with these results.

## Current four-case comparison

All 36 fresh two-rank processes completed successfully on the same two H200
GPUs, without other GPU jobs during the measurement window. Each algorithm has
three trials per configuration. The cases rotate order between repetitions.
Every trial runs 10 warmup and 20 steady optimizer windows; latency is the
**median of three per-trial means**, with the range of those means in parentheses.
Each measured window uses the maximum elapsed time across the two ranks.

| Case | Runtime | Training path and compilation |
| --- | --- | --- |
| FSDP (Torch 2.13) | Torch 2.13.0, Triton 3.7.1 | Original FSDP1 `TrainerCore` and `BF16Optimizer`, common FP32 benchmark buffers; no model/block compilation |
| FSDP (Torch 2.14) | Torch 2.14.0, Triton 3.8.0 | Same original FSDP1 training path and options |
| Native Titan CUDA | Torch 2.14.0, Triton 3.8.0, TorchTitan 0.3.0 | Native `Trainer.train()` and inherited training step, FSDP2, decoder/projection compilation and native CUDA graphs |
| Graph full, diagnostic | Same 2.14 environment | Native `GraphTrainer`, SimpleFSDP, full joint forward/backward Inductor compilation and CUDA graphs |

Torch 2.13 is the repository's default pinned dependency. The TorchTitan v0.3.0
integration runs in a separate Torch 2.14/CUDA 13 environment. Both environments
use Transformers 5.12.1; the original FSDP path does not import TorchTitan.
FlexAttention's own internal compilation remains active in the FSDP cases.
This measures complete configurations, including compilation/capture benefits;
it does not isolate the effect of FSDP1 versus FSDP2 alone. The native engine is
provided by [PR #920](https://github.com/sgl-project/SpecForge/pull/920).

| Algorithm | Trainable draft | FSDP (Torch 2.13) ms/window | FSDP (Torch 2.14) ms/window | Native Titan CUDA ms/window | Native latency reduction vs FSDP on Torch 2.13 / 2.14 |
| --- | ---: | ---: | ---: | ---: | ---: |
| DFlash v1 | 537.4M | 270.66 (270.29–271.51) | 271.52 (271.48–272.15) | 224.94 (224.39–225.30) | 16.78% / 17.35% |
| DFlash2 | 632.4M | 252.24 (252.20–253.24) | 253.07 (252.94–255.03) | 204.63 (204.34–204.87) | 18.86% / 19.26% |
| DSpark | 615.2M | 458.01 (457.71–458.39) | 459.24 (459.16–459.35) | 413.00 (412.31–413.32) | 9.90% / 10.09% |

Latency reductions are medians of paired per-trial reductions. Equivalent
paired throughput ratios versus FSDP (Torch 2.13) are 1.2016× / 1.2325× / 1.1099×.
Exact trainable counts are 537,427,200 / 632,360,192 / 615,221,249; each case also
has 777,912,320 frozen embedding/head parameters. The controlled geometry is
five layers, hidden size 2560, intermediate size 9728, vocabulary 151936 and
block size 16. Each algorithm uses its own architecture and objective. These
are controlled recipes, not claims about the sizes of every released draft.

Peak allocated memory below is the **maximum across the three trials**, after
warmup. Captured pools and allocator reservations mean allocated memory alone
does not measure required GPU capacity; each raw record also includes peak
reserved memory, construction time and the full warmup timing vector.

| Algorithm | FSDP (Torch 2.13) GiB | FSDP (Torch 2.14) GiB | Native Titan CUDA GiB |
| --- | ---: | ---: | ---: |
| DFlash v1 | 16.003 | 15.892 | 6.089 |
| DFlash2 | 18.464 | 18.354 | 6.910 |
| DSpark | 22.205 | 22.095 | 6.673 |

### Inputs, precision and numerical boundary

All four cases share DP2, `SHARD_GRAD_OP`, batch 1 per rank, accumulation 2,
sequence length 4096, 512 anchors, objective chunks of 128 blocks, the default
fused head/convolution kernels and no separate decoder activation checkpointing.
The current native and Graph paths both retain the DFlash2 fused-convolution
route. GraphTrainer's default selective-recomputation passes remain active.
Initial persistent draft state, nonpersistent buffers, teacher tables, features,
all 30 windows of sampled anchors and resolved model configurations have
matching fingerprints or recorded values within each algorithm. Adam uses
learning rate 1e-4, betas (0.9, 0.999), epsilon
1e-8, zero weight decay and gradient clipping at 1. Optimizer/scheduler settings
are verified from pinned source rather than independently fingerprinted at
runtime. Both runtimes use BF16 forward
computation and FP32 Adam moments, but their gradient policies differ:

| Policy | Original FSDP1 | Native Titan / GraphTrainer |
| --- | --- | --- |
| Parameter storage | BF16 parameters plus FP32 master weights | FP32 parameters |
| Gradient accumulation/reduction | BF16 | FP32 |
| Forward computation / Adam moments | BF16 / FP32 | BF16 / FP32 |

The benchmark preserves fresh FP32 `rotary_emb.inv_freq` and
`original_inv_freq` buffers in both engines, instead of allowing the legacy
initialization to round them through BF16. This is a benchmark-only
normalization of model state; production FSDP defaults are unchanged. Thus the
FSDP rows exercise the original training backend with a controlled common
buffer reference, not untouched default model initialization. Names, shapes,
dtypes and contents are hashed separately from persistent `state_dict` entries,
checked on both ranks after preparation and again after training.

A six-launch full-geometry preflight passed the same first-window loss tolerance
(`atol=1e-5`, `rtol=1e-4`): first-window FSDP (Torch 2.14)/native differences were
9.54e-6 / 4.29e-5 / 2.77e-5 for DFlash v1 / DFlash2 / DSpark. Each first-window
objective pools two accumulation microbatches per rank; this
check does not compare every individual forward loss. The final 30-window p5
matrix separately verifies that persistent draft/teacher state and the complete
cached-feature/anchor contracts match p4; the untracked FSDP buffer discrepancy
was corrected.

The following observed maximum loss differences span all 30 windows and all
three repetitions. They are reported without assigning a post-hoc tolerance to
the FSDP/native comparison. The remaining differences prevent any claim that
the timing table establishes equivalent training trajectories. Their full cause has not been isolated;
recording different precision policies does not prove that precision alone
explains the trajectories.

| Algorithm | FSDP (Torch 2.13) vs FSDP (Torch 2.14) max absolute loss difference | FSDP (Torch 2.14) vs Native Titan CUDA max absolute loss difference |
| --- | ---: | ---: |
| DFlash v1 | 0.009086 | 0.100455 |
| DFlash2 | 0.046537 | 0.141664 |
| DSpark | 0.001615 | 0.002534 |

The native/Graph comparison uses the same recorded precision policy and a
strict gate declared before launch:
`abs(graph_loss - native_loss) <= 1e-5 + 1e-4 * abs(native_loss)` at every one of
30 windows. All nine pairs failed. The matrix deliberately returns failure
while retaining the separate FSDP/native timing rows and these Graph diagnostics:

| Algorithm | Graph full diagnostic ms/window (range) | Max absolute loss difference vs native | Failed windows across three trials |
| --- | ---: | ---: | ---: |
| DFlash v1 | 220.73 (220.12–220.85) | 0.040533 | 78 / 90 |
| DFlash2 | 209.95 (209.95–210.04) | 0.042302 | 82 / 90 |
| DSpark | 297.20 (296.98–297.27) | 0.005556 | 81 / 90 |

These Graph numbers are not matched-quality speedups. Focused probes corrected
compile-dependent convolution routing and repeated SimpleFSDP parameter reads.
Separate controls also showed BF16 gradient summation/reduction sensitivity;
a tiny FP32 full-Graph comparison had first-gradient relative L2 error
2.95e-7. Those controls do not override the failed BF16 trajectory gate or
establish real-data convergence. Full-Graph selector gradients can also be
nondeterministic under the default nondeterministic execution setting.

### Timing scope and reproduction

Every window includes feature preparation, CPU anchor sampling, token/mask H2D
transfers, forward/backward, clipping and optimizer work. Hidden features are
already cached on GPU (two synthetic batches per rank). FSDP's existing trainer
also reduces its normal scalar/ratio diagnostics; Titan uses its native metrics
path. Construction, hashing, checkpoint/export, evaluation, teacher capture,
feature transport and speculative serving are outside steady windows.
Fresh processes share compiler disk caches, so warmup values are observed
startup costs rather than guaranteed cold-cache compilation costs. Falling loss
on these reused synthetic features is not a quality evaluation.

The source audit matched all 168 SpecForge Python files to core `5a35ff35` before
launch. The three executed driver/helper/matrix files are byte-identical to
benchmark code
[`8bf59172`](https://github.com/yushengsu-thu/SpecForge/commit/8bf591727ad3b3a2287c777997f8c6da352f28cd).
Every trial retains its full configuration, runtime versions, parameter and
buffer fingerprints, timings, losses and memory records. The matrix enforces
algorithm/runtime identity, matching source/input/buffer contracts and the
unchanged strict Graph loss tolerance.

Use the final core checkout and isolated Torch 2.13/2.14 environments described
in the [setup guide](https://github.com/yushengsu-thu/SpecForge/blob/5a35ff35b0ba939242f294907892a2d8955a52b9/docs/sections/basic_usage/torchtitan.md):

```bash
python scripts/training_backend_matrix.py \
  --specforge-root /path/to/frozen-core \
  --driver-root /path/to/frozen-benchmark \
  --python /path/to/torch214/bin/python \
  --python213 /path/to/torch213/bin/python \
  --gpus 4,5 --output /path/to/new-results --execute
```

Without `--execute`, the runner writes a plan without launching GPU jobs. It
rejects existing results, mixed inputs/snapshots and mislabeled algorithm or
runtime records. The [current evidence archive](https://github.com/yushengsu-thu/SpecForge/blob/8256d23f4cb5146cb2ac7cd926ba57f55d7fae99/artifacts/torchtitan-native-benchmark/extensions-p5/README.md) contains all raw trials,
exact launch commands, frozen drivers, source hashes, preflight results and
the final identity audit.

[Numerical corrections and independent gradient/lifecycle controls](https://github.com/yushengsu-thu/SpecForge/blob/8256d23f4cb5146cb2ac7cd926ba57f55d7fae99/artifacts/torchtitan-graph-loss-fix/README.md)
are archived separately from performance claims.

The [superseded p4 archive](https://github.com/yushengsu-thu/SpecForge/tree/e8b17f3b20c79b898712a72998e45ee4ad41792d/artifacts/torchtitan-native-benchmark/extensions-p4)
retains the buffer-mismatched experiment unchanged. Its state-dict-only hashes
missed 63 of 64 rounded RoPE frequency entries; the FSDP/native same-model claim
was withdrawn. Its native/Graph comparison had common buffers and still failed
the strict loss gate. No p4 timing is pooled with this matrix.

## Historical appendix: earlier source snapshots

The following measurements are retained for provenance. The `c70a5f4` extension
matrix used a different convolution route and compiler/cache behavior, while
the initial native experiment used `36eb793`. Their aggregation was the median
of per-trial **medians**, unlike the current median of means. Their numerical
boundaries and speed observations apply only to those recorded snapshots.
Historical FSDP/native contracts lacked the nonpersistent-buffer checks added
in p5 and must not be read as proof of fully matched runtime model state.

## Historical graph and CUDA graph extension comparison

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

## Historical reproduction and evidence

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
