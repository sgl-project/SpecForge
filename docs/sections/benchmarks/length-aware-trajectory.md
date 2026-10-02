# BF16 length-scheduling trajectory diagnostic

The 50-update follow-up found small heldout synthetic-loss differences, but
**did not resolve the failed BF16 parameter-update gate or establish model
quality**. Gradient-norm trajectories also differed materially, including when
repeating the unchanged baseline. Keep these observations separate from a
real-model convergence or serving-acceptance result.

## Reproduce

From the repository root on two CUDA GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
  -m benchmarks.length_aware_trajectory \
  --output trajectory-bf16-nonidentity-control.json
```

The recorded run used two H200 GPUs, PyTorch 2.13.0+cu130, CUDA 13.0, BF16,
SDPA, FSDP `SHARD_GRAD_OP`, and TF32 disabled. Production sources were from
`e8e51f2cf26e56174c375dae8e64af904dd09e6d`; the added diagnostic script and its
dependencies are identified by SHA256 hashes in the JSON.

The fixture is a random DFlash2 draft with hidden size 128, two layers,
vocabulary 1024, block size 8, and 16 fixed anchors per sample. Each GPU has
microbatch size 2 and accumulation 4. Lengths are 64/256/512/1024. Four distinct
16-sample training windows are cycled for 50 optimizer updates; a disjoint
16-sample synthetic heldout window is evaluated before training and after every
update. Seeds are 42, 43 and 44. AdamW uses the existing benchmark's constant
learning rate 0.001 and gradient clipping at 1.0.

Each seed runs five variants from identical initial weights:

- Baseline source order, shuffled deterministically within each window.
- An exact repeat of the baseline execution plan, to measure repeatability.
- A permutation of intact global microbatch rounds, preserving each sample's
  rank assignment and its microbatch partners. If shuffling produces the
  identity permutation, the control rotates whole rounds instead.
- Production online length scheduling within each optimizer window.
- Production offline bucketing confined to that same optimizer window.

All variants preserve the sample IDs in each optimizer window and the fixed
per-sample anchors. This deliberately excludes normal offline redistribution
across optimizer windows and native random-anchor trajectory differences. The
test uses production model, strategy, `TrainerCore`, FSDP, and `BF16Optimizer`.

## Declared checks

Before running, the diagnostic declared these checks: finite losses, gradients,
BF16 parameters and FP32 master parameters; equal supervised-position counts
at every update; and at most 1% relative heldout-loss deviation from baseline at
any evaluated step. This 1% check is a bounded synthetic diagnostic, not a
model-quality tolerance. The earlier one-update BF16 thresholds remain
`atol=0.0002`, `rtol=0.03`, requiring both elementwise agreement and relative L2
error at most 3%. Their results are reported separately and were not relaxed.

The JSON retains every step's loss, gradient norm, supervision count and heldout
loss, plus first-update gradient/update comparisons and final parameter
comparisons. Timing includes instrumented evaluation and is not a throughput
benchmark.

## Recorded results (2026-10-02)

All 15 trajectories completed 50 updates: 750 optimizer updates in total.
Losses, gradients and parameters remained finite, and all paired supervision
counts matched at every update. All heldout-loss diagnostic comparisons were
below the predeclared 1% threshold.

The table reports the largest observed relative deviation over all three seeds
and 51 heldout evaluations. Parameter ranges are the final BF16 parameter
relative L2 errors across the three seeds, globally reduced across shards.

| Comparison to baseline | Maximum heldout-loss deviation | Final parameter relative L2 error | Maximum preclip gradient-norm deviation |
| --- | ---: | ---: | ---: |
| Exact baseline repeat | 0.0860% | 0.790–1.081% | 200.9% |
| Intact-microbatch permutation | 0.1584% | 0.914–1.193% | 233.4% |
| Online length scheduling | 0.0792% | 0.931–1.152% | 201.7% |
| Offline length bucketing | 0.2362% | 0.945–1.233% | 215.2% |

The large gradient-norm differences are not caused by dividing by an almost-zero
baseline: at seed 44, update 44, the baseline norm was 3.844 while the exact
baseline repeat was 11.566 and online scheduling was 11.596. All are preclip
norms; the configured clip threshold is 1.0. No gradient-norm agreement gate
was declared, but these differences are material and should not be hidden by
the small heldout-loss differences.

The original first-update loss and gradient checks passed. The parameter-update
gate still failed for both features on every seed:

| Variant | Seed 42 update relative L2 error | Seed 43 | Seed 44 |
| --- | ---: | ---: | ---: |
| Exact baseline repeat | 0% | 0% | 0% |
| Intact-microbatch permutation | **3.825%** | 0% | **3.505%** |
| Online length scheduling | **5.243%** | **4.624%** | **4.908%** |
| Offline length bucketing | **5.035%** | **4.701%** | **4.888%** |

All permutation layouts are nonidentity, including seed 43; a nonidentity
ordering can still produce an identical first update. The unchanged baseline
itself diverged on later updates in this runtime. That demonstrates limited
repeatability in this experiment, not proof that feature-induced differences
are harmless. The causes of the full trajectory differences have not been
isolated.

The baseline's heldout synthetic loss increased from 13.720 to 13.760,
13.783 to 13.801, and 13.728 to 13.827 for seeds 42/43/44. Random features and
labels do not measure useful language-model learning, so these runs cannot
establish convergence, acceptance length, serving accuracy, or a safe quality
tradeoff. Real-model validation remains necessary, especially for offline
bucketing across multiple optimizer windows.

Raw results:

- [Reported run with nonidentity permutation control](../../../benchmarks/results/length-aware-h200/trajectory-bf16-nonidentity-control.json).
- [Initial run](../../../benchmarks/results/length-aware-h200/trajectory-bf16-initial.json),
  retained for transparency: its shuffled control could occasionally be the
  identity. The entire experiment was rerun after making that control
  explicitly nonidentity; thresholds and model configuration stayed unchanged.
