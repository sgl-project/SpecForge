# Length-aware training validation

`benchmarks/length_aware_training.py` compares FIFO/shuffled order with the
production online window scheduler or offline length bucketing. It runs the
actual EAGLE3/DFlash2 training models, strategies, `TrainerCore`, FSDP (or DDP
with `--sharding NO_SHARD`), and AdamW on synthetic teacher features.

This is a trainer microbenchmark. It does not run a teacher, feature transport,
or serving, and it cannot establish convergence or acceptance-length quality.
Offline bucketing can move samples between optimizer steps during normal
training. The numerical test deliberately uses one optimizer window to isolate
batching correctness from that expected change in optimization trajectory.

## Numerical gate

Run on two CUDA GPUs from the repository root:

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.length_aware_training \
  --algorithm dflash2 --schedule offline --batch-size 2 \
  --lengths 32,64,96,128 --num-anchors 8 --hidden-size 64 --vocab-size 128 \
  --accumulation-steps 4 --skip-performance --output offline-parity.json
```

Also run DFlash2 with `--schedule online`, and EAGLE3 with
`--algorithm eagle3 --schedule online --batch-size 1`. EAGLE3 batch size greater
than one is intentionally unsupported: its loss normalization includes padded
rows, so regrouping samples would change the objective.

The gate compares the accumulated objective, normalized gradients before
clipping, and one AdamW parameter update. DFlash2 uses fixed random anchors per
sample only for this numerical comparison. The FP32 defaults are
`atol=2e-5`, `rtol=5e-4`, with TF32 disabled; gradient and update relative L2 errors
must also be at most `rtol`. An unsuccessful gate exits nonzero and still saves
its error measurements in JSON.

For a separate BF16 rounding check, add:

```bash
--parity-dtype bfloat16 --atol 0.0002 --rtol 0.03
```

Those tolerances test numerical agreement under lower precision, not model
quality. Do not compare differently sampled anchors and call the resulting
loss difference a batching regression.

Use `--permutation-control --skip-performance` to diagnose ordinary accumulation
order sensitivity: it reorders intact microbatches, preserving sample membership,
padding and rank assignment in each microbatch. It uses the same pass thresholds
as the feature comparison. The JSON also reports total-parameter relative error
and the gradient pair at the largest update discrepancy. These diagnostics do
not override a failed update gate.

## Performance and negative control

```bash
torchrun --standalone --nproc-per-node=2 -m benchmarks.length_aware_training \
  --algorithm dflash2 --schedule offline --batch-size 2 \
  --lengths 128,256,1024,2048 --hidden-size 256 --vocab-size 4096 \
  --num-anchors 32 --accumulation-steps 8 \
  --warmup 3 --steps 10 --repeats 3 --skip-parity --output offline-perf.json
```

Repeat with `--pattern uniform`, then run the online scheduler and EAGLE3 online
batch-size-one cases. A second, larger DFlash2 workload uses
`--lengths 512,2048,4096,8192 --hidden-size 512` with the remaining settings
unchanged. Report both workloads, including cases with negligible benefit.

Performance runs use the native production anchor sampler. Actual supervised
positions are counted from the loss metrics; source samples and valid context
tokens are identical between variants. Each repetition resets the model,
excludes warmup, and alternates variant order (A/B, B/A, A/B). JSON includes all
optimizer-window timing samples, throughput, peak allocated memory, padding,
and source-file hashes.

The recorded CUDA microstep duration difference between ranks is an imbalance
indicator, not a direct GPU-idle or NCCL-wait measurement. Teacher capture,
feature storage/transport, scheduler overhead and startup/compilation are
outside the reported steady-state timing. Real-workload validation must measure
those separately and compare held-out loss plus serving acceptance and speed.

Metadata-only checks can run without PyTorch or CUDA:

```bash
python -m benchmarks.length_aware_training --dry-run \
  --schedule offline --output layout.json
```

## Recorded H200 results (2026-10-02)

Environment: two NVIDIA H200 GPUs, Python 3.12.3, PyTorch 2.13.0+cu130,
CUDA runtime 13.0, Transformers 5.12.1. Base revision:
`53398a8f01ae47175bee8459c5b5cca3848c8a7e`, plus this change. The GPU checkout was
transferred without `.git`; per-run source hashes in the JSON identify the
source snapshot. The benchmark subsequently gained additional diagnostics;
those additions do not change the production training objective or timing path.

All performance rows use SDPA, BF16, `SHARD_GRAD_OP`, accumulation 8, block size
8, and native anchor sampling. Each variant runs three warmup windows and ten
measured windows, repeated three times with model resets and alternating order.
DFlash2 uses two draft layers, 32 anchors, and microbatch size 2. EAGLE3 uses its
single-layer architecture, TTT length 3, and microbatch size 1. Vocabulary size
is 4096. The small workload uses hidden size 256 and lengths
128/256/1024/2048; the larger DFlash2 workload uses hidden size 512 and lengths
512/2048/4096/8192. Uniform controls use the longest length in each workload.

| Workload | Scheduling | Baseline window ms | Enabled window ms | Median paired speedup | Range over three repeats |
| --- | --- | ---: | ---: | ---: | --- |
| DFlash2 small | Online | 133.454 | 133.351 | 0.9989x | 0.9974–1.0008x |
| DFlash2 small | Offline | 133.654 | 133.587 | 1.0045x | 0.9956–1.0081x |
| DFlash2 larger | Online | 136.105 | 135.927 | 1.0044x | 1.0005–1.0097x |
| DFlash2 larger | Offline | 138.123 | 137.315 | 1.0059x | 0.9970–1.0061x |
| EAGLE3 small | Online | 175.760 | 166.506 | 1.0529x | 1.0417–1.0638x |

Window columns are medians of repetition medians; speedup is the median of
paired ratios, so it need not equal the ratio of those two columns. These
ranges are observed ranges, not confidence intervals.

DFlash2 padding fell from 34.15% to zero in the small workload, and from 30.12%
to zero in the larger workload, **without a meaningful measured step-time
improvement**. Its uniform-control medians were 1.0010–1.0058x, with individual
repeat ratios spanning approximately 0.994–1.012x. EAGLE3's uniform control also
improved by 1.0265x (range 1.0089–1.0510x), overlapping the heterogeneous case.
The full EAGLE3 speedup therefore cannot confidently be attributed to length
scheduling from these runs alone. This experiment does not demonstrate an
end-to-end training speedup for either feature.

Native random anchors preserve sample count, context-token count and anchor
budget, but the number of valid labels at sequence ends can differ slightly.
For example, the first small DFlash2 online repetition processed 70,655 versus
70,604 supervised draft positions across the ten measured windows. JSON reports
actual supervised-position throughput as well as context-token throughput.

The numerical fixture uses hidden size 64, vocabulary 128, lengths 32/64/96/128,
8 fixed per-sample anchors, accumulation 4 and the same production objectives.
Errors below are the maximum over the two ranks.

| Numerical comparison | Loss absolute difference | Gradient relative L2 error | Update relative L2 error | Gate |
| --- | ---: | ---: | ---: | --- |
| DFlash2 online FP32 | 0 | 1.99e-7 | 5.82e-6 | Pass |
| DFlash2 offline FP32 | 0 | 1.54e-7 | 7.12e-6 | Pass |
| EAGLE3 online FP32 | 9.54e-7 | 1.40e-7 | 7.92e-6 | Pass |
| DFlash2 online BF16 | 1.14e-5 | 0.446% | 5.59% | **Fail: update** |
| DFlash2 offline BF16 | 1.14e-5 | 0.418% | 5.14% | **Fail: update** |
| EAGLE3 online BF16 | 9.54e-7 | 0.400% | 3.62% | **Fail: update** |

All BF16 loss and gradient gates passed, but the predeclared 3% relative-update
gate failed; its thresholds were not widened after observing results. Total
updated-parameter relative errors were 0.0921%, 0.0906%, and 0.1528%, respectively.
Those smaller total-weight errors do not override the failed update gate.

The intact-microbatch permutation control also failed the same BF16 update
gate: DFlash2 had 0.360% gradient error and 3.83% update error; EAGLE3 had 0.398%
gradient error and 3.41% update error. At the largest update discrepancy, small
gradients changed sign (for example, `1.19e-7` versus `-1.19e-7`). This is
consistent with accumulation-order rounding amplified by the first AdamW update.
It does **not** establish that all feature-induced differences are harmless, or
that trained-model quality is preserved. BF16 update agreement and real-model
held-out/serving validation remain unresolved.

All 20 raw reports (about 203 KB), including failed gates and the earlier
fixed-anchor performance diagnostic, are retained in
[`benchmarks/results/length-aware-h200`](../../../benchmarks/results/length-aware-h200).
The fixed-anchor performance file is explicitly a diagnostic; the main table
uses the native-sampler reports. No real-model convergence or serving benchmark
was run.
