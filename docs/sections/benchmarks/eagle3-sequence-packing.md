# EAGLE3 sequence packing: implementation and H200 measurements

Measured on 2026-10-02, based on upstream
[`53398a8f01ae47175bee8459c5b5cca3848c8a7e`](https://github.com/sgl-project/SpecForge/tree/53398a8f01ae47175bee8459c5b5cca3848c8a7e),
with the local `codex/sequence-packing` changes.

## Result and scope

Packing improves training throughput when a microbatch contains substantially
different sequence lengths. In the final H200 run below, 47% and 64% padding
produced 1.50x and 1.96x median-step speedups. The mean-based throughput gains
were 1.44x and 1.61x, including observed scheduling outliers. Equal-length
samples showed only a small timing difference. This is not a universal
"biggest lever": batch size one has no inter-sample padding to remove.

This report measures **offline text EAGLE3 with FlexAttention**. The switch
also supports text DFlash/DFlash2 and online server-capture consumers; see the
[current support contract](../basic_usage/training.md#sequence-packing).
The EAGLE3 measurements below do not establish DFlash/DFlash2 or end-to-end
online throughput. FA/USP, multimodal positions, compact teacher, trimmed loss
positions, and EAGLE3 LK objectives remain unsupported.

## How it works

```text
training.sequence_packing
  → offline provider.build_packed_collator
  → DataCollatorWithPacking: [B, max(L)] → [1, sum(L)]
  → Eagle3TrainStrategy: shift teacher/input/mask within each document
  → OnlineEagle3Model: per-document TTT shifts + original loss denominator
  → LlamaFlexAttention: isolated causal prefixes + diagonal TTT cache suffixes
```

The collator keeps sample order and logical microbatch size, and emits document
lengths, reset positions, and the original padded loss denominator. The attention
mask prevents information flow between documents. RoPE uses the maximum
individual document length, preserving dynamic NTK behavior. Every TTT shift
zeros each document's tail rather than importing the next document's tokens.
The plain masked loss is rescaled by `sum(L) / (B * max(L))`, preserving the
existing padded objective and effective learning rate. The number of samples,
optimizer steps, accumulation steps, and checkpoint cursor do not change.

The implementation packs only the existing logical microbatch. It does not
reorder examples or greedily combine additional microbatches. `data.max_length`
continues to limit individual documents; packed rows can exceed that length.

## Reproducible training-step benchmark

- One NVIDIA H200, Python 3.12.3, Torch 2.13.0+cu130, CUDA 13.0,
  Transformers 5.12.1.
- One real EAGLE3 draft layer, hidden size 2048, intermediate size 8192,
  16 attention heads / 4 KV heads, target and draft vocabulary 32000.
- BF16, TTT length 7, microbatch 4, seed 1729, 25% prompt mask.
- Same features and initial weights for both paths. Frozen target-head
  projection, production loss/backward and `BF16Optimizer` update are included.
- Five warmup steps (including compilation), then 20 synchronized wall-clock
  measurements per mode. Compilation is excluded from steady-state timing.
- Synthetic features are already on the GPU. Capture, file I/O, bulk H2D,
  distributed communication, serving, and training convergence are excluded.
- Useful tokens/s counts original unpadded input tokens once and uses the
  **mean** step time. It is not derived from the median or multiplied by TTT.
  Memory is peak allocated CUDA memory, including model/optimizer/input state.

| Original sequence lengths | Padding | P50 step ms, padded → packed | P50 speedup | Mean step ms, padded → packed | Useful tokens/s, padded → packed | Peak GiB, padded → packed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024, 1024, 1024, 1024 | 0.00% | 80.03 → 76.61 | 1.04x | 87.82 → 76.68 | 46,643 → 53,419 | 13.09 → 9.17 |
| 512, 768, 1024, 2048 | 46.88% | 135.94 → 90.35 | 1.50x | 147.45 → 102.27 | 29,515 → 42,554 | 23.98 → 9.60 |
| 128, 256, 512, 2048 | 64.06% | 136.79 → 69.67 | 1.96x | 142.04 → 88.27 | 20,726 → 33,354 | 23.91 → 7.21 |

The full samples retain outliers; for example CPU scheduling produced occasional
long steps. A separate 50-step repeat before the final FSDP metadata conversion
measured median speedups of 1.03x, 1.51x and 1.94x for the same three profiles,
with mean speedups of 1.04x, 1.57x and 1.87x. Treat the difference between median
and mean as part of the measurement, not guaranteed deployment throughput.

The equal-length case also reduces peak memory. Packing changes teacher-table
layout to batch dimension one; the TTT adapter can retain contiguous slices
instead of making per-depth copies across padded batch strides. Thus observed
memory savings are not solely proportional to removed padding.

An additional larger draft (hidden 4096, intermediate 14336, 32 heads / 8 KV
heads) with `[128, 256, 512, 2048]` measured 249.50 → 113.15 ms P50 (2.21x),
271.96 → 115.71 ms mean, and 33.03 → 12.97 GiB peak. This measurement predates
the final FSDP tuple/int metadata conversion; the numerical path is the same,
but use the table above for the final-source timing evidence.

Commands from the repository root:

```bash
PYTHONPATH=. python scripts/benchmark_sequence_packing.py \
  --preset tiny --dtype float32 --correctness-only \
  --output artifacts/sequence-packing/tiny-fp32.json
PYTHONPATH=. python scripts/benchmark_sequence_packing.py \
  --preset medium --warmup 5 --steps 20 \
  --output artifacts/sequence-packing/medium-bf16-current.json
```

## Correctness and integration evidence

- Latest FP32 benchmark: three profiles passed, including documents shorter
  than TTT. Scalar loss matched exactly; maximum trainable-gradient absolute
  difference was approximately 4.1e-10.
- BF16 medium: loss, every depth's loss, and all trainable parameter gradients
  passed the recorded tolerances. Maximum gradient relative L2 difference was
  approximately 0.00658; BF16 equality is numerical, not bitwise.
- Seven production GPU tests passed, including FP32/BF16/dynamic-NTK parity,
  cross-document isolation, mixed CPU/GPU fields, zero supervision, short
  documents, and FSDP metadata compatibility.
- CPU regression: 162 tests and 523 subtests passed; the GPU lifecycle test was
  skipped in this CPU run and executed separately.
- Full offline lifecycle passed: variable-length files, packed train/eval
  loaders, 2 optimizer steps / 4 microsteps, partial eval batch, finite metrics,
  both checkpoints, and original sample counters 4 and 8.
- This EAGLE3 lifecycle used single-rank FSDP, which selects NO_SHARD. The
  subsequent [DFlash/online validation report](dflash-sequence-packing.md)
  includes a two-rank FULL_SHARD comparison and live Mooncake coverage for
  EAGLE3, DFlash and DFlash2. Long-run convergence/acceptance and speculative
  serving performance remain outside these measurements.

Committed evidence includes the [medium BF16 results](sequence-packing-results/eagle3-medium-bf16.json),
[tiny FP32 comparisons](sequence-packing-results/eagle3-tiny-fp32.json), and
[larger-draft results](sequence-packing-results/eagle3-large-bf16.json).
These JSON files contain settings, individual timings, numerical comparisons
and measured source hashes. See the [evidence inventory](sequence-packing-results/README.md)
for provenance and limitations.

Additional validation logs, `source-verification.json` and exact measured source
snapshots remain local under `artifacts/sequence-packing/`; they are not committed
with this report. The local source verification found all seven measured source
ASTs equivalent to the final implementation; changes at that point were
formatting only.

## Enabling packing

```bash
specforge train \
  --config examples/configs/offline/colocated/qwen3-8b-eagle3-offline.yaml \
  training.batch_size=4 \
  training.attention_backend=flex_attention \
  training.sequence_packing=true
```

Choose the same batch size and gradient accumulation for the baseline and
packed run. Increasing batch size at the same time changes the training
schedule and invalidates a simple A/B comparison.

Text DFlash/DFlash2 packing is also implemented, with document-aware context
attention and positions, boundary-safe labels and the same per-document anchor
sampling budget. Its [implementation and measurements](dflash-sequence-packing.md)
use a separate attention and proposal path; EAGLE3 timing gains do not predict
DFlash/DFlash2 gains.
