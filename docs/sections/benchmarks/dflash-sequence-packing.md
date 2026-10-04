# DFlash/DFlash2 sequence packing and online validation

Based on upstream [`53398a8f01ae47175bee8459c5b5cca3848c8a7e`](https://github.com/sgl-project/SpecForge/tree/53398a8f01ae47175bee8459c5b5cca3848c8a7e)
plus the local `codex/sequence-packing` changes, tested on 2026-10-02.

The final H200 benchmark with 512 anchors per document and 64% context padding
measured **1.39x DFlash** and **1.47x DFlash2** training-step speedups, with about
29% lower peak allocated memory. These are consumer-compute results. A separate
[real Qwen3-4B online pipeline benchmark](online-sequence-packing.md) measured
**1.041× DFlash** and **1.045× DFlash2** on 128 ShareGPT conversations, including
target capture and transport but excluding startup and final checkpoint.

## Supported execution

Text EAGLE3, DFlash, and DFlash2 support packing for offline features and online
server capture. DFlash2 remains `training.strategy: dflash` with a
`DFlash2DraftModel` draft config. Enable packing on an existing supported config:

```yaml
training:
  attention_backend: flex_attention
  sequence_packing: true
```

For example, add those overrides to
`examples/configs/online/disaggregated/external/qwen3-4b-dflash-online.yaml`.
Packing is opt-in. It packs the existing microbatch and does not reorder samples
or change the number of samples per optimizer step. Other algorithms, USP,
multimodal positions, compact teacher, and loss-position trimming are unsupported.
DFlash/DFlash2 LK and D-PACE objectives are supported; EAGLE3 LK is unsupported.

## Call chain and numerical contract

```text
SGLang /generate capture (one request per original sample)
  -> MooncakeFeatureStore + SampleRef
  -> online consumer / FeatureDataLoader
  -> provider.build_packed_collator: [B, max(L)] -> [1, sum(L)]
  -> DFlashTrainStrategy: host document lengths and valid-anchor counts
  -> OnlineDFlashModel: original sampler, isolated context/proposal positions
  -> DFlashDraftModel or DFlash2DraftModel
  -> restore [B, anchors, block] for original losses and metrics
  -> trainer optimizer / original sample-ID acknowledgements / checkpoint
```

The original sampler runs once on the original `[B, max(L)]` loss-mask shape,
preserving its random draws, each document's anchor budget, anchor order, and
keep mask. Only the small sampling mask is padded. Context hidden states, token
IDs, and optional target final states are concatenated without padding.

Attention never crosses a document boundary. Full and sliding masks respect
per-document starts; target labels cannot pass document ends, and teacher
predecessors cannot precede document starts. RoPE positions reset per document.
DFlash2 convolutions retain complete proposal blocks. Loss reductions, D-PACE
sequence weights, selector objectives, and diagnostics retain their original
batch and anchor axes.

The packed attention mask uses conservative sparse tile ranges, with the exact
per-token predicate for partial tiles. This avoids constructing a dense
`total_queries * total_keys` mask across unrelated documents. Where integer
features are on the CPU, the strategy also supplies per-document valid-anchor
counts: invalid padded proposal slots skip the backbone and outputs are
scattered back into the original loss layout. Direct GPU-only callers without
those counts retain all proposal slots. No CUDA `nonzero()` or host readback is
needed to size the compact proposals.

Online target requests, capture tensors, transport protocol, queue order, and
acknowledged IDs are unchanged. Packing affects the consumer's draft training;
it does not itself accelerate target capture or speculative serving.

## Why concatenation alone was slower

The first correct implementation removed context padding but still generated
a dense FlexAttention mask before converting it to sparse blocks. With four
samples, the flattened query/key grid also contained cross-document regions
that were entirely masked. A CUDA profiler measured DFlash mask construction
at 8.61 ms padded versus 29.43 ms packed in the 512-anchor profile, while the
backbone remained roughly 22 ms. The initial packed implementation regressed
median whole-step time by about 20%. This motivated the sparse tile builder and
invalid-proposal elimination; concatenation alone is not a reliable speedup.

## Validation and benchmark scope

The production comparison checks exact sampled anchors and keep masks, loss,
all loss terms, and every trainable gradient before timing. Model tests also
compare all detailed metrics, cover FP32/BF16, full/hybrid sliding attention,
D-PACE variants, LK lambda/TV, DFlash2 selectors and nonzero convolutions, short
and unsupervised documents, and cross-document perturbation isolation.

Final regression runs passed 204 CPU tests with 540 subtests (nine GPU/live
tests skipped in that CPU run), and 57 GPU model/host-sync tests with 45
subtests. The final live gate and strategy-metadata checks passed five tests
with seven subtests. These counts describe separate suites, not unique tests
summed across repeated runs.

A separate two-rank `FULL_SHARD` probe also passed for EAGLE3, DFlash, and
DFlash2. The ranks used different document lengths, and the packed local loss
and all trainable gradients matched the padded reference after its gradients
were averaged across ranks. This verifies sharded forwards/backwards and host
packing metadata, not multi-node online throughput. The following command uses
the retained local probe, which is not committed with this report:

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONPATH=. python -m torch.distributed.run \
  --standalone --nproc-per-node=2 \
  artifacts/sequence-packing/check_packed_fsdp.py
```

The live online gate uses a tiny eight-layer Llama target and an FSDP training
consumer, with an isolated copy of SGLang 0.5.18 and the repository's capture
patch. The initial gate used separate H200s; the final optimized gate colocated
both processes on one H200 after unrelated work occupied the original devices.
Actual captures pass through Mooncake TCP,
`RefDistributor`, a SQLite durable ledger, and the packed feature loader.
EAGLE3, DFlash, and DFlash2 each execute two optimizer steps/four microsteps,
acknowledge all eight original sample IDs, and write checkpoints. This is
functional online evidence, not an end-to-end throughput benchmark, RDMA
validation, or pretrained-model convergence evidence.

The training-step benchmark uses one H200, BF16, two actual draft layers,
hidden size 2048, intermediate size 8192, 16 attention heads / 4 KV heads,
vocabulary 32000, block size 16, microbatch four, and 25% prompt masking.
Frozen target weights and features are synthetic. It includes the production
strategy's CPU integer-feature processing/transfers, forward, backward, and
`BF16Optimizer` update. Hidden features are GPU resident. Target capture,
hidden-feature I/O/H2D, distributed communication, and serving are excluded.

Five warmup steps include compilation; 20 synchronized wall-clock steps measure
steady-state performance. Useful tokens/s divides the original unpadded input
tokens by mean step time. The benchmark GPU was dedicated to these measurements;
other GPUs on the node also ran unrelated validation workloads.

| Algorithm | Anchors/document | Lengths | Padding | P50 ms padded → packed | P50 speedup | Mean ms padded → packed | Useful tokens/s padded → packed | Peak GiB padded → packed |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| DFlash | 512 | 128, 256, 512, 2048 | 64.06% | 102.68 → 73.68 | **1.39x** | 102.74 → 73.74 | 28,655 → 39,923 | 11.82 → 8.34 |
| DFlash2 | 512 | 128, 256, 512, 2048 | 64.06% | 105.68 → 72.01 | **1.47x** | 105.65 → 72.00 | 27,865 → 40,891 | 13.28 → 9.44 |
| DFlash | 128 | 128, 256, 512, 2048 | 64.06% | 35.38 → 30.50 | 1.16x | 35.42 → 30.50 | 83,107 → 96,540 | 5.81 → 5.43 |
| DFlash2 | 128 | 128, 256, 512, 2048 | 64.06% | 35.85 → 30.81 | 1.16x | 36.10 → 30.87 | 81,542 → 95,379 | 5.08 → 4.80 |
| DFlash | 128 | 1024, 1024, 1024, 1024 | 0% | 33.35 → 31.23 | 1.07x | 33.56 → 31.22 | 122,061 → 131,179 | 5.67 → 5.59 |
| DFlash2 | 128 | 1024, 1024, 1024, 1024 | 0% | 34.63 → 31.35 | 1.10x | 34.71 → 31.55 | 118,005 → 129,824 | 4.94 → 4.98 |

The 512-anchor mixed-length case preserves all 1178 valid sampled blocks and
skips 870 invalid slots from the padded 2048-slot grid. The backbone therefore
processes 18,848 proposal tokens instead of 32,768. Context rows fall from 8192
to 2944. Equal-length gains mainly reflect the sparse mask builder; they are
not evidence of removed context padding.

Numerical checks use exact anchor/keep-mask comparisons and tolerance-based
loss/gradient comparisons. BF16 execution is not bitwise equivalent: one
DFlash2 128-anchor run's loss differed by about 2% after 25 optimizer updates,
despite passing initial loss/all-gradient checks. This experiment does not
establish convergence, checkpoint quality, or serving acceptance equivalence.
Evaluate those separately on a representative pretrained target and dataset.

```bash
PYTHONPATH=. python scripts/benchmark_dflash_sequence_packing.py \
  --preset tiny --dtype float32 --correctness-only \
  --output artifacts/sequence-packing/dflash-tiny.json
PYTHONPATH=. python scripts/benchmark_dflash_sequence_packing.py \
  --preset medium --anchors 512 --warmup 5 --steps 20 \
  --output artifacts/sequence-packing/dflash-medium.json
```

Committed evidence includes the [512-anchor BF16 results](sequence-packing-results/dflash-family-512anchors-bf16.json),
[128-anchor BF16 results](sequence-packing-results/dflash-family-128anchors-bf16.json),
and [tiny-model FP32 comparisons](sequence-packing-results/dflash-family-tiny-fp32.json).
These preserve all timed samples, memory, settings, numerical comparisons and
measured source hashes. The [evidence inventory](sequence-packing-results/README.md)
also links the subsequent real-target online measurements.

Exact source snapshots, source-verification files, capture/consumer service logs,
regression logs and the ad hoc `check_packed_fsdp.py` probe remain local under
`artifacts/sequence-packing/`; they are not committed with this report. The
two-rank command above describes that retained local probe, not a file shipped
in this repository. The committed runtime tests cover the supported model,
strategy and lifecycle behavior.

Do not extrapolate the [EAGLE3 measurements](eagle3-sequence-packing.md) to
DFlash/DFlash2. Their proposal workload differs, and a producer or network
bottleneck can limit end-to-end online gains even when consumer compute improves.
