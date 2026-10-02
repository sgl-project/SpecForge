# DFlash2 length scheduling: profiler follow-up

This diagnostic investigates why the synthetic DFlash2 experiment in
[the validation report](length-aware-training.md) reduced context padding without
a meaningful step-time improvement. It does not change the model, objective,
optimizer, or production scheduler. It profiles the existing trainer fixture.

## Reproduction and scope

Run on two idle CUDA devices from the repository root:

```bash
CUDA_VISIBLE_DEVICES=0,1 OMP_NUM_THREADS=1 \
torchrun --standalone --nproc-per-node=2 -m benchmarks.profile_length_aware_training \
  --algorithm dflash2 --schedule online --batch-size 2 --accumulation-steps 8 \
  --lengths 512,2048,4096,8192 --hidden-size 512 --vocab-size 4096 \
  --draft-layers 2 --num-anchors 32 --block-size 8 \
  --attention-backend sdpa --perf-dtype bfloat16 --sharding SHARD_GRAD_OP \
  --warmup 3 --steps 2 --output profile_h512_online_a32.json
```

The utility reuses `make_orders`, `build_features`, `collate_batches`, and
`make_runner` from `benchmarks.length_aware_training`. Native anchor sampling,
the production model/strategy/TrainerCore/FSDP, and AdamW remain enabled. Both
variants start from the same model initialization. Each is warmed up for three
optimizer windows, followed by two profiled windows (16 microsteps per rank).
The CUDA synchronization at each window boundary is included in its annotated
span. Chrome traces and operator tables are written beside the JSON summary.

All durations here include profiler overhead. They are **not throughput or
speedup results**. Kernel-duration sums can overlap across streams. The kernel
union measures time covered by at least one recorded kernel; it does not measure
SM occupancy. NCCL kernel duration includes protocol/progress and possible peer
waiting, so it must not be described as pure transfer time or GPU idle time.

## Recorded evidence (2026-10-02)

The corrected run uses the same 32-anchor, H512 larger workload as the original
performance table. It ran on two H200 GPUs with PyTorch 2.13.0+cu130, CUDA 13.0,
and the PR checkout `e8e51f2cf26e56174c375dae8e64af904dd09e6d`. Source hashes,
per-rank values, kernel names, CPU operator summaries, and trace SHA256s are
retained in
[`profile-h512-online.json`](../../../benchmarks/results/length-aware-h200/profile-h512-online.json).
The full Chrome traces are retained outside Git; they can be regenerated with
the command above. An earlier diagnostic accidentally used the parser's default
16 anchors; its traces are retained separately and are not used below.

Each entry totals the two profiled optimizer windows on one rank:

| Diagnostic | Baseline rank 0 | Baseline rank 1 | Length-aware rank 0 | Length-aware rank 1 |
| --- | ---: | ---: | ---: | ---: |
| Annotated host span, ms | 440.21 | 461.92 | 430.15 | 429.97 |
| Kernel + memcpy timeline union, ms | 47.85 | 110.82 | 45.45 | 51.17 |
| Non-NCCL kernel-duration sum, ms | 44.21 | 53.10 | 42.14 | 42.14 |
| SDPA forward + backward kernel-duration sum, ms | 5.69 | 7.95 | 5.11 | 5.12 |
| CUDA kernel count | 11,652 | 11,718 | 11,648 | 11,648 |
| CPU launch-API duration sum, ms | 51.91 | 52.29 | 51.09 | 51.75 |
| NCCL kernel-duration sum, ms | 0.35 | 54.65 | 0.34 | 6.14 |
| NCCL kernel count | 20 | 20 | 20 | 20 |

Global context padding falls from 30.12% to zero. The trace confirms that
attention actually executes cuDNN SM90 flash SDPA forward/backward kernels; a
slow math-attention fallback is not the observed explanation. Attention kernel
work becomes shorter, but the approximately 11,650 kernel launches barely
change. There are still 1,168 `aten::mm` calls, 2,274 `aten::mul` calls, 128 fused
grouped-convolution backward calls, and 48 FSDP forward scopes per rank.

The largest ordinary CPU self-time entries include `cudaLaunchKernel`
(42.6–43.5 ms), FSDP forward (37.5–38.4 ms), `aten::mm` (22.1–22.6 ms),
`aten::mul` (16.1–16.4 ms), and grouped-convolution backward (14.8–15.0 ms).
These are operator-attribution measurements, not disjoint percentages of the
optimizer-window wall time. User annotation `profile::microstep` also contains
unattributed Python/framework work; it is not a GPU kernel.

**Inference limited to this synthetic fixture:** host dispatch/framework work
is a substantial constraint, while the amount of padding-sensitive GPU compute
is small. That is consistent with the independent unprofiled experiment showing
little step-time gain after padding removal. Profiler instrumentation increases
host overhead, and another independent experiment used the other GPU pair on
the same machine; this diagnostic does not establish a production bottleneck or
a precise host/GPU time fraction. No speedup ratio is inferred from this table.
The rank-asymmetric NCCL durations are reported rather than reclassified as
network cost or claimed GPU-wait improvements.

## Why the fixture does not settle production performance

The draft's query side operates on the sampled proposal blocks. Reducing context
padding changes the context projections, attention KV length, and related
operations, while the fixed anchor/block budget leaves much of the draft-query
and objective work unchanged. The relevant code is
`OnlineDFlashModel._forward_draft_blocks` and
`Qwen3DFlashAttention._compute_qkv`, which separately project target context and
draft queries. The number of launched operations therefore need not fall in
proportion to context padding.

The synthetic fixture also differs substantially from the checked-in Qwen3.8
27B DFlash2 recipe:

| Setting | Profiled synthetic fixture | Qwen3.8 27B DFlash2 recipe |
| --- | --- | --- |
| Draft hidden size | 512 | 5120 |
| Draft layers | 2 | 5 |
| Vocabulary | 4096 | 248320 |
| Anchor budget | 32 | 512 |
| Block size | 8 | 8 |
| Maximum draft query tokens per sample | 256 | 4096 |
| Attention | Full context, SDPA | Sliding window 2048, FlexAttention |
| Objective chunk size in blocks | 0 | 128 |
| Loss | D-PACE | DFlash |
| Selector rank / top-k | 8 / 4 | 256 / 16 |
| Teacher features | Random, preloaded | Captured from target model |

Recipe values come from
[`configs/qwen3.8-27b-dflash2.json`](../../../configs/qwen3.8-27b-dflash2.json)
and
[`qwen3.8-27b-dflash2-disaggregated.yaml`](../../../examples/configs/online/disaggregated/external/qwen3.8-27b-dflash2-disaggregated.yaml).
The anchor budget can be reduced by the number of eligible tokens in an actual
sample; the query-token figures above are maxima.

The original microbenchmark is still useful for checking sample conservation,
padding, numerical sensitivity, and the particular small-model timing. Extending
only its context length does not turn it into a representative production model.
It excludes capture, storage, transport, scheduler dispatch, and startup. A real
training-step benchmark and capture-to-training measurement are needed to decide
whether either feature improves the user's workload.
