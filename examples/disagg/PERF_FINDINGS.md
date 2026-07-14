# Domino Disagg — Performance Findings & Tuning Guide

Perf investigation of online domino-disagg training on a single 8×H200 node
(1 patched SGLang capture server + 7 FSDP draft trainers, Qwen3-8B target +
~1.1 B DFlash draft). All throughput numbers are **total-samples ÷ total-time
integrated over ≥390 s** — single ~20 s windows are unreliable here (bursty
supply, ~1.8× spread).

## Headline

Config/env tuning alone took the 1srv+DP7 setup from **28.8 → 50.1 samples/s
(+74%)** with util/SM/mem healthy on all 8 GPUs (server util 98% / SM 76%;
trainers util 77–100% / SM 76–85%). No model or default-behavior change — every
lever below is an env knob that is off by default.

## Best config (settings only)

```bash
U=http://127.0.0.1:30000
export DISAGG_REF_CHANNEL=/shared/control/domino-dp7.refs.jsonl
export DISAGG_DB=/shared/control/domino-dp7.sqlite
export DISAGG_SERVER_URLS="$U,$U,$U,$U,$U,$U,$U,$U"   # 8 concurrent producer workers
export CLONE_ON_FETCH=0        # skip the redundant clone on the mooncake zero-copy path
export LOADER_PREFETCH=2       # background-thread batch prefetch (hide fetch latency)
export SERVER_MEM_FRACTION=0.5 # right-size the server KV reservation (126 GB -> ~78 GB)
ACCUM=8 BATCH_SIZE=2 MOONCAKE_PROTOCOL=rdma MOONCAKE_RDMA_DEVICES=mlx5_0,...,mlx5_7
# then run the producer/consumer roles through examples/disagg/run_role.sh
```

The channel and database paths are required launch state, not tuning knobs.
Choose fresh paths for each attempt, and place `DISAGG_DB` where every consumer
rank can see it. The public CLI does not resume online runs in this PR.

## What each lever does (and why)

| Lever | Effect | Mechanism |
|---|---|---|
| `DISAGG_SERVER_URLS` = URL repeated N× | breaks the single-producer ceiling | N concurrent rollout workers with disjoint leases drive one server (the blocking HTTP prefill call releases the GIL) |
| `ACCUM=8` | 460 → ~280 ms/microstep | `no_sync` grad accumulation amortizes the FSDP reduce-scatter across 8 microsteps (comm drops to ~1 ms/microstep) |
| `CLONE_ON_FETCH=0` | −15 ms/batch | the mooncake zero-copy `get()` already allocates a fresh tensor; the extra defensive clone is redundant |
| `LOADER_PREFETCH=2` | removes fetch from the step | a background thread materializes batches ahead so the training step never pays get/collate latency inline |
| `SERVER_MEM_FRACTION=0.5` | server 126 → 78 GB | the default 0.85 hoards KV cache the capture-only server never uses; no perf cost |

Also added (server-side, in `patches/sglang/.../spec-capture.patch`): the capture
sink keeps its hidden-state slices on GPU and does one `torch.cat` + a single D2H
per request instead of a per-prefill-batch unpinned copy (`d2h` 5–8 → ~3.8 ms/sample).

## The pipeline is a supply/demand seesaw on 8 GPUs

At 50 samples/s the system is **supply-bound**: the single server can just barely
feed 7 trainers (loaders wait ~40 ms/batch; producer `in_flight` stays low). The
GPU split is a genuine trade-off, and 1:7 wins:

| Split | Throughput | Regime |
|---|---|---|
| **1 srv + 7 trn** | **50.1/s** | supply-bound (best) |
| 2 srv + 6 trn (b2) | 39.0/s | demand-bound (only 6 trainers) |
| 2 srv + 6 trn (b8) | 39.5/s | demand-bound (bigger batch changes nothing) |

Each trainer GPU ≈ 7 samples/s; one server ≈ 52–57 samples/s. Seven trainers
(~50/s demand) are almost exactly balanced by one server, so trading a trainer
for a second server loses more than it gains.

## Bigger batch does NOT help (and ~100% memory is an anti-goal)

| batch | accum | throughput | trainer mem | trainer util |
|---|---|---|---|---|
| **2** | **8** | **50.1/s** | 45 GB (~31%) | 77–100% |
| 4 | 4 | 47.5/s | 65 GB (~45%) | 64–99% |
| 8 | 2 | 47.2/s | 113 GB (~79%) | 100% |

Memory does climb toward full with batch (b8 → 113 GB, would OOM ~b12–16) but
throughput *falls* — bigger batch just makes each fetch 4× heavier (`get_ms`
20→90) while per-sample compute stays flat. Low trainer memory (~31%) is the
**efficient** state, not a symptom. The trainer is compute-bound at a healthy
**~44% MFU** (measured — see the MFU section below), not memory-bound;
"full util/memory" ≠ "fast".

## Why is per-sample trainer demand so low? (num_anchors)

Per-microstep compute (batch 2, `PROFILE_STEPS`, cuda-synchronized — trust the
composition, not the absolute):

| num_anchors | fwd | bwd | opt | data_wait |
|---|---|---|---|---|
| 256 (default) | ~88 ms | ~150 ms | 1.3 ms | ~40 ms (14%) |
| 64 | ~43 ms | ~58 ms | 1.3 ms | ~145 ms (38%) — now supply-starved |

The domino/DFlash draft training forward **expands every sample to
`num_anchors × block_size` = 256 × 16 = 4096 draft positions** (independent of
sequence length), each pushed through the 5-layer draft plus a full
151,936-vocab head. Fitting `compute ≈ fixed + k·anchors`: **~53 ms fixed + ~0.73
ms/anchor**, so **~78% of the ~240 ms/microstep compute at 256 anchors is the
anchor expansion.** Cutting anchors to 64 more than halves compute — and the
trainer immediately becomes supply-starved (data_wait 40 → 145 ms), which both
confirms the anchor expansion is the demand driver *and* re-exposes the server as
the next wall.

So the low demand is **by design, not inefficiency**: domino deliberately does
256× the per-token prediction work to densify the draft's training signal. The
optimizer/comm is a non-factor under `ACCUM=8` (1.3 ms/microstep).

**`num_anchors` is NOT a throughput lever — do not reduce it for speed.** The
metric that matters is training signal per second (anchor-updates/s), not
sequences/s. Because each anchor is a supervised training target, and there is a
fixed ~53 ms/microstep per-sequence overhead (base forward + embedding + GRU +
the vocab head over the base positions), *fewer* anchors amortize that fixed cost
over less signal: at 256 anchors ≈ 2,130 anchor-updates/s vs at 64 anchors ≈
1,280 anchor-updates/s. So dropping anchors makes sequences/s rise but
training-signal/s **fall ~40%**, and — since the pipeline is supply-bound —
forces you to capture ~4× more sequences for the same signal, piling load onto
the exact bottleneck. Treat `num_anchors` purely as a quality/data-efficiency
setting (validated on the acceptance-length curve); the launcher keeps it fixed
at 256. The real memory/compute lever is the **full-vocab loss layer** (next
section), which is reducible without touching the training signal.

## MFU / FLOPs (both sides ≈ 43–44%, compute-bound)

**Trainer — measured** (`bench_domino_mfu.py`, isolated single GPU, no data-path
/ no DP comm, bf16, real qwen3-8b-domino shapes, `num_anchors=256`, seq 768; FLOPs
counted with `torch.utils.flop_counter`, timed with CUDA events):

| bsz | step | per-sample | FLOP/sample (fwd+bwd) | achieved | MFU | peak mem |
|---|---|---|---|---|---|---|
| 2 | 211 ms | 105.6 ms | ~45 T | 430 TFLOP/s | **43.5%** | 25 GB |
| 4 | 416 ms | 103.9 ms | ~45 T | 437 TFLOP/s | 44.1% | 46 GB |
| 8 | 836 ms | 104.6 ms | ~45 T | 434 TFLOP/s | 43.9% | 87 GB |

So the ~1.1 B draft trains at a **healthy ~44% MFU** (typical LLM training is
30–50%), i.e. it is **compute-bound, not stalled**. (This supersedes an earlier
"MFU ~14% / occupied-but-stalled" note, which came from a `cuda.synchronize`-
distorted timing and was wrong.) Per-sample time is **flat across batch** →
confirms it is compute-throughput-bound, not launch/latency-bound, so batch buys
nothing. Memory is ~10 GB/sample (linear) — not a constraint. "Slow in
samples/s" just reflects that one sample is ~45 TFLOP (fwd+bwd) ≈ 9× a normal
1.1 B forward, because of the 256-anchor × double-151,936-vocab structure — the
work is inherent to domino, executed efficiently.

**Inference (target prefill) — estimated ≈ 43% MFU.** From the 50/s server log:
prefill runs at **~27,000 tokens/s** (mean over 1,080 batches) at ~100% duty;
average prompt ≈ 550 tokens (confirmed two ways: loader bytes, and
49/s × 550 = 27k tok/s). For the 8 B target, `2 × params × tokens` =
2 × 8e9 × 27,000 ≈ **430 TFLOP/s ≈ 43% MFU** — a normal prefill efficiency, not a
capture slowdown. (Estimate from tokens/s, not a direct FLOP count; ignores
attention FLOPs and the 5-aux-layer capture overhead.)

## Remaining ceilings (need code, not config)

1. **Supply ≈ 50/s is PREFILL-COMPUTE-BOUND, not sink-bound.** At ≥8 concurrent
   producer workers the server GPU prefills ~100% of the time at ~27k tok/s
   (~43% MFU on the 8 B target, ~550-tok prompts → ~49 prompts/s). The mooncake
   sink thread *keeps up* (~44–57/s ≥ supply) — it is the **next** wall (~55/s,
   single thread), not the current one. So an **async/parallel sink is only a
   second-order lever** (worthwhile *after* prefill is sped up), NOT the primary
   fix. The real supply levers: (a) higher prefill MFU — CUDA graphs for the
   capture/piecewise-prefill path, bigger prefill batches, lighter aux-hidden-
   state extraction+D2H (async copy stream) → maybe 43%→~55% ≈ +30%; (b) fewer
   aux layers (a training change); (c) a 2nd inference GPU (the seesaw — steals a
   trainer; 2srv+6t measured 39/s, worse on 8 GPUs).
2. **Trainer demand ≈ 50/s at ~44% MFU** (compute-bound, above). Because the
   system is supply-bound, raising trainer MFU (CUDA-graphing the small anchor-
   block kernels, kernel fusion, reduced-vocab head) does NOT raise system
   throughput today — it only makes the trainer idle more. It matters only once
   supply is lifted past the trainer, or to cut trainer GPU cost. `num_anchors`
   is **not** a lever here (reducing it lowers training-signal/s — see above).

## Profiling knobs (all env-gated, default-off)

`PROFILE_PRODUCER=N` (`[prod]`/`[prod-http]`), `PROFILE_LOADER=N` (`[loader rK]`),
`PROFILE_STORE=N` (`[store rK]`), `PROFILE_DISTRIB=secs` (`[dist]`),
`PROFILE_STEPS=N` (`[profile]`/`[profile2]` data-wait vs fwd/bwd/opt),
`PROFILE_TORCH=N` (rank-0 torch.profiler), `FSDP_SHARDING=NO_SHARD`.
