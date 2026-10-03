# TorchTitan backend: controlled H200 training comparison

This experiment compares SpecForge's default FSDP1 backend with the optional
TorchTitan 0.3.0 FSDP2 backend. It measures actual DFlash, DFlash2, and DSpark
training with cached synthetic features. It does not measure target capture,
feature transport, checkpoint I/O, convergence, acceptance, or serving speed.

The benchmark's FSDP baseline works with SpecForge main. Running
`--backend torchtitan` requires an installed checkout containing
[the backend in PR #920](https://github.com/sgl-project/SpecForge/pull/920), plus
its optional TorchTitan dependency. This benchmark change does not provide the
backend itself. The original measured source and evidence are preserved in
[the immutable experiment snapshot](https://github.com/yushengsu-thu/SpecForge/tree/f381af478657365c6575950b4b7eb5f9dab97bc1).

## Configuration

- Base: SpecForge `53398a8f01ae47175bee8459c5b5cca3848c8a7e`; implementation
  commit `525e67c`.
- Hardware: the same two NVIDIA H200 GPUs on one node for every timed run.
- Runtime: PyTorch 2.13.0+cu130, Transformers 5.12.1, TorchTitan 0.3.0.
- BF16, `SHARD_GRAD_OP`, FlexAttention, no decoder activation checkpointing.
  Algorithm-default objective chunking/checkpointing remains enabled; the
  DFlash2 fused unary head skips objective checkpointing.
- Controlled common backbone: 5 layers, hidden size 2560, intermediate size
  9728, vocabulary 151936, block size 16. Each algorithm keeps its actual architecture and
  loss, including DFlash2 convolution/selector and DSpark's custom heads.
  This is a custom comparison geometry, not a released DFlash2 checkpoint.
- Sequence length 4096; 512 anchors; objective chunks of 128 blocks; batch 1
  per rank; accumulation 2. Each optimizer window contains 16,384 input context
  tokens across both ranks. This is not an accepted-token throughput metric.
- Independently seeded random frozen target embedding/output tables and
  deterministic cached hidden states. Exact initial model, teacher-table, and
  per-rank feature fingerprints match within every backend comparison.
  Two cached batches are reused; their rapidly decreasing losses are not
  evidence of convergence on real training data.
- Three fresh processes per algorithm/backend, alternating A/B then B/A then
  A/B; 10 warmup and 20 measured optimizer windows per process.
- Each window completes CUDA work before stopping the timer; time is reduced
  with MAX across ranks. Peak allocated memory includes frozen teacher tables
  and the resident feature cache. Model initialization and hashing are untimed.

Run the same command for `fsdp` and `torchtitan`, and for `dflash`, `dflash2`,
and `dspark`, changing the output filename for each independent trial:

```bash
torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_backends.py \
  --backend torchtitan --algorithm dflash2 --sharding SHARD_GRAD_OP \
  --attention flex_attention --seq-length 4096 --num-anchors 512 \
  --objective-chunk-blocks 128 --batch-size 1 --accumulation-steps 2 \
  --warmup-steps 10 --steps 20 --repeats 1 --output /tmp/titan-dflash2-r0.json
```

Use independent processes for peak-memory comparisons. An exploratory
in-process repeat retained allocations after model replacement; those trials
are excluded from the results below.

## Results

The table reports the median of three per-process median optimizer-window
times. Parentheses contain the range of the three medians. Speed ratio is
FSDP time divided by TorchTitan time; values above 1 favor TorchTitan.

| Algorithm | Trainable draft | FSDP1 ms/window | TorchTitan ms/window | Speed ratio |
| --- | ---: | ---: | ---: | ---: |
| DFlash v1 | 537.4M | 270.00 (268.97–270.28) | 269.83 (269.60–270.25) | 1.001× |
| DFlash2 | 632.4M | 250.28 (249.83–250.62) | 250.66 (250.16–250.84) | 0.998× |
| DSpark | 615.2M | 459.57 (459.45–459.95) | 459.62 (459.49–459.74) | 1.000× |

| Algorithm | FSDP1 peak GiB | TorchTitan peak GiB |
| --- | ---: | ---: |
| DFlash v1 | 16.003 | 16.011 |
| DFlash2 | 18.462 | 18.491 |
| DSpark | 22.205 | 22.220 |

The measured differences are below 1%; this workload shows no meaningful
training-speed or peak-memory improvement from switching backends.

These results apply to this single-node, two-GPU workload. They do not establish
multi-node scaling, benefits for larger drafts, or performance with other
sequence lengths, anchor counts, precision, or parallelism configurations.

The adapter preserves the existing model kernels, loss and FP32-master
optimizer, and only changes the sharding backend. It does not enable
TorchTitan compilation, tensor/context/pipeline parallelism, or FP8. A backend
switch alone should therefore not be interpreted as enabling those features.

## Numerical and runtime validation

The distributed numerical gate passes all six algorithm/sharding cases on
both PyTorch 2.13.0 and 2.14.0: DFlash, DFlash2 and DSpark with `FULL_SHARD` and
`SHARD_GRAD_OP`, BF16, two ranks, accumulation 2, distinct inputs and unequal
valid-token counts across ranks/microbatches.

The independent unwrapped reference explicitly reconstructs each algorithm's
existing normalization. DFlash normalizes the complete optimizer window;
DSpark normalizes token counts globally inside each microbatch. FSDP1 matches
the reference FP32 masters and Adam moments exactly. Across the TorchTitan
cases, the maximum first-moment relative L2 error is 0.0857%, maximum grad-norm
relative error is 8.44e-6, and minimum FP32-update cosine is 0.999615. Cross-backend
BF16 results are not bitwise identical; strict elementwise parameter equality
is not claimed.

Every TorchTitan case passes bitwise-identical checkpoint continuation for
weights, FP32 masters, Adam moments, RNG and the next update. Frozen teacher
tables stay unchanged and replicated, and exported draft tensors retain their
ordinary full CPU tensor format and parameter names. DSpark also exercises
actual global gradient clipping.

A separate public-entry smoke uses `python -m specforge.cli train`, its
configured two-rank launcher, real tiny DFlash2, offline generated features,
BF16, `FULL_SHARD`, and accumulation 2. The interrupted one-step run resumed to
step two matches an uninterrupted two-step run bitwise for all 36 draft
tensors, both ranks' optimizer/scheduler/RNG state, and training counters.
This fixture has 25,952 draft parameters and verifies wiring/resume, not model
quality or full-size end-to-end throughput.

Raw timings, comparison contracts and source hashes are in
[`artifacts/torchtitan/h200-dp2.json`](https://github.com/yushengsu-thu/SpecForge/blob/f381af478657365c6575950b4b7eb5f9dab97bc1/artifacts/torchtitan/h200-dp2.json).
Numerical and public-entry evidence is in
[`artifacts/torchtitan/validation.json`](https://github.com/yushengsu-thu/SpecForge/blob/f381af478657365c6575950b4b7eb5f9dab97bc1/artifacts/torchtitan/validation.json).
The measured benchmark script differs from the archived script only in
standard-library import ordering and whitespace; normalized executable ASTs
match. The raw report preserves the exact measured script hash. The standalone
copy additionally documents its optional backend requirement; benchmark
behavior is unchanged.
