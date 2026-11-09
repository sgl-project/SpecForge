# Frozen vocabulary projection benchmark

This standalone benchmark uses a randomly initialized frozen, bias-free Linear and synthetic hidden states. It downloads no model or data and imports no SpecForge model. It reproduces the layout boundary in the DFlash-family objective: a chunk of a larger [batch, anchors, slots, hidden] tensor is reshaped into a three-dimensional head input.

It compares three layouts using the same weight, hidden values, and fixed random upstream gradient:

- native: the three-dimensional strided head input;
- contiguous: pack that input while preserving its three-dimensional Linear call and hook shape;
- flatten: pack token dimensions into a two-dimensional Linear call and restore output dimensions.

Layout conversion happens inside every measured invocation. The teacher phase measures no-grad forward; the draft phase measures forward plus dHidden with the weight frozen. These are projection-only measurements with an arbitrary fixed upstream gradient, not CE/L1 training, an optimizer step, or end-to-end training throughput.

## CPU sanity checks

The default sizes are deliberately small and use CPU:

    python benchmarks/bench_vocab_projection.py --checks-only
    python -m unittest tests.test_utils.test_vocab_projection_benchmark -v

The numerical checks compare output shapes/dtypes, output values, input-gradient relative L2 and cosine, finite values, frozen weight gradients, and RNG state. Gradient elementwise violations are reported; --strict-gradient-elements also requires every element to satisfy rtol/atol. The aggregate gradient gates remain enabled by default. FP64+autocast is explicitly rejected because autocast does not lower FP64.

## Synthetic CUDA reproduction

Run on an explicitly allocated idle device. The script does not reserve a GPU. For a large BF16 example matching a [4, 896, 2560] head input:

    python benchmarks/bench_vocab_projection.py \
      --device cuda:0 --dtype bfloat16 --batch 4 \
      --anchors 512 --chunk 128 --offset 384 --block-size 7 \
      --hidden-size 2560 --vocab-size 151936 \
      --warmup 5 --iterations 10 --repeats 3 \
      --output projection.json

Other representative shapes use the same B=4, K=7, H=2560 and V=151936:

| anchors | chunk | offset | head input | flattened input |
| --- | --- | --- | --- | --- |
| 176 | 48 | 128 | [4,336,2560] | [1344,2560] |
| 486 | 102 | 384 | [4,714,2560] | [2856,2560] |
| 512 | 128 | 384 | [4,896,2560] | [3584,2560] |

The middle head dimension is per batch element; it is not the flattened row count. Use --describe-only with these arguments to inspect shapes and allocation sizes without importing Torch or running work. B=1 and --layout contiguous provide useful controls where packing may be a no-op. --autocast bfloat16 additionally tests an FP32 input/weight with autocast, including its conversion costs.

## Timing, memory, and traces

Each native/candidate contrast uses repeated ABBA blocks after warmup. CUDA events explicitly record on the current stream for the tensor's selected device, including cuda:1 when the process's current device differs. The report includes device-event time and host wall time; CPU reports wall time only. Compare paired repeat ratios rather than treating one ordering as a performance conclusion.

Peak statistics are reset for each timed block after setup and numerical checks. Absolute allocated peaks include resident weight, full hidden storage and, when the draft phase is requested, the fixed gradient probe. Deltas subtract that block's resident baseline. Reserved memory also reflects the allocator's earlier cache; neither reserved delta nor allocated delta is the whole process footprint. Setup/check transient peaks are outside the measurement window. Run --phases teacher alone to avoid allocating the draft probe when measuring a teacher-only footprint.

Use --profile-dir traces for separate untimed profiler invocations. Reports include Linear/mm/bmm/contiguous/copy operator input shapes and CPU/device time. Summed profiler times can overlap and are not a critical-path measurement. No hardware-counter, HBM-bandwidth, or MFU claim is made.

## Recorded synthetic CUDA run

Validated on one NVIDIA A100-SXM4-40GB, PyTorch 2.13.0+cu129 / CUDA 12.9, BF16, TF32 disabled, frozen random weight [151936,2560], seed 0. Each phase/layout uses five warmups and three paired ABBA repeats, with ten invocations per timed block. Layout conversion is included in each invocation. The two alternatives each have their own paired native contrast.

| m in [4,m,2560] | Teacher native → contiguous | Teacher native → flatten | Draft native → contiguous | Draft native → flatten |
| ---: | ---: | ---: | ---: | ---: |
| 336 | 7.98 → 4.24 (1.88x) | 7.98 → 4.24 (1.88x) | 15.57 → 8.60 (1.81x) | 15.59 → 8.60 (1.81x) |
| 714 | 16.15 → 8.93 (1.81x) | 16.16 → 8.96 (1.80x) | 31.31 → 17.50 (1.79x) | 31.27 → 17.59 (1.78x) |
| 896 | 18.52 → 10.93 (1.69x) | 18.52 → 10.99 (1.69x) | 37.08 → 21.63 (1.71x) | 37.40 → 21.66 (1.72x) |

All forward comparisons were bitwise equal. Input-gradient relative L2 differences were 0.265%–0.387%, with cosine at least 0.9999925, within the predeclared checks. Frozen-weight-gradient and RNG checks passed. The largest timed allocated peak was about 2.86 GiB, with the persistent allocations described above included.

Packing while preserving the 3D call shape and explicitly flattening had similar component timings in this run. These results describe the isolated random-weight projection, with no model/data downloads or optimizer; full-model training measurements are separate.

## Relationship to existing work

This complements the flattened/cached projection work in SpecForge PR #974 and the historical #964 proposal. It adds a reproducible measurement tool, including a same-head-shape contiguous alternative, rather than another runtime projection flag or caching implementation. The benchmark can help assess a targeted main-branch fix and the larger optional optimization stack independently.
