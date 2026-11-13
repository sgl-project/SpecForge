# Frozen vocabulary projection benchmark

This standalone benchmark uses a randomly initialized frozen, bias-free Linear and synthetic hidden states. It downloads no model or data and imports no SpecForge model. It reproduces the layout boundary in the DFlash-family objective: a chunk of a larger [batch, anchors, slots, hidden] tensor is reshaped into a three-dimensional head input.

It compares three layouts using the same weight, hidden values, and fixed random upstream gradient:

- native: the three-dimensional strided head input;
- contiguous: pack that input while preserving its three-dimensional Linear call and hook shape;
- flatten: pack token dimensions into a two-dimensional Linear call and restore output dimensions.

Layout conversion happens inside every measured invocation. The teacher phase measures no-grad forward; the draft phase measures forward plus dHidden with the weight frozen. These are projection-only measurements with an arbitrary fixed upstream gradient, not CE/L1 training, an optimizer step, or end-to-end training throughput.

## Training helper change

The shared DFlash-family training helper makes the three-dimensional input contiguous before calling the existing LM-head module. This covers draft and teacher projections, including DSpark objective chunks and checkpoint recomputation. It retains the head's three-dimensional call shape, module forward/hooks, full vocabulary, and the caller's losses and normalization. It does not alter the separate inference-time draft-head implementations.

An already-contiguous input needs no packing allocation. A strided chunk can require a hidden-state copy; the benchmark includes that conversion cost. Input stride and storage aliasing can change even though the module call shape is retained.

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

## Actual shared training helper validation

The patched production `OnlineDFlashModel._project_logits` was also called directly instead of the benchmark's mirrored contiguous branch, on the same A100/PyTorch/BF16 setup above. B=3 and B=2 match uneven-DP local batch shapes. Five warmups, three paired ABBA repeats and ten invocations per block include packing. The six cases made 804 actual-helper calls and used an 8 GiB PyTorch allocator cap, with zero target capture queries and zero optimizer updates.

| B | m in [B,m,2560] | Teacher native → actual contiguous (ms) | Draft forward+dHidden native → actual contiguous (ms) |
|---:|---:|---:|---:|
| 3 | 336 | 5.954 → 3.064 (1.94×) | 11.513 → 6.043 (1.90×) |
| 3 | 714 | 12.074 → 6.763 (1.79×) | 23.870 → 13.354 (1.79×) |
| 3 | 896 | 13.898 → 8.400 (1.65×) | 26.139 → 16.364 (1.60×) |
| 2 | 336 | 3.958 → 2.276 (1.74×) | 8.565 → 4.434 (1.93×) |
| 2 | 714 | 8.007 → 4.611 (1.74×) | 15.492 → 9.084 (1.71×) |
| 2 | 896 | 9.276 → 5.510 (1.68×) | 17.846 → 10.858 (1.64×) |

Actual-helper versus flatten teacher/draft outputs and draft dHidden were bitwise equal in every case. Native-strided versus either alternative forward outputs were bitwise equal; dHidden relative L2 differences were 0.265%–0.405%, with cosine at least 0.9999918, within the original checks. These remain projection-only results, not training-step or inference-TPS speedups.

| B | m | Teacher allocated peak increase (MiB) | Draft allocated peak increase (MiB) |
|---:|---:|---:|---:|
| 3 | 336 | 4.922 | 0.000 |
| 3 | 714 | 10.459 | 0.000 |
| 3 | 896 | 14.000 | 0.000 |
| 2 | 336 | 3.281 | 0.000 |
| 2 | 714 | 6.973 | 0.000 |
| 2 | 896 | 8.750 | 0.000 |

These are differences in the maximum allocated peak over each native/contiguous contrast. Absolute peaks include the resident weight, full hidden storage and the draft gradient probe because both phases were requested. They do not measure the whole process footprint or optimizer memory. Allocator granularity can differ from the logical hidden-copy size.

To execute the actual training helper, run this adapter from the repository root on an explicitly allocated GPU. This example uses cuda:0 and B=3, m=896. Use B=3/2 and `(anchors,chunk,offset)` values `(176,48,128)`, `(486,102,384)` or `(512,128,384)` for the six cases.

```bash
python - --device cuda:0 --dtype bfloat16 --batch 3 \
  --anchors 512 --chunk 128 --offset 384 --block-size 7 \
  --hidden-size 2560 --vocab-size 151936 \
  --warmup 5 --iterations 10 --repeats 3 --output actual_helper.json <<'PY'
import importlib.util
from types import SimpleNamespace

import torch
from tests.test_utils.test_dflash_losses import OnlineDFlashModel

spec = importlib.util.spec_from_file_location(
    "projection_bench", "benchmarks/bench_vocab_projection.py"
)
bench = importlib.util.module_from_spec(spec)
spec.loader.exec_module(bench)
project_reference = bench.project
checks_reference = bench.numerical_checks
torch.cuda.set_per_process_memory_fraction(
    (8 << 30) / torch.cuda.get_device_properties(0).total_memory, 0
)

def project(case, variant):
    if variant == "contiguous":
        return OnlineDFlashModel._project_logits(
            SimpleNamespace(lm_head=case.head), case.hidden
        )
    return project_reference(case, variant)

def checks(case):
    result = checks_reference(case)
    for phase in case.args.phases:
        left, left_grad = bench.invoke(case, "contiguous", phase, True)
        right, right_grad = bench.invoke(case, "flatten", phase, True)
        assert left.equal(right)
        if phase == "draft":
            assert left_grad.equal(right_grad)
        del left, left_grad, right, right_grad
    case.hidden.grad = None
    return result

bench.project = project
bench.numerical_checks = checks
bench.main()
PY
```

The test module supplies the existing lightweight loader for the production class method; no draft/target model is constructed by this adapter. The standalone benchmark itself still imports no SpecForge model. The adapter checks exact helper/flatten equality in addition to the standard native-reference diagnostics.
