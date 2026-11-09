# Copyright 2026 The SpecForge team. All rights reserved.
# Licensed under the Apache License, Version 2.0.
"""Standalone synthetic benchmark for strided frozen vocabulary projections.

No model/data downloads are used. CPU defaults are a small numerical sanity
check. CUDA runs must be scheduled on an otherwise idle, explicitly allocated
device. Timings include layout conversion; they are projection-only results.
"""
import argparse
import contextlib
import json
import math
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

VARIANTS = ("native", "contiguous", "flatten")
PHASES = ("teacher", "draft")


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--dtype",
        choices=("float64", "float32", "bfloat16", "float16"),
        default="float32",
    )
    parser.add_argument("--autocast", choices=("bfloat16", "float16"))
    parser.add_argument("--batch", type=int, default=2)
    parser.add_argument("--anchors", type=int, default=8)
    parser.add_argument("--chunk", type=int, default=3)
    parser.add_argument("--offset", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=7)
    parser.add_argument("--hidden-size", type=int, default=32)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument(
        "--layout", choices=("strided", "contiguous"), default="strided"
    )
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cpu-threads", type=int, default=2)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--atol", type=float)
    parser.add_argument("--max-grad-relative-l2", type=float)
    parser.add_argument("--min-grad-cosine", type=float, default=0.9999)
    parser.add_argument("--strict-gradient-elements", action="store_true")
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--describe-only", action="store_true")
    parser.add_argument("--profile-dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def validate_args(args):
    for name in (
        "batch",
        "anchors",
        "chunk",
        "block_size",
        "hidden_size",
        "vocab_size",
        "iterations",
        "repeats",
        "cpu_threads",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"{name} must be positive")
    if args.offset < 0 or args.offset + args.chunk > args.anchors:
        raise ValueError("offset + chunk must fit within anchors")
    if args.warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if args.dtype == "float64" and args.autocast is not None:
        raise ValueError(
            "autocast does not lower float64; omit --autocast or use float32"
        )
    for name in ("rtol", "atol", "max_grad_relative_l2"):
        value = getattr(args, name)
        if value is not None and (not math.isfinite(value) or value < 0):
            raise ValueError(f"{name} must be finite and nonnegative")
    if not math.isfinite(args.min_grad_cosine) or not -1 <= args.min_grad_cosine <= 1:
        raise ValueError("min_grad_cosine must be finite and within [-1, 1]")


def make_case(args, torch):
    validate_args(args)
    device = torch.device(args.device)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("this benchmark supports CPU and CUDA")
    if device.type == "cpu":
        torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(args.seed)
    dtype = getattr(torch, args.dtype)
    effective_dtype = getattr(torch, args.autocast) if args.autocast else dtype
    head = (
        torch.nn.Linear(
            args.hidden_size,
            args.vocab_size,
            bias=False,
            device=device,
            dtype=dtype,
        )
        .requires_grad_(False)
        .eval()
    )
    full = torch.randn(
        args.batch,
        args.anchors,
        args.block_size,
        args.hidden_size,
        device=device,
        dtype=dtype,
    )
    selected = full[:, args.offset : args.offset + args.chunk]
    if args.layout == "contiguous":
        selected = selected.contiguous()
    hidden = selected.detach().requires_grad_(True)
    probe = None
    if "draft" in args.phases:
        probe = torch.randn(
            args.batch,
            args.chunk,
            args.block_size,
            args.vocab_size,
            device=device,
            dtype=effective_dtype,
        )
    low_precision = effective_dtype in (torch.bfloat16, torch.float16)
    rtol = args.rtol if args.rtol is not None else (0.02 if low_precision else 1e-5)
    atol = args.atol if args.atol is not None else (0.002 if low_precision else 1e-6)
    grad_limit = args.max_grad_relative_l2
    if grad_limit is None:
        grad_limit = 0.01 if low_precision else 1e-5
    return SimpleNamespace(
        args=args,
        torch=torch,
        head=head,
        hidden=hidden,
        probe=probe,
        rtol=rtol,
        atol=atol,
        grad_limit=grad_limit,
    )


def project(case, variant):
    """Mirror the current four-dimensional wrapper and two layout alternatives."""
    hidden = case.hidden
    batch, chunks, slots, width = hidden.shape
    head_input = hidden.reshape(batch, chunks * slots, width)
    if variant == "native":
        logits = case.head(head_input)
    elif variant == "contiguous":
        # Preserve the three-dimensional head call/hook shape.
        logits = case.head(head_input.contiguous())
    elif variant == "flatten":
        logits = case.head(head_input.reshape(-1, width))
    else:
        raise ValueError(variant)
    return logits.reshape(batch, chunks, slots, -1)


def invoke(case, variant, phase, capture=False):
    case.hidden.grad = None
    grad_context = (
        case.torch.no_grad() if phase == "teacher" else contextlib.nullcontext()
    )
    amp_context = (
        case.torch.autocast(
            case.hidden.device.type, dtype=getattr(case.torch, case.args.autocast)
        )
        if case.args.autocast
        else contextlib.nullcontext()
    )
    with grad_context, amp_context:
        output = project(case, variant)
    if phase == "draft":
        output.backward(case.probe)
    if capture:
        return output.detach(), (
            None if case.hidden.grad is None else case.hidden.grad.detach()
        )
    return None


def compare(expected, actual, rtol, atol):
    """Bounded-memory diagnostics; FP64 accumulation is outside timed regions."""
    if expected.shape != actual.shape or expected.dtype != actual.dtype:
        raise AssertionError("output/gradient shape or dtype changed")
    left, right = expected.reshape(-1), actual.reshape(-1)
    max_abs = 0.0
    error_sq = ref_sq = actual_sq = inner = 0.0
    violations = nonfinite = 0
    for start in range(0, left.numel(), 1 << 20):
        a = left[start : start + (1 << 20)].double()
        b = right[start : start + (1 << 20)].double()
        finite = a.isfinite() & b.isfinite()
        nonfinite += int((~finite).sum().item())
        error = (a - b).abs()
        if error.numel():
            max_abs = max(max_abs, float(error.max().item()))
        violations += int((error > atol + rtol * a.abs()).sum().item())
        error_sq += float(error.square().sum().item())
        ref_sq += float(a.square().sum().item())
        actual_sq += float(b.square().sum().item())
        inner += float((a * b).sum().item())
    relative_l2 = (
        math.sqrt(error_sq / ref_sq)
        if ref_sq
        else (0.0 if error_sq == 0 else float("inf"))
    )
    cosine = (
        inner / math.sqrt(ref_sq * actual_sq)
        if ref_sq and actual_sq
        else (1.0 if ref_sq == actual_sq == 0 else 0.0)
    )
    return dict(
        shape=list(expected.shape),
        dtype=str(expected.dtype),
        max_abs=max_abs,
        relative_l2=relative_l2,
        cosine=max(-1.0, min(1.0, cosine)),
        elementwise_violations=violations,
        nonfinite_values=nonfinite,
        bitwise_equal=expected.equal(actual),
    )


def numerical_checks(case):
    checks = []
    cpu_rng = case.torch.get_rng_state().clone()
    cuda_rng = (
        case.torch.cuda.get_rng_state(case.hidden.device).clone()
        if case.hidden.is_cuda
        else None
    )
    for phase in case.args.phases:
        reference_output, reference_gradient = invoke(case, "native", phase, True)
        for variant in VARIANTS[1:]:
            output, gradient = invoke(case, variant, phase, True)
            output_check = compare(reference_output, output, case.rtol, case.atol)
            if (
                output_check["nonfinite_values"]
                or output_check["elementwise_violations"]
            ):
                raise AssertionError(
                    f"{phase}/{variant} output check failed: {output_check}"
                )
            gradient_check = None
            if phase == "draft":
                gradient_check = compare(
                    reference_gradient,
                    gradient,
                    case.rtol,
                    case.atol,
                )
                if (
                    gradient_check["nonfinite_values"]
                    or gradient_check["relative_l2"] > case.grad_limit
                    or gradient_check["cosine"] < case.args.min_grad_cosine
                    or (
                        case.args.strict_gradient_elements
                        and gradient_check["elementwise_violations"]
                    )
                ):
                    raise AssertionError(
                        f"{variant} dHidden check failed: {gradient_check}"
                    )
            elif reference_gradient is not None or gradient is not None:
                raise AssertionError(
                    "teacher branch unexpectedly produced input gradients"
                )
            checks.append(
                dict(
                    phase=phase,
                    variant=variant,
                    output=output_check,
                    input_gradient=gradient_check,
                )
            )
            del output, gradient
        del reference_output, reference_gradient
    if case.head.weight.grad is not None:
        raise AssertionError("frozen weight received a gradient")
    if not cpu_rng.equal(case.torch.get_rng_state()):
        raise AssertionError("projection consumed CPU RNG")
    if cuda_rng is not None and not cuda_rng.equal(
        case.torch.cuda.get_rng_state(case.hidden.device)
    ):
        raise AssertionError("projection consumed CUDA RNG")
    case.hidden.grad = None
    return checks


def timed_block(case, variant, phase):
    torch = case.torch
    cuda = case.hidden.is_cuda
    case.hidden.grad = None
    if cuda:
        torch.cuda.synchronize(case.hidden.device)
        torch.cuda.reset_peak_memory_stats(case.hidden.device)
        allocated = torch.cuda.memory_allocated(case.hidden.device)
        reserved = torch.cuda.memory_reserved(case.hidden.device)
        stream = torch.cuda.current_stream(case.hidden.device)
        start, end = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall_start = time.perf_counter()
    if cuda:
        start.record(stream)
    for _ in range(case.args.iterations):
        invoke(case, variant, phase)
    if cuda:
        end.record(stream)
        end.synchronize()
    wall_seconds = time.perf_counter() - wall_start
    result = dict(
        variant=variant,
        phase=phase,
        wall_ms_per_iteration=wall_seconds * 1000 / case.args.iterations,
    )
    if cuda:
        peak = torch.cuda.max_memory_allocated(case.hidden.device)
        peak_reserved = torch.cuda.max_memory_reserved(case.hidden.device)
        result.update(
            cuda_ms_per_iteration=start.elapsed_time(end) / case.args.iterations,
            baseline_allocated_bytes=allocated,
            peak_allocated_bytes=peak,
            peak_allocated_delta_bytes=max(0, peak - allocated),
            baseline_reserved_bytes=reserved,
            peak_reserved_bytes=peak_reserved,
            peak_reserved_delta_bytes=max(0, peak_reserved - reserved),
        )
    return result


def measure(case):
    blocks, pairs = [], []
    for phase in case.args.phases:
        for variant in VARIANTS:
            for _ in range(case.args.warmup):
                invoke(case, variant, phase)
        if case.hidden.is_cuda:
            case.torch.cuda.synchronize(case.hidden.device)
        for candidate in VARIANTS[1:]:
            for repeat in range(case.args.repeats):
                # Two independent ABBA contrasts, rather than timing A/B/C once.
                order = ("native", candidate, candidate, "native")
                current = []
                for position, variant in enumerate(order):
                    block = timed_block(case, variant, phase)
                    block.update(candidate=candidate, repeat=repeat, position=position)
                    current.append(block)
                blocks.extend(current)
                key = (
                    "cuda_ms_per_iteration"
                    if case.hidden.is_cuda
                    else "wall_ms_per_iteration"
                )
                a = statistics.mean([current[0][key], current[3][key]])
                b = statistics.mean([current[1][key], current[2][key]])
                pairs.append(
                    dict(
                        phase=phase,
                        candidate=candidate,
                        repeat=repeat,
                        native_ms=a,
                        candidate_ms=b,
                        component_speedup=a / b,
                    )
                )
    summary = []
    for phase in case.args.phases:
        for candidate in VARIANTS[1:]:
            rows = [
                x for x in pairs if x["phase"] == phase and x["candidate"] == candidate
            ]
            summary.append(
                dict(
                    phase=phase,
                    candidate=candidate,
                    median_native_ms=statistics.median(x["native_ms"] for x in rows),
                    median_candidate_ms=statistics.median(
                        x["candidate_ms"] for x in rows
                    ),
                    median_component_speedup=statistics.median(
                        x["component_speedup"] for x in rows
                    ),
                )
            )
    return dict(
        order="ABBA for each native/candidate contrast",
        blocks=blocks,
        pairs=pairs,
        summary=summary,
    )


def profile(case):
    if case.args.profile_dir is None:
        return []
    torch = case.torch
    directory = case.args.profile_dir
    directory.mkdir(parents=True, exist_ok=True)
    rows = []
    activities = [torch.profiler.ProfilerActivity.CPU]
    if case.hidden.is_cuda:
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    for phase in case.args.phases:
        for variant in VARIANTS:
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                profile_memory=True,
            ) as profiler:
                invoke(case, variant, phase)
                if case.hidden.is_cuda:
                    torch.cuda.synchronize(case.hidden.device)
            trace = directory / f"{phase}_{variant}.trace.json"
            profiler.export_chrome_trace(str(trace))
            operations = []
            for event in profiler.key_averages(group_by_input_shape=True):
                if event.key in (
                    "aten::linear",
                    "aten::mm",
                    "aten::bmm",
                    "aten::contiguous",
                    "aten::clone",
                    "aten::copy_",
                ):
                    operations.append(
                        dict(
                            name=event.key,
                            count=event.count,
                            input_shapes=event.input_shapes,
                            cpu_time_total_us=event.cpu_time_total,
                            device_time_total_us=getattr(event, "device_time_total", 0),
                        )
                    )
            rows.append(
                dict(
                    phase=phase,
                    variant=variant,
                    trace=str(trace),
                    operations=operations,
                )
            )
    return rows


def run(args, torch):
    case = make_case(args, torch)
    checks = numerical_checks(case)
    head_input = case.hidden.reshape(
        args.batch, args.chunk * args.block_size, args.hidden_size
    )
    metadata = dict(
        torch_version=torch.__version__,
        cuda_runtime=torch.version.cuda,
        device=str(case.hidden.device),
        dtype=str(case.hidden.dtype),
        autocast=args.autocast,
        weight_shape=list(case.head.weight.shape),
        weight_frozen=not case.head.weight.requires_grad,
        hidden_shape=list(case.hidden.shape),
        hidden_stride=list(case.hidden.stride()),
        head_input_shape=list(head_input.shape),
        head_input_stride=list(head_input.stride()),
        head_input_contiguous=head_input.is_contiguous(),
        float32_matmul_precision=torch.get_float32_matmul_precision(),
    )
    if case.hidden.is_cuda:
        properties = torch.cuda.get_device_properties(case.hidden.device)
        metadata.update(
            device_name=properties.name,
            device_total_memory_bytes=properties.total_memory,
            allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        )
    timing = None if args.checks_only else measure(case)
    profiling = profile(case)
    return dict(
        metadata=metadata,
        arguments={
            k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
        },
        numerical_checks=checks,
        timing=timing,
        profiling=profiling,
        scope="Synthetic frozen-head projection only; no training/E2E speedup or MFU claim.",
        layout_copy_cost_included=True,
        peak_scope=(
            "Window starts after setup/numerical checks. Absolute peaks include "
            "resident weight/full hidden/probe; deltas subtract the block baseline. "
            "Reserved memory also includes allocator cache from earlier operations."
        ),
        profiling_scope="Separate untimed invocations; summed operation times can overlap.",
        strict_gradient_elementwise_check=args.strict_gradient_elements,
    )


def main():
    parser = make_parser()
    args = parser.parse_args()
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    if args.describe_only:
        element_bytes = dict(float64=8, float32=4, bfloat16=2, float16=2)
        output_dtype = args.autocast or args.dtype
        print(
            json.dumps(
                dict(
                    head_input_shape=[
                        args.batch,
                        args.chunk * args.block_size,
                        args.hidden_size,
                    ],
                    flattened_shape=[
                        args.batch * args.chunk * args.block_size,
                        args.hidden_size,
                    ],
                    frozen_weight_shape=[args.vocab_size, args.hidden_size],
                    full_hidden_bytes=args.batch
                    * args.anchors
                    * args.block_size
                    * args.hidden_size
                    * element_bytes[args.dtype],
                    weight_bytes=args.vocab_size
                    * args.hidden_size
                    * element_bytes[args.dtype],
                    output_bytes=args.batch
                    * args.chunk
                    * args.block_size
                    * args.vocab_size
                    * element_bytes[output_dtype],
                    note="Allocation sizes only; peak and timing require actual execution.",
                    torch_imported=False,
                )
            )
        )
        return
    # --help, --describe-only and argument validation do not import Torch.
    import torch

    report = run(args, torch)
    text = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + chr(10))
    print(text)


if __name__ == "__main__":
    main()
