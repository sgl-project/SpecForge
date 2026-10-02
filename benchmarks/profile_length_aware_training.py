"""Profile the length-aware benchmark without changing the training path.

Accepts the arguments of ``benchmarks.length_aware_training``. ``--steps`` is the
number of profiled optimizer windows, and ``--output`` names the JSON summary.
Chrome traces and readable operator tables are written beside that summary.
These timings include profiler overhead and are not speedup measurements.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from collections import defaultdict
from pathlib import Path

from benchmarks import length_aware_training as fixture


def _event_row(event):
    return {
        "name": event.key,
        "count": event.count,
        "self_cpu_ms": event.self_cpu_time_total / 1000,
        "cpu_total_ms": event.cpu_time_total / 1000,
        "self_device_ms": event.self_device_time_total / 1000,
        "device_total_ms": event.device_time_total / 1000,
    }


def _merge_intervals(intervals):
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return sum(end - start for start, end in merged)


def _timeline_summary(trace_path):
    events = json.loads(trace_path.read_text())["traceEvents"]
    windows = [
        event
        for event in events
        if event.get("name") == "profile::optimizer_window" and event.get("ph") == "X"
    ]
    kernels = [event for event in events if event.get("cat") == "kernel"]
    memcpy = [event for event in events if event.get("cat") == "gpu_memcpy"]
    start = min(event["ts"] for event in windows)
    end = max(event["ts"] + event["dur"] for event in windows)

    def clipped_intervals(items):
        return [
            (max(start, event["ts"]), min(end, event["ts"] + event["dur"]))
            for event in items
            if event["ts"] < end and event["ts"] + event["dur"] > start
        ]

    kernel_intervals = clipped_intervals(kernels)
    copy_intervals = clipped_intervals(memcpy)
    nccl = [event for event in kernels if "nccl" in event["name"].lower()]
    attention = [event for event in kernels if "sdpa" in event["name"].lower()]
    kernel_groups = defaultdict(lambda: {"count": 0, "duration_sum_ms": 0.0})
    for event in kernels:
        kernel_groups[event["name"]]["count"] += 1
        kernel_groups[event["name"]]["duration_sum_ms"] += event["dur"] / 1000
    launches = [
        event
        for event in events
        if event.get("cat") in {"cuda_runtime", "cuda_driver"}
        and "LaunchKernel" in event.get("name", "")
        and event.get("ph") == "X"
    ]
    return {
        "profiled_window_span_ms": (end - start) / 1000,
        "kernel_union_ms_within_window_span": _merge_intervals(kernel_intervals) / 1000,
        "kernel_or_memcpy_union_ms_within_window_span": _merge_intervals(
            kernel_intervals + copy_intervals
        )
        / 1000,
        "kernel_duration_sum_ms": sum(event["dur"] for event in kernels) / 1000,
        "kernel_count": len(kernels),
        "nccl_kernel_duration_sum_ms": sum(event["dur"] for event in nccl) / 1000,
        "nccl_kernel_count": len(nccl),
        "non_nccl_kernel_duration_sum_ms": sum(
            event["dur"] for event in kernels if "nccl" not in event["name"].lower()
        )
        / 1000,
        "sdpa_kernel_duration_sum_ms": sum(event["dur"] for event in attention) / 1000,
        "sdpa_kernel_count": len(attention),
        "cpu_kernel_launch_calls": len(launches),
        "cpu_kernel_launch_duration_sum_ms": sum(event["dur"] for event in launches)
        / 1000,
        "top_cuda_kernels": [
            {"name": name, **values}
            for name, values in sorted(
                kernel_groups.items(),
                key=lambda item: item[1]["duration_sum_ms"],
                reverse=True,
            )[:15]
        ],
        "notes": [
            "Profiler-instrumented timeline, not a production utilization or speedup measurement.",
            "Kernel sums may overlap across streams. Union counts time with at least one recorded kernel active; it does not measure SM occupancy.",
            "NCCL kernel duration includes protocol/progress and possible peer waiting; it is not a pure network transfer measurement.",
        ],
    }


def main():
    import torch
    import torch.distributed as dist
    from torch.profiler import ProfilerActivity, profile, record_function

    from specforge.distributed import init_distributed
    from specforge.training.strategies.base import StepContext

    args = fixture.parse_args()
    if args.dry_run or args.permutation_control:
        raise ValueError("Profiling requires the normal GPU benchmark path")
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    init_distributed(timeout=180, tp_size=1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    rank, world = dist.get_rank(), dist.get_world_size()
    refs = fixture.make_refs(args, world)
    orders = fixture.make_orders(args, refs, world)
    dtype = getattr(torch, args.perf_dtype)
    features, anchors = fixture.build_features(args, refs, dtype)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "world_size": world,
        },
        "source_sha256": {
            str(Path(__file__).name): hashlib.sha256(
                Path(__file__).read_bytes()
            ).hexdigest(),
            str(Path(fixture.__file__).name): hashlib.sha256(
                Path(fixture.__file__).read_bytes()
            ).hexdigest(),
        },
        "variants": {},
        "scope": "Profiler diagnostic only; native anchors and production TrainerCore/FSDP/AdamW. No speedup or accuracy conclusion can be drawn from profiled wall times.",
    }
    for variant, order in orders.items():
        batches = fixture.collate_batches(
            args,
            fixture.rank_batches(order, rank, world, args.batch_size),
            features,
        )
        model, core, observed = fixture.make_runner(args, dtype, anchors, False)
        for _ in range(args.warmup):
            fixture.one_window(args, model, core, batches)
        torch.cuda.synchronize()
        dist.barrier()
        observed["supervised_positions"].clear()
        elapsed = []
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            for _ in range(args.steps):
                with record_function("profile::optimizer_window"):
                    start = time.perf_counter()
                    for batch in batches:
                        with record_function("profile::microstep"):
                            model._benchmark_sample_ids = batch.sample_ids
                            core.train_step(
                                batch,
                                StepContext(
                                    global_step=0,
                                    total_steps=1000,
                                    collect_detailed_metrics=False,
                                ),
                            )
                    torch.cuda.synchronize()
                    elapsed.append((time.perf_counter() - start) * 1000)
                prof.step()
        stem = args.output.with_suffix("")
        trace_path = Path(f"{stem}.{variant}.rank{rank}.trace.json")
        prof.export_chrome_trace(str(trace_path))
        averages = list(prof.key_averages())
        Path(f"{stem}.{variant}.rank{rank}.operators.txt").write_text(
            prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=40)
            + "\n"
            + prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=40)
        )
        local = {
            "rank": rank,
            "profiled_optimizer_window_wall_ms": elapsed,
            "top_self_cpu": [
                _event_row(event)
                for event in sorted(
                    averages, key=lambda item: item.self_cpu_time_total, reverse=True
                )[:40]
            ],
            "top_self_device": [
                _event_row(event)
                for event in sorted(
                    averages, key=lambda item: item.self_device_time_total, reverse=True
                )[:40]
            ],
            "timeline": _timeline_summary(trace_path),
            "trace_file": trace_path.name,
            "supervised_draft_positions": (
                int(torch.stack(observed["supervised_positions"]).sum().item())
                if observed["supervised_positions"]
                else None
            ),
        }
        output["variants"][variant] = {
            "layout": fixture.layout_report(args, order, world),
            "by_rank": fixture.gather(local),
        }
        del model, core, observed, batches, prof
        gc.collect()
        torch.cuda.empty_cache()
        dist.barrier()
    if rank == 0:
        args.output.write_text(json.dumps(output, indent=2) + "\n")
        print(f"Wrote profile summary: {args.output}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
