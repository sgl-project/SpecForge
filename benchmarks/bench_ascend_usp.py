"""Measure an offline Ascend training recipe under torchrun.

Use batch size 1, one SP group, and equal accumulation windows across
topologies. Launch once per configuration and use a fresh output directory.
The JSON contains synchronized trainer-core timings and per-rank allocator
peaks. Input tokens exclude right padding and USP teacher lookahead.
"""

import argparse
import json
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist

from specforge.cli import main as train_cli
from specforge.config import load_config
from specforge.training.controller import TrainerCore


def summarize(directory):
    reports = [
        json.loads(path.read_text()) for path in sorted(directory.glob("rank-*.json"))
    ]
    if not reports:
        raise ValueError("no rank measurements found")
    if len(reports) != reports[0]["world_size"]:
        raise ValueError("incomplete rank measurements")
    durations = []
    tokens = 0
    for step in zip(*(r["records"] for r in reports), strict=True):
        if any(r["warmup"] != step[0]["warmup"] for r in step):
            raise ValueError("rank warmup boundaries differ")
        if not step[0]["warmup"]:
            durations.append(max(r["seconds"] for r in step))
            tokens += sum(r["tokens"] for r in step)
    records = [item for report in reports for item in report["records"]]
    return {
        "measured_tokens": tokens,
        "compute_seconds": sum(durations),
        "input_tokens_per_second": tokens / sum(durations),
        "peak_allocated_gib_per_device": max(r["allocated_bytes"] for r in records)
        / 2**30,
        "peak_reserved_gib_per_device": max(r["reserved_bytes"] for r in records)
        / 2**30,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--summarize", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup-updates", type=int, default=1)
    args = parser.parse_args()
    if args.summarize:
        print(json.dumps(summarize(args.output), indent=2))
        return 0
    cfg = load_config(str(args.config))
    world = int(os.environ["WORLD_SIZE"])
    rank = int(os.environ["RANK"])
    training = cfg.training
    if (
        training.batch_size != 1
        or training.tp_size != 1
        or world != training.sp_ulysses_size * training.sp_ring_size
        or args.warmup_updates < 1
    ):
        parser.error("use batch size 1, TP=1, DP=1 and at least one warmup update")
    lookahead = (
        training.objective_options.get("ttt_length", 7)
        if training.attention_backend == "usp"
        else 0
    )
    original = TrainerCore.train_step
    records = []
    completed = 0

    def measured_step(trainer, batch, ctx=None):
        nonlocal completed
        mask = batch.tensors["attention_mask"]
        owned = mask[:, : mask.shape[1] - lookahead]
        count = int(owned.sum().item())
        torch.npu.synchronize()
        dist.barrier()
        torch.npu.synchronize()
        torch.npu.reset_peak_memory_stats()
        start = time.perf_counter()
        result = original(trainer, batch, ctx)
        torch.npu.synchronize()
        elapsed = time.perf_counter() - start
        records.append(
            dict(
                warmup=completed < args.warmup_updates,
                seconds=elapsed,
                tokens=count,
                allocated_bytes=torch.npu.max_memory_allocated(),
                reserved_bytes=torch.npu.max_memory_reserved(),
                optimizer_stepped=result.optimizer_stepped,
            )
        )
        completed += int(result.optimizer_stepped)
        return result

    args.output.mkdir(parents=True, exist_ok=True)
    # Each process measures its own allocator. Rank files need no extra timed
    # collectives and can be combined after the normal CLI tears down HCCL.
    TrainerCore.train_step = measured_step
    try:
        status = train_cli(["train", "-c", str(args.config)])
    finally:
        TrainerCore.train_step = original
    if status:
        return status
    if completed <= args.warmup_updates:
        raise ValueError("recipe ended before the first measured optimizer update")
    with (args.output / f"rank-{rank}.json").open("x") as file:
        json.dump({"world_size": world, "records": records}, file)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
