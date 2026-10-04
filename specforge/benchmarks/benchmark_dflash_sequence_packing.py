"""Measure DFlash and DFlash2 packing separately, with identical sampled anchors.

Example (CUDA, no model downloads)::

    PYTHONPATH=. python scripts/benchmark_dflash_sequence_packing.py \
        --preset tiny --dtype float32 --correctness-only --output tiny.json
    PYTHONPATH=. python scripts/benchmark_dflash_sequence_packing.py \
        --preset medium --steps 20 --output perf.json

Production model, collators, strategy, and BF16Optimizer are used. Frozen target
embeddings/head and captured features are synthetic. The timed region includes
the strategy's CPU integer-feature processing/transfers, forward, backward, and
optimizer update. Hidden features are GPU resident. Capture, hidden-feature I/O
and H2D, distributed communication, and serving are outside this benchmark.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.algorithms.common.hidden_states_data import (
    build_collator,
    build_packed_collator,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.optimizer import BF16Optimizer
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import DFlashTrainStrategy, StepContext

PRESETS = {
    "tiny": dict(
        hidden_size=64,
        intermediate_size=128,
        layers=2,
        heads=4,
        kv_heads=2,
        vocab_size=128,
        block_size=4,
        anchors=8,
        capture_layers=2,
        conv_group_size=4,
        conv_kernel_size=2,
        selector_rank=4,
        selector_top_k=8,
        lengths=[[9, 17, 31, 64], [32, 32, 32, 32]],
    ),
    "medium": dict(
        hidden_size=2048,
        intermediate_size=8192,
        layers=2,
        heads=16,
        kv_heads=4,
        vocab_size=32000,
        block_size=16,
        anchors=128,
        capture_layers=2,
        conv_group_size=32,
        conv_kernel_size=4,
        selector_rank=16,
        selector_top_k=16,
        lengths=[[1024, 1024, 1024, 1024], [128, 256, 512, 2048]],
    ),
}
CONTEXT = StepContext(global_step=10, total_steps=100, collect_detailed_metrics=False)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="medium")
    parser.add_argument(
        "--algorithm", choices=["dflash", "dflash2", "both"], default="both"
    )
    for key in PRESETS["tiny"]:
        if key != "lengths":
            parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--lengths", action="append")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--sliding-window", type=int)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--objective-chunk-blocks", type=int, default=128)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for key, value in PRESETS[args.preset].items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if isinstance(args.lengths[0], str):
        args.lengths = [
            [int(value) for value in row.split(",")] for row in args.lengths
        ]
    args.atol = (
        args.atol
        if args.atol is not None
        else (2e-5 if args.dtype == "float32" else 2e-3)
    )
    args.rtol = (
        args.rtol
        if args.rtol is not None
        else (2e-4 if args.dtype == "float32" else 2e-2)
    )
    if any(not row or min(row) < 4 for row in args.lengths):
        parser.error("each batch must contain sequence lengths >= 4")
    if args.steps < 1 or args.warmup < 1 or args.anchors < 1:
        parser.error("steps, warmup, and anchors must be positive")
    if args.hidden_size % args.heads or args.heads % args.kv_heads:
        parser.error("hidden_size/heads and heads/kv_heads must be integral")
    if args.correctness_only and args.skip_correctness:
        parser.error("correctness-only and skip-correctness cannot be combined")
    return args


def build_strategy(args, algorithm):
    torch.manual_seed(args.seed)
    config = Qwen3Config(
        architectures=[
            "DFlash2DraftModel" if algorithm == "dflash2" else "DFlashDraftModel"
        ],
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=args.layers,
        num_target_layers=args.capture_layers + 4,
        num_attention_heads=args.heads,
        num_key_value_heads=args.kv_heads,
        head_dim=args.hidden_size // args.heads,
        vocab_size=args.vocab_size,
        attention_dropout=0.0,
        max_position_embeddings=max(map(max, args.lengths)) + args.block_size,
        layer_types=["sliding_attention" if args.sliding_window else "full_attention"]
        * args.layers,
        use_sliding_window=bool(args.sliding_window),
        sliding_window=args.sliding_window,
        dflash_config={
            "block_size": args.block_size,
            "mask_token_id": args.vocab_size - 1,
            "target_layer_ids": list(range(1, args.capture_layers + 1)),
            "conv_group_size": args.conv_group_size,
            "conv_kernel_size": args.conv_kernel_size,
            "selector_rank": args.selector_rank,
            "selector_top_k": args.selector_top_k,
        },
    )
    config._attn_implementation = "flex_attention"
    draft_type = DFlash2DraftModel if algorithm == "dflash2" else DFlashDraftModel
    draft = draft_type(config)
    model = OnlineDFlashModel(
        draft_model=draft,
        target_lm_head=nn.Linear(
            args.hidden_size, args.vocab_size, bias=False
        ).requires_grad_(False),
        target_embed_tokens=nn.Embedding(
            args.vocab_size, args.hidden_size
        ).requires_grad_(False),
        mask_token_id=args.vocab_size - 1,
        block_size=args.block_size,
        attention_backend="flex_attention",
        num_anchors=args.anchors,
        objective_chunk_blocks=args.objective_chunk_blocks,
        loss_decay_gamma=7.0,
        selector_loss_alpha=1.0 if algorithm == "dflash2" else 0.0,
        teacher_metrics=False,
    ).to("cuda", getattr(torch, args.dtype))
    return DFlashTrainStrategy(model.train())


def make_features(args, lengths):
    generator = torch.Generator().manual_seed(args.seed + 1)
    features = []
    for length in lengths:
        loss_mask = torch.ones(1, length, dtype=torch.long)
        loss_mask[:, : length // 4] = 0
        loss_mask[:, -1] = 0
        features.append(
            {
                "input_ids": torch.randint(
                    0, args.vocab_size - 1, (1, length), generator=generator
                ),
                "loss_mask": loss_mask,
                "hidden_states": torch.randn(
                    1,
                    length,
                    args.capture_layers * args.hidden_size,
                    generator=generator,
                ).to(getattr(torch, args.dtype)),
            }
        )
    return features


def make_batch(features, mode, algorithm):
    collator = build_collator() if mode == "padded" else build_packed_collator()
    tensors = collator(features)
    tensors["hidden_states"] = tensors["hidden_states"].cuda()
    return TrainBatch(
        sample_ids=[str(i) for i in range(len(features))],
        strategy=algorithm,
        tensors=tensors,
        metadata={},
    )


def compare(reference, actual, args):
    reference, actual = reference.float(), actual.float()
    delta = actual - reference
    return {
        "pass": bool(torch.allclose(reference, actual, atol=args.atol, rtol=args.rtol)),
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2_diff": float(delta.norm()) / max(float(reference.norm()), 1e-30),
    }


def correctness(strategy, features, args, algorithm):
    model = strategy.trainable_module()
    results = {}
    for mode in ["padded", "packed"]:
        model.zero_grad(set_to_none=True)
        batch = make_batch(features, mode, algorithm)
        recorded_anchors = []
        sampler = model._sample_anchor_positions

        def record(*pos, **kwargs):
            anchors, keep = sampler(*pos, **kwargs)
            recorded_anchors.append((anchors.cpu(), keep.cpu()))
            return anchors, keep

        torch.cuda.manual_seed(args.seed + 100)
        with mock.patch.object(model, "_sample_anchor_positions", side_effect=record):
            output = strategy.forward_loss(batch, CONTEXT)
        output.loss.backward()
        results[mode] = {
            "loss": output.loss.detach().float().cpu(),
            "loss_terms": torch.stack(output.loss_terms).detach().float().cpu(),
            "anchors": recorded_anchors,
            "grads": {
                name: None if p.grad is None else p.grad.detach().float().cpu()
                for name, p in model.named_parameters()
                if p.requires_grad
            },
        }
        del output, batch
    baseline, packed = results["padded"], results["packed"]
    checks = {
        key: compare(baseline[key], packed[key], args) for key in ["loss", "loss_terms"]
    }
    checks["loss_values"] = [float(baseline["loss"]), float(packed["loss"])]
    checks["identical_sampled_anchors"] = len(baseline["anchors"]) == len(
        packed["anchors"]
    ) and all(
        torch.equal(a, b) and torch.equal(ka, kb)
        for (a, ka), (b, kb) in zip(baseline["anchors"], packed["anchors"])
    )
    checks["sampled_anchor_count"] = sum(
        int(keep.sum()) for _, keep in baseline["anchors"]
    )
    checks["gradients"] = {}
    for name, ref in baseline["grads"].items():
        value = packed["grads"][name]
        checks["gradients"][name] = (
            {"pass": ref is None and value is None, "missing_gradient": True}
            if ref is None or value is None
            else compare(ref, value, args)
        )
    checks["pass"] = (
        checks["identical_sampled_anchors"]
        and all(checks[key]["pass"] for key in ["loss", "loss_terms"])
        and all(row["pass"] for row in checks["gradients"].values())
    )
    model.zero_grad(set_to_none=True)
    return checks


def measure(strategy, initial_state, features, lengths, args, algorithm, mode):
    model = strategy.trainable_module()
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    batch = make_batch(features, mode, algorithm)
    optimizer = BF16Optimizer(
        model,
        lr=args.learning_rate,
        warmup_ratio=0.0,
        total_steps=args.warmup + args.steps + 1,
        lr_scheduler="constant",
    )
    torch.cuda.manual_seed(args.seed + 100)

    def step():
        output = strategy.forward_loss(batch, CONTEXT)
        output.loss.backward()
        optimizer.step()
        return output.loss.detach()

    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - start
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        loss = step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - start)
    mean, median = statistics.mean(times), statistics.median(times)
    result = {
        "mean_step_ms": mean * 1000,
        "p50_step_ms": median * 1000,
        "useful_tokens_per_second": sum(lengths) / mean,
        "p50_useful_tokens_per_second": sum(lengths) / median,
        "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "baseline_allocated_gib": baseline_bytes / 2**30,
        "warmup_seconds_including_compile": warmup_seconds,
        "step_ms": [duration * 1000 for duration in times],
        "final_loss": float(loss),
    }
    optimizer = batch = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def provenance():
    root = Path(__file__).resolve().parents[2]
    files = [
        "scripts/benchmark_dflash_sequence_packing.py",
        "specforge/benchmarks/benchmark_dflash_sequence_packing.py",
        "specforge/algorithms/common/dflash_family_model.py",
        "specforge/algorithms/common/hidden_states_data.py",
        "specforge/modeling/draft/dflash.py",
        "specforge/modeling/draft/dflash2.py",
        "specforge/modeling/packed_dflash.py",
        "specforge/training/strategies/base.py",
    ]
    result = {
        "files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in files
        }
    }
    try:
        result["head"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        result["head"] = "unavailable (copied snapshot is identified by file hashes)"
    return result


def save(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    settings = {**vars(args), "output": str(args.output)}
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": provenance(),
        "settings": settings,
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        },
        "scope": "Single GPU; synthetic frozen target components and captured features; GPU-resident hidden features plus CPU integer features; production strategy forward/backward/BF16Optimizer; excludes capture, hidden-feature I/O/H2D, distributed training and serving.",
        "cases": [],
    }
    algorithms = ["dflash", "dflash2"] if args.algorithm == "both" else [args.algorithm]
    for algorithm in algorithms:
        strategy = build_strategy(args, algorithm)
        initial_state = {
            name: value.detach().cpu().clone()
            for name, value in strategy.trainable_module().state_dict().items()
        }
        for lengths in args.lengths:
            strategy.trainable_module().load_state_dict(initial_state)
            features = make_features(args, lengths)
            case = {
                "algorithm": algorithm,
                "lengths": lengths,
                "padding_fraction": 1 - sum(lengths) / (len(lengths) * max(lengths)),
                "useful_tokens": sum(lengths),
            }
            report["cases"].append(case)
            print(json.dumps({"case_start": algorithm, "lengths": lengths}), flush=True)
            if not args.skip_correctness:
                case["correctness"] = correctness(strategy, features, args, algorithm)
                save(report, args.output)
                print(
                    json.dumps(
                        {
                            "correctness": case["correctness"]["pass"],
                            "anchors": case["correctness"]["sampled_anchor_count"],
                        }
                    ),
                    flush=True,
                )
                if not case["correctness"]["pass"]:
                    raise AssertionError(f"Packed parity failed; inspect {args.output}")
            if not args.correctness_only:
                for mode in ["padded", "packed"]:
                    case[mode] = measure(
                        strategy,
                        initial_state,
                        features,
                        lengths,
                        args,
                        algorithm,
                        mode,
                    )
                    save(report, args.output)
                    print(
                        json.dumps(
                            {"algorithm": algorithm, "mode": mode, **case[mode]}
                        ),
                        flush=True,
                    )
                case["p50_speedup"] = (
                    case["padded"]["p50_step_ms"] / case["packed"]["p50_step_ms"]
                )
                case["mean_speedup"] = (
                    case["padded"]["mean_step_ms"] / case["packed"]["mean_step_ms"]
                )
            save(report, args.output)
            del features
        del strategy, initial_state
        gc.collect()
        torch.cuda.empty_cache()
    print(f"Report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
