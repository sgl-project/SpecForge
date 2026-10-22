"""Bounded BF16 trajectory diagnostic for DFlash2 length scheduling.

Run from the repository root::

    CUDA_VISIBLE_DEVICES=0,1 torchrun --standalone --nproc-per-node=2 \
        -m benchmarks.length_aware_trajectory --output trajectory.json

This is a synthetic numerical diagnostic, not a convergence or serving test.
Every variant sees identical sample membership in each optimizer window and
identical per-sample anchors. Offline redistribution ACROSS windows is excluded.
The declared 1% heldout-loss diagnostic threshold does not replace the existing
one-update BF16 tolerance (atol=0.0002, rtol=0.03), which is reported separately.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import random
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

from benchmarks.length_aware_training import (
    build_features,
    collate_batches,
    gather,
    layout_report,
    make_orders,
    make_refs,
    make_runner,
    rank_batches,
)

# Declared before any run. These are diagnostic gates, not quality guarantees.
HELDOUT_LOSS_RELATIVE_LIMIT = 0.01
ONE_UPDATE_ATOL = 0.0002
ONE_UPDATE_RTOL = 0.03
VARIANTS = ("baseline", "baseline_repeat", "permutation", "online", "offline")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seeds", default="42,43,44")
    parser.add_argument("--train-windows", type=int, default=4)
    parser.add_argument("--accumulation-steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--lengths", default="64,256,512,1024")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--draft-layers", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    args = parser.parse_args()
    args.seeds = [int(value) for value in args.seeds.split(",")]
    args.lengths = [int(value) for value in args.lengths.split(",")]
    for name in (
        "steps",
        "train_windows",
        "accumulation_steps",
        "batch_size",
        "hidden_size",
        "vocab_size",
        "draft_layers",
        "num_anchors",
        "block_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if not args.seeds or not args.lengths or args.hidden_size % 32:
        parser.error("seeds/lengths must be nonempty and hidden-size divisible by 32")
    return args


def experiment_args(args, seed):
    return SimpleNamespace(
        **{key: value for key, value in vars(args).items() if key != "output"},
        seed=seed,
        algorithm="dflash2",
        pattern="heterogeneous",
        schedule="online",
        permutation_control=False,
        attention_backend="sdpa",
        sharding="SHARD_GRAD_OP",
    )


def fixture(args, world):
    windows = []
    count = world * args.batch_size * args.accumulation_steps
    for window in range(args.train_windows + 1):
        window_args = SimpleNamespace(**vars(args))
        window_args.seed = args.seed + 100 * window
        refs = make_refs(window_args, world)
        windows.append(
            [
                replace(ref, sample_id=str(window * count + index))
                for index, ref in enumerate(refs)
            ]
        )
    return windows[:-1], windows[-1]


def orders_for_step(args, refs, world, step):
    baseline = list(refs)
    random.Random(args.seed + step * 17).shuffle(baseline)
    result = {"baseline": baseline, "baseline_repeat": baseline}
    for name in ("permutation", "online", "offline"):
        order_args = SimpleNamespace(**vars(args))
        order_args.seed = args.seed + step * 17
        order_args.schedule = "offline" if name == "offline" else "online"
        order_args.permutation_control = name == "permutation"
        result[name] = make_orders(order_args, baseline, world)["length_aware"]
        if (
            name == "permutation"
            and result[name] == baseline
            and args.accumulation_steps > 1
        ):
            # A shuffled control can accidentally be the identity. Rotate whole
            # rounds in that case; preserve rank assignment and batch contents.
            width = world * args.batch_size
            result[name] = baseline[width:] + baseline[:width]
        assert Counter(ref.sample_id for ref in result[name]) == Counter(
            ref.sample_id for ref in baseline
        )
    return result


def global_tensor_comparison(reference, actual):
    import torch
    import torch.distributed as dist

    delta = actual - reference
    packed = torch.tensor(
        [
            float(delta.double().square().sum()),
            float(reference.double().square().sum()),
        ],
        dtype=torch.float64,
        device="cuda",
    )
    dist.all_reduce(packed)
    maximum = torch.tensor(
        float(delta.abs().max()) if delta.numel() else 0.0, device="cuda"
    )
    dist.all_reduce(maximum, op=dist.ReduceOp.MAX)
    close = torch.tensor(
        int(
            torch.allclose(
                reference, actual, atol=ONE_UPDATE_ATOL, rtol=ONE_UPDATE_RTOL
            )
        ),
        device="cuda",
    )
    dist.all_reduce(close, op=dist.ReduceOp.MIN)
    relative = math.sqrt(float(packed[0]) / max(float(packed[1]), 1e-60))
    return {
        "relative_l2_error": relative,
        "max_abs_diff": float(maximum),
        "allclose_at_original_bf16_tolerance": bool(close),
        "original_bf16_tolerance_passed": bool(close) and relative <= ONE_UPDATE_RTOL,
    }


def evaluate(model, core, batches, step, total_steps):
    import torch
    import torch.distributed as dist

    from specforge.training.strategies.base import StepContext

    model.eval()
    pairs = []
    with torch.no_grad():
        for batch in batches:
            model._benchmark_sample_ids = batch.sample_ids
            out = core.strategy.forward_loss(
                batch,
                StepContext(
                    global_step=step,
                    total_steps=total_steps,
                    collect_detailed_metrics=False,
                ),
            )
            if out.loss_terms is None:
                raise RuntimeError("DFlash objective must expose additive loss_terms")
            pairs.append(
                torch.stack([value.detach().float() for value in out.loss_terms])
            )
    objective = torch.stack(pairs).sum(dim=0)
    dist.all_reduce(objective)
    model.train()
    return float(objective[0] / objective[1])


def run_variant(args, variant, windows, heldout, features, anchors):
    import torch
    import torch.distributed as dist

    from specforge.training.strategies.base import StepContext

    world, rank = dist.get_world_size(), dist.get_rank()
    heldout_batches = collate_batches(
        args, rank_batches(heldout, rank, world, args.batch_size), features
    )
    model, core, observed = make_runner(args, torch.bfloat16, anchors, True)
    trace = [
        {
            "step": 0,
            "heldout_loss": evaluate(model, core, heldout_batches, 0, args.steps),
        }
    ]
    first = {}
    finite = True
    started = time.monotonic()
    for step in range(args.steps):
        refs = windows[step % len(windows)]
        order = orders_for_step(args, refs, world, step)[variant]
        batches = collate_batches(
            args, rank_batches(order, rank, world, args.batch_size), features
        )
        observed["objective"].clear()
        observed["supervised_positions"].clear()
        updates = 0
        for batch in batches:
            model._benchmark_sample_ids = batch.sample_ids
            result = core.train_step(
                batch,
                StepContext(
                    global_step=step,
                    total_steps=args.steps,
                    collect_detailed_metrics=False,
                ),
            )
            updates += int(result.optimizer_stepped)
        assert updates == 1 and core.accumulation_remainder == 0
        pair = torch.stack(
            [
                torch.stack([value.float() for value in values])
                for values in observed["objective"]
            ]
        ).sum(dim=0)
        dist.all_reduce(pair)
        count = torch.stack(observed["supervised_positions"]).sum()
        dist.all_reduce(count)
        loss = float(pair[0] / pair[1])
        grad_norm = float(core.backend.optimizer.last_grad_norm)
        heldout_loss = evaluate(model, core, heldout_batches, step + 1, args.steps)
        state_finite = all(
            bool(torch.isfinite(value).all())
            for value in (
                core.backend.optimizer.fp32_params + core.backend.optimizer.model_params
            )
        ) and bool(torch.isfinite(observed["gradients"]).all())
        finite = (
            finite
            and state_finite
            and all(math.isfinite(value) for value in (loss, grad_norm, heldout_loss))
        )
        trace.append(
            {
                "step": step + 1,
                "train_loss": loss,
                "heldout_loss": heldout_loss,
                "preclip_gradient_norm": grad_norm,
                "supervised_positions": int(count),
                "finite": state_finite,
            }
        )
        if step == 0:
            first = {
                name: observed[name].detach().cpu()
                for name in ("gradients", "updates", "initial_weights")
            }
        if rank == 0 and (step + 1) % 10 == 0:
            print(
                json.dumps(
                    {
                        "seed": args.seed,
                        "variant": variant,
                        "step": step + 1,
                        "heldout_loss": heldout_loss,
                    }
                ),
                flush=True,
            )
    final = {
        "bf16_parameters": torch.cat(
            [
                value.detach().float().flatten()
                for value in core.backend.optimizer.model_params
            ]
        ).cpu(),
        "fp32_master_parameters": torch.cat(
            [value.detach().flatten() for value in core.backend.optimizer.fp32_params]
        ).cpu(),
    }
    report = {
        "trace": trace,
        "finite_on_all_ranks": all(gather(finite)),
        "elapsed_seconds": time.monotonic() - started,
    }
    del model, core, observed, heldout_batches, batches
    gc.collect()
    torch.cuda.empty_cache()
    return report, first, final


def compare(
    reference, actual, first_reference, first_actual, final_reference, final_actual
):
    a, b = reference["trace"], actual["trace"]
    relative = [
        abs(y["heldout_loss"] - x["heldout_loss"]) / max(abs(x["heldout_loss"]), 1e-30)
        for x, y in zip(a, b)
    ]
    count_equal = all(
        x["supervised_positions"] == y["supervised_positions"]
        for x, y in zip(a[1:], b[1:])
    )
    first_loss_error = abs(a[1]["train_loss"] - b[1]["train_loss"])
    first_update = {
        key: global_tensor_comparison(first_reference[key], first_actual[key])
        for key in ("gradients", "updates")
    }
    first_update["loss"] = {
        "reference": a[1]["train_loss"],
        "actual": b[1]["train_loss"],
        "absolute_error": first_loss_error,
        "original_bf16_tolerance_passed": first_loss_error
        <= ONE_UPDATE_ATOL + ONE_UPDATE_RTOL * abs(a[1]["train_loss"]),
    }
    first_update["original_bf16_gate_passed"] = all(
        value["original_bf16_tolerance_passed"] for value in first_update.values()
    )
    return {
        "heldout_loss_max_relative_deviation": max(relative),
        "heldout_loss_final_relative_deviation": relative[-1],
        "heldout_loss_mean_relative_deviation": sum(relative) / len(relative),
        "heldout_loss_relative_deviation_by_step": relative,
        "supervised_positions_equal_every_update": count_equal,
        "gradient_norm_max_relative_deviation": max(
            abs(y["preclip_gradient_norm"] - x["preclip_gradient_norm"])
            / max(abs(x["preclip_gradient_norm"]), 1e-30)
            for x, y in zip(a[1:], b[1:])
        ),
        "first_update": first_update,
        "final_parameters": {
            key: global_tensor_comparison(final_reference[key], final_actual[key])
            for key in final_reference
        },
        "diagnostic_gate_passed": (
            actual["finite_on_all_ranks"]
            and reference["finite_on_all_ranks"]
            and count_equal
            and max(relative) <= HELDOUT_LOSS_RELATIVE_LIMIT
        ),
    }


def main():
    import os

    import torch
    import torch.distributed as dist

    from specforge.distributed import init_distributed

    args = parse_args()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    init_distributed(timeout=120, tp_size=1)
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "environment": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "world_size": world,
            "dtype": "bfloat16",
            "tf32": False,
        },
        "predeclared_gates": {
            "finite_training_loss_gradients_master_parameters_and_heldout_loss": True,
            "equal_supervised_positions_per_update": True,
            "max_relative_heldout_loss_deviation": HELDOUT_LOSS_RELATIVE_LIMIT,
            "original_one_update_bf16_atol": ONE_UPDATE_ATOL,
            "original_one_update_bf16_rtol": ONE_UPDATE_RTOL,
            "diagnostic_gate_does_not_replace_original_one_update_gate": True,
        },
        "limitations": [
            "Synthetic random model, labels and teacher features; no real-model convergence or serving accuracy evidence.",
            "Same source sample membership per optimizer window; offline redistribution across optimizer windows is NOT exercised.",
            "Fixed per-sample anchors remove objective sampling differences; native random anchor trajectory is NOT exercised.",
            "The heldout set is disjoint synthetic data; a small loss difference does not establish useful language-model quality.",
            "Elapsed time includes evaluation/collation/instrumentation and is NOT a performance benchmark.",
        ],
        "seeds": [],
    }
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            "benchmarks/length_aware_trajectory.py",
            "benchmarks/length_aware_training.py",
            "specforge/optimizer.py",
            "specforge/training/controller.py",
            "specforge/algorithms/common/dflash_family_model.py",
        )
    }
    for seed in args.seeds:
        config = experiment_args(args, seed)
        windows, heldout = fixture(config, world)
        all_refs = [ref for window in windows + [heldout] for ref in window]
        features, anchors = build_features(config, all_refs, torch.bfloat16)
        seed_result = {"seed": seed, "variants": {}, "comparisons_to_baseline": {}}
        seed_result["first_window_layouts"] = {
            name: layout_report(config, order, world)
            for name, order in orders_for_step(config, windows[0], world, 0).items()
        }
        baseline_first, baseline_final = None, None
        for variant in VARIANTS:
            result, first, final = run_variant(
                config, variant, windows, heldout, features, anchors
            )
            seed_result["variants"][variant] = result
            if variant == "baseline":
                baseline_first, baseline_final = first, final
            else:
                seed_result["comparisons_to_baseline"][variant] = compare(
                    seed_result["variants"]["baseline"],
                    result,
                    baseline_first,
                    first,
                    baseline_final,
                    final,
                )
        report["seeds"].append(seed_result)
        if rank == 0:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
    report["all_diagnostic_gates_passed"] = all(
        comparison["diagnostic_gate_passed"]
        for seed_result in report["seeds"]
        for comparison in seed_result["comparisons_to_baseline"].values()
    )
    if rank == 0:
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "all_diagnostic_gates_passed": report[
                        "all_diagnostic_gates_passed"
                    ],
                }
            ),
            flush=True,
        )
    dist.barrier()
    dist.destroy_process_group()
    if not report["all_diagnostic_gates_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
