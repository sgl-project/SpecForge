"""GPU A/B benchmark for length-aware training, using the production trainer.

Examples (from the repository root)::

    torchrun --standalone --nproc-per-node=2 -m benchmarks.length_aware_training \
        --algorithm dflash2 --schedule online --batch-size 2 --output online.json
    torchrun --standalone --nproc-per-node=2 -m benchmarks.length_aware_training \
        --algorithm dflash2 --schedule offline --batch-size 2 --output offline.json
    torchrun --standalone --nproc-per-node=2 -m benchmarks.length_aware_training \
        --algorithm eagle3 --schedule online --batch-size 1 --output eagle3.json

Repeat with ``--pattern uniform`` for the negative control. ``--dry-run`` only
checks metadata, sample conservation and padding; it needs no torch or GPU.

This is a synthetic trainer microbenchmark, NOT a capture-to-training benchmark,
convergence experiment, or serving acceptance test. It uses real SpecForge
models, strategies, TrainerCore, FSDP/DDP, and AdamW. Teacher features are random
and preloaded on CPU. DFlash anchors are fixed per sample for numerical parity so
changing sample order cannot silently change the supervised work. Performance
uses native anchor sampling and reports the actual supervised-position count. Offline
bucketing is confined to ONE optimizer window for the numerical parity check;
normal training may reorder samples across optimizer windows and will not have
an identical optimization trajectory.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import random
import statistics
import subprocess
import time
import types
from collections import Counter
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--algorithm", choices=("eagle3", "dflash2"), default="dflash2")
    parser.add_argument("--schedule", choices=("online", "offline"), default="online")
    parser.add_argument(
        "--pattern", choices=("heterogeneous", "uniform"), default="heterogeneous"
    )
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--accumulation-steps", type=int, default=8)
    parser.add_argument("--lengths", default="64,256,512,1024")
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--draft-layers", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=16)
    parser.add_argument("--block-size", type=int, default=8)
    parser.add_argument("--ttt-length", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument(
        "--repeats",
        type=int,
        default=3,
        help="Independent model resets; alternate A/B order between repeats.",
    )
    parser.add_argument(
        "--parity-dtype", choices=("float32", "bfloat16"), default="float32"
    )
    parser.add_argument(
        "--sharding",
        choices=("SHARD_GRAD_OP", "FULL_SHARD", "NO_SHARD"),
        default="SHARD_GRAD_OP",
    )
    parser.add_argument(
        "--attention-backend", choices=("sdpa", "flex_attention"), default="sdpa"
    )
    parser.add_argument(
        "--perf-dtype", choices=("float32", "bfloat16"), default="bfloat16"
    )
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=5e-4)
    parser.add_argument("--skip-parity", action="store_true")
    parser.add_argument("--skip-performance", action="store_true")
    parser.add_argument(
        "--permutation-control",
        action="store_true",
        help="Numerical diagnostic: reorder intact microbatches without length grouping.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--world-size",
        type=int,
        default=2,
        help="Metadata-only world size for --dry-run.",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.lengths = [int(value) for value in args.lengths.split(",")]
    if args.permutation_control and not (args.skip_performance or args.dry_run):
        parser.error(
            "--permutation-control is a numerical diagnostic; use --skip-performance"
        )
    for name in (
        "batch_size",
        "accumulation_steps",
        "hidden_size",
        "vocab_size",
        "draft_layers",
        "num_anchors",
        "block_size",
        "ttt_length",
        "steps",
        "repeats",
        "world_size",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not args.lengths or min(args.lengths) < args.block_size + args.num_anchors + 2:
        parser.error(
            "lengths must leave room for the requested fixed supervised anchors"
        )
    if args.hidden_size % 32:
        parser.error("hidden-size must be divisible by 32")
    if args.algorithm == "eagle3" and (
        args.schedule != "online" or args.batch_size != 1
    ):
        parser.error(
            "EAGLE3 is supported here only for online batch-size=1; rebatching changes its padded-row loss normalization"
        )
    return args


def make_refs(args, world):
    from specforge.runtime.contracts import SampleRef

    count = world * args.batch_size * args.accumulation_steps
    lengths = (
        [max(args.lengths)] * count
        if args.pattern == "uniform"
        else [args.lengths[index % len(args.lengths)] for index in range(count)]
    )
    random.Random(args.seed).shuffle(lengths)
    return [
        SampleRef(
            sample_id=str(index),
            run_id="length-benchmark",
            source_task_id=None,
            feature_store_uri="synthetic://cpu",
            feature_keys={},
            feature_specs={},
            strategy="eagle3" if args.algorithm == "eagle3" else "dflash",
            num_tokens=length,
        )
        for index, length in enumerate(lengths)
    ]


def make_orders(args, refs, world):
    if args.schedule == "online":
        from specforge.runtime.data_plane.ref_distributor import _length_grouped_window

        grouped = _length_grouped_window(
            refs, dp_size=world, refs_per_rank_batch=args.batch_size
        )
    else:
        from specforge.data.length_bucketing import bucket_by_length

        grouped = bucket_by_length(
            refs,
            length_fn=lambda ref: ref.num_tokens,
            batch_size=args.batch_size,
            dp_size=world,
            length_bucket_size=len(refs),
            seed=args.seed,
            epoch=0,
        )
    if args.permutation_control:
        width = world * args.batch_size
        rounds = [
            list(refs[start : start + width]) for start in range(0, len(refs), width)
        ]
        random.Random(args.seed + 1).shuffle(rounds)
        grouped = [ref for round_refs in rounds for ref in round_refs]
        for rank in range(world):
            before = rank_batches(refs, rank, world, args.batch_size)
            after = rank_batches(grouped, rank, world, args.batch_size)
            assert Counter(
                tuple(ref.sample_id for ref in batch) for batch in before
            ) == Counter(tuple(ref.sample_id for ref in batch) for batch in after)
    assert Counter(ref.sample_id for ref in refs) == Counter(
        ref.sample_id for ref in grouped
    )
    return {"baseline": list(refs), "length_aware": list(grouped)}


def rank_batches(order, rank, world, batch_size):
    local = order[rank::world]
    return [
        local[start : start + batch_size] for start in range(0, len(local), batch_size)
    ]


def layout_report(args, order, world):
    batches = [
        rank_batches(order, rank, world, args.batch_size) for rank in range(world)
    ]
    padded = [
        [max(ref.num_tokens for ref in batch) * len(batch) for batch in local]
        for local in batches
    ]
    valid = sum(ref.num_tokens for ref in order)
    return {
        "sample_ids_by_rank_and_microstep": [
            [[ref.sample_id for ref in batch] for batch in local] for local in batches
        ],
        "valid_context_tokens": valid,
        "padded_context_positions": sum(map(sum, padded)),
        "padding_fraction": 1 - valid / sum(map(sum, padded)),
        "padded_context_positions_by_rank": list(map(sum, padded)),
        "sum_of_microstep_max_padded_positions": sum(
            max(row[step] for row in padded) for step in range(args.accumulation_steps)
        ),
    }


def build_model(args, dtype, *, fixed_sample_anchors=False):
    import torch
    from transformers import LlamaConfig, Qwen3Config

    torch.manual_seed(args.seed)
    if args.algorithm == "eagle3":
        from specforge.algorithms.eagle3.model import OnlineEagle3Model
        from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3

        config = LlamaConfig(
            architectures=["LlamaForCausalLMEagle3"],
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            vocab_size=args.vocab_size,
            draft_vocab_size=args.vocab_size,
            max_position_embeddings=max(args.lengths) + 32,
            pad_token_id=0,
            tie_word_embeddings=False,
            attention_dropout=0.0,
        )
        draft = LlamaForCausalLMEagle3(config, attention_backend=args.attention_backend)
        draft.freeze_embedding()
        model = OnlineEagle3Model(
            draft, length=args.ttt_length, attention_backend=args.attention_backend
        )
    else:
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
        from specforge.modeling.draft.dflash2 import DFlash2DraftModel

        config = Qwen3Config(
            architectures=["DFlash2DraftModel"],
            hidden_size=args.hidden_size,
            intermediate_size=args.hidden_size * 4,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=args.hidden_size // 4,
            num_hidden_layers=args.draft_layers,
            num_target_layers=4,
            vocab_size=args.vocab_size,
            max_position_embeddings=max(args.lengths) + 32,
            layer_types=["full_attention"] * args.draft_layers,
            attention_dropout=0.0,
            dflash_config={
                "block_size": args.block_size,
                "mask_token_id": 0,
                "target_layer_ids": [1],
                "conv_group_size": 16,
                "conv_kernel_size": 2,
                "selector_rank": 8,
                "selector_top_k": 4,
            },
        )
        config._attn_implementation = args.attention_backend
        draft = DFlash2DraftModel(config)
        head = torch.nn.Linear(
            args.hidden_size, args.vocab_size, bias=False
        ).requires_grad_(False)
        embed = torch.nn.Embedding(args.vocab_size, args.hidden_size).requires_grad_(
            False
        )
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embed,
            mask_token_id=0,
            block_size=args.block_size,
            num_anchors=args.num_anchors,
            attention_backend=args.attention_backend,
            loss_type="dpace",
            selector_loss_alpha=1.0,
            teacher_metrics=False,
            objective_chunk_blocks=0,
        )

        def fixed_anchors(self, seq_len, loss_mask, device, max_valid_anchors=None):
            # Benchmark-only: one precomputed random anchor set per source sample.
            positions = torch.stack(
                [
                    self._benchmark_anchors[sample_id]
                    for sample_id in self._benchmark_sample_ids
                ]
            )
            return positions, torch.ones_like(positions, dtype=torch.bool)

        if fixed_sample_anchors:
            model._sample_anchor_positions = types.MethodType(fixed_anchors, model)
    return model.to(device=torch.cuda.current_device(), dtype=dtype)


def build_features(args, refs, dtype):
    import torch

    features, anchors = {}, {}
    for ref in refs:
        gen = torch.Generator().manual_seed(args.seed + 1000 + int(ref.sample_id))
        length = ref.num_tokens
        mask = torch.ones(1, length, dtype=torch.long)
        mask[:, : length // 4] = 0
        mask[:, -1] = 0
        item = {
            "input_ids": torch.randint(1, args.vocab_size, (1, length), generator=gen),
            "loss_mask": mask,
        }
        width = args.hidden_size * (3 if args.algorithm == "eagle3" else 1)
        hidden = torch.randn(1, length, width, generator=gen).to(dtype)
        if args.algorithm == "eagle3":
            item.update(
                loss_mask=mask.unsqueeze(-1),
                hidden_state=hidden,
                attention_mask=torch.ones_like(mask),
                target=torch.randn(1, length, args.vocab_size, generator=gen).to(dtype),
            )
        else:
            item["hidden_states"] = hidden
            candidates = torch.arange(length // 4, length - args.block_size)
            if len(candidates) < args.num_anchors:
                raise ValueError(
                    "Not enough full-block supervised anchors; increase --lengths or reduce --num-anchors"
                )
            positions = (
                candidates[
                    torch.randperm(len(candidates), generator=gen)[: args.num_anchors]
                ]
                .sort()
                .values
            )
            anchors[ref.sample_id] = positions.cuda()
        features[ref.sample_id] = item
    return features, anchors


def collate_batches(args, local_batches, features):
    from specforge.runtime.contracts import TrainBatch

    if args.algorithm == "eagle3":
        from specforge.algorithms.eagle3.data import build_server_collator

        collator = build_server_collator()
    else:
        from specforge.algorithms.common.hidden_states_data import build_collator

        collator = build_collator()
    return [
        TrainBatch(
            sample_ids=[ref.sample_id for ref in batch],
            strategy="eagle3" if args.algorithm == "eagle3" else "dflash",
            tensors={
                key: tensor.pin_memory()
                for key, tensor in collator(
                    [features[ref.sample_id] for ref in batch]
                ).items()
            },
            metadata={"target_repr": "logits"} if args.algorithm == "eagle3" else {},
        )
        for batch in local_batches
    ]


def make_runner(args, dtype, anchors, capture):
    import torch

    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import (
        DFlashTrainStrategy,
        Eagle3TrainStrategy,
    )

    model = build_model(args, dtype, fixed_sample_anchors=capture)
    model._benchmark_anchors = anchors
    pc = ParallelConfig.from_distributed(
        sharding_strategy=args.sharding, param_dtype=dtype
    )
    backend = FSDPTrainingBackend(
        pc,
        optimizer_factory=lambda module: BF16Optimizer(
            module,
            lr=1e-3,
            max_grad_norm=1.0,
            warmup_ratio=0.0,
            total_steps=1000,
            lr_scheduler="constant",
        ),
    )
    backend.prepare_model(model, optimizer_target=model.draft_model)
    strategy = (
        Eagle3TrainStrategy(backend.module)
        if args.algorithm == "eagle3"
        else DFlashTrainStrategy(backend.module)
    )
    observed = {
        "objective": [],
        "gradients": None,
        "updates": None,
        "supervised_positions": [],
    }
    original_forward = strategy.forward_loss

    def forward(batch, ctx=None):
        out = original_forward(batch, ctx)
        if "accuracy_denom" in out.metrics:
            observed["supervised_positions"].append(
                out.metrics["accuracy_denom"].detach()
            )
        if capture:
            pair = (
                out.loss_terms
                if out.loss_terms is not None
                else (out.loss, out.loss.new_ones(()))
            )
            observed["objective"].append(tuple(value.detach() for value in pair))
        return out

    strategy.forward_loss = forward
    if capture:
        original_step = backend.optimizer.step

        def step(**kwargs):
            params = backend.optimizer.model_params
            observed["gradients"] = torch.cat(
                [
                    (
                        parameter.grad.detach().float().flatten().clone()
                        if parameter.grad is not None
                        else torch.zeros_like(parameter, dtype=torch.float32).flatten()
                    )
                    for parameter in params
                ]
            )
            before = torch.cat(
                [parameter.detach().float().flatten().clone() for parameter in params]
            )
            observed["initial_weights"] = before
            result = original_step(**kwargs)
            observed["updates"] = (
                torch.cat(
                    [parameter.detach().float().flatten() for parameter in params]
                )
                - before
            )
            return result

        backend.optimizer.step = step
    return (
        model,
        TrainerCore(strategy, backend, accumulation_steps=args.accumulation_steps),
        observed,
    )


def one_window(args, model, core, batches, *, timings=False):
    import torch

    from specforge.training.strategies.base import StepContext

    events = []
    for batch in batches:
        model._benchmark_sample_ids = batch.sample_ids
        start, end = (
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            if timings
            else (None, None)
        )
        if timings:
            start.record()
        core.train_step(
            batch,
            StepContext(
                global_step=0, total_steps=1000, collect_detailed_metrics=False
            ),
        )
        if timings:
            end.record()
            events.append((start, end))
    if timings:
        torch.cuda.synchronize()
        return [start.elapsed_time(end) for start, end in events]
    return None


def gather(value):
    import torch.distributed as dist

    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


def parity_run(args, orders, refs):
    import torch
    import torch.distributed as dist

    dtype = getattr(torch, args.parity_dtype)
    features, anchors = build_features(args, refs, dtype)
    results = {}
    for name, order in orders.items():
        batches = collate_batches(
            args,
            rank_batches(
                order, dist.get_rank(), dist.get_world_size(), args.batch_size
            ),
            features,
        )
        model, core, observed = make_runner(args, dtype, anchors, True)
        one_window(args, model, core, batches)
        objective = torch.stack(
            [torch.stack(pair) for pair in observed["objective"]]
        ).sum(dim=0)
        dist.all_reduce(objective)
        results[name] = {
            "loss": float((objective[0] / objective[1]).item()),
            "gradients": observed["gradients"].cpu(),
            "updates": observed["updates"].cpu(),
            "initial_weights": observed["initial_weights"].cpu(),
        }
        del model, core, observed, batches
        gc.collect()
        torch.cuda.empty_cache()
    reference, actual = results["baseline"], results["length_aware"]
    report = {
        "reference_loss": reference["loss"],
        "length_aware_loss": actual["loss"],
        "loss_abs_diff": abs(reference["loss"] - actual["loss"]),
        "dtype": args.parity_dtype,
        "atol": args.atol,
        "rtol": args.rtol,
    }
    passes = [report["loss_abs_diff"] <= args.atol + args.rtol * abs(reference["loss"])]
    for name in ("gradients", "updates"):
        ref, value = reference[name], actual[name]
        relative_l2_error = float((ref - value).norm() / ref.norm().clamp_min(1e-30))
        passed = (
            bool(torch.allclose(ref, value, atol=args.atol, rtol=args.rtol))
            and relative_l2_error <= args.rtol
        )
        passes.append(passed)
        report[name] = {
            "passed": passed,
            "max_abs_diff": float((ref - value).abs().max()) if ref.numel() else 0.0,
            "relative_l2_error": relative_l2_error,
            "reference_l2_norm": float(ref.norm()),
        }
    update_diff = reference["updates"] - actual["updates"]
    if update_diff.numel():
        worst = int(update_diff.abs().argmax())
        report["updates"]["gradient_pair_at_max_update_diff"] = [
            float(reference["gradients"][worst]),
            float(actual["gradients"][worst]),
        ]
        report["updated_parameter_relative_l2_error"] = float(
            update_diff.norm()
            / (reference["initial_weights"] + reference["updates"])
            .norm()
            .clamp_min(1e-30)
        )
        report["gradients"]["sign_disagreement_count"] = int(
            (reference["gradients"].sign() != actual["gradients"].sign()).sum()
        )
    report["comparison"] = (
        "fixed_microbatch_permutation_control"
        if args.permutation_control
        else "length_aware"
    )
    report["passed"] = all(passes)
    return {
        "passed": all(item["passed"] for item in gather(report)),
        "by_rank": gather(report),
        "scope": "One optimizer window; identical sample IDs, per-sample DFlash anchors and production objective. TF32 disabled; not a convergence or serving accuracy result.",
    }


def _performance_pair(args, orders, refs):
    import torch
    import torch.distributed as dist

    dtype = getattr(torch, args.perf_dtype)
    features, anchors = build_features(args, refs, dtype)
    results = {}
    for name, order in orders.items():
        batches = collate_batches(
            args,
            rank_batches(
                order, dist.get_rank(), dist.get_world_size(), args.batch_size
            ),
            features,
        )
        model, core, observed = make_runner(args, dtype, anchors, False)
        for _ in range(args.warmup):
            one_window(args, model, core, batches)
        torch.cuda.synchronize()
        dist.barrier()
        torch.cuda.reset_peak_memory_stats()
        observed["supervised_positions"].clear()
        local = []
        for _ in range(args.steps):
            dist.barrier()
            torch.cuda.synchronize()
            start = time.perf_counter()
            microstep_ms = one_window(args, model, core, batches, timings=True)
            local.append(
                {
                    "optimizer_window_ms": (time.perf_counter() - start) * 1000,
                    "microstep_cuda_ms": microstep_ms,
                }
            )
        peak = torch.cuda.max_memory_allocated()
        per_rank = gather(local)
        peak_by_rank = gather(peak)
        windows = [
            max(rank[step]["optimizer_window_ms"] for rank in per_rank)
            for step in range(args.steps)
        ]
        skews = [
            max(rank[step]["microstep_cuda_ms"][micro] for rank in per_rank)
            - min(rank[step]["microstep_cuda_ms"][micro] for rank in per_rank)
            for step in range(args.steps)
            for micro in range(args.accumulation_steps)
        ]
        seconds = sum(windows) / 1000
        results[name] = {
            "optimizer_window_ms_median": statistics.median(windows),
            "optimizer_window_ms_min": min(windows),
            "optimizer_window_ms_max": max(windows),
            "optimizer_window_ms_samples": windows,
            "valid_context_tokens_per_second": sum(ref.num_tokens for ref in refs)
            * args.steps
            / seconds,
            "samples_per_second": len(refs) * args.steps / seconds,
            "microstep_rank_time_skew_ms_mean": statistics.mean(skews),
            "peak_allocated_bytes_by_rank": peak_by_rank,
            "rank_time_skew_note": "Difference in recorded CUDA microstep durations, including collectives; not a direct measurement of GPU idle or NCCL wait time.",
        }
        if args.algorithm == "dflash2":
            positions = torch.stack(observed["supervised_positions"]).sum()
            dist.all_reduce(positions)
            count = int(positions.item())
            results[name]["supervised_draft_positions"] = count
            results[name]["supervised_draft_positions_per_second"] = count / seconds
            results[name]["anchor_sampling"] = "native production random sampler"
        del model, core, batches, observed
        gc.collect()
        torch.cuda.empty_cache()
    results["median_speedup"] = (
        results["baseline"]["optimizer_window_ms_median"]
        / results["length_aware"]["optimizer_window_ms_median"]
    )
    results["scope"] = (
        "Preloaded synthetic CPU features through production model, strategy, TrainerCore, FSDP/DDP and AdamW; excludes teacher capture, feature storage/transport and scheduling latency. Warmup is excluded. Both variants update weights."
    )
    return results


def performance_run(args, orders, refs):
    pairs = []
    for repeat in range(args.repeats):
        ordered = orders if repeat % 2 == 0 else dict(reversed(list(orders.items())))
        pair = _performance_pair(args, ordered, refs)
        pair["variant_execution_order"] = list(ordered)
        pairs.append(pair)
    return {
        "repetitions": pairs,
        "median_speedup": statistics.median(pair["median_speedup"] for pair in pairs),
        "min_speedup": min(pair["median_speedup"] for pair in pairs),
        "max_speedup": max(pair["median_speedup"] for pair in pairs),
        "baseline_window_ms_median": statistics.median(
            pair["baseline"]["optimizer_window_ms_median"] for pair in pairs
        ),
        "length_aware_window_ms_median": statistics.median(
            pair["length_aware"]["optimizer_window_ms_median"] for pair in pairs
        ),
        "scope": pairs[0]["scope"],
    }


def main():
    args = parse_args()
    world = args.world_size if args.dry_run else int(os.environ.get("WORLD_SIZE", "1"))
    refs = make_refs(args, world)
    orders = make_orders(args, refs, world)
    report = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "world_size": world,
        "layouts": {
            name: layout_report(args, order, world) for name, order in orders.items()
        },
        "limitations": [
            "Synthetic random models/features, not evidence of real-model convergence or serving acceptance.",
            "This fixture is one optimizer window; offline sample order across multiple optimizer windows can change the trajectory.",
            "No guaranteed speedup; uniform-length runs are the required negative control.",
        ],
    }
    try:
        report["git_sha"] = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        report["git_dirty"] = bool(
            subprocess.check_output(["git", "status", "--porcelain"], text=True).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        report["git_sha"] = None
    root = Path(__file__).resolve().parents[1]
    report["source_sha256"] = {
        path: hashlib.sha256((root / path).read_bytes()).hexdigest()
        for path in (
            "benchmarks/length_aware_training.py",
            "specforge/data/length_bucketing.py",
            "specforge/runtime/data_plane/ref_distributor.py",
            "specforge/training/controller.py",
            "specforge/training/backend.py",
            "specforge/training/strategies/base.py",
            "specforge/algorithms/eagle3/model.py",
            "specforge/algorithms/common/dflash_family_model.py",
            "specforge/modeling/draft/dflash2.py",
        )
    }
    rank = 0
    if not args.dry_run:
        import torch
        import torch.distributed as dist

        from specforge.distributed import init_distributed

        if not torch.cuda.is_available():
            raise RuntimeError(
                "GPU benchmark requires CUDA; use --dry-run for metadata checks"
            )
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29593")
        init_distributed(timeout=120, tp_size=1)
        rank = dist.get_rank()
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        report["environment"] = {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "tf32": False,
        }
        if not args.skip_parity:
            report["numerical_parity"] = parity_run(args, orders, refs)
        if not args.skip_performance:
            report["performance"] = performance_run(args, orders, refs)
        dist.barrier()
        dist.destroy_process_group()
    if rank == 0:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(
            json.dumps(
                {
                    "output": str(args.output),
                    "parity_passed": report.get("numerical_parity", {}).get("passed"),
                    "median_speedup": report.get("performance", {}).get(
                        "median_speedup"
                    ),
                },
                indent=2,
            )
        )
    if not report.get("numerical_parity", {}).get("passed", True):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
