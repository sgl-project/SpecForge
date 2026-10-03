#!/usr/bin/env python3
"""Benchmark real draft training with deterministic, cached synthetic features.

Run each backend in a fresh torchrun process on the same GPUs, for example::

    torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_backends.py \
        --backend fsdp --algorithm dflash2 --output work/fsdp-dflash2.json
    torchrun --standalone --nproc-per-node=2 scripts/benchmark_training_backends.py \
        --backend torchtitan --algorithm dflash2 --output work/titan-dflash2.json

The FSDP baseline works with SpecForge main. The optional ``--backend torchtitan``
mode requires an installed checkout containing the backend from PR #920
(https://github.com/sgl-project/SpecForge/pull/920) and its optional dependency.
This benchmark does not add or install the TorchTitan backend.

For reported results, run three independent torchrun launches per backend with
--repeats 1 and distinct output filenames, alternating backend order (A/B then
B/A). Framework caches or retained allocations may survive a model replacement
inside one process; in-process repeats are not independent peak-memory trials.
Repeat with DP4 and algorithms dflash/dspark. The default controlled recipe uses
the full Qwen3-4B draft geometry and vocabulary for every algorithm. It is a
custom benchmark recipe, not a released DFlash2 checkpoint. --recipe stock uses
the repository configs verbatim (DFlash2's Qwen3.5-4B vocabulary is different).

This measures TrainerCore + real forward/backward/optimizer work, with frozen
random target tables and cached random hidden states. It excludes target model
inference, feature production/transport, checkpoint I/O, and model construction.
It does not measure convergence or speculative serving speed. Throughput counts
input context tokens, not accepted tokens or unique supervised draft positions.

For correctness, use --tiny --precision bf16 --check-resume and --snapshot-dir
for BOTH backends; then run --compare-snapshots LEFT.pt RIGHT.pt. Validation
snapshots and resume replay are deliberately restricted to tiny models.
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
STOCK_CONFIGS = {
    "dflash": "qwen3-4b-dflash.json",
    "dflash2": "qwen3.5-4b-dflash2.json",
    "dspark": "qwen3-4b-dspark.json",
}
ARCHITECTURES = {
    "dflash": "DFlashDraftModel",
    "dflash2": "DFlash2DraftModel",
    "dspark": "DSparkDraftModel",
}


def resolve_config(algorithm, recipe="controlled", *, tiny=False, config_path=None):
    """Resolve a fully recorded recipe without importing GPU dependencies."""
    source = (
        Path(config_path)
        if config_path
        else REPO_ROOT
        / "configs"
        / (STOCK_CONFIGS[algorithm] if recipe == "stock" else STOCK_CONFIGS["dflash"])
    )
    config = json.loads(source.read_text())
    if config_path and config.get("architectures") != [ARCHITECTURES[algorithm]]:
        raise ValueError("--config architecture does not match --algorithm")
    if not config_path and recipe == "controlled":
        config["architectures"] = [ARCHITECTURES[algorithm]]
        config.pop("auto_map", None)
        method = config["dflash_config"]
        if algorithm == "dflash2":
            method.update(
                conv_group_size=16,
                conv_kernel_size=2,
                selector_rank=256,
                selector_top_k=16,
            )
        elif algorithm == "dspark":
            method.update(
                attention_mode="gqa",
                projector_type="dspark",
                markov_head_type="vanilla",
                markov_rank=256,
                confidence_head_alpha=1.0,
                enable_confidence_head=True,
                confidence_head_with_markov=True,
            )
    if tiny:
        config.update(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=2,
            head_dim=8,
            num_target_layers=4,
            vocab_size=128,
            max_position_embeddings=4096,
            layer_types=["full_attention"] * 2,
            block_size=4,
            bos_token_id=1,
            eos_token_id=2,
            pad_token_id=0,
            rope_scaling=None,
            rope_theta=10000.0,
        )
        config.pop("rope_parameters", None)
        method = config["dflash_config"]
        method.update(target_layer_ids=[1, 2], mask_token_id=127, block_size=4)
        if algorithm == "dflash2":
            method.update(conv_group_size=4, selector_rank=8, selector_top_k=4)
        if algorithm == "dspark":
            method["markov_rank"] = 8
    # Match AutoDraftModelConfig: a draft owns no target embedding/head to tie.
    config["tie_word_embeddings"] = False
    return config, str(source.resolve())


def summarize_times(seconds, input_tokens_per_window):
    if not seconds or any(not math.isfinite(x) or x <= 0 for x in seconds):
        raise ValueError("step durations must be non-empty, finite and positive")
    ordered = sorted(seconds)
    return {
        "optimizer_step_seconds_mean": statistics.mean(seconds),
        "optimizer_step_seconds_median": statistics.median(seconds),
        "optimizer_step_seconds_p95": ordered[math.ceil(0.95 * len(ordered)) - 1],
        "input_context_tokens_per_second": input_tokens_per_window
        * len(seconds)
        / sum(seconds),
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--backend", choices=("fsdp", "torchtitan"))
    parser.add_argument("--algorithm", choices=tuple(ARCHITECTURES), default="dflash")
    parser.add_argument(
        "--recipe", choices=("controlled", "stock"), default="controlled"
    )
    parser.add_argument(
        "--config", type=Path, help="Explicit draft JSON; overrides recipe selection"
    )
    parser.add_argument(
        "--tiny",
        action="store_true",
        help="Small real architecture for correctness, not performance claims",
    )
    parser.add_argument("--precision", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument(
        "--sharding", choices=("FULL_SHARD", "SHARD_GRAD_OP"), default="FULL_SHARD"
    )
    parser.add_argument(
        "--attention", choices=("sdpa", "flex_attention", "eager"), default="sdpa"
    )
    parser.add_argument("--seq-length", type=int, default=1024)
    parser.add_argument(
        "--batch-size", type=int, default=1, help="Per-rank microbatch size"
    )
    parser.add_argument("--accumulation-steps", type=int, default=2)
    parser.add_argument("--num-anchors", type=int, default=64)
    parser.add_argument("--objective-chunk-blocks", type=int, default=16)
    parser.add_argument("--cache-batches", type=int, default=2)
    parser.add_argument(
        "--warmup-steps",
        type=int,
        default=5,
        help="Optimizer windows, excluded from timing",
    )
    parser.add_argument(
        "--steps", type=int, default=20, help="Measured optimizer windows per repeat"
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=1,
        help="In-process model resets; use 1 and three independent torchrun launches for reported measurements",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--activation-checkpointing", action="store_true")
    parser.add_argument("--detailed-metrics", action="store_true")
    parser.add_argument(
        "--feature-file",
        type=str,
        help="Local torch tensor-list file; {rank} is substituted. Must contain all four normalized feature tensors.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--source-revision", help="Provenance label for source archives without .git"
    )
    parser.add_argument(
        "--check-resume",
        action="store_true",
        help="Tiny-only disk save/restore/replay check after measurement",
    )
    parser.add_argument(
        "--snapshot-dir",
        type=Path,
        help="Tiny-only rank-0 final draft snapshots for backend parity",
    )
    parser.add_argument(
        "--compare-snapshots", nargs=2, type=Path, metavar=("LEFT", "RIGHT")
    )
    parser.add_argument("--rtol", type=float, default=1e-3)
    parser.add_argument("--atol", type=float, default=1e-5)
    return parser


def validate_args(args):
    if any(not math.isfinite(x) or x < 0 for x in (args.rtol, args.atol)):
        raise ValueError("comparison tolerances must be finite and nonnegative")
    if args.compare_snapshots:
        return
    if args.backend is None or args.output is None:
        raise ValueError("--backend and --output are required for a benchmark")
    for name in (
        "seq_length",
        "batch_size",
        "accumulation_steps",
        "num_anchors",
        "cache_batches",
        "steps",
        "repeats",
    ):
        if getattr(args, name) < 1:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    if args.warmup_steps < 0 or args.objective_chunk_blocks < 0:
        raise ValueError("warmup steps and objective chunk blocks must be nonnegative")
    if args.learning_rate <= 0 or not math.isfinite(args.learning_rate):
        raise ValueError("learning rate must be finite and positive")
    if (args.check_resume or args.snapshot_dir) and not args.tiny:
        raise ValueError("resume checks and validation snapshots require --tiny")
    if args.backend == "torchtitan" and args.precision != "bf16":
        raise ValueError("TorchTitan backend currently requires --precision bf16")


def _tensor_fingerprint(values):
    """Exact content hash, evaluated outside timing before sharding."""
    import torch

    digest = hashlib.sha256()
    for name, tensor in sorted(values.items()):
        tensor = tensor.detach().cpu().contiguous()
        digest.update(f"{name}:{tuple(tensor.shape)}:{tensor.dtype}".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _build_model(args, config, device, dtype):
    import torch
    from torch import nn
    from transformers import Qwen3Config

    from specforge.algorithms.common.dflash_family_model import (
        OnlineDFlashModel,
        OnlineDSparkModel,
    )
    from specforge.modeling.auto import AutoDraftModel

    old_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(dtype)
        torch.manual_seed(args.seed)
        draft_config = Qwen3Config.from_dict(copy.deepcopy(config))
        draft_config._attn_implementation = args.attention
        draft = AutoDraftModel.from_config(draft_config, torch_dtype=dtype)
        if args.activation_checkpointing:
            draft.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        draft_hash = _tensor_fingerprint(draft.state_dict())
        # Independent seed makes the target identical across algorithms, too.
        torch.manual_seed(args.seed + 1)
        embedding = nn.Embedding(config["vocab_size"], config["hidden_size"])
        head = nn.Linear(config["hidden_size"], config["vocab_size"], bias=False)
        nn.init.normal_(embedding.weight, std=0.02)
        nn.init.normal_(head.weight, std=0.02)
        embedding.requires_grad_(False)
        head.requires_grad_(False)
        teacher_hash = _tensor_fingerprint(
            {"embedding": embedding.weight, "head": head.weight}
        )
        kwargs = dict(
            draft_model=draft,
            target_lm_head=head,
            target_embed_tokens=embedding,
            mask_token_id=config["dflash_config"]["mask_token_id"],
            block_size=draft.block_size,
            attention_backend=args.attention,
            num_anchors=args.num_anchors,
            objective_chunk_blocks=args.objective_chunk_blocks,
        )
        if args.algorithm == "dspark":
            model = OnlineDSparkModel(**kwargs)
        else:
            model = OnlineDFlashModel(**kwargs, teacher_metrics=args.detailed_metrics)
        counts = {
            "trainable_parameters": sum(
                p.numel() for p in model.parameters() if p.requires_grad
            ),
            "frozen_parameters": sum(
                p.numel() for p in model.parameters() if not p.requires_grad
            ),
        }
        return (
            model.to(device=device, dtype=dtype),
            draft,
            counts,
            draft_hash,
            teacher_hash,
        )
    finally:
        torch.set_default_dtype(old_dtype)


def _make_batches(args, config, rank, device, dtype):
    import torch

    from specforge.runtime.contracts import TrainBatch

    capture_width = (
        len(config["dflash_config"]["target_layer_ids"]) * config["hidden_size"]
    )
    if args.feature_file:
        raw_batches = torch.load(
            args.feature_file.format(rank=rank), map_location="cpu", weights_only=True
        )
        if not isinstance(raw_batches, list) or not raw_batches:
            raise ValueError(
                "feature file must be a nonempty list of tensor dictionaries"
            )
    else:
        raw_batches = []
        for index in range(args.cache_batches):
            generator = torch.Generator().manual_seed(
                args.seed + 1000 + rank * 10000 + index
            )
            shape = (args.batch_size, args.seq_length)
            mask = torch.ones(shape, dtype=torch.float32)
            # Vary supervision by rank/cache entry, exercising token weighting.
            tail = (rank + index) % max(1, args.seq_length // 4)
            if tail:
                mask[:, -tail:] = 0
            raw_batches.append(
                {
                    "input_ids": torch.randint(
                        0, config["vocab_size"], shape, generator=generator
                    ),
                    "loss_mask": mask,
                    "hidden_states": torch.randn(
                        *shape, capture_width, generator=generator, dtype=dtype
                    ),
                    "target_last_hidden_states": torch.randn(
                        *shape, config["hidden_size"], generator=generator, dtype=dtype
                    ),
                }
            )
    expected = {
        "input_ids": (args.batch_size, args.seq_length),
        "loss_mask": (args.batch_size, args.seq_length),
        "hidden_states": (args.batch_size, args.seq_length, capture_width),
        "target_last_hidden_states": (
            args.batch_size,
            args.seq_length,
            config["hidden_size"],
        ),
    }
    batches, fingerprints = [], []
    cache_bytes = 0
    for index, tensors in enumerate(raw_batches):
        if not isinstance(tensors, dict) or any(
            name not in tensors for name in expected
        ):
            raise ValueError(f"batch {index} must contain {sorted(expected)}")
        for name, shape in expected.items():
            if (
                not isinstance(tensors[name], torch.Tensor)
                or tuple(tensors[name].shape) != shape
            ):
                raise ValueError(f"batch {index} {name} must have shape {shape}")
        if (
            tensors["input_ids"].min() < 0
            or tensors["input_ids"].max() >= config["vocab_size"]
        ):
            raise ValueError("feature token id outside target vocabulary")
        normalized = {
            name: tensor.to(
                dtype=(
                    torch.long
                    if name == "input_ids"
                    else torch.float32 if name == "loss_mask" else dtype
                )
            )
            for name, tensor in tensors.items()
            if name in expected
        }
        fingerprints.append(_tensor_fingerprint(normalized))
        # Match prepositioned feature consumers: hidden states on GPU; small
        # integer/mask inputs on pinned host memory for CPU anchor counting.
        for name in normalized:
            if name in ("input_ids", "loss_mask"):
                normalized[name] = normalized[name].pin_memory()
            else:
                normalized[name] = normalized[name].to(device)
                cache_bytes += (
                    normalized[name].numel() * normalized[name].element_size()
                )
        batches.append(
            TrainBatch(
                sample_ids=[
                    f"rank{rank}-cache{index}-sample{n}" for n in range(args.batch_size)
                ],
                strategy="dspark" if args.algorithm == "dspark" else "dflash",
                tensors=normalized,
            )
        )
    return batches, fingerprints, cache_bytes


def _run_window(core, batches, args, step):
    from specforge.training.strategies.base import StepContext

    for micro in range(args.accumulation_steps):
        batch = batches[(step * args.accumulation_steps + micro) % len(batches)]
        result = core.train_step(
            batch,
            StepContext(
                global_step=step,
                total_steps=100000,
                collect_detailed_metrics=args.detailed_metrics,
            ),
        )
    if not result.optimizer_stepped or core.accumulation_remainder:
        raise AssertionError("benchmark window did not end at an optimizer boundary")
    return result


def _cpu_tree(value):
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    return copy.deepcopy(value)


def _assert_state_equal(actual, expected, path="state"):
    """Compare tensor trees including optimizer string/None metadata."""
    import torch

    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0, msg=path)
    elif isinstance(expected, dict):
        if actual.keys() != expected.keys():
            raise AssertionError(f"{path}: state keys differ")
        for key in expected:
            _assert_state_equal(actual[key], expected[key], f"{path}.{key}")
    elif isinstance(expected, (tuple, list)):
        if len(actual) != len(expected):
            raise AssertionError(f"{path}: state sequence lengths differ")
        for index, item in enumerate(expected):
            _assert_state_equal(actual[index], item, f"{path}[{index}]")
    elif actual != expected:
        raise AssertionError(f"{path}: {actual!r} != {expected!r}")


def _check_resume(backend, core, batches, args, step, directory):
    """Write backend state, advance, restore from disk and exactly replay."""
    import torch
    import torch.distributed as dist

    from specforge.training.controller import TrainerCore

    rank = dist.get_rank()
    directory.mkdir(parents=True, exist_ok=True)
    state = backend.state_dict()
    if rank == 0:
        torch.save(state["model"], directory / "model.pt")
    torch.save(
        {key: value for key, value in state.items() if key != "model"},
        directory / f"rank{rank}.pt",
    )
    del state
    dist.barrier()
    expected_loss = _run_window(core, batches, args, step).loss
    expected = _cpu_tree(backend.state_dict())
    restored = torch.load(
        directory / f"rank{rank}.pt", map_location="cpu", weights_only=False
    )
    restored["model"] = torch.load(
        directory / "model.pt", map_location="cpu", weights_only=True
    )
    backend.load_state_dict(restored)
    replay_core = TrainerCore(
        core.strategy, backend, accumulation_steps=args.accumulation_steps
    )
    actual_loss = _run_window(replay_core, batches, args, step).loss
    actual = _cpu_tree(backend.state_dict())
    # Exact same-backend replay checks optimizer masters/moments/scheduler as
    # well as ordinary full model weights, not merely the reported loss.
    _assert_state_equal(actual["optimizer"], expected["optimizer"])
    if rank == 0:
        torch.testing.assert_close(actual["model"], expected["model"], rtol=0, atol=0)
    if expected_loss != actual_loss:
        raise AssertionError(f"resume loss mismatch: {expected_loss} != {actual_loss}")
    return {"passed": True, "replayed_loss": actual_loss, "directory": str(directory)}


def _run_repeat(args, config, repeat):
    import torch
    import torch.distributed as dist

    from specforge.optimizer import BF16Optimizer
    from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
    from specforge.training.controller import TrainerCore
    from specforge.training.strategies.base import (
        DFlashTrainStrategy,
        DSparkTrainStrategy,
    )

    rank, world = dist.get_rank(), dist.get_world_size()
    device = torch.device("cuda", torch.cuda.current_device())
    dtype = torch.bfloat16 if args.precision == "bf16" else torch.float32
    model, draft, counts, draft_hash, teacher_hash = _build_model(
        args, config, device, dtype
    )
    pc = ParallelConfig.from_distributed(
        sharding_strategy=args.sharding, param_dtype=dtype
    )
    if pc.sharding_strategy != args.sharding:
        raise ValueError("FSDP_SHARDING environment conflicts with --sharding")
    backend_type = FSDPTrainingBackend
    if args.backend == "torchtitan":
        from specforge.training.torchtitan_backend import TorchTitanTrainingBackend

        backend_type = TorchTitanTrainingBackend
    backend = backend_type(
        pc,
        optimizer_factory=lambda module: BF16Optimizer(
            module,
            lr=args.learning_rate,
            weight_decay=0.0,
            max_grad_norm=1.0,
            total_steps=100000,
            warmup_ratio=0.0,
            lr_scheduler="constant",
        ),
    )
    wrapped = backend.prepare_model(model, optimizer_target=draft)
    strategy_type = (
        DSparkTrainStrategy if args.algorithm == "dspark" else DFlashTrainStrategy
    )
    core = TrainerCore(
        strategy_type(wrapped), backend, accumulation_steps=args.accumulation_steps
    )
    batches, fingerprints, cache_bytes = _make_batches(
        args, config, rank, device, dtype
    )
    rank_fingerprints = [None] * world
    dist.all_gather_object(rank_fingerprints, fingerprints)
    # Reset AFTER construction/wrapping so backend setup cannot alter anchors.
    torch.manual_seed(args.seed + 2000 + rank)
    losses, seconds = [], []
    for step in range(args.warmup_steps):
        loss = _run_window(core, batches, args, step).loss
        if not math.isfinite(loss):
            raise ValueError(f"non-finite warmup loss: {loss}")
        losses.append(loss)
    torch.cuda.synchronize()
    dist.barrier()
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    for step in range(args.warmup_steps, args.warmup_steps + args.steps):
        dist.barrier()
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = _run_window(core, batches, args, step)
        torch.cuda.synchronize()
        local_seconds = time.perf_counter() - started
        elapsed = torch.tensor(local_seconds, dtype=torch.float64, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        seconds.append(elapsed.item())
        loss = result.loss
        if not math.isfinite(loss):
            raise ValueError(f"non-finite measured loss: {loss}")
        losses.append(loss)
    memory = torch.tensor(
        [
            baseline_bytes,
            torch.cuda.max_memory_allocated(),
            torch.cuda.max_memory_reserved(),
            cache_bytes,
        ],
        dtype=torch.int64,
        device=device,
    )
    dist.all_reduce(memory, op=dist.ReduceOp.MAX)
    baseline, peak_allocated, peak_reserved, cache_max = memory.tolist()
    input_tokens = args.batch_size * args.seq_length * args.accumulation_steps * world
    contract = {
        "config": config,
        "algorithm": args.algorithm,
        "world_size": world,
        "seed": args.seed,
        "precision": args.precision,
        "sharding": args.sharding,
        "attention": args.attention,
        "activation_checkpointing": args.activation_checkpointing,
        "detailed_metrics": args.detailed_metrics,
        "batch_size": args.batch_size,
        "seq_length": args.seq_length,
        "num_anchors": args.num_anchors,
        "objective_chunk_blocks": args.objective_chunk_blocks,
        "accumulation_steps": args.accumulation_steps,
        "warmup_steps": args.warmup_steps,
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "initial_draft_sha256": draft_hash,
        "frozen_target_sha256": teacher_hash,
        "feature_sha256_by_rank": rank_fingerprints,
    }
    record = {
        "repeat": repeat,
        "backend": args.backend,
        **counts,
        "comparison_contract": contract,
        "step_seconds_max_rank": seconds,
        "losses_including_warmup": losses,
        **summarize_times(seconds, input_tokens),
        "baseline_allocated_bytes_max_rank": baseline,
        "peak_allocated_bytes_max_rank": peak_allocated,
        "peak_reserved_bytes_max_rank": peak_reserved,
        "feature_cache_gpu_bytes_max_rank": cache_max,
        "input_context_tokens_per_optimizer_window": input_tokens,
    }
    if args.snapshot_dir:
        state = backend.state_dict()
        if rank == 0:
            args.snapshot_dir.mkdir(parents=True, exist_ok=True)
            path = (
                args.snapshot_dir / f"{args.algorithm}-{args.backend}-repeat{repeat}.pt"
            )
            torch.save(
                {
                    "comparison_contract": contract,
                    "backend": args.backend,
                    "draft_state_dict": _cpu_tree(
                        core.strategy.checkpoint_state_filter(state["model"])
                    ),
                    "losses": losses,
                },
                path,
            )
            record["validation_snapshot"] = str(path)
        del state
    if args.check_resume:
        directory = args.output.parent / f"{args.output.stem}-resume-repeat{repeat}"
        record["resume_check"] = _check_resume(
            backend, core, batches, args, args.warmup_steps + args.steps, directory
        )
    return record


def compare_snapshots(paths, *, rtol, atol):
    import torch

    left, right = [
        torch.load(path, map_location="cpu", weights_only=True) for path in paths
    ]
    if left["comparison_contract"] != right["comparison_contract"]:
        raise ValueError(
            "snapshots have different initialization, input, or training contracts"
        )
    if not left["draft_state_dict"] or not right["draft_state_dict"]:
        raise ValueError("snapshot has no draft parameters")
    torch.testing.assert_close(
        left["draft_state_dict"], right["draft_state_dict"], rtol=rtol, atol=atol
    )
    torch.testing.assert_close(
        torch.tensor(left["losses"]),
        torch.tensor(right["losses"]),
        rtol=rtol,
        atol=atol,
    )
    max_error = max(
        (
            left["draft_state_dict"][name].float()
            - right["draft_state_dict"][name].float()
        )
        .abs()
        .max()
        .item()
        for name in left["draft_state_dict"]
    )
    return {
        "passed": True,
        "left_backend": left["backend"],
        "right_backend": right["backend"],
        "max_draft_parameter_absolute_error": max_error,
        "rtol": rtol,
        "atol": atol,
    }


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as error:
        parser.error(str(error))
    if args.compare_snapshots:
        print(
            json.dumps(
                compare_snapshots(
                    args.compare_snapshots, rtol=args.rtol, atol=args.atol
                ),
                indent=2,
            )
        )
        return
    config, source = resolve_config(
        args.algorithm, args.recipe, tiny=args.tiny, config_path=args.config
    )
    block_size = config.get("block_size", config["dflash_config"].get("block_size"))
    if args.seq_length < block_size + 2:
        parser.error("--seq-length must be at least block_size + 2")
    # Ordinary `python script.py --help` remains dependency-light.
    sys.path.insert(0, str(REPO_ROOT))
    import torch
    import torch.distributed as dist

    from specforge.distributed import destroy_distributed, init_distributed

    if not torch.cuda.is_available():
        raise RuntimeError("this performance harness requires CUDA and torchrun")
    if "RANK" not in os.environ:
        raise RuntimeError(
            "launch with torchrun --standalone --nproc-per-node=2 (or 4)"
        )
    os.environ["SPECFORGE_DEVICE"] = "cuda"
    init_distributed()
    try:
        revision = args.source_revision
        if revision is None:
            try:
                revision = subprocess.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=REPO_ROOT,
                    text=True,
                    stderr=subprocess.DEVNULL,
                ).strip()
            except (OSError, subprocess.CalledProcessError):
                revision = "unavailable (source archive); inspect source_sha256"
        source_files = (
            "scripts/benchmark_training_backends.py",
            "specforge/training/backend.py",
            "specforge/training/torchtitan_backend.py",
            "specforge/training/controller.py",
            "specforge/optimizer.py",
        )
        report = {
            "schema_version": 1,
            "specforge_revision": revision,
            "source_sha256": {
                name: hashlib.sha256((REPO_ROOT / name).read_bytes()).hexdigest()
                for name in source_files
                if (REPO_ROOT / name).is_file()
            },
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "world_size": dist.get_world_size(),
            "algorithm": args.algorithm,
            "backend": args.backend,
            "recipe": "custom" if args.config else args.recipe,
            "tiny": args.tiny,
            "repeat_isolation": (
                "fresh process (one model run)"
                if args.repeats == 1
                else "in-process model resets; retained allocations may affect memory"
            ),
            "source_config": source,
            "features": (
                "cached local features"
                if args.feature_file
                else "cached synthetic features"
            ),
            "frozen_target": "synthetic independently seeded random embedding and LM head",
            "measurement": "real TrainerCore, max-rank synchronized optimizer-window wall time; no feature production/transport or checkpoint I/O",
            "stock_recipe_note": "DFlash2 stock uses Qwen3.5-4B vocab 248320; DFlash/DSpark Qwen3-4B use 151936. Compare backends within one algorithm/recipe.",
            "kernel_environment": {
                key: value
                for key, value in os.environ.items()
                if key.startswith("SPECFORGE_")
            },
            "results": [],
        }
        for repeat in range(args.repeats):
            record = _run_repeat(args, config, repeat)
            report["results"].append(record)
            if dist.get_rank() == 0:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    json.dumps(
                        {
                            key: record[key]
                            for key in (
                                "repeat",
                                "backend",
                                "optimizer_step_seconds_median",
                                "input_context_tokens_per_second",
                                "peak_allocated_bytes_max_rank",
                            )
                        }
                    )
                )
            # A function boundary releases all wrapped models/optimizers before
            # the next repeat. Processes should still be fresh between backends.
            gc.collect()
            torch.cuda.empty_cache()
            dist.barrier()
    finally:
        destroy_distributed()


if __name__ == "__main__":
    main()
