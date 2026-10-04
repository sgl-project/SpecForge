"""Compare padded and packed EAGLE3 training on identical synthetic features.

This exercises the production collators, Eagle3TrainStrategy, OnlineEagle3Model,
LlamaForCausalLMEagle3, TargetHead preprocessing/projection, and BF16Optimizer.
No model downloads are needed. It measures one GPU with resident input features;
target feature generation, disk loading, H2D transfers, and distributed training
are outside the measurement. Synthetic features do not establish model quality
or speculative serving speedups.

Run a strict small FP32 gate before the representative BF16 benchmark::

    PYTHONPATH=. python scripts/benchmark_sequence_packing.py --preset tiny \
        --dtype float32 --correctness-only --output /tmp/packing-correctness.json
    PYTHONPATH=. python scripts/benchmark_sequence_packing.py --preset medium \
        --warmup 5 --steps 20 --output /tmp/packing-perf.json

Each --lengths argument defines one batch (repeat it for several padding ratios).
For example: --lengths 1024,1024,1024,1024 --lengths 128,256,512,2048.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import LlamaConfig

from specforge.algorithms.eagle3.data import DataCollatorWithPacking
from specforge.algorithms.eagle3.model import OnlineEagle3Model
from specforge.data.utils import DataCollatorWithPadding
from specforge.modeling.draft.llama3_eagle import LlamaForCausalLMEagle3
from specforge.modeling.target.target_head import TargetHead
from specforge.optimizer import BF16Optimizer
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import Eagle3TrainStrategy


class SyntheticTargetHead(TargetHead):
    """Random frozen head with the production forward and preprocess methods."""

    def __init__(self, hidden_size: int, vocab_size: int):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.config = SimpleNamespace(hidden_size=hidden_size, vocab_size=vocab_size)
        self.fc = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.freeze_weights()


PRESETS = {
    "tiny": {
        "hidden_size": 128,
        "intermediate_size": 256,
        "num_heads": 4,
        "num_kv_heads": 2,
        "vocab_size": 512,
        "draft_vocab_size": 256,
        "lengths": [[8, 17, 31, 64], [2, 3, 5, 17], [32, 32, 32, 32]],
    },
    "medium": {
        "hidden_size": 2048,
        "intermediate_size": 8192,
        "num_heads": 16,
        "num_kv_heads": 4,
        "vocab_size": 32000,
        "draft_vocab_size": 32000,
        "lengths": [
            [1024, 1024, 1024, 1024],
            [512, 768, 1024, 2048],
            [128, 256, 512, 2048],
        ],
    },
    "large": {
        "hidden_size": 4096,
        "intermediate_size": 14336,
        "num_heads": 32,
        "num_kv_heads": 8,
        "vocab_size": 32000,
        "draft_vocab_size": 32000,
        "lengths": [[128, 256, 512, 2048]],
    },
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=PRESETS, default="medium")
    for key in PRESETS["tiny"]:
        if key != "lengths":
            parser.add_argument("--" + key.replace("_", "-"), type=int)
    parser.add_argument("--target-hidden-size", type=int)
    parser.add_argument("--lengths", action="append", help="comma-separated lengths")
    parser.add_argument("--ttt-length", type=int, default=7)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1729)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--prompt-fraction", type=float, default=0.25)
    parser.add_argument("--correctness-only", action="store_true")
    parser.add_argument("--skip-correctness", action="store_true")
    parser.add_argument("--atol", type=float)
    parser.add_argument("--rtol", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    preset = PRESETS[args.preset]
    for key, value in preset.items():
        if getattr(args, key) is None:
            setattr(args, key, value)
    if isinstance(args.lengths[0], str):
        args.lengths = [[int(item) for item in row.split(",")] for row in args.lengths]
    args.target_hidden_size = args.target_hidden_size or args.hidden_size
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
    if any(not row or min(row) < 1 for row in args.lengths):
        parser.error("each batch must contain positive sequence lengths")
    if args.steps < 1 or args.warmup < 1 or args.ttt_length < 1:
        parser.error("steps, warmup, and ttt-length must be positive")
    if not 0 <= args.prompt_fraction < 1:
        parser.error("prompt-fraction must be in [0, 1)")
    if args.draft_vocab_size > args.vocab_size:
        parser.error("draft-vocab-size cannot exceed vocab-size")
    if args.hidden_size % args.num_heads or args.num_heads % args.num_kv_heads:
        parser.error(
            "hidden-size / num-heads and num-heads / num-kv-heads must be integral"
        )
    if args.correctness_only and args.skip_correctness:
        parser.error("correctness-only and skip-correctness are incompatible")
    return args


def build_strategy(args):
    torch.manual_seed(args.seed)
    config = LlamaConfig(
        hidden_size=args.hidden_size,
        target_hidden_size=args.target_hidden_size,
        intermediate_size=args.intermediate_size,
        num_attention_heads=args.num_heads,
        num_key_value_heads=args.num_kv_heads,
        num_hidden_layers=1,
        vocab_size=args.vocab_size,
        draft_vocab_size=args.draft_vocab_size,
        max_position_embeddings=max(map(max, args.lengths)) + args.ttt_length,
        pad_token_id=0,
        attention_dropout=0.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=False,
    )
    draft = LlamaForCausalLMEagle3(config, attention_backend="flex_attention")
    # Exercise the real target-to-draft mapping, including a reduced vocabulary.
    selected = torch.randperm(args.vocab_size)[: args.draft_vocab_size].sort().values
    draft.t2d.zero_()
    draft.t2d[selected] = True
    draft.d2t.copy_(selected - torch.arange(args.draft_vocab_size))
    draft.freeze_embedding()
    dtype = getattr(torch, args.dtype)
    model = OnlineEagle3Model(
        draft, length=args.ttt_length, attention_backend="flex_attention"
    ).to(device="cuda", dtype=dtype)
    head = (
        SyntheticTargetHead(args.target_hidden_size, args.vocab_size)
        .to(device="cuda", dtype=dtype)
        .eval()
    )
    model.train()
    return Eagle3TrainStrategy(model, target_head=head)


def make_features(args, lengths):
    generator = torch.Generator().manual_seed(args.seed + 1)
    dtype = getattr(torch, args.dtype)
    features = []
    for length in lengths:
        loss_mask = torch.ones(1, length, dtype=torch.long)
        loss_mask[:, : int(length * args.prompt_fraction)] = 0
        loss_mask[:, -1] = 0  # The production offline normalizer does this.
        features.append(
            {
                "input_ids": torch.randint(
                    1, args.vocab_size, (1, length), generator=generator
                ),
                "attention_mask": torch.ones(1, length, dtype=torch.long),
                "loss_mask": loss_mask,
                "hidden_state": torch.randn(
                    1, length, 3 * args.target_hidden_size, generator=generator
                ).to(dtype),
                "target": torch.randn(
                    1, length, args.target_hidden_size, generator=generator
                ).to(dtype),
            }
        )
    return features


def make_batch(features, mode):
    collator = (
        DataCollatorWithPadding() if mode == "padded" else DataCollatorWithPacking()
    )
    tensors = collator(features)
    # Keep packing control metadata on CPU, matching the production loader.
    tensors = {
        name: (
            value if name in {"sequence_lengths", "loss_denominator"} else value.cuda()
        )
        for name, value in tensors.items()
    }
    return TrainBatch(
        sample_ids=[str(i) for i in range(len(features))],
        strategy="eagle3",
        tensors=tensors,
        metadata={"target_repr": "hidden_state"},
    )


def tensor_comparison(reference, actual, args):
    reference, actual = reference.float(), actual.float()
    delta = actual - reference
    reference_norm = float(reference.norm())
    return {
        "pass": bool(torch.allclose(reference, actual, atol=args.atol, rtol=args.rtol)),
        "max_abs_diff": float(delta.abs().max()),
        "relative_l2_diff": float(delta.norm()) / max(reference_norm, 1e-30),
        "reference_l2": reference_norm,
    }


def check_correctness(strategy, features, args):
    model = strategy.trainable_module()
    outputs = {}
    for mode in ("padded", "packed"):
        model.zero_grad(set_to_none=True)
        batch = make_batch(features, mode)
        result = strategy.forward_loss(batch)
        result.loss.backward()
        outputs[mode] = {
            "loss": result.loss.detach().float().cpu(),
            "plosses": torch.stack(result.metrics["plosses"]).float().cpu(),
            "grads": {
                name: None if param.grad is None else param.grad.detach().float().cpu()
                for name, param in model.named_parameters()
                if param.requires_grad
            },
        }
        del batch, result
    checks = {
        key: tensor_comparison(outputs["padded"][key], outputs["packed"][key], args)
        for key in ("loss", "plosses")
    }
    checks["loss_padded"] = float(outputs["padded"]["loss"])
    checks["loss_packed"] = float(outputs["packed"]["loss"])
    checks["gradients"] = {}
    for name, reference in outputs["padded"]["grads"].items():
        actual = outputs["packed"]["grads"][name]
        checks["gradients"][name] = (
            {"pass": reference is None and actual is None, "missing_gradient": True}
            if reference is None or actual is None
            else tensor_comparison(reference, actual, args)
        )
    checks["pass"] = all(checks[key]["pass"] for key in ("loss", "plosses")) and all(
        row["pass"] for row in checks["gradients"].values()
    )
    model.zero_grad(set_to_none=True)
    return checks


def time_mode(strategy, features, lengths, mode, args, initial_state):
    model = strategy.trainable_module()
    model.load_state_dict(initial_state)
    model.zero_grad(set_to_none=True)
    batch = make_batch(features, mode)
    optimizer = BF16Optimizer(
        model,
        lr=args.learning_rate,
        warmup_ratio=0.0,
        total_steps=args.warmup + args.steps + 1,
        lr_scheduler="constant",
    )

    def step():
        output = strategy.forward_loss(batch)
        output.loss.backward()
        optimizer.step()
        return output.loss.detach()

    torch.cuda.synchronize()
    warm_start = time.perf_counter()
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter() - warm_start
    torch.cuda.reset_peak_memory_stats()
    baseline_bytes = torch.cuda.memory_allocated()
    durations = []
    for _ in range(args.steps):
        torch.cuda.synchronize()
        start = time.perf_counter()
        final_loss = step()
        torch.cuda.synchronize()
        durations.append(time.perf_counter() - start)
    peak_bytes = torch.cuda.max_memory_allocated()
    mean_seconds = statistics.mean(durations)
    result = {
        "mean_step_ms": mean_seconds * 1000,
        "p50_step_ms": statistics.median(durations) * 1000,
        "stdev_step_ms": (
            statistics.stdev(durations) * 1000 if len(durations) > 1 else 0
        ),
        "useful_tokens_per_second": sum(lengths) / mean_seconds,
        "useful_ttt_positions_per_second": sum(lengths)
        * args.ttt_length
        / mean_seconds,
        "peak_allocated_gib": peak_bytes / 2**30,
        "baseline_allocated_gib": baseline_bytes / 2**30,
        "peak_increment_gib": (peak_bytes - baseline_bytes) / 2**30,
        "warmup_seconds_including_compile": warmup_seconds,
        "step_ms": [duration * 1000 for duration in durations],
        "final_loss": float(final_loss),
    }
    optimizer = batch = None
    model.zero_grad(set_to_none=True)
    gc.collect()
    torch.cuda.empty_cache()
    return result


def source_state():
    root = Path(__file__).resolve().parents[2]
    source_paths = [
        "scripts/benchmark_sequence_packing.py",
        "specforge/benchmarks/benchmark_sequence_packing.py",
        "specforge/algorithms/eagle3/data.py",
        "specforge/algorithms/eagle3/model.py",
        "specforge/modeling/draft/llama3_eagle.py",
        "specforge/modeling/packed_sequence.py",
        "specforge/training/strategies/base.py",
    ]
    result = {
        "files_sha256": {
            name: hashlib.sha256((root / name).read_bytes()).hexdigest()
            for name in source_paths
        }
    }

    def git(*command):
        return subprocess.check_output(
            ["git", *command], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()

    try:
        result.update(head=git("rev-parse", "HEAD"), dirty=git("status", "--short"))
    except (OSError, subprocess.CalledProcessError):
        result["head"] = "unavailable (file hashes identify copied snapshot)"
    return result


def write_report(report, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This benchmark requires a CUDA GPU")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_num_threads(4)
    settings = vars(args).copy()
    settings["output"] = str(args.output)
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "source": source_state(),
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "transformers": importlib.metadata.version("transformers"),
        },
        "settings": settings,
        "scope": "single-GPU synthetic offline features; resident inputs; production forward/backward/BF16Optimizer; excludes capture, I/O, transfer, distributed communication, and serving",
        "cases": [],
    }
    strategy = build_strategy(args)
    initial_state = {
        key: value.detach().cpu().clone()
        for key, value in strategy.trainable_module().state_dict().items()
    }
    for lengths in args.lengths:
        strategy.trainable_module().load_state_dict(initial_state)
        features = make_features(args, lengths)
        case = {
            "lengths": lengths,
            "useful_tokens": sum(lengths),
            "padded_tokens": len(lengths) * max(lengths),
            "padding_fraction": 1 - sum(lengths) / (len(lengths) * max(lengths)),
            "raw_supervised_tokens": sum(
                int(item["loss_mask"].sum()) for item in features
            ),
        }
        report["cases"].append(case)
        print(
            json.dumps(
                {"case_start": lengths, "padding_fraction": case["padding_fraction"]}
            ),
            flush=True,
        )
        if not args.skip_correctness:
            case["correctness"] = check_correctness(strategy, features, args)
            write_report(report, args.output)
            print(
                json.dumps({"correctness_pass": case["correctness"]["pass"]}),
                flush=True,
            )
            if not case["correctness"]["pass"]:
                raise AssertionError(
                    f"Packed loss/gradient parity failed; inspect {args.output}"
                )
        if not args.correctness_only:
            for mode in ("padded", "packed"):
                case[mode] = time_mode(
                    strategy, features, lengths, mode, args, initial_state
                )
                write_report(report, args.output)
                print(json.dumps({"mode": mode, **case[mode]}), flush=True)
            case["speedup"] = (
                case["padded"]["mean_step_ms"] / case["packed"]["mean_step_ms"]
            )
            case["peak_memory_reduction_fraction"] = 1 - (
                case["packed"]["peak_allocated_gib"]
                / case["padded"]["peak_allocated_gib"]
            )
        write_report(report, args.output)
        del features
    print(f"Report written to {args.output}", flush=True)


if __name__ == "__main__":
    main()
