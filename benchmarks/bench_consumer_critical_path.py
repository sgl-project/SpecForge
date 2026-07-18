"""Synthetic microbenchmarks for SpecForge's consumer critical path.

No model weights or datasets are downloaded. Run one process on one GPU, for
example::

    CUDA_VISIBLE_DEVICES=0,1,2,3 python benchmarks/bench_consumer_critical_path.py \
        --device cuda:0 --warmup 4 --iterations 20
"""

from __future__ import annotations

import argparse
import gc
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import torch
import torch.nn.functional as F
from transformers import Qwen3Config

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from specforge.modeling.draft.dflash import DFlashDraftModel  # noqa: E402
from specforge.ops.fused_linear_cross_entropy import (  # noqa: E402
    frozen_linear_cross_entropy,
)
from specforge.optimizer import BF16Optimizer  # noqa: E402
from specforge.runtime.data_plane.feature_dataloader import (  # noqa: E402
    FeatureDataLoader,
)
from specforge.runtime.data_plane.feature_store import LocalFeatureStore  # noqa: E402
from specforge.runtime.data_plane.sample_ref_queue import SampleRefQueue  # noqa: E402
from specforge.training.strategies.registry import resolve_strategy  # noqa: E402


def _timing_summary(samples_ms: list[float]) -> dict[str, float]:
    ordered = sorted(samples_ms)
    p95_index = min(len(ordered) - 1, int(0.95 * len(ordered)))
    return {
        "mean_ms": round(statistics.fmean(samples_ms), 4),
        "median_ms": round(statistics.median(samples_ms), 4),
        "p95_ms": round(ordered[p95_index], 4),
    }


def _speedup(baseline: dict[str, float], optimized: dict[str, float]) -> float:
    return round(baseline["mean_ms"] / optimized["mean_ms"], 4)


def _training_ms(result: dict[str, dict[str, float]]) -> float:
    return round(result["forward"]["mean_ms"] + result["backward"]["mean_ms"], 4)


def _measure_cuda(
    operation: Callable[[], object],
    *,
    warmup: int,
    iterations: int,
    before: Callable[[], None] | None = None,
) -> dict[str, float]:
    for _ in range(warmup):
        if before is not None:
            before()
        result = operation()
        del result
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        if before is not None:
            before()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
        del result
    return _timing_summary(samples)


def _measure_backward(
    prepare_loss: Callable[[], torch.Tensor],
    clear_gradients: Callable[[], None],
    *,
    warmup: int,
    iterations: int,
) -> dict[str, float]:
    for _ in range(warmup):
        clear_gradients()
        loss = prepare_loss()
        loss.backward()
    torch.cuda.synchronize()

    samples = []
    for _ in range(iterations):
        clear_gradients()
        loss = prepare_loss()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        loss.backward()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return _timing_summary(samples)


def _clear_cuda() -> None:
    gc.collect()
    torch.cuda.empty_cache()


class _DelayedPinnedStore(LocalFeatureStore):
    """One-shot pinned reads with a repeatable backend materialization delay."""

    def __init__(self, store_id: str, delay_s: float) -> None:
        super().__init__(store_id, retain_on_release=True)
        self.delay_s = delay_s

    def get(self, sample_ref, *, device="cpu", names=None, pin_memory=False):
        time.sleep(self.delay_s)
        tensors, handle = super().get(
            sample_ref,
            device=device,
            names=names,
            pin_memory=pin_memory,
        )
        return tensors, handle


def _loader_run(
    *,
    device: str,
    warmup: int,
    iterations: int,
    sequence_length: int,
    hidden_size: int,
    delay_s: float,
    consumer_delay_s: float,
    prefetch: bool,
) -> dict[str, float]:
    count = warmup + iterations
    store = _DelayedPinnedStore(f"bench-loader-{int(prefetch)}", delay_s)
    template = {
        "input_ids": torch.arange(sequence_length).view(1, -1),
        "loss_mask": torch.ones(1, sequence_length),
        "hidden_states": torch.randn(
            1, sequence_length, hidden_size, dtype=torch.bfloat16
        ),
    }
    refs = []
    for index in range(count):
        source = {
            name: tensor.clone().pin_memory() for name, tensor in template.items()
        }
        refs.append(
            store.put(
                source,
                sample_id=f"sample-{index}",
                metadata={
                    "run_id": "benchmark",
                    "strategy": "dflash",
                    "num_tokens": sequence_length,
                },
            )
        )
    queue = SampleRefQueue()
    queue.put(refs)
    strategy = resolve_strategy("dflash")
    loader = FeatureDataLoader(
        store,
        queue,
        batch_size=1,
        collate_fn=strategy.make_online_collate(),
        device=device,
        feature_names=sorted(strategy.required_features),
        clone_on_fetch=False,
        drop_last=False,
        strategy="dflash",
        gc_interval_s=None,
        prefetch_batches=2 if prefetch else 0,
        prefetch_to_device=prefetch,
    )
    iterator = iter(loader)
    for _ in range(warmup):
        next(iterator)
        time.sleep(consumer_delay_s)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        next(iterator)
        time.sleep(consumer_delay_s)
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    next(iterator, None)  # ACK the final yielded batch and drain the worker.

    bytes_per_batch = sum(
        tensor.numel() * tensor.element_size() for tensor in template.values()
    )
    return {
        "batches_s": round(iterations / elapsed, 4),
        "effective_gb_s": round(bytes_per_batch * iterations / elapsed / 1e9, 4),
        "mean_ms": round(elapsed * 1000.0 / iterations, 4),
    }


def benchmark_loader(args) -> dict[str, object]:
    baseline = _loader_run(
        device=args.device,
        warmup=args.warmup,
        iterations=args.loader_iterations,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        delay_s=args.materialization_delay_ms / 1000.0,
        consumer_delay_s=args.consumer_delay_ms / 1000.0,
        prefetch=False,
    )
    prefetched = _loader_run(
        device=args.device,
        warmup=args.warmup,
        iterations=args.loader_iterations,
        sequence_length=args.sequence_length,
        hidden_size=args.hidden_size,
        delay_s=args.materialization_delay_ms / 1000.0,
        consumer_delay_s=args.consumer_delay_ms / 1000.0,
        prefetch=True,
    )
    return {
        "baseline": baseline,
        "prefetch_h2d": prefetched,
        "speedup": _speedup(baseline, prefetched),
        "steady_batches": args.loader_iterations,
        "materialization_delay_ms": args.materialization_delay_ms,
        "consumer_delay_ms": args.consumer_delay_ms,
    }


def benchmark_kv_projection(args) -> dict[str, object]:
    device = torch.device(args.device)
    output_size = args.hidden_size // 4
    k_proj = torch.nn.Linear(args.hidden_size, output_size, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    v_proj = torch.nn.Linear(args.hidden_size, output_size, bias=False).to(
        device=device, dtype=torch.bfloat16
    )
    target = torch.randn(
        args.batch_size,
        args.context_length,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    noise = torch.randn(
        args.batch_size,
        args.draft_length,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )

    def split():
        return (
            torch.cat((k_proj(target), k_proj(noise)), dim=1),
            torch.cat((v_proj(target), v_proj(noise)), dim=1),
        )

    def combined():
        values = torch.cat((target, noise), dim=1)
        return k_proj(values), v_proj(values)

    def clear():
        k_proj.zero_grad(set_to_none=True)
        v_proj.zero_grad(set_to_none=True)
        target.grad = None
        noise.grad = None

    split_forward = _measure_cuda(split, warmup=args.warmup, iterations=args.iterations)
    combined_forward = _measure_cuda(
        combined, warmup=args.warmup, iterations=args.iterations
    )
    split_backward = _measure_backward(
        lambda: sum(output.float().square().mean() for output in split()),
        clear,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    combined_backward = _measure_backward(
        lambda: sum(output.float().square().mean() for output in combined()),
        clear,
        warmup=args.warmup,
        iterations=args.iterations,
    )
    split_result = {"forward": split_forward, "backward": split_backward}
    combined_result = {
        "forward": combined_forward,
        "backward": combined_backward,
    }
    return {
        "split": split_result,
        "combined": combined_result,
        "split_training_ms": _training_ms(split_result),
        "combined_training_ms": _training_ms(combined_result),
        "training_speedup": round(
            _training_ms(split_result) / _training_ms(combined_result), 4
        ),
        "forward_speedup": _speedup(split_forward, combined_forward),
        "backward_speedup": _speedup(split_backward, combined_backward),
    }


def _draft_config(args) -> Qwen3Config:
    head_dim = args.hidden_size // args.attention_heads
    config = Qwen3Config(
        hidden_size=args.hidden_size,
        intermediate_size=args.intermediate_size,
        num_hidden_layers=1,
        num_attention_heads=args.attention_heads,
        num_key_value_heads=args.kv_heads,
        head_dim=head_dim,
        vocab_size=1024,
        max_position_embeddings=args.context_length + args.draft_length + 32,
        rms_norm_eps=1e-6,
        attention_dropout=0.0,
    )
    config.block_size = 4
    config.num_target_layers = 4
    config.dflash_config = {"mask_token_id": 0, "target_layer_ids": [1]}
    config._attn_implementation = "sdpa"
    return config


def benchmark_draft(args) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(31)
    model = DFlashDraftModel(_draft_config(args)).to(
        device=device, dtype=torch.bfloat16
    )
    noise = torch.randn(
        args.batch_size,
        args.draft_length,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    target = torch.randn(
        args.batch_size,
        args.context_length,
        args.hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    positions = torch.arange(
        args.context_length + args.draft_length, device=device
    ).expand(args.batch_size, -1)

    def forward():
        return model(
            position_ids=positions,
            noise_embedding=noise,
            target_hidden=target,
            attention_mask=None,
        )

    torch.cuda.reset_peak_memory_stats(device)
    forward_result = _measure_cuda(
        forward, warmup=args.warmup, iterations=args.iterations
    )
    backward_result = _measure_backward(
        lambda: forward().float().square().mean(),
        lambda: model.zero_grad(set_to_none=True),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    forward = None
    model = noise = target = positions = None
    _clear_cuda()
    return {
        "forward": forward_result,
        "backward": backward_result,
        "peak_memory_mb": round(peak_mb, 2),
    }


def _benchmark_linear_ce_backend(args, backend: str) -> dict[str, object]:
    device = torch.device(args.device)
    torch.manual_seed(41)
    hidden = torch.randn(
        args.ce_tokens,
        args.ce_hidden_size,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        args.vocab_size,
        args.ce_hidden_size,
        device=device,
        dtype=torch.bfloat16,
    )
    targets = torch.randint(args.vocab_size, (args.ce_tokens,), device=device)
    token_weights = torch.linspace(0.0, 1.5, args.ce_tokens, device=device)

    def forward():
        if backend == "liger":
            return frozen_linear_cross_entropy(hidden, weight, targets)
        logits = hidden @ weight.t()
        return (
            F.cross_entropy(logits, targets, reduction="none"),
            (logits.argmax(dim=-1) == targets).float(),
        )

    def prepare_loss():
        loss, _accuracy = forward()
        return (loss * token_weights).sum()

    torch.cuda.reset_peak_memory_stats(device)
    forward_result = _measure_cuda(
        forward, warmup=args.warmup, iterations=args.iterations
    )
    backward_result = _measure_backward(
        prepare_loss,
        lambda: setattr(hidden, "grad", None),
        warmup=args.warmup,
        iterations=args.iterations,
    )
    peak_mb = torch.cuda.max_memory_allocated(device) / 1024**2
    prepare_loss = forward = None
    hidden = weight = targets = token_weights = None
    _clear_cuda()
    return {
        "forward": forward_result,
        "backward": backward_result,
        "peak_memory_mb": round(peak_mb, 2),
    }


def benchmark_linear_ce(args) -> dict[str, object]:
    torch_result = _benchmark_linear_ce_backend(args, "torch")
    liger_result = _benchmark_linear_ce_backend(args, "liger")
    return {
        "torch": torch_result,
        "liger": liger_result,
        "torch_training_ms": _training_ms(torch_result),
        "liger_training_ms": _training_ms(liger_result),
        "training_speedup": round(
            _training_ms(torch_result) / _training_ms(liger_result), 4
        ),
        "forward_speedup": _speedup(torch_result["forward"], liger_result["forward"]),
        "backward_speedup": _speedup(
            torch_result["backward"], liger_result["backward"]
        ),
        "peak_memory_ratio": round(
            torch_result["peak_memory_mb"] / liger_result["peak_memory_mb"], 4
        ),
    }


def _benchmark_optimizer_backend(args, backend: str) -> dict[str, float]:
    device = torch.device(args.device)
    model = torch.nn.Sequential(
        *[
            torch.nn.Linear(
                args.optimizer_size,
                args.optimizer_size,
                bias=False,
                device=device,
                dtype=torch.bfloat16,
            )
            for _ in range(args.optimizer_layers)
        ]
    )
    optimizer = BF16Optimizer(
        model,
        lr=1e-3,
        max_grad_norm=0.25,
        warmup_ratio=0.0,
        total_steps=args.warmup + args.iterations + 1,
        adamw_backend=backend,
    )
    gradients = [torch.randn_like(parameter) for parameter in model.parameters()]

    def assign_gradients():
        for parameter, gradient in zip(model.parameters(), gradients):
            parameter.grad = gradient

    result = _measure_cuda(
        optimizer.step,
        warmup=args.warmup,
        iterations=args.iterations,
        before=assign_gradients,
    )
    assign_gradients = None
    optimizer = model = gradients = None
    _clear_cuda()
    return result


def benchmark_optimizer(args) -> dict[str, object]:
    torch_result = _benchmark_optimizer_backend(args, "torch")
    fused_result = _benchmark_optimizer_backend(args, "fused")
    return {
        "torch": torch_result,
        "fused": fused_result,
        "speedup": _speedup(torch_result, fused_result),
    }


def parse_args(argv: Iterable[str] | None = None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--loader-iterations", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=1024)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument("--draft-length", type=int, default=128)
    parser.add_argument("--hidden-size", type=int, default=1024)
    parser.add_argument("--intermediate-size", type=int, default=3072)
    parser.add_argument("--attention-heads", type=int, default=16)
    parser.add_argument("--kv-heads", type=int, default=4)
    parser.add_argument("--ce-hidden-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=151936)
    parser.add_argument("--ce-tokens", type=int, default=8192)
    parser.add_argument("--optimizer-size", type=int, default=2048)
    parser.add_argument("--optimizer-layers", type=int, default=2)
    parser.add_argument("--materialization-delay-ms", type=float, default=2.0)
    parser.add_argument("--consumer-delay-ms", type=float, default=2.0)
    parser.add_argument(
        "--components",
        nargs="+",
        choices=("all", "loader", "kv", "draft", "linear_ce", "optimizer"),
        default=("all",),
    )
    args = parser.parse_args(argv)
    if args.warmup < 1 or args.iterations < 1 or args.loader_iterations < 1:
        parser.error("warmup and iteration counts must be positive")
    if args.materialization_delay_ms < 0 or args.consumer_delay_ms < 0:
        parser.error("synthetic delays must be non-negative")
    if torch.device(args.device).type != "cuda" or not torch.cuda.is_available():
        parser.error("the critical-path benchmark requires an available CUDA device")
    return args


def main() -> None:
    args = parse_args()
    torch.cuda.set_device(torch.device(args.device))
    components = set(args.components)
    if "all" in components:
        components = {"loader", "kv", "draft", "linear_ce", "optimizer"}

    runners = {
        "loader": benchmark_loader,
        "kv": benchmark_kv_projection,
        "draft": benchmark_draft,
        "linear_ce": benchmark_linear_ce,
        "optimizer": benchmark_optimizer,
    }
    results = {
        "environment": {
            "device": args.device,
            "gpu": torch.cuda.get_device_name(torch.device(args.device)),
            "torch": torch.__version__,
            "warmup": args.warmup,
            "iterations": args.iterations,
        },
        "configuration": {
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "draft_length": args.draft_length,
            "hidden_size": args.hidden_size,
            "ce_hidden_size": args.ce_hidden_size,
            "vocab_size": args.vocab_size,
            "ce_tokens": args.ce_tokens,
            "materialization_delay_ms": args.materialization_delay_ms,
            "consumer_delay_ms": args.consumer_delay_ms,
        },
        "results": {},
    }
    for name in runners:
        if name in components:
            results["results"][name] = runners[name](args)
    print(json.dumps(results, sort_keys=True))


if __name__ == "__main__":
    main()
