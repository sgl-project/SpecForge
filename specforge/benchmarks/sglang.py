"""Benchmark a running SGLang server.

Request scheduling is adapted from z-lab/dflash's MIT benchmark:
https://github.com/z-lab/dflash/blob/main/dflash/benchmark.py

The runner is speculative-algorithm agnostic and consumes SGLang's optional
speculative-decoding metadata when the server returns it.
"""

from __future__ import annotations

import json
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from typing import Any, Optional

DATASETS: dict[str, dict[str, Any]] = {
    "gsm8k": {
        "load_args": ("openai/gsm8k", "main"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['question']}\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        ),
    },
    "math500": {
        "load_args": ("HuggingFaceH4/MATH-500",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            f"{row['problem']}\nPlease reason step by step, and put your final "
            "answer within \\boxed{}."
        ),
    },
    "humaneval": {
        "load_args": ("openai/openai_humaneval",),
        "load_kwargs": {"split": "test"},
        "format": lambda row: (
            "Write a solution to the following problem and make sure that it "
            f"passes the tests:\n```python\n{row['prompt']}\n```"
        ),
    },
    "mbpp": {
        "load_args": ("google-research-datasets/mbpp", "sanitized"),
        "load_kwargs": {"split": "test"},
        "format": lambda row: row["prompt"],
    },
    "mt-bench": {
        "load_args": ("HuggingFaceH4/mt_bench_prompts",),
        "load_kwargs": {"split": "train"},
        "format": lambda row: row["prompt"],
        "multi_turn": True,
    },
}


@dataclass(frozen=True)
class BenchmarkResult:
    backend: str
    dataset: str
    samples: int
    output_tokens: int
    latency_seconds: float
    throughput_tokens_per_second: float
    average_acceptance_length: Optional[float] = None
    spec_verify_count: Optional[int] = None


def _load_prompts(name: str, max_samples: Optional[int]) -> list[list[str]]:
    from datasets import load_dataset

    if max_samples is not None and max_samples <= 0:
        raise ValueError("--max-samples must be positive")
    descriptor = DATASETS[name]
    dataset = load_dataset(
        *descriptor["load_args"],
        **descriptor["load_kwargs"],
    )
    prompts: list[list[str]] = []
    for row in dataset:
        formatted = descriptor["format"](row)
        turns = list(formatted) if descriptor.get("multi_turn") else [formatted]
        prompts.append(turns)
    if not prompts:
        raise ValueError(f"dataset {name!r} did not contain any prompts")
    if max_samples is not None and len(prompts) > max_samples:
        random.Random(42).shuffle(prompts)
        prompts = prompts[:max_samples]
    return prompts


def _apply_chat_template(tokenizer, messages, enable_thinking: bool) -> str:
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": enable_thinking,
    }
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking")
        return tokenizer.apply_chat_template(messages, **kwargs)


def _send_sglang(args, prompt: str) -> dict[str, Any]:
    import requests

    response = requests.post(
        args.base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                "top_k": args.top_k,
                "max_new_tokens": args.max_new_tokens,
            },
        },
        timeout=args.timeout_seconds,
    )
    response.raise_for_status()
    payload = response.json()
    return payload[0] if isinstance(payload, list) else payload


def _run_sglang(args) -> BenchmarkResult:
    import requests
    from transformers import AutoTokenizer

    if args.concurrency <= 0:
        raise ValueError("--concurrency must be positive")
    if args.num_prompts <= 0:
        raise ValueError("--num-prompts must be positive")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=args.trust_remote_code,
    )
    dataset = _load_prompts(args.dataset, args.max_samples)
    prompt_count = args.num_prompts
    warmup_count = args.concurrency
    prompts = [
        _apply_chat_template(
            tokenizer,
            [{"role": "user", "content": dataset[index % len(dataset)][0]}],
            args.enable_thinking,
        )
        for index in range(prompt_count + warmup_count)
    ]

    try:
        requests.get(
            args.base_url.rstrip("/") + "/flush_cache",
            timeout=min(args.timeout_seconds, 60),
        ).raise_for_status()
    except requests.RequestException:
        print("Warning: /flush_cache failed. Continuing.")

    with ThreadPoolExecutor(max_workers=warmup_count) as executor:
        list(
            executor.map(
                lambda prompt: _send_sglang(args, prompt), prompts[:warmup_count]
            )
        )
    prompts = prompts[warmup_count:]

    total_tokens = 0
    verify_count = 0
    acceptance_lengths: list[float] = []
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [executor.submit(_send_sglang, args, prompt) for prompt in prompts]
        for future in as_completed(futures):
            output = future.result()
            metadata = output.get("meta_info", {}) or {}
            total_tokens += int(metadata.get("completion_tokens", 0))
            verify_count += int(metadata.get("spec_verify_ct", 0))
            if metadata.get("spec_accept_length") is not None:
                try:
                    acceptance_lengths.append(float(metadata["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass
    elapsed = time.perf_counter() - start
    return BenchmarkResult(
        backend="sglang",
        dataset=args.dataset,
        samples=prompt_count,
        output_tokens=total_tokens,
        latency_seconds=elapsed,
        throughput_tokens_per_second=total_tokens / max(elapsed, 1e-12),
        average_acceptance_length=(
            statistics.fmean(acceptance_lengths) if acceptance_lengths else None
        ),
        spec_verify_count=verify_count or None,
    )


def _print_result(result: BenchmarkResult) -> None:
    print(f"Backend: {result.backend}")
    print(f"Dataset: {result.dataset} ({result.samples} completed prompts/turns)")
    print(f"Output throughput: {result.throughput_tokens_per_second:.2f} tok/s")
    if result.average_acceptance_length is not None:
        print(f"Average acceptance length: {result.average_acceptance_length:.3f}")
    if result.spec_verify_count is not None:
        print(f"Speculative verify count: {result.spec_verify_count}")


def run(args) -> int:
    random.seed(42)
    result = _run_sglang(args)
    _print_result(result)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            json.dump(asdict(result), output_file, indent=2, sort_keys=True)
            output_file.write("\n")
    return 0


__all__ = ["BenchmarkResult", "DATASETS", "run"]
