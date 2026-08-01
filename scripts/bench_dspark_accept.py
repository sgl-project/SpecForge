"""Accept-length benchmark: dflash/benchmark.py sglang server path with all
DeepSpec workloads adapted in (prompt formats copied verbatim from
DeepSpec/eval_datasets/convert_eval_datasets_to_jsonl.py).

Measures ONLY spec_accept_length (no baseline, no speedup). Sampling and stop
ids follow the NDA Inkling serving contract.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

random.seed(42)

CACHE_DIR = Path(os.environ.get("BENCH_CACHE", "bench_cache"))
REASONING_SUFFIX = (
    "\nPlease reason step by step, and put your final answer within \\boxed{}."
)

# Default stop ids follow the Inkling serving contract (its tokenizer
# defines no eos; 200001 is a legal mid-stream header and must NOT be a
# stop id). Override with --stop-token-ids for other targets.
STOP_TOKEN_IDS = [200006, 200000, 200002, 200003]


def fmt_math(row):
    return [row["problem"] + REASONING_SUFFIX]


def fmt_livecodebench(row):
    if "messages" in row and row["messages"]:
        turns = [m["content"] for m in row["messages"] if m["role"] == "user"]
        if turns:
            return turns
    question = row["question_content"]
    starter_code = row.get("starter_code") or ""
    code_template = starter_code if starter_code else "# YOUR CODE HERE"
    format_label = (
        "Use the following code structure:"
        if starter_code
        else "Write your code in the following format:"
    )
    return [
        "You are an expert Python programmer. You will be given a question "
        "(problem specification) and will generate a correct Python program "
        "that matches the specification and passes all tests. You will NOT "
        "return anything except for the program\n\n"
        "### Question:\n" + question + "\n\n"
        "### Format: " + format_label + "\n"
        "```python\n" + code_template + "\n```\n\n"
        "### Answer: (use the provided format with backticks)"
    ]


def fmt_alpaca(row):
    if row["input"]:
        return [row["instruction"] + "\n\nInput:\n" + row["input"]]
    return [row["instruction"]]


DATASETS = {
    "gsm8k": dict(
        load_args=("openai/gsm8k", "main"),
        split="test",
        fmt=lambda r: [r["question"] + REASONING_SUFFIX],
    ),
    "math500": dict(load_args=("HuggingFaceH4/MATH-500",), split="test", fmt=fmt_math),
    "aime24": dict(load_args=("HuggingFaceH4/aime_2024",), split="train", fmt=fmt_math),
    "aime25": dict(load_args=("MathArena/aime_2025",), split="train", fmt=fmt_math),
    "alpaca": dict(load_args=("tatsu-lab/alpaca",), split="train", fmt=fmt_alpaca),
    "mt-bench": dict(
        load_args=("HuggingFaceH4/mt_bench_prompts",),
        split="train",
        fmt=lambda r: list(r["prompt"]),
    ),
    "humaneval": dict(
        load_args=("openai/openai_humaneval",),
        split="test",
        fmt=lambda r: [
            "Write a solution to the following problem and make "
            "sure that it passes the tests:\n```python\n" + r["prompt"] + "\n```"
        ],
    ),
    "mbpp": dict(
        load_args=("google-research-datasets/mbpp", "sanitized"),
        split="test",
        fmt=lambda r: [r["prompt"]],
    ),
    "lbpp": dict(
        load_args=("CohereLabs/lbpp",),
        split="test",
        fmt=lambda r: [r["instruction"]],
        parquet_files=("python/test.parquet",),
    ),
    "swe-bench": dict(
        load_args=("princeton-nlp/SWE-bench_Lite",),
        split="test",
        fmt=lambda r: [
            "Problem Statement:\n"
            + r["problem_statement"]
            + "\nPlease fix the issue described above."
        ],
    ),
    "livecodebench": dict(
        load_args=("livecodebench/code_generation_lite",),
        split="test",
        fmt=fmt_livecodebench,
        jsonl_files=(
            "test.jsonl",
            "test2.jsonl",
            "test3.jsonl",
            "test4.jsonl",
            "test5.jsonl",
            "test6.jsonl",
        ),
    ),
    "arena-hard-v2": dict(loader="arena_hard"),
}


def _load_rows(name):
    cfg = DATASETS[name]
    if cfg.get("loader") == "arena_hard":
        from huggingface_hub import hf_hub_download

        src = hf_hub_download(
            repo_id="lmarena-ai/arena-hard-auto",
            filename="data/arena-hard-v2.0/question.jsonl",
            repo_type="dataset",
        )
        with open(src, encoding="utf-8") as f:
            return [{"turns": [json.loads(line)["prompt"]]} for line in f]
    if cfg.get("jsonl_files"):
        from huggingface_hub import hf_hub_download

        rows = []
        for fn in cfg["jsonl_files"]:
            src = hf_hub_download(
                repo_id=cfg["load_args"][0], filename=fn, repo_type="dataset"
            )
            with open(src, encoding="utf-8") as f:
                rows.extend(json.loads(line) for line in f if line.strip())
        return [{"turns": cfg["fmt"](r)} for r in rows]
    if cfg.get("parquet_files"):
        import pandas as pd
        from huggingface_hub import hf_hub_download

        rows = []
        for fn in cfg["parquet_files"]:
            src = hf_hub_download(
                repo_id=cfg["load_args"][0], filename=fn, repo_type="dataset"
            )
            rows.extend(pd.read_parquet(src).to_dict("records"))
        return [{"turns": cfg["fmt"](r)} for r in rows]
    from datasets import load_dataset

    ds = load_dataset(*cfg["load_args"], split=cfg["split"])
    return [{"turns": cfg["fmt"](r)} for r in ds]


def load_and_cache(name):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / (name + ".jsonl")
    if not path.exists():
        rows = _load_rows(name)
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, path)
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def send_one(base_url, text, args):
    resp = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": text,
            "sampling_params": {
                "temperature": args.temperature,
                "top_p": args.top_p,
                # dflash/benchmark.py defaults top_k=1, which silently
                # turns any temperature into greedy decoding and inflates
                # agreement. -1 = full distribution; do not "clean this up".
                "top_k": -1,
                "max_new_tokens": args.max_new_tokens,
                "stop_token_ids": args.stop_token_ids,
            },
        },
        timeout=args.timeout_s,
    )
    resp.raise_for_status()
    out = resp.json()
    return out if isinstance(out, dict) else out[0]


def apply_chat_template(tokenizer, turns, reasoning_effort):
    kwargs = {
        "tokenize": False,
        "add_generation_prompt": True,
        "reasoning_effort": reasoning_effort,
    }
    try:
        return tokenizer.apply_chat_template(turns, **kwargs)
    except TypeError:
        kwargs.pop("reasoning_effort")
        return tokenizer.apply_chat_template(turns, **kwargs)


def run_workload(name, tokenizer, args):
    dataset = load_and_cache(name)
    rng = random.Random(42)
    rng.shuffle(dataset)

    total = args.num_prompts + args.warmup
    prompts = []
    for i in range(total):
        item = dataset[i % len(dataset)]
        prompts.append(
            apply_chat_template(
                tokenizer,
                [{"role": "user", "content": item["turns"][0]}],
                args.reasoning_effort,
            )
        )

    try:
        requests.get(args.base_url + "/flush_cache", timeout=60).raise_for_status()
    except Exception:
        pass

    if args.warmup:
        with ThreadPoolExecutor(max_workers=args.warmup) as pool:
            list(
                pool.map(
                    lambda p: send_one(args.base_url, p, args), prompts[: args.warmup]
                )
            )
    measured = prompts[args.warmup :]

    accept_lengths, total_tokens = [], 0
    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(send_one, args.base_url, p, args) for p in measured]
        for fut in tqdm(as_completed(futures), total=len(measured), desc=name):
            out = fut.result()
            meta = out.get("meta_info", {}) or {}
            total_tokens += int(meta.get("completion_tokens", 0))
            if "spec_accept_length" in meta:
                try:
                    accept_lengths.append(float(meta["spec_accept_length"]))
                except (TypeError, ValueError):
                    pass
    elapsed = time.perf_counter() - start

    mean_accept = statistics.mean(accept_lengths) if accept_lengths else float("nan")
    result = {
        "workload": name,
        "num_prompts": len(measured),
        "with_accept_meta": len(accept_lengths),
        "mean_accept_length": mean_accept,
        "output_tokens": total_tokens,
        "elapsed_s": round(elapsed, 1),
    }
    print(json.dumps(result))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workloads", nargs="+", default=list(DATASETS.keys()))
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--num-prompts", type=int, default=128)
    ap.add_argument("--warmup", type=int, default=8)
    ap.add_argument("--concurrency", type=int, default=32)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--reasoning-effort", type=float, default=0.99)
    ap.add_argument("--timeout-s", type=int, default=3600)
    ap.add_argument(
        "--stop-token-ids",
        type=int,
        nargs="+",
        default=STOP_TOKEN_IDS,
        help="Stop token ids (default: Inkling serving contract).",
    )
    ap.add_argument("--output-json", default="bench_cache/accept_results.json")
    ap.add_argument("--trust-remote-code", action="store_true")
    args = ap.parse_args()

    if args.num_prompts <= 0:
        ap.error("--num-prompts must be positive")
    if args.warmup < 0:
        ap.error("--warmup must be non-negative")
    if args.concurrency <= 0:
        ap.error("--concurrency must be positive")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer,
        trust_remote_code=args.trust_remote_code,
    )

    results = []
    for name in args.workloads:
        results.append(run_workload(name, tokenizer, args))
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

    print(
        "\n==== ACCEPT LENGTH SUMMARY (temp=%.1f top_p=%.2f block=7) ===="
        % (args.temperature, args.top_p)
    )
    for r in results:
        print(
            "%-16s %.4f  (n=%d)"
            % (r["workload"], r["mean_accept_length"], r["with_accept_meta"])
        )


if __name__ == "__main__":
    main()
