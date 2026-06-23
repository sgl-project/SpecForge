#!/usr/bin/env python3
"""Collect target-model rollouts with strict token accounting.

This is intentionally narrower than ``regenerate_train_data.py``: it is for
on-policy Qwen-style rollout collection before DFlash activation caching.
"""

import argparse
import json
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

from openai import OpenAI
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser(description="Collect Qwen rollout JSONL")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--tokenizer-path", type=str, default=None)
    parser.add_argument("--input-file-path", type=str, required=True)
    parser.add_argument("--output-file-path", type=str, required=True)
    parser.add_argument("--server-address", type=str, nargs="+", required=True)
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--max-total-length", type=int, default=4096)
    parser.add_argument("--max-response-tokens", type=int, default=1536)
    parser.add_argument("--min-response-budget", type=int, default=32)
    parser.add_argument("--eos-margin", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--timeout-s", type=int, default=3600)
    parser.add_argument(
        "--keep-truncated",
        action="store_true",
        help="Keep samples that hit the response cap or exceed max_total_length.",
    )
    return parser.parse_args()


def count_chat_tokens(tokenizer, messages: list[dict[str, Any]], add_generation_prompt: bool) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=add_generation_prompt,
        )
    )


def build_query_kwargs(args, messages, max_tokens: int) -> dict:
    kwargs = {
        "model": args.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": args.temperature,
        "stream": False,
        "timeout": args.timeout_s,
    }
    if args.top_p is not None:
        kwargs["top_p"] = args.top_p
    extra_body = {}
    if args.top_k is not None:
        extra_body["top_k"] = args.top_k
    if extra_body:
        kwargs["extra_body"] = extra_body
    return kwargs


def collect_one(args, tokenizer, server_address: str, source: dict[str, Any]) -> dict[str, Any]:
    client = OpenAI(base_url=f"http://{server_address}/v1", api_key="None")
    source_id = source.get("id")
    messages = source.get("conversations", [])
    if not messages:
        return {"id": source_id, "status": "error", "error": "empty conversations"}
    if messages[0].get("role") == "assistant":
        return {
            "id": source_id,
            "status": "error",
            "error": "conversation starts with assistant",
        }

    regenerated: list[dict[str, Any]] = []
    turn_stats = []
    for message in messages:
        role = message.get("role")
        if role == "system":
            regenerated.append({"role": "system", "content": message.get("content", "")})
            continue
        if role == "assistant":
            continue
        if role != "user":
            return {
                "id": source_id,
                "status": "error",
                "error": f"invalid role: {role}",
            }

        regenerated.append({"role": "user", "content": message.get("content", "")})
        prompt_tokens = count_chat_tokens(tokenizer, regenerated, add_generation_prompt=True)
        response_budget = args.max_total_length - prompt_tokens - args.eos_margin
        if response_budget < args.min_response_budget:
            return {
                "id": source_id,
                "status": "skipped",
                "error": "insufficient response budget",
                "meta": {
                    "prompt_tokens": prompt_tokens,
                    "max_total_length": args.max_total_length,
                    "min_response_budget": args.min_response_budget,
                },
            }

        max_tokens = min(args.max_response_tokens, response_budget)
        try:
            resp = client.chat.completions.create(
                **build_query_kwargs(args, regenerated, max_tokens)
            )
        except Exception as exc:
            return {"id": source_id, "status": "error", "error": str(exc)}

        choice = resp.choices[0]
        response_text = choice.message.content or ""
        regenerated.append({"role": "assistant", "content": response_text})
        total_tokens = count_chat_tokens(tokenizer, regenerated, add_generation_prompt=False)
        usage = getattr(resp, "usage", None)
        completion_tokens = (
            getattr(usage, "completion_tokens", None)
            if usage is not None
            else None
        )
        turn_stats.append(
            {
                "prompt_tokens": prompt_tokens,
                "max_tokens": max_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens_after_turn": total_tokens,
                "finish_reason": getattr(choice, "finish_reason", None),
                "truncated": getattr(choice, "finish_reason", None) == "length"
                or total_tokens > args.max_total_length,
            }
        )

    truncated = any(turn["truncated"] for turn in turn_stats)
    result = {
        "id": source_id,
        "conversations": regenerated,
        "status": "success",
        "meta": {
            "target_model": args.model,
            "server_address": server_address,
            "max_total_length": args.max_total_length,
            "max_response_tokens": args.max_response_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "turns": turn_stats,
            "total_tokens": (
                turn_stats[-1]["total_tokens_after_turn"] if turn_stats else 0
            ),
            "truncated": truncated,
        },
    }
    if truncated and not args.keep_truncated:
        result["status"] = "skipped"
        result["error"] = "truncated"
    return result


def load_processed_ids(*paths: Path) -> set[Any]:
    processed = set()
    for path in paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "id" in row:
                    processed.add(row["id"])
    return processed


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path or args.model)
    output_path = Path(args.output_file_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    error_path = output_path.with_name(output_path.stem + "_error.jsonl")

    processed_ids = load_processed_ids(output_path, error_path) if args.resume else set()
    mode = "a" if args.resume and processed_ids else "w"

    total_lines = sum(1 for _ in open(args.input_file_path))
    if args.num_samples is not None:
        total_lines = min(total_lines, args.num_samples)
    print(
        f"Collecting up to {total_lines} rollout samples "
        f"({len(processed_ids)} ids already processed)"
    )

    with (
        open(args.input_file_path) as input_f,
        open(output_path, mode) as output_f,
        open(error_path, mode) as error_f,
        ThreadPoolExecutor(max_workers=args.concurrency * len(args.server_address)) as pool,
    ):
        futures = set()
        server_idx = 0
        submitted = 0
        completed = 0
        pbar = tqdm(total=total_lines, initial=min(len(processed_ids), total_lines), desc="rollouts")

        def submit(source: dict[str, Any]):
            nonlocal server_idx, submitted
            server = args.server_address[server_idx]
            server_idx = (server_idx + 1) % len(args.server_address)
            futures.add(pool.submit(collect_one, args, tokenizer, server, source))
            submitted += 1

        for line in input_f:
            if submitted + len(processed_ids) >= total_lines:
                break
            source = json.loads(line)
            if source.get("id") in processed_ids:
                continue
            submit(source)
            if len(futures) >= args.concurrency * len(args.server_address):
                done, futures = wait(futures, return_when=FIRST_COMPLETED)
                for fut in done:
                    row = fut.result()
                    handle = output_f if row.get("status") == "success" else error_f
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                    handle.flush()
                    completed += 1
                    pbar.update(1)

        while futures:
            done, futures = wait(futures, return_when=FIRST_COMPLETED)
            for fut in done:
                row = fut.result()
                handle = output_f if row.get("status") == "success" else error_f
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                completed += 1
                pbar.update(1)

        pbar.close()
        print(f"Rollout collection finished: new_rows={completed}")


if __name__ == "__main__":
    main()
