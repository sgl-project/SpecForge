#!/usr/bin/env python3
# coding=utf-8
"""Mimic user traffic against a live-capture SGLang server.

Reads ShareGPT-style conversations (``{"id", "conversations": [{"role",
"content"}, ...]}`` JSONL — the same file the driven disaggregated recipes use
as ``data.train_data_path``), builds a chat prompt up to the last assistant
turn, and POSTs plain ``/generate`` requests exactly like a real user: no
``spec_capture`` field, sampled decoding. With the server launched with
``--spec-capture-intake-url``, every request feeds drafter training.

Example:
    python scripts/online_live/live_traffic_mimic.py \\
        --server-url http://127.0.0.1:30000 \\
        --config examples/configs/qwen3-4b-dspark-live.yaml
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument(
        "--config", default=None, help="run YAML; supplies the tokenizer default"
    )
    parser.add_argument(
        "--data",
        default="./cache/dataset/sharegpt_train.jsonl",
        help="ShareGPT JSONL path",
    )
    parser.add_argument(
        "--tokenizer", default=None, help="HF tokenizer id/path (or use --config)"
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--qps", type=float, default=0.0, help="0 = unthrottled")
    parser.add_argument("--max-requests", type=int, default=0, help="0 = all")
    # Matches data.max_length in the disagg recipes (the intake token cap).
    parser.add_argument("--max-prompt-tokens", type=int, default=3072)
    # Temporarily 1 so live runs compare directly against the driven
    # disaggregated recipe (prefill-only capture).
    parser.add_argument("--max-new-tokens", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--loop", action="store_true", help="repeat the dataset")
    parser.add_argument("--log-interval-s", type=float, default=10.0)
    return parser.parse_args()


def iter_prompts(args, tokenizer):
    """Yield token-id prompts: the full conversation through the last assistant turn.

    Matches the driven disagg-online producer, which prefills whole
    conversations (assistant replies included) with max_new_tokens=1.
    """
    rng = random.Random(args.seed)
    with open(args.data, encoding="utf-8") as stream:
        records = [json.loads(line) for line in stream if line.strip()]
    rng.shuffle(records)
    for record in itertools.cycle(records) if args.loop else records:
        turns = [
            {"role": turn["role"], "content": turn["content"]}
            for turn in record.get("conversations", [])
            if turn.get("role") in ("system", "user", "assistant")
        ]
        while turns and turns[-1]["role"] != "assistant":
            turns.pop()
        if not turns:
            continue
        text = tokenizer.apply_chat_template(
            turns, add_generation_prompt=False, tokenize=False
        )
        input_ids = tokenizer(text, add_special_tokens=False).input_ids
        if len(input_ids) > args.max_prompt_tokens:
            continue
        yield input_ids


def main() -> None:
    from transformers import AutoTokenizer

    args = parse_args()
    if args.tokenizer is None:
        if args.config is None:
            raise SystemExit("pass --tokenizer or --config")
        import yaml

        with open(args.config, encoding="utf-8") as stream:
            args.tokenizer = yaml.safe_load(stream)["model"]["target_model_path"]
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    url = f"{args.server_url.rstrip('/')}/generate"
    stats = {"sent": 0, "ok": 0, "error": 0}
    lock = threading.Lock()
    start = time.monotonic()

    def send(input_ids) -> None:
        try:
            body = json.dumps(
                {
                    "input_ids": list(input_ids),
                    "sampling_params": {
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new_tokens,
                    },
                }
            ).encode("utf-8")
            request = Request(
                url, data=body, headers={"Content-Type": "application/json"}
            )
            with urlopen(request, timeout=args.timeout_s):
                outcome = "ok"
        except Exception as exc:  # noqa: BLE001 — a user request may just fail
            outcome = "error"
            print(f"request failed: {exc}", flush=True)
        with lock:
            stats[outcome] += 1

    last_log = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        for input_ids in iter_prompts(args, tokenizer):
            if args.max_requests and stats["sent"] >= args.max_requests:
                break
            if args.qps > 0:
                target = start + stats["sent"] / args.qps
                delay = target - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            executor.submit(send, input_ids)
            stats["sent"] += 1
            now = time.monotonic()
            if now - last_log >= args.log_interval_s:
                rate = stats["sent"] / (now - start)
                print(f"{stats} ({rate:.2f} req/s)", flush=True)
                last_log = now
    print(f"done: {stats} in {time.monotonic() - start:.1f}s", flush=True)


if __name__ == "__main__":
    main()
