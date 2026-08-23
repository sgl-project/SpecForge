#!/usr/bin/env python3
"""Accept-length probe over a preformatted-prompt jsonl against a running
sglang DSPARK server.

Rows must carry the full rendered prompt in a ``text`` field (e.g. the
0813-traffic test set produced by transform_traffic_recordings.py); prompts
are sent raw to ``/generate`` — no chat template. Reports the average
speculative acceptance length from ``meta_info``.

    python3 scripts/bench_accept_jsonl.py \
        --jsonl /personal/dataset/prod/dsv4_flash_0813_32k_test.jsonl \
        --base-url http://127.0.0.1:31000 --num-prompts 64 --output-json out.json
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed


def send(base_url: str, prompt: str, max_new_tokens: int, timeout: int):
    import requests

    response = requests.post(
        base_url.rstrip("/") + "/generate",
        json={
            "text": prompt,
            "sampling_params": {
                "temperature": 0.0,
                "max_new_tokens": max_new_tokens,
            },
        },
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload[0] if isinstance(payload, list) else payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:31000")
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--max-prompt-chars", type=int, default=None,
                        help="skip rows with longer rendered prompts")
    parser.add_argument("--timeout-seconds", type=int, default=3600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args()

    rows = []
    with open(args.jsonl, encoding="utf-8") as stream:
        for line in stream:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text")
            if not text:
                continue
            if args.max_prompt_chars and len(text) > args.max_prompt_chars:
                continue
            rows.append(text)
    if not rows:
        raise SystemExit("no usable rows (need a non-empty 'text' field)")
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.num_prompts]
    print(f"probing {len(rows)} prompts from {args.jsonl}")

    results = []
    errors = 0
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(send, args.base_url, text, args.max_new_tokens,
                        args.timeout_seconds)
            for text in rows
        ]
        for future in as_completed(futures):
            try:
                payload = future.result()
            except Exception as exc:  # noqa: BLE001
                errors += 1
                print(f"  request failed: {exc}")
                continue
            meta = payload.get("meta_info", {})
            accept = meta.get("spec_accept_length")
            verify = meta.get("spec_verify_ct")
            tokens = meta.get("completion_tokens")
            if accept is not None:
                results.append(
                    {"accept": accept, "verify": verify, "tokens": tokens}
                )
    if not results:
        raise SystemExit(f"no successful spec responses ({errors} errors)")

    total_tokens = sum(r["tokens"] or 0 for r in results)
    total_verify = sum(r["verify"] or 0 for r in results)
    per_request = sum(r["accept"] for r in results) / len(results)
    token_weighted = (total_tokens / total_verify) if total_verify else None
    print(f"completed: {len(results)}  errors: {errors}")
    print(f"average acceptance length (per-request mean): {per_request:.3f}")
    if token_weighted is not None:
        print(f"average acceptance length (token-weighted): {token_weighted:.3f}")
    summary = {
        "jsonl": args.jsonl,
        "num_completed": len(results),
        "num_errors": errors,
        "average_acceptance_length": per_request,
        "token_weighted_acceptance_length": token_weighted,
        "max_new_tokens": args.max_new_tokens,
    }
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as stream:
            json.dump(summary, stream, indent=2)
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()
