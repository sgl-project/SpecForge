"""Measure single-request multimodal DFlash serving on a fixed set of held-out images."""

import argparse
import base64
import hashlib
import json
import statistics
import time
from pathlib import Path


def measure_stream(events, incremental=False):
    """Exclude prefill and every token in the first emitted chunk from decode timing."""
    ids = []
    first_time = last_time = None
    first_count = previous_count = 0
    meta = {}
    for timestamp, chunk in events:
        if "error" in chunk:
            raise ValueError(f"SGLang error: {chunk['error']}")
        meta = chunk["meta_info"]
        if (meta.get("finish_reason") or {}).get("type") == "abort":
            raise ValueError(f"aborted generation: {meta['finish_reason']}")
        count = int(meta["completion_tokens"])
        if count < previous_count:
            raise ValueError("non-monotone completion count")
        if count > previous_count:
            if first_time is None:
                first_time, first_count = timestamp, count
            previous_count, last_time = count, timestamp
        if incremental:
            ids.extend(chunk.get("output_ids", []))
        elif chunk.get("output_ids"):
            ids = chunk["output_ids"]
    verifies = int(meta.get("spec_verify_ct", 0))
    if verifies <= 0 or not ids:
        raise ValueError("missing speculative verify count or output IDs from SGLang")
    if first_time is None or first_time == last_time:
        raise ValueError(
            "decode timing needs two distinct token emissions; check stream interval and reply length"
        )
    return {
        "completion_tokens": previous_count,
        "verify_count": verifies,
        "accept_length": previous_count / verifies,
        "decode_tokens": previous_count - first_count,
        "decode_seconds": last_time - first_time,
        "output_ids": ids,
    }


def compare_runs(baseline, mmflash):
    ids = [r["id"] for r in baseline]
    if not ids or len(ids) != len(set(ids)) or ids != [r["id"] for r in mmflash]:
        raise ValueError("comparison requires matching, unique held-out IDs in order")

    if any(
        a["request_sha256"] != b["request_sha256"] for a, b in zip(baseline, mmflash)
    ):
        raise ValueError(
            "the two arms used different prompts, images or sampling parameters"
        )

    def summarize(rows):
        return {
            "mean_accept_length": statistics.mean(r["accept_length"] for r in rows),
            "tokens_per_verify": sum(r["completion_tokens"] for r in rows)
            / sum(r["verify_count"] for r in rows),
            "client_decode_tok_s": sum(r["decode_tokens"] for r in rows)
            / sum(r["decode_seconds"] for r in rows),
        }

    before, after = summarize(baseline), summarize(mmflash)
    return {
        "samples": len(ids),
        "dflash": before,
        "mmflash": after,
        "delta_mean_accept_length": after["mean_accept_length"]
        - before["mean_accept_length"],
        "decode_speed_ratio": after["client_decode_tok_s"]
        / before["client_decode_tok_s"],
        "different_greedy_outputs": [
            a["id"]
            for a, b in zip(baseline, mmflash)
            if a["output_ids"] != b["output_ids"]
        ],
    }


def stream_events(response):
    for line in response.iter_lines(chunk_size=None):
        if line.startswith(b"data:"):
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            yield time.perf_counter(), json.loads(data)


def benchmark(args):
    import requests
    from transformers import AutoProcessor

    base = args.url.rstrip("/")
    info = requests.get(base + "/server_info", timeout=30)
    info.raise_for_status()
    server = info.json()
    if str(server.get("speculative_algorithm", "")).upper() != "DFLASH":
        raise ValueError("expected a DFLASH server")
    if Path(server["speculative_draft_model_path"]).resolve() != args.draft.resolve():
        raise ValueError("server is serving a different draft")
    processor = AutoProcessor.from_pretrained(server["model_path"])
    records = [json.loads(s) for s in args.input.read_text().splitlines() if s.strip()]
    if not records:
        raise ValueError("empty held-out set")
    results = []
    with requests.Session() as session:
        # Unmeasured warm-up, same prompt and disabled radix cache in both arms.
        for index, row in enumerate([records[0], *records]):
            question = (
                row["conversations"][0]["content"].removeprefix("<image>").strip()
            )
            messages = [
                {
                    "role": "user",
                    "content": [{"type": "image"}, {"type": "text", "text": question}],
                }
            ]
            prompt = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            image = base64.b64encode(Path(row["image"]).read_bytes()).decode("ascii")
            payload = {
                "text": prompt,
                "image_data": ["data:image/png;base64," + image],
                "stream": True,
                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
            }
            request_hash = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode()
            ).hexdigest()
            with session.post(
                base + "/generate", json=payload, stream=True, timeout=(10, 180)
            ) as response:
                response.raise_for_status()

                result = measure_stream(
                    stream_events(response),
                    incremental=server.get("incremental_streaming_output", False),
                )
                if index > 0:
                    results.append(
                        {"id": row["id"], "request_sha256": request_hash, **result}
                    )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    run = sub.add_parser("run")
    run.add_argument("--draft", type=Path, required=True)
    run.add_argument("--url", default="http://127.0.0.1:30000")
    run.add_argument("--input", required=True, type=Path)
    run.add_argument("--out", required=True, type=Path)
    compare = sub.add_parser("compare")
    compare.add_argument("--dflash", required=True, type=Path)
    compare.add_argument("--mmflash", required=True, type=Path)
    args = parser.parse_args()
    if args.action == "run":
        args.out.write_text(json.dumps(benchmark(args), indent=2) + "\n")
    else:
        report = compare_runs(
            json.loads(args.dflash.read_text()), json.loads(args.mmflash.read_text())
        )
        print(json.dumps(report, indent=2))
        if report["different_greedy_outputs"]:
            raise SystemExit(
                "inspect the mismatching greedy responses before interpreting the speed comparison"
            )


if __name__ == "__main__":
    main()
