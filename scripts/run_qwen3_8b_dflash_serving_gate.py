#!/usr/bin/env python3
"""Gate one Domino overfit artifact through real SGLang DFLASH chat serving."""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence

import requests


def longest_prefix_match(left: Sequence[int], right: Sequence[int]) -> int:
    for index, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return index
    return min(len(left), len(right))


def load_prompt_artifact(path: str) -> Dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        artifact = json.load(handle)
    messages = artifact.get("prompt_messages")
    target = artifact.get("target_suffix")
    if not isinstance(messages, list) or not messages:
        raise ValueError("prompt artifact must contain nonempty prompt_messages")
    if not isinstance(target, str) or not target:
        raise ValueError("prompt artifact must contain nonempty target_suffix")
    return artifact


def build_chat_payload(
    artifact: Dict[str, Any], model: str, max_tokens: int
) -> Dict[str, Any]:
    messages: List[Dict[str, Any]] = []
    for message in artifact["prompt_messages"]:
        clean = dict(message)
        clean.pop("reasoning_content", None)
        messages.append(clean)
    return {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "stream": False,
        "return_meta_info": True,
        "separate_reasoning": True,
        "chat_template_kwargs": {
            "enable_thinking": bool(artifact.get("enable_thinking", False))
        },
    }


def request_messages_with_reasoning_content(payload: Dict[str, Any]) -> int:
    return sum("reasoning_content" in message for message in payload["messages"])


def evaluate_response(
    *,
    response_json: Dict[str, Any],
    server_info: Dict[str, Any],
    payload: Dict[str, Any],
    target_ids: Sequence[int],
    encode,
    block_size: int,
) -> Dict[str, Any]:
    choices = response_json.get("choices") or []
    choice = choices[0] if choices and isinstance(choices[0], dict) else {}
    message = choice.get("message") or {}
    generated_reasoning = message.get("reasoning_content") or ""
    generated_content = message.get("content") or ""
    generated_text = generated_reasoning + generated_content
    generated_ids = encode(generated_text)

    # This is intentionally choice metadata, not aggregate /server_info metrics.
    choice_meta_info = choice.get("meta_info")
    spec_accept_length = (
        choice_meta_info.get("spec_accept_length")
        if isinstance(choice_meta_info, dict)
        else None
    )
    prefix_match = longest_prefix_match(generated_ids, target_ids)
    algorithm = server_info.get("speculative_algorithm")
    reasoning_count = request_messages_with_reasoning_content(payload)

    errors = []
    if algorithm != "DFLASH":
        errors.append(
            f"server speculative_algorithm is {algorithm!r}, expected 'DFLASH'"
        )
    if reasoning_count:
        errors.append("request history contains reasoning_content")
    if spec_accept_length is None:
        errors.append("missing choices[0].meta_info.spec_accept_length")
    elif float(spec_accept_length) < block_size:
        errors.append(
            f"spec_accept_length {spec_accept_length} < block_size {block_size}"
        )
    if prefix_match < block_size:
        errors.append(f"target prefix match {prefix_match} < block_size {block_size}")

    return {
        "passed": not errors,
        "endpoint": "/v1/chat/completions",
        "request_messages_with_reasoning_content": reasoning_count,
        "sglang_server_info_before": server_info,
        "choice_meta_info": choice_meta_info,
        "spec_accept_length": spec_accept_length,
        "target_prefix_match_tokens": prefix_match,
        "generated_tokens": len(generated_ids),
        "generated_reasoning": generated_reasoning,
        "generated_content": generated_content,
        "target_tokens": len(target_ids),
        "clean_block_tokens": block_size if not errors else 0,
        "errors": errors,
        "request_payload": payload,
        "sglang_response": response_json,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", required=True)
    parser.add_argument("--prompt-json-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.max_tokens < args.block_size:
        raise SystemExit("--block-size must be positive and --max-tokens >= block size")

    from transformers import AutoTokenizer

    artifact = load_prompt_artifact(args.prompt_json_path)
    payload = build_chat_payload(artifact, args.served_model, args.max_tokens)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    def encode(text):
        return tokenizer.encode(text, add_special_tokens=False)

    target_ids = encode(artifact["target_suffix"])

    server_url = args.server_url.rstrip("/")
    info_response = requests.get(f"{server_url}/server_info", timeout=30)
    info_response.raise_for_status()
    response = requests.post(
        f"{server_url}/v1/chat/completions", json=payload, timeout=args.timeout
    )
    response.raise_for_status()
    result = evaluate_response(
        response_json=response.json(),
        server_info=info_response.json(),
        payload=payload,
        target_ids=target_ids,
        encode=encode,
        block_size=args.block_size,
    )
    result["server_url"] = server_url
    result["endpoint"] = f"{server_url}/v1/chat/completions"

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(
            "real SGLang DFLASH serving gate failed: " + "; ".join(result["errors"])
        )


if __name__ == "__main__":
    main()
