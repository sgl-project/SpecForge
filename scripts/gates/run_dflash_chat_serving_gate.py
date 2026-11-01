"""Gate one overfit draft artifact through real SGLang DFLASH chat serving.

This script is tied to the SGLang DFLASH serving API, not to a particular draft
architecture. It accepts either a single-record training JSONL file or the
legacy prompt artifact produced by ``select_overfit_sample.py``.

Future DFLASH draft methods can reuse this checker when they expose the same
OpenAI chat endpoint behavior and per-choice ``spec_accept_length`` metadata. If a
method needs a different request shape, acceptance metric, or output comparison,
add a sibling checker under this directory instead of weakening this strict gate.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Sequence


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


def load_training_jsonl(
    path: str,
    tokenizer,
    *,
    enable_thinking: bool,
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    row = None
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            if row is not None:
                raise ValueError("training JSONL must contain exactly one record")
            row = json.loads(line)
    if row is None:
        raise ValueError("training JSONL must contain exactly one record")

    conversations = row.get("conversations")
    if (
        not isinstance(conversations, list)
        or len(conversations) < 2
        or not all(isinstance(message, dict) for message in conversations)
    ):
        raise ValueError(
            "training JSONL record must contain at least two conversations"
        )
    assistant = conversations[-1]
    if assistant.get("role") != "assistant":
        raise ValueError("last training conversation must have role='assistant'")
    content = assistant.get("content")
    reasoning = assistant.get("reasoning_content", "")
    if not isinstance(content, str) or not isinstance(reasoning, str):
        raise ValueError("assistant content and reasoning_content must be strings")
    if not content.strip() and not reasoning.strip():
        raise ValueError("assistant target must not be empty")

    # Match SpecForge's training parser.  The general ``qwen`` parser drops
    # reasoning_content, while ``qwen3-thinking`` preserves it.  It also
    # prepends the template's default system prompt when the source starts with
    # a user turn.  Replaying the raw JSONL without these two normalizations can
    # produce a different prompt and, for Qwen templates, can even make the
    # rendered full conversation cease to share the rendered prompt prefix.
    normalized_conversations = []
    for message in conversations:
        clean = dict(message)
        if not enable_thinking:
            clean.pop("reasoning_content", None)
        normalized_conversations.append(clean)
    if (
        system_prompt
        and normalized_conversations
        and normalized_conversations[0].get("role") != "system"
    ):
        normalized_conversations.insert(0, {"role": "system", "content": system_prompt})

    prompt_messages = normalized_conversations[:-1]
    print("Question:", prompt_messages)
    template_kwargs = {
        "tokenize": False,
        "add_special_tokens": False,
        "enable_thinking": enable_thinking,
    }
    flat_prompt = tokenizer.apply_chat_template(
        prompt_messages,
        add_generation_prompt=True,
        **template_kwargs,
    )
    flat_train_text = tokenizer.apply_chat_template(
        normalized_conversations,
        add_generation_prompt=False,
        **template_kwargs,
    )
    if not flat_train_text.startswith(flat_prompt):
        raise ValueError("flattened train text does not start with flattened prompt")
    target_suffix = flat_train_text[len(flat_prompt) :]
    if not target_suffix:
        raise ValueError("rendered assistant target must not be empty")
    return {
        "id": row.get("id"),
        "prompt_messages": prompt_messages,
        "target_suffix": target_suffix,
        "enable_thinking": enable_thinking,
    }


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
    if algorithm not in {"DFLASH", "DSPARK"}:
        errors.append(
            f"server speculative_algorithm is {algorithm!r}, "
            "expected one of ['DFLASH', 'DSPARK']"
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


def result_summary(
    result: Dict[str, Any],
    output_path: str,
) -> Dict[str, Any]:
    return {
        "passed": result["passed"],
        "input_format": result["input_format"],
        "spec_accept_length": result["spec_accept_length"],
        "target_prefix_match_tokens": result["target_prefix_match_tokens"],
        "generated_tokens": result["generated_tokens"],
        "target_tokens": result["target_tokens"],
        "clean_block_tokens": result["clean_block_tokens"],
        "errors": result["errors"],
        "result_path": os.path.abspath(output_path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server-url", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--served-model", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--data-path",
        help="single-record JSONL in the conversations format used for training",
    )
    source.add_argument(
        "--prompt-json-path",
        help="legacy prompt artifact produced by select_overfit_sample.py",
    )
    parser.add_argument(
        "--enable-thinking",
        action="store_true",
        help="enable tokenizer thinking mode when rendering --data-path",
    )
    parser.add_argument(
        "--system-prompt",
        default="You are a helpful assistant.",
        help=(
            "system prompt inserted when the JSONL starts with a user turn; "
            "defaults to the SpecForge qwen template value"
        ),
    )
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--block-size", type=int, default=16)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--timeout", type=float, default=1800.0)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="print the complete result instead of the concise summary",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.block_size <= 0 or args.max_tokens < args.block_size:
        raise SystemExit("--block-size must be positive and --max-tokens >= block size")

    import requests
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if args.data_path:
        artifact = load_training_jsonl(
            args.data_path,
            tokenizer,
            enable_thinking=args.enable_thinking,
            system_prompt=args.system_prompt,
        )
        input_path = args.data_path
        input_format = "training_jsonl"
    else:
        artifact = load_prompt_artifact(args.prompt_json_path)
        input_path = args.prompt_json_path
        input_format = "prompt_artifact"
    payload = build_chat_payload(artifact, args.served_model, args.max_tokens)

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
    result["input_path"] = os.path.abspath(input_path)
    result["input_format"] = input_format

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "w", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    displayed = result if args.verbose else result_summary(result, args.output_path)
    print(json.dumps(displayed, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(
            "real SGLang DFLASH serving gate failed: " + "; ".join(result["errors"])
        )


if __name__ == "__main__":
    main()
