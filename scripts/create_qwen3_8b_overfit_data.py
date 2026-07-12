#!/usr/bin/env python3
"""Select one clean single-turn Domino sample and record its serving contract."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
import types
from typing import Any, Dict, Iterable, Optional, Tuple


THINK_MARKERS = ("<think>", "</think>")


def iter_jsonl(path: str) -> Iterable[Tuple[int, Dict[str, Any]]]:
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if line.strip():
                yield index, json.loads(line)


def single_turn_candidate(
    row: Dict[str, Any],
    min_assistant_chars: int,
    reasoning_policy: str = "forbidden",
) -> Optional[Dict[str, Any]]:
    if row.get("status", "success") != "success":
        return None
    conversations = row.get("conversations")
    if not isinstance(conversations, list) or len(conversations) != 2:
        return None
    user, assistant = conversations
    if user.get("role") != "user" or assistant.get("role") != "assistant":
        return None
    user_content = user.get("content")
    assistant_content = assistant.get("content")
    if not isinstance(user_content, str) or not user_content.strip():
        return None
    if not isinstance(assistant_content, str):
        return None
    if len(assistant_content.strip()) < min_assistant_chars:
        return None
    reasoning_content = assistant.get("reasoning_content", "")
    if not isinstance(reasoning_content, str):
        return None
    if any(
        marker in text.lower()
        for text in (assistant_content, reasoning_content)
        for marker in THINK_MARKERS
    ):
        return None
    has_reasoning = bool(reasoning_content.strip())
    if reasoning_policy == "forbidden" and has_reasoning:
        return None
    if reasoning_policy == "required" and not has_reasoning:
        return None
    return {
        "id": row.get("id"),
        "conversations": [dict(user), dict(assistant)],
        "status": "success",
    }


def load_processing_stack(model_path: str, chat_template: str):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    package_name = "_specforge_qwen3_overfit"
    root_package = types.ModuleType(package_name)
    root_package.__path__ = []
    sys.modules[package_name] = root_package

    data_package_name = f"{package_name}.data"
    data_package = types.ModuleType(data_package_name)
    data_package.__path__ = []
    sys.modules[data_package_name] = data_package

    distributed_module = types.ModuleType(f"{package_name}.distributed")
    distributed_module.get_draft_sp_group = lambda: None
    distributed_module.get_sp_ring_group = lambda: None
    sys.modules[f"{package_name}.distributed"] = distributed_module

    for module_name in ("template", "parse", "preprocessing"):
        full_name = f"{data_package_name}.{module_name}"
        module_path = os.path.join(root_dir, "specforge", "data", f"{module_name}.py")
        spec = importlib.util.spec_from_file_location(full_name, module_path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[full_name] = module
        spec.loader.exec_module(module)

    template_module = sys.modules[f"{data_package_name}.template"]
    preprocessing_module = sys.modules[f"{data_package_name}.preprocessing"]
    return (
        tokenizer,
        template_module.TEMPLATE_REGISTRY.get(chat_template),
        preprocessing_module.preprocess_conversations,
    )


def loss_token_count(
    candidate: Dict[str, Any],
    processing_stack,
    *,
    max_length: int,
    train_only_last_turn: bool,
) -> int:
    tokenizer, template, preprocess_conversations = processing_stack
    processed = preprocess_conversations(
        tokenizer,
        [candidate["conversations"]],
        template,
        max_length=max_length,
        train_only_last_turn=train_only_last_turn,
    )
    if len(processed["loss_mask"]) != 1:
        raise ValueError("preprocessing did not return exactly one loss mask")
    return int(processed["loss_mask"][0].sum().item())


def build_prompt_artifact(
    candidate: Dict[str, Any],
    processing_stack,
    *,
    enable_thinking: bool,
    source_row_index: int,
    loss_tokens: int,
) -> Dict[str, Any]:
    """Render the exact chat prompt/target boundary used by real serving."""
    tokenizer = processing_stack[0]
    messages = candidate["conversations"]
    prompt_messages = []
    for message in messages[:-1]:
        clean = dict(message)
        clean.pop("reasoning_content", None)
        prompt_messages.append(clean)
    template_kwargs = {
        "tokenize": False,
        "add_special_tokens": False,
        "enable_thinking": enable_thinking,
    }
    flat_prompt = tokenizer.apply_chat_template(
        prompt_messages, add_generation_prompt=True, **template_kwargs
    )
    flat_train_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=False, **template_kwargs
    )
    if not flat_train_text.startswith(flat_prompt):
        raise ValueError("flattened train text does not start with flattened prompt")
    full_input_tokens = len(tokenizer.encode(flat_train_text, add_special_tokens=False))
    return {
        "source_row_index": source_row_index,
        "id": candidate.get("id"),
        "prompt_messages": prompt_messages,
        "target_suffix": flat_train_text[len(flat_prompt) :],
        "enable_thinking": enable_thinking,
        "loss_tokens": loss_tokens,
        "full_input_tokens": full_input_tokens,
    }


def create_single_sample(
    input_path: str,
    output_path: str,
    *,
    model_path: str,
    chat_template: str = "qwen",
    max_length: int = 512,
    min_loss_tokens: int = 32,
    min_assistant_chars: int = 128,
    train_only_last_turn: bool = False,
    reasoning_policy: str = "forbidden",
    prompt_output_path: Optional[str] = None,
    enable_thinking: bool = False,
    require_untruncated: bool = False,
    processing_stack=None,
) -> Tuple[int, Dict[str, Any], int]:
    if os.path.exists(output_path):
        raise FileExistsError(f"refusing to overwrite existing output: {output_path}")
    if prompt_output_path and os.path.exists(prompt_output_path):
        raise FileExistsError(
            f"refusing to overwrite existing prompt artifact: {prompt_output_path}"
        )
    if reasoning_policy not in {"forbidden", "required", "allow"}:
        raise ValueError(f"unknown reasoning policy: {reasoning_policy}")
    if processing_stack is None:
        processing_stack = load_processing_stack(model_path, chat_template)
    for index, row in iter_jsonl(input_path):
        candidate = single_turn_candidate(
            row, min_assistant_chars, reasoning_policy=reasoning_policy
        )
        if candidate is None:
            continue
        loss_tokens = loss_token_count(
            candidate,
            processing_stack,
            max_length=max_length,
            train_only_last_turn=train_only_last_turn,
        )
        if loss_tokens < min_loss_tokens:
            continue
        artifact = None
        if prompt_output_path:
            artifact = build_prompt_artifact(
                candidate,
                processing_stack,
                enable_thinking=enable_thinking,
                source_row_index=index,
                loss_tokens=loss_tokens,
            )
            if require_untruncated and artifact["full_input_tokens"] > max_length:
                continue
        parent = os.path.dirname(os.path.abspath(output_path))
        os.makedirs(parent, exist_ok=True)
        with open(output_path, "x", encoding="utf-8") as handle:
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
        if prompt_output_path and artifact is not None:
            prompt_parent = os.path.dirname(os.path.abspath(prompt_output_path))
            os.makedirs(prompt_parent, exist_ok=True)
            with open(prompt_output_path, "x", encoding="utf-8") as handle:
                json.dump(artifact, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
        return index, candidate, loss_tokens
    raise ValueError(
        "no clean single-turn sample found; expected a successful user/assistant "
        f"row with at least {min_assistant_chars} assistant characters, "
        f"reasoning_policy={reasoning_policy!r}, no <think> markers, and at least "
        f"{min_loss_tokens} post-preprocessing loss tokens"
        + (f", without exceeding {max_length} tokens" if require_untruncated else "")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-data-path", required=True)
    parser.add_argument("--output-data-path", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--chat-template", default="qwen")
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--min-loss-tokens", type=int, default=32)
    parser.add_argument("--min-assistant-chars", type=int, default=128)
    parser.add_argument("--train-only-last-turn", action="store_true")
    parser.add_argument(
        "--reasoning-policy",
        choices=["forbidden", "required", "allow"],
        default="forbidden",
    )
    parser.add_argument("--prompt-output-path")
    parser.add_argument("--enable-thinking", action="store_true")
    parser.add_argument("--require-untruncated", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index, row, loss_tokens = create_single_sample(
        args.input_data_path,
        args.output_data_path,
        model_path=args.model_path,
        chat_template=args.chat_template,
        max_length=args.max_length,
        min_loss_tokens=args.min_loss_tokens,
        min_assistant_chars=args.min_assistant_chars,
        train_only_last_turn=args.train_only_last_turn,
        reasoning_policy=args.reasoning_policy,
        prompt_output_path=args.prompt_output_path,
        enable_thinking=args.enable_thinking,
        require_untruncated=args.require_untruncated,
    )
    print(
        json.dumps(
            {
                "source_row_index": index,
                "id": row.get("id"),
                "loss_tokens": loss_tokens,
                "output_data_path": os.path.abspath(args.output_data_path),
                "prompt_output_path": (
                    os.path.abspath(args.prompt_output_path)
                    if args.prompt_output_path
                    else None
                ),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
