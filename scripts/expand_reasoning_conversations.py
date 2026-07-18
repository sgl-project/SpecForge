"""Expand regenerated reasoning conversations into generation-event samples."""

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    from scripts.conversation_validation import (
        validate_conversation as validate_basic_conversation,
    )
except ModuleNotFoundError:
    from conversation_validation import (
        validate_conversation as validate_basic_conversation,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Convert conversation-level regenerated reasoning JSONL into one "
            "training row per assistant generation event."
        )
    )
    parser.add_argument("--input-file-path", required=True)
    parser.add_argument("--output-file-path", required=True)
    parser.add_argument(
        "--skipped-file-path",
        default=None,
        help="Path for skipped invalid rows. Defaults to <output>_skipped.jsonl.",
    )
    return parser.parse_args()


def skipped_path(output_file_path: str) -> str:
    if not output_file_path.endswith(".jsonl"):
        raise ValueError("output file path must end in .jsonl")
    return f"{output_file_path[:-len('.jsonl')]}_skipped.jsonl"


def validate_paths(input_path: str, output_path: str, skip_path: str) -> None:
    paths = [Path(path).resolve() for path in (input_path, output_path, skip_path)]
    if len(set(paths)) != len(paths):
        raise ValueError("input, output, and skipped paths must be distinct")
    if not paths[0].is_file():
        raise ValueError(f"input file does not exist: {input_path}")
    for path in paths[1:]:
        if path.suffix != ".jsonl":
            raise ValueError(f"output path must end in .jsonl: {path}")
        if path.exists():
            raise ValueError(f"refusing to overwrite existing output: {path}")


def iter_jsonl(path: str) -> Iterable[Tuple[int, Any]]:
    with open(path, "r", encoding="utf-8") as handle:
        for line_index, line in enumerate(handle):
            if line.strip():
                yield line_index, json.loads(line)


def validate_conversation(messages: Any) -> Optional[str]:
    if isinstance(messages, list):
        expected_role = "user"
        for index, message in enumerate(messages):
            if not isinstance(message, dict):
                break
            role = message.get("role")
            if role == "system":
                continue
            if role == "assistant":
                return (
                    f"Invalid conversation role order at position {index}: "
                    f"expected {expected_role}, got {role}"
                )
            break

    invalid_reason = validate_basic_conversation(
        messages,
        error_style="regeneration",
    )
    if invalid_reason is not None:
        return invalid_reason

    saw_assistant = False
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue

        reasoning = message.get("reasoning_content")
        if not isinstance(reasoning, str) or not reasoning.strip():
            return (
                f"Invalid assistant reasoning_content at position {index}: "
                "expected non-empty string"
            )
        saw_assistant = True

    if not saw_assistant:
        return "Data does not contain an assistant message"
    if messages[-1].get("role") != "assistant":
        return "Conversation ends with a user message"
    return None


def visible_history_message(message: Dict[str, Any]) -> Dict[str, Any]:
    visible = dict(message)
    if visible.get("role") == "assistant":
        visible.pop("reasoning_content", None)
    return visible


def expand_row(row: Any, source_row_index: int) -> List[Dict[str, Any]]:
    if not isinstance(row, dict):
        raise ValueError("Expected a JSON object")
    if row.get("status") != "success":
        raise ValueError("status must be 'success'")
    source_id = row.get("id")
    if not isinstance(source_id, str) or not source_id.strip():
        raise ValueError("id must be a non-empty string")
    messages = row.get("conversations")
    invalid_reason = validate_conversation(messages)
    if invalid_reason is not None:
        raise ValueError(invalid_reason)

    history: List[Dict[str, Any]] = []
    events: List[Dict[str, Any]] = []
    assistant_turn_index = 0

    for message in messages:
        role = message["role"]
        if role != "assistant":
            history.append(dict(message))
            continue

        event_messages = [dict(item) for item in history]
        event_messages.append(dict(message))
        event = {
            "id": f"{source_id}#turn{assistant_turn_index}",
            "source_id": source_id,
            "source_row_index": source_row_index,
            "assistant_turn_index": assistant_turn_index,
            "conversations": event_messages,
        }
        if "status" in row:
            event["status"] = row["status"]
        events.append(event)

        history.append(visible_history_message(message))
        assistant_turn_index += 1

    return events


def main() -> None:
    args = parse_args()
    try:
        skip_path = args.skipped_file_path or skipped_path(args.output_file_path)
        validate_paths(args.input_file_path, args.output_file_path, skip_path)
    except ValueError as exc:
        raise SystemExit(f"invalid paths: {exc}") from exc
    stats: Counter = Counter()

    with (
        open(args.output_file_path, "w", encoding="utf-8") as output_handle,
        open(skip_path, "w", encoding="utf-8") as skipped_handle,
    ):
        for source_row_index, row in iter_jsonl(args.input_file_path):
            stats["input_rows"] += 1
            try:
                events = expand_row(row, source_row_index)
            except ValueError as exc:
                skipped = dict(row) if isinstance(row, dict) else {"data": row}
                skipped["status"] = "skipped"
                skipped["error"] = str(exc)
                skipped["source_row_index"] = source_row_index
                skipped_handle.write(json.dumps(skipped, ensure_ascii=False) + "\n")
                stats["skipped_rows"] += 1
                continue

            assistant_turns = len(events)
            if assistant_turns > 1:
                stats["multi_turn_input_rows"] += 1
            stats["assistant_turns"] += assistant_turns
            for event in events:
                output_handle.write(json.dumps(event, ensure_ascii=False) + "\n")
                stats["output_event_samples"] += 1

    input_rows = stats["input_rows"]
    avg_events = stats["output_event_samples"] / input_rows if input_rows else 0.0
    print(f"input rows: {stats['input_rows']}")
    print(f"output event samples: {stats['output_event_samples']}")
    print(f"skipped rows: {stats['skipped_rows']}")
    print(f"assistant turns: {stats['assistant_turns']}")
    print(f"multi-turn input rows: {stats['multi_turn_input_rows']}")
    print(f"avg events per input row: {avg_events:.3f}")
    print(f"skipped file: {skip_path}")


if __name__ == "__main__":
    main()
