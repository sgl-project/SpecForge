"""
Validate regenerated SFT-style training data quality.

This is a quality gate that should be run BEFORE regen output is fed to
prepare_hidden_states.py / train_eagle3.py. It catches the kinds of silent
data-pollution issues that previously broke the Qwen3.5-122B-A10B MTP run:

  1. assistant.content == None / empty (regen truncated, or reasoning_content
     hijacked the actual answer)
  2. assistant has a reasoning_content field that the original SFT data never
     had (causes chat_template to wrap real answer inside <think>...</think>)
  3. user message contains '/no_think' but assistant still produced a real
     reasoning trace (target model didn't honor the directive at regen time)

Usage:
    python scripts/validate_regen_data.py \
        --regen-file /path/to/train_regen_v2.jsonl \
        --reference-file /path/to/train.jsonl \
        --strict

Exits non-zero (only with --strict) if any quality gate fails.
"""

import argparse
import json
import sys
from typing import Iterator, Tuple, Dict, Any


def iter_jsonl(path: str) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            try:
                yield i, json.loads(line)
            except json.JSONDecodeError as e:
                print(f"[warn] line {i} parse error: {e}", file=sys.stderr)


def _is_real_thinking(content, reasoning_content) -> bool:
    """A turn is 'really thinking' if it has non-empty reasoning either as a
    standalone reasoning_content field or inside a non-empty <think>...</think>
    block in content."""
    if isinstance(reasoning_content, str) and reasoning_content.strip():
        return True
    if isinstance(content, str) and "<think>" in content and "</think>" in content:
        try:
            inside = content.split("<think>", 1)[1].split("</think>", 1)[0]
            if inside.strip():
                return True
        except IndexError:
            pass
    return False


def stats_for_data(rows: Iterator[Tuple[int, Dict[str, Any]]]) -> Dict[str, int]:
    s = {
        "n_lines": 0,
        "n_with_messages": 0,
        "n_with_conversations": 0,
        "n_assistant": 0,
        "n_assistant_content_null": 0,
        "n_assistant_content_empty": 0,
        "n_assistant_with_rc_field": 0,
        "n_assistant_with_rc_nonempty": 0,
        "n_assistant_has_think_tag": 0,
        "n_user_no_think_pairs": 0,
        "n_user_no_think_violations": 0,
        "n_extra_keys_assistant": 0,
    }

    for _, row in rows:
        s["n_lines"] += 1
        if "messages" in row:
            s["n_with_messages"] += 1
        if "conversations" in row:
            s["n_with_conversations"] += 1
        # Loader chooses 'conversations' before 'messages' (see specforge.utils),
        # so validate the same field that training will actually read.
        msgs = row.get("conversations") or row.get("messages") or []

        last_user_no_think = False
        for m in msgs:
            if not isinstance(m, dict):
                continue
            role = m.get("role")
            if role == "user":
                c = m.get("content", "")
                last_user_no_think = (
                    isinstance(c, str) and "/no_think" in c
                )
            elif role == "assistant":
                s["n_assistant"] += 1
                content = m.get("content")
                rc = m.get("reasoning_content", "__MISSING__")
                # extra key audit
                for k in m.keys():
                    if k not in ("role", "content", "tool_calls", "reasoning_content"):
                        s["n_extra_keys_assistant"] += 1
                if content is None:
                    s["n_assistant_content_null"] += 1
                elif content == "":
                    s["n_assistant_content_empty"] += 1
                elif isinstance(content, str) and "<think>" in content:
                    s["n_assistant_has_think_tag"] += 1
                if rc != "__MISSING__":
                    s["n_assistant_with_rc_field"] += 1
                    if rc not in (None, ""):
                        s["n_assistant_with_rc_nonempty"] += 1
                if last_user_no_think:
                    s["n_user_no_think_pairs"] += 1
                    rc_val = rc if rc != "__MISSING__" else None
                    if _is_real_thinking(content, rc_val):
                        s["n_user_no_think_violations"] += 1
                # consume one user-assistant pair
                last_user_no_think = False
    return s


def pct(num: int, denom: int) -> str:
    if denom == 0:
        return "  n/a"
    return f"{100 * num / denom:5.2f}%"


def print_block(name: str, s: Dict[str, int]) -> None:
    a = max(s["n_assistant"], 1)
    print(f"--- {name} ---")
    print(
        f"  rows: {s['n_lines']}, "
        f"has 'messages': {s['n_with_messages']}, "
        f"has 'conversations': {s['n_with_conversations']}"
    )
    print(f"  assistant turns: {s['n_assistant']}")
    print(
        f"    content == None:                  "
        f"{s['n_assistant_content_null']:>7}  ({pct(s['n_assistant_content_null'], a)})"
    )
    print(
        f"    content == '':                    "
        f"{s['n_assistant_content_empty']:>7}  ({pct(s['n_assistant_content_empty'], a)})"
    )
    print(
        f"    content has <think> tag:          "
        f"{s['n_assistant_has_think_tag']:>7}  ({pct(s['n_assistant_has_think_tag'], a)})"
    )
    print(
        f"    has 'reasoning_content' field:    "
        f"{s['n_assistant_with_rc_field']:>7}  ({pct(s['n_assistant_with_rc_field'], a)})"
    )
    print(
        f"      └ non-empty:                    "
        f"{s['n_assistant_with_rc_nonempty']:>7}  ({pct(s['n_assistant_with_rc_nonempty'], a)})"
    )
    print(
        f"    extra keys beyond {'role/content/tool_calls/reasoning_content'}: "
        f"{s['n_extra_keys_assistant']}"
    )
    print(f"  /no_think pairs (user→assistant):    {s['n_user_no_think_pairs']:>7}")
    print(
        f"    └ violations (real thinking emitted): "
        f"{s['n_user_no_think_violations']:>5}  "
        f"({pct(s['n_user_no_think_violations'], max(s['n_user_no_think_pairs'], 1))})"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--regen-file", required=True, help="jsonl to validate")
    parser.add_argument(
        "--reference-file",
        default=None,
        help="optional: original SFT jsonl to print as comparison baseline",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit with non-zero status if any quality gate fails",
    )
    parser.add_argument(
        "--max-null-content-rate",
        type=float,
        default=0.0,
        help="max allowed rate of assistant.content == None (default: 0.0)",
    )
    parser.add_argument(
        "--max-empty-content-rate",
        type=float,
        default=0.0,
        help="max allowed rate of assistant.content == '' (default: 0.0)",
    )
    parser.add_argument(
        "--max-rc-field-rate",
        type=float,
        default=0.0,
        help=(
            "max allowed rate of assistant turns having a 'reasoning_content' "
            "field. Original SFT data has 0%%; if regen uses "
            "--inline-reasoning-into-content, this should also be 0%%."
        ),
    )
    parser.add_argument(
        "--max-no-think-violation-rate",
        type=float,
        default=0.05,
        help=(
            "max allowed rate of /no_think violations "
            "(user said /no_think but assistant still produced reasoning). "
            "Default 5%%."
        ),
    )
    args = parser.parse_args()

    print(f"[regen] reading {args.regen_file}")
    regen = stats_for_data(iter_jsonl(args.regen_file))
    print_block("Regen", regen)

    if args.reference_file:
        print()
        print(f"[ref]   reading {args.reference_file}")
        ref = stats_for_data(iter_jsonl(args.reference_file))
        print_block("Reference (original SFT data)", ref)

    a = max(regen["n_assistant"], 1)
    nt = max(regen["n_user_no_think_pairs"], 1)
    null_rate = regen["n_assistant_content_null"] / a
    empty_rate = regen["n_assistant_content_empty"] / a
    rc_rate = regen["n_assistant_with_rc_field"] / a
    nt_violation_rate = regen["n_user_no_think_violations"] / nt

    print()
    print("=== Quality Gates ===")
    failed = []

    def gate(name, value, threshold):
        ok = value <= threshold
        marker = "OK  " if ok else "FAIL"
        print(
            f"  [{marker}] {name:<35} value={value*100:5.2f}%  threshold<={threshold*100:5.2f}%"
        )
        if not ok:
            failed.append(f"{name}: {value*100:.2f}% > {threshold*100:.2f}%")

    gate("content == None rate", null_rate, args.max_null_content_rate)
    gate("content == '' rate", empty_rate, args.max_empty_content_rate)
    gate("reasoning_content field rate", rc_rate, args.max_rc_field_rate)
    gate("/no_think violation rate", nt_violation_rate, args.max_no_think_violation_rate)

    print()
    if failed:
        print("RESULT: FAIL")
        for x in failed:
            print(f"  - {x}")
        if args.strict:
            sys.exit(1)
    else:
        print("RESULT: PASS  (all quality gates green)")


if __name__ == "__main__":
    main()
