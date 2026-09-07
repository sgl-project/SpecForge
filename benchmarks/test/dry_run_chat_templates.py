#!/usr/bin/env python3
"""Dry-run for the chat-template path in benchmarker/utils.py.

Renders the prompt shapes the SGL function helpers emit and checks that the
multi-turn delta-append exactly reconstructs apply_chat_template's output.
No SGLang server, no GPU, no generation.

Usage:
    python benchmarks/dry_run_chat_templates.py [--model HF_ID]...
"""
import argparse
import sys
import textwrap
from typing import List

sys.path.insert(0, "benchmarks")
from benchmarker.utils import (  # noqa: E402
    _assistant_for_replay,
    _delta,
    _has_chat_template,
    _render_chat,
    set_chat_template_context,
)
from sglang.lang.chat_template import get_chat_template_by_model_path  # noqa: E402

FAKE_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe."
)
FAKE_Q1 = "Write a one-paragraph travel blog post about visiting Hawaii."
FAKE_A1 = (
    "Hawaii's volcanic shores blend turquoise breakers with black-sand beaches "
    "where sea turtles bask at dusk."
)
FAKE_Q2 = "Now rewrite the same paragraph from the perspective of the sea turtle."


def banner(title: str) -> None:
    print()
    print("=" * 80)
    print(title)
    print("=" * 80)


def show(label: str, text: str, max_chars: int = 600) -> None:
    snippet = text if len(text) <= max_chars else text[:max_chars] + f"... [{len(text) - max_chars} more chars]"
    print(f"-- {label} ({len(text)} chars) --")
    print(textwrap.indent(snippet, "    "))
    print()


def _simulate_raw_answer(prompt_1: str, clean: str) -> str:
    """Approximate the raw model output: harmony-wrapped for gpt-oss, plain otherwise."""
    if prompt_1.endswith("<|start|>assistant"):
        return f"<|channel|>final<|message|>{clean}<|end|>"
    return clean


def dry_run_multi_turn() -> dict:
    msgs: List[dict] = [
        {"role": "system", "content": FAKE_SYSTEM},
        {"role": "user", "content": FAKE_Q1},
    ]
    prompt_1 = _render_chat(msgs, add_generation_prompt=True)
    answer_1_raw = _simulate_raw_answer(prompt_1, FAKE_A1)
    answer_1_for_replay = _assistant_for_replay(answer_1_raw)

    msgs.append({"role": "assistant", "content": answer_1_for_replay})
    msgs.append({"role": "user", "content": FAKE_Q2})
    prompt_2_full = _render_chat(msgs, add_generation_prompt=True)

    cur_text = prompt_1 + answer_1_raw  # mirrors actual SGLang state
    return {
        "prompt_1": prompt_1,
        "answer_1_raw": answer_1_raw,
        "answer_1_for_replay": answer_1_for_replay,
        "prompt_2_full": prompt_2_full,
        "delta": _delta(cur_text, prompt_2_full),
        "path": "exact-prefix" if prompt_2_full.startswith(cur_text) else "longest-common-prefix",
        "cur_text": cur_text,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="HF model id or local path. May be passed multiple times.",
    )
    args = parser.parse_args()
    if not args.model:
        args.model = [
            "openai/gpt-oss-20b",
            "Qwen/Qwen3-8B",
            "nreHieW/Llama-3.1-8B-Instruct",
        ]

    for model_path in args.model:
        banner(f"Model: {model_path}")
        try:
            set_chat_template_context(model_path)
        except Exception as e:
            print(f"Failed to load tokenizer for {model_path}: {e}")
            continue

        # Side-by-side: what SGLang's lang frontend would emit (the path the
        # bench used before our fix) vs. the HF tokenizer's chat_template.
        sgl_tmpl = get_chat_template_by_model_path(model_path)
        sgl_msgs = [
            {"role": "system", "content": FAKE_SYSTEM},
            {"role": "user", "content": FAKE_Q1},
        ]
        print(f"   sglang frontend template: name={sgl_tmpl.name!r}")
        show("SGLang lang frontend output (legacy path)", sgl_tmpl.get_prompt(sgl_msgs))

        if not _has_chat_template():
            print(
                f"Tokenizer for {model_path} has no chat_template; "
                "helpers would fall back to SGLang's lang frontend registry."
            )
            continue

        # Probe each message shape that benchmarks actually send. mtbench
        # always has [system, user], gsm8k has [user] only.
        from benchmarker.utils import _render_chat as render
        probes = [
            ("[user]  (gsm8k-style, no system)",
             [{"role": "user", "content": "Question: 2+2?\nAnswer:"}]),
            ("[system, user]  (mtbench-style)",
             [{"role": "system", "content": FAKE_SYSTEM},
              {"role": "user", "content": FAKE_Q1}]),
            ("[user]  (long few-shot, gsm8k-realistic)",
             [{"role": "user",
               "content": "Question: A pumpkin patch...\nAnswer: Step 1...\n\n" * 5
                           + "Question: How many...\nAnswer:"}]),
        ]
        for label, msgs in probes:
            try:
                out = render(msgs, add_generation_prompt=True)
                print(f"   {label:55s}  OK  ({len(out)} chars)")
            except Exception as e:
                print(f"   {label:55s}  FAIL  {type(e).__name__}: {e}")

        mt = dry_run_multi_turn()
        show("turn-1 prompt", mt["prompt_1"])
        show("raw answer_1 (what SGLang state holds)", mt["answer_1_raw"])
        show("cleaned answer_1 (added to messages list)", mt["answer_1_for_replay"])
        show("full turn-2 prompt", mt["prompt_2_full"])
        show("delta appended for turn-2", mt["delta"])
        print(f"   delta path: {mt['path']}")
        ok = (mt["cur_text"] + mt["delta"]) == mt["prompt_2_full"]
        if ok:
            print("   reconstruct check: OK")
        else:
            matched = sum(
                1 for a, b in zip(mt["cur_text"] + mt["delta"], mt["prompt_2_full"]) if a == b
            )
            print(f"   reconstruct check: APPROXIMATE — {matched}/{len(mt['prompt_2_full'])} chars match")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
