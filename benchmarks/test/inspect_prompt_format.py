#!/usr/bin/env python3
"""End-to-end check of what the SGLang server actually receives.

Sends one short prompt through the bench's SGL function path twice:
  1. Legacy: sgl.system/user/assistant -> SGLang lang frontend registry.
  2. Fixed: pre-rendered with HF tokenizer.apply_chat_template.
Prints state.text() after each call so you can compare what the server
processed in each case.

Requires a running SGLang server (use ./benchmarks/<target>_run.sh server).

Usage:
    python benchmarks/inspect_prompt_format.py --port 30000

Optional --model (defaults to whatever the running server reports).
"""
import argparse
import sys
from argparse import Namespace

import sglang as sgl
from sglang import set_default_backend
from sglang.test.test_utils import select_sglang_backend

sys.path.insert(0, "benchmarks")
from benchmarker.utils import set_chat_template_context  # noqa: E402

FAKE_SYSTEM = (
    "You are a helpful, respectful and honest assistant. Always answer as "
    "helpfully as possible, while being safe."
)
FAKE_Q = "Reply in exactly 5 words."


@sgl.function
def legacy_func(s, question):
    s += sgl.system(FAKE_SYSTEM)
    s += sgl.user(question)
    s += sgl.assistant(sgl.gen("answer", max_tokens=16))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="localhost")
    p.add_argument("--port", type=int, default="30000")
    p.add_argument("--model", default=None, help="HF id; default = whatever the server reports")
    args = p.parse_args()

    backend = select_sglang_backend(
        Namespace(host=f"http://{args.host}", port=args.port, backend="srt-no-parallel")
    )
    set_default_backend(backend)
    server_model = backend.model_info["model_path"]
    model_path = args.model or server_model
    print(f"server model_path: {server_model}")
    print(f"frontend chat template (legacy path): {backend.chat_template.name!r}\n")

    # 1) Legacy path: do NOT set chat-template context.
    set_chat_template_context(None)
    states = legacy_func.run_batch([{"question": FAKE_Q}], temperature=0, num_threads=1)
    print("=" * 80)
    print("LEGACY PATH (sgl.system / sgl.user / sgl.assistant)")
    print("=" * 80)
    print(states[0].text())

    # 2) Fixed path: use HF tokenizer's chat template via our helper.
    set_chat_template_context(model_path)
    from benchmarker.utils import create_simple_sgl_function

    fixed_func = create_simple_sgl_function(
        function_name="fixed_inspect",
        answer_key="answer",
        system_prompt=FAKE_SYSTEM,
        max_tokens=16,
    )
    states = fixed_func.run_batch([{"question": FAKE_Q}], temperature=0, num_threads=1)
    print("\n" + "=" * 80)
    print("FIXED PATH (HF tokenizer.apply_chat_template)")
    print("=" * 80)
    print(states[0].text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
