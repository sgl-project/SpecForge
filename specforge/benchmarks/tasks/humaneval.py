# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""HumanEval: Python function completion scored by running the unit tests."""

from __future__ import annotations

import re
import textwrap
from typing import Any, Dict, Iterable, Optional

from specforge.benchmarks.tasks.answers import extract_python_code, run_python_tests
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

PROMPT = (
    "Write a solution to the following problem and make sure that it passes "
    "the tests:\n```python\n{prompt}\n```"
)


def assemble_program(code: str, label: Dict[str, Any]) -> str:
    """Combine generated code with the HumanEval prompt and test harness.

    Models either emit the whole function or just its body.  When the entry
    point is not defined in the output, the original signature is prepended.
    """
    entry_point = label["entry_point"]
    defines_entry = re.search(rf"^\s*def\s+{re.escape(entry_point)}\s*\(", code, re.M)
    if defines_entry:
        program = code
    else:
        body = textwrap.indent(textwrap.dedent(code), "    ")
        program = label["prompt"].rstrip("\n") + "\n" + body
    return f"{program}\n\n{label['test']}\n\ncheck({entry_point})\n"


@TASKS.register
class HumanEvalTask(BenchmarkTask):
    name = "humaneval"
    description = "HumanEval; pass rate of generated code against the tests"
    max_new_tokens = 1024

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("openai/openai_humaneval", split="test")
        for row in self.limit(rows):
            yield Sample(
                turns=[PROMPT.format(prompt=row["prompt"])],
                label={
                    "prompt": row["prompt"],
                    "test": row["test"],
                    "entry_point": row["entry_point"],
                },
            )

    def score(self, output: str, label: Dict[str, Any]) -> Optional[bool]:
        code = extract_python_code(output)
        if code is None:
            return False
        return run_python_tests(assemble_program(code, label))
