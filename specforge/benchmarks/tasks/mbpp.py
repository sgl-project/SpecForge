# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MBPP (sanitized): short Python tasks scored by running the listed asserts."""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

from specforge.benchmarks.tasks.answers import extract_python_code, run_python_tests
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

PROMPT = (
    "You are an expert Python programmer, and here is your task: {text}\n"
    "Your code should pass these tests:\n\n{tests}\n"
)


def assemble_program(code: str, label: Dict[str, Any]) -> str:
    return "\n".join([label["setup"], code, *label["tests"], ""])


@TASKS.register
class MBPPTask(BenchmarkTask):
    name = "mbpp"
    description = "MBPP sanitized test split; pass rate against the reference asserts"
    max_new_tokens = 1024

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("google-research-datasets/mbpp", "sanitized", split="test")
        for row in self.limit(rows):
            tests: List[str] = list(row.get("test_list") or [])
            setup = "\n".join(
                [*(row.get("test_imports") or []), row.get("test_setup_code") or ""]
            ).strip()
            yield Sample(
                turns=[PROMPT.format(text=row["prompt"], tests="\n".join(tests))],
                label={"setup": setup, "tests": tests},
            )

    def score(self, output: str, label: Dict[str, Any]) -> Optional[bool]:
        code = extract_python_code(output)
        if code is None:
            return False
        return run_python_tests(assemble_program(code, label))
