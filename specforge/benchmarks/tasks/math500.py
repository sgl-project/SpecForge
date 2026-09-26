# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MATH-500: competition math, scored on the boxed final answer."""

from __future__ import annotations

from typing import Iterable, Optional

from specforge.benchmarks.tasks.answers import (
    REASON_STEP_BY_STEP,
    extract_boxed,
    math_answers_equal,
)
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample


@TASKS.register
class Math500Task(BenchmarkTask):
    name = "math500"
    description = "MATH-500 test split; accuracy on the boxed answer"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("HuggingFaceH4/MATH-500", split="test")
        for row in self.limit(rows):
            yield Sample(
                turns=[row["problem"] + REASON_STEP_BY_STEP],
                label=str(row["answer"]).strip(),
            )

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return math_answers_equal(extract_boxed(output), label)
