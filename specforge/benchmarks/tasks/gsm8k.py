# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GSM8K: grade-school math word problems, scored on the final number."""

from __future__ import annotations

from typing import Iterable, Optional

from specforge.benchmarks.tasks.answers import (
    REASON_STEP_BY_STEP,
    extract_final_number,
    extract_last_number,
    numbers_equal,
)
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample


@TASKS.register
class GSM8KTask(BenchmarkTask):
    name = "gsm8k"
    description = "GSM8K test split; accuracy on the final numeric answer"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("openai/gsm8k", "main", split="test")
        for row in self.limit(rows):
            # Reference answers end with ``#### <number>``.
            label = extract_last_number(row["answer"].split("####")[-1])
            yield Sample(turns=[row["question"] + REASON_STEP_BY_STEP], label=label)

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return numbers_equal(extract_final_number(output), label)
