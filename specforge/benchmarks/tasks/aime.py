# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""AIME 2024: olympiad problems with integer answers and long reasoning."""

from __future__ import annotations

from typing import Iterable, Optional

from specforge.benchmarks.tasks.answers import (
    REASON_STEP_BY_STEP,
    extract_final_number,
    numbers_equal,
)
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample


@TASKS.register
class AIMETask(BenchmarkTask):
    name = "aime"
    description = "AIME 2024; accuracy on the integer answer, long generations"
    max_new_tokens = 32768

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("Maxwell-Jia/AIME_2024", split="train")
        for row in self.limit(rows):
            yield Sample(
                turns=[row["Problem"] + REASON_STEP_BY_STEP],
                label=str(row["Answer"]).strip(),
            )

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return numbers_equal(extract_final_number(output), label)
