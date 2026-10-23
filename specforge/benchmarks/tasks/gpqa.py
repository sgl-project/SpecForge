# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""GPQA (main): graduate-level science questions with shuffled choices."""

from __future__ import annotations

import random
from typing import Iterable, Optional

from specforge.benchmarks.tasks.answers import extract_choice
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample
from specforge.benchmarks.tasks.mmlu import format_multiple_choice


@TASKS.register
class GPQATask(BenchmarkTask):
    name = "gpqa"
    description = "GPQA main; multiple-choice accuracy with seeded choice order"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rng = random.Random(self.seed)
        rows = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
        for row in self.limit(rows):
            choices = [
                row["Incorrect Answer 1"],
                row["Incorrect Answer 2"],
                row["Incorrect Answer 3"],
            ]
            gold = rng.randint(0, 3)
            choices.insert(gold, row["Correct Answer"])
            yield Sample(
                turns=[
                    format_multiple_choice(
                        row["Question"],
                        {
                            letter: str(choice).strip()
                            for letter, choice in zip("ABCD", choices)
                        },
                    )
                ],
                label="ABCD"[gold],
            )

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return extract_choice(output) == label
