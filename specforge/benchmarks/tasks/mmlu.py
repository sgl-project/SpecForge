# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MMLU: multiple-choice questions across academic subjects."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from specforge.benchmarks.tasks.answers import extract_choice
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

MULTIPLE_CHOICE_PROMPT = """\
Answer the following multiple choice question. The last line of your response \
should be of the following format: 'Answer: $LETTER' (without quotes) where \
LETTER is one of ABCD. Think step by step before answering.

{question}

A) {A}
B) {B}
C) {C}
D) {D}"""


def format_multiple_choice(question: str, choices: Dict[str, Any]) -> str:
    return MULTIPLE_CHOICE_PROMPT.format(question=question.strip(), **choices)


@TASKS.register
class MMLUTask(BenchmarkTask):
    name = "mmlu"
    description = "MMLU test split; multiple-choice accuracy, subset = HF config name"
    supports_subsets = True

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        def rows():
            for config in self.subset or ["all"]:
                yield from load_dataset("cais/mmlu", config, split="test")

        for row in self.limit(rows()):
            choices = {
                letter: str(choice).strip()
                for letter, choice in zip("ABCD", row["choices"])
            }
            yield Sample(
                turns=[format_multiple_choice(row["question"], choices)],
                label="ABCD"[int(row["answer"])],
            )

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return extract_choice(output) == label
