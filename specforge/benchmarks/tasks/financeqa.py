# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""FinanceQA: financial questions with optional context, unscored."""

from __future__ import annotations

from typing import Iterable

from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

WITH_CONTEXT = """\
Given the following context:

{context}

Can you answer the following question?

{question}"""


@TASKS.register
class FinanceQATask(BenchmarkTask):
    name = "financeqa"
    description = "FinanceQA test split; throughput and acceptance only"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("AfterQuery/FinanceQA", split="test")
        for row in self.limit(rows):
            question = row["question"].strip()
            if row.get("context"):
                question = WITH_CONTEXT.format(
                    context=row["context"].strip(), question=question
                )
            yield Sample(turns=[question])
