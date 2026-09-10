# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SimpleQA: short factual questions, unscored (needs a judge model)."""

from __future__ import annotations

from typing import Iterable

from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample


@TASKS.register
class SimpleQATask(BenchmarkTask):
    name = "simpleqa"
    description = "SimpleQA test split; throughput and acceptance only"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("basicv8vc/SimpleQA", split="test")
        for row in self.limit(rows):
            yield Sample(turns=[row["problem"].strip()])
