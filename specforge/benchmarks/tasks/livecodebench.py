# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""LiveCodeBench: competitive programming prompts, unscored."""

from __future__ import annotations

from typing import Iterable

from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample


@TASKS.register
class LiveCodeBenchTask(BenchmarkTask):
    name = "livecodebench"
    description = "LiveCodeBench code generation; throughput and acceptance only"

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        rows = load_dataset("livecodebench/code_generation", split="test")
        for row in self.limit(rows):
            yield Sample(turns=[row["question_content"].strip()])
