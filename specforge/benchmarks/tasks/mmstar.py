# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""MMStar: multiple-choice visual questions for vision-language models."""

from __future__ import annotations

import os
import re
import shutil
import tempfile
from typing import Iterable, Optional

from specforge.benchmarks.tasks.answers import extract_choice
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

_OPTION_LINE = re.compile(r"^([A-Z])[.:)]\s*(.*)$")


def count_options(question: str) -> int:
    """Number of ``A. ...`` style option lines after ``Options:``."""
    if "Options:" not in question:
        return 0
    _, options = question.split("Options:", 1)
    return sum(1 for line in options.splitlines() if _OPTION_LINE.match(line.strip()))


@TASKS.register
class MMStarTask(BenchmarkTask):
    name = "mmstar"
    description = "MMStar val split; multiple-choice accuracy on image questions"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._image_dir: Optional[str] = None

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        self._image_dir = tempfile.mkdtemp(prefix="specforge-mmstar-")
        rows = load_dataset("Lin-Chen/MMStar", split="val")
        for index, row in enumerate(self.limit(rows)):
            path = os.path.join(self._image_dir, f"{index}.jpg")
            row["image"].convert("RGB").save(path, "JPEG")
            question = row["question"]
            choices = "ABCDEFGHIJ"[: max(count_options(question), 4)]
            yield Sample(
                turns=[question],
                label={
                    "answer": str(row["answer"]).strip().upper(),
                    "choices": choices,
                },
                image=path,
            )

    def score(self, output: str, label: dict) -> Optional[bool]:
        return extract_choice(output, label["choices"]) == label["answer"]

    def cleanup(self) -> None:
        if self._image_dir and os.path.isdir(self._image_dir):
            shutil.rmtree(self._image_dir, ignore_errors=True)
        self._image_dir = None
