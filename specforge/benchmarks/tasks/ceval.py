# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""C-Eval: Chinese multiple-choice exams across 52 subjects."""

from __future__ import annotations

from typing import Iterable, List, Optional

from specforge.benchmarks.tasks.answers import extract_choice
from specforge.benchmarks.tasks.base import TASKS, BenchmarkTask, Sample

SUBJECTS = (
    "accountant", "advanced_mathematics", "art_studies", "basic_medicine",
    "business_administration", "chinese_language_and_literature", "civil_servant",
    "clinical_medicine", "college_chemistry", "college_economics", "college_physics",
    "college_programming", "computer_architecture", "computer_network",
    "discrete_mathematics", "education_science", "electrical_engineer",
    "environmental_impact_assessment_engineer", "fire_engineer", "high_school_biology",
    "high_school_chemistry", "high_school_chinese", "high_school_geography",
    "high_school_history", "high_school_mathematics", "high_school_physics",
    "high_school_politics", "ideological_and_moral_cultivation", "law",
    "legal_professional", "logic", "mao_zedong_thought", "marxism",
    "metrology_engineer", "middle_school_biology", "middle_school_chemistry",
    "middle_school_geography", "middle_school_history", "middle_school_mathematics",
    "middle_school_physics", "middle_school_politics", "modern_chinese_history",
    "operating_system", "physician", "plant_protection", "probability_and_statistics",
    "professional_tour_guide", "sports_science", "tax_accountant",
    "teacher_qualification", "urban_and_rural_planner", "veterinary_medicine",
)  # fmt: skip


def format_question(question: str, options: List[str]) -> str:
    lines = [question, "", "选项："]
    lines += [f"{chr(65 + index)}. {option}" for index, option in enumerate(options)]
    lines += ["", "请从A、B、C、D中选择一个答案。"]
    return "\n".join(lines)


@TASKS.register
class CEvalTask(BenchmarkTask):
    name = "ceval"
    description = "C-Eval validation split; multiple-choice accuracy per subject subset"
    supports_subsets = True

    def load(self) -> Iterable[Sample]:
        from datasets import load_dataset

        subjects = self.subset or list(SUBJECTS)
        unknown = sorted(set(subjects) - set(SUBJECTS))
        if unknown:
            raise ValueError(f"unknown C-Eval subjects {unknown}")

        def rows():
            for subject in subjects:
                # The test split withholds answers; validation is the scored one.
                yield from load_dataset("ceval/ceval-exam", subject, split="val")

        for row in self.limit(rows()):
            options = [row[letter] for letter in "ABCD"]
            yield Sample(
                turns=[format_question(row["question"], options)],
                label=str(row["answer"]).strip().upper(),
            )

    def score(self, output: str, label: Optional[str]) -> Optional[bool]:
        return extract_choice(output) == label
