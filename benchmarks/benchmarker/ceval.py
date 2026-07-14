"""
C-Eval benchmark evaluation script.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import concatenate_datasets, load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def extract_answer(answer_str: str) -> str:
    """Extract the answer choice (A, B, C, D) from the model output."""
    # Try to find the answer in various formats
    answer_str = answer_str.strip().upper()

    # Direct match for single letter
    match = re.search(r"\b([ABCD])\b", answer_str)
    if match:
        return match.group(1)

    # Try to find answer in parentheses or brackets
    for pattern in [
        r"\(([ABCD])\)",
        r"\[([ABCD])\]",
        r"答案[：:]\s*([ABCD])",
        r"Answer[：:]\s*([ABCD])",
    ]:
        match = re.search(pattern, answer_str, re.IGNORECASE)
        if match:
            return match.group(1).upper()

    # Try to find the first occurrence of A, B, C, or D
    match = re.search(r"([ABCD])", answer_str)
    if match:
        return match.group(1)

    return None


def format_question(question: str, options: List[str]) -> str:
    """Format the question with options."""
    prompt = question + "\n\n选项：\n"
    for i, option in enumerate(options):
        prompt += f"{chr(65 + i)}. {option}\n"
    prompt += "\n请从A、B、C、D中选择一个答案。"
    return prompt


@BENCHMARKS.register("ceval")
class CEvalBenchmarker(Benchmarker):
    """C-Eval benchmark implementation."""

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        if subset is None:
            subset = "all"
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[str]]:
        """Load and preprocess C-Eval dataset."""
        all_configs = [
            "accountant",
            "advanced_mathematics",
            "art_studies",
            "basic_medicine",
            "business_administration",
            "chinese_language_and_literature",
            "civil_servant",
            "clinical_medicine",
            "college_chemistry",
            "college_economics",
            "college_physics",
            "college_programming",
            "computer_architecture",
            "computer_network",
            "discrete_mathematics",
            "education_science",
            "electrical_engineer",
            "environmental_impact_assessment_engineer",
            "fire_engineer",
            "high_school_biology",
            "high_school_chemistry",
            "high_school_chinese",
            "high_school_geography",
            "high_school_history",
            "high_school_mathematics",
            "high_school_physics",
            "high_school_politics",
            "ideological_and_moral_cultivation",
            "law",
            "legal_professional",
            "logic",
            "mao_zedong_thought",
            "marxism",
            "metrology_engineer",
            "middle_school_biology",
            "middle_school_chemistry",
            "middle_school_geography",
            "middle_school_history",
            "middle_school_mathematics",
            "middle_school_physics",
            "middle_school_politics",
            "modern_chinese_history",
            "operating_system",
            "physician",
            "plant_protection",
            "probability_and_statistics",
            "professional_tour_guide",
            "sports_science",
            "tax_accountant",
            "teacher_qualification",
            "urban_and_rural_planner",
            "veterinary_medicine",
        ]

        # Select configs to load
        if self.subset == "all":
            configs_to_load = all_configs
        else:
            for subset in self.subset:
                assert (
                    subset in all_configs
                ), f"Subset {subset} not found in C-Eval dataset"
            configs_to_load = self.subset

        # Load datasets
        try:
            datasets = []
            for config in configs_to_load:
                try:
                    ds = load_dataset("ceval/ceval-exam", name=config, split="test")
                    datasets.append(ds)
                    print(f"Loaded config '{config}' with {len(ds)} samples")
                except Exception as e:
                    print(f"Warning: Failed to load config '{config}': {e}")
            if len(datasets) == 0:
                raise ValueError("No configs could be loaded")
            dataset = concatenate_datasets(datasets)
            print(
                f"Successfully loaded C-Eval dataset with all configs (total: {len(dataset)} samples)"
            )
        except Exception as e:
            print(e)
            print(f"Failed to load C-Eval dataset from 'ceval/ceval-exam': {e}")
            print("Please ensure the dataset is available or install it manually.")
            print("You can try: pip install datasets")
            print("Or download from: https://huggingface.co/datasets/ceval/ceval-exam")
            return [], []

        # Process questions
        questions = []
        labels = []
        for idx, item in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            # Handle different dataset formats
            question_text = None
            if "question" in item:
                question_text = item["question"]
            elif "inputs" in item:
                question_text = item["inputs"]
            elif "problem" in item:
                question_text = item["problem"]
            elif "content" in item:
                question_text = item["content"]

            if not question_text:
                continue

            # Get options - C-Eval typically has options as a list or dict
            options = None
            if "options" in item:
                options = item["options"]
                if isinstance(options, dict):
                    # Convert dict to list in order A, B, C, D
                    options = [
                        options.get("A", ""),
                        options.get("B", ""),
                        options.get("C", ""),
                        options.get("D", ""),
                    ]
                elif isinstance(options, list):
                    # Ensure we have 4 options
                    while len(options) < 4:
                        options.append("")
            elif "choices" in item:
                options = item["choices"]
                if isinstance(options, dict):
                    options = [
                        options.get("A", ""),
                        options.get("B", ""),
                        options.get("C", ""),
                        options.get("D", ""),
                    ]
            else:
                # Try to construct options from A, B, C, D fields
                options = [
                    item.get("A", item.get("option_A", "")),
                    item.get("B", item.get("option_B", "")),
                    item.get("C", item.get("option_C", "")),
                    item.get("D", item.get("option_D", "")),
                ]

            # Filter out empty options
            if options:
                options = [str(opt).strip() for opt in options if opt]
                if len(options) < 2:  # Need at least 2 options
                    continue
            else:
                continue

            # Get answer
            answer = None
            if "answer" in item:
                answer = str(item["answer"]).upper().strip()
            elif "target" in item:
                answer = str(item["target"]).upper().strip()
            elif "label" in item:
                answer = str(item["label"]).upper().strip()
            elif "correct" in item:
                answer = str(item["correct"]).upper().strip()

            # Validate answer
            if answer and answer in ["A", "B", "C", "D"]:
                # Format question
                formatted_question = format_question(question_text, options)
                questions.append({"question": formatted_question})
                labels.append(answer)

        if len(questions) == 0:
            print("No valid questions found. Please check the dataset format.")
            print(
                "Sample item keys:",
                list(dataset[0].keys()) if len(dataset) > 0 else "No items",
            )
            return [], []

        return questions, labels

    def create_sgl_function(self):
        """Create SGL function for C-Eval."""
        return create_simple_sgl_function(
            function_name="get_ceval_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def extract_answer(self, output: str, label: Any = None) -> str:
        """Extract answer choice from model output."""
        return extract_answer(output)

    def compute_accuracy(self, predictions: List[str], labels: List[str]) -> float:
        """Compute accuracy metric."""
        correct = 0
        valid_count = 0
        for i in range(len(predictions)):
            if predictions[i] is not None:  # Only count valid predictions
                valid_count += 1
                if predictions[i] == labels[i]:
                    correct += 1
        return correct / valid_count if valid_count > 0 else 0.0
