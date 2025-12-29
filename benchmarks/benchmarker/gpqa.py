import random
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function

GPQA_QUERY_TEMPLATE = """
Answer the following multiple choice question. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

{Question}

A) {A}
B) {B}
C) {C}
D) {D}
""".strip()


def generate_question(row: Dict[str, Any]) -> str:
    gold_index = random.randint(0, 3)
    choices = [
        row["Incorrect Answer 1"],
        row["Incorrect Answer 2"],
        row["Incorrect Answer 3"],
    ]
    choices.insert(gold_index, row["Correct Answer"])

    question = GPQA_QUERY_TEMPLATE.format(
        Question=row["Question"].strip(),
        A=choices[0].strip(),
        B=choices[1].strip(),
        C=choices[2].strip(),
        D=choices[3].strip(),
    )

    # 0 means A, 1 means B, 2 means C, 3 means D
    answer = ["A", "B", "C", "D"][gold_index]
    return question, answer


@BENCHMARKS.register("gpqa")
class GPQABenchmarker(Benchmarker):
    """GPQA benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Load and preprocess GPQA dataset."""
        # Read data
        ds = load_dataset("Idavidrein/gpqa", "gpqa_main")["train"]

        questions = []
        labels = []
        for i in range((len(ds))):
            if self.num_samples is not None and i >= self.num_samples:
                break

            question_text, answer = generate_question(ds[i])
            questions.append({"question": question_text})
            labels.append(answer)
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[int]:
        if "Answer: " not in output:
            return None
        return output.split("Answer: ")[1].strip()

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels or len(labels) == 0:
            return None
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if len(labels) > 0 else 0.0

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="get_gpqa_mcq_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )
