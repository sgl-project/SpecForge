"""
GSM8K benchmark evaluation script.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def generate_question(row: Dict[str, Any]) -> str:
    question = row["question_content"].strip()
    return question


@BENCHMARKS.register("livecodebench")
class LCBBenchmarker(Benchmarker):
    """LiveCodeBench benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        # Read data
        ds = load_dataset("livecodebench/code_generation")["test"]

        questions = []
        labels = []
        for i in range((len(ds))):
            if self.num_samples is not None and i >= self.num_samples:
                break

            question_text = generate_question(ds[i])
            questions.append({"question": question_text})
            labels.append(None)
        return questions, labels

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="get_livecodebench_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )
