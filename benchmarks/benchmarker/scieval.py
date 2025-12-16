"""
GSM8K benchmark evaluation script.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function

QUESTION_PROMPT = """Given a question and four options, please select the right answer. The last line of your response should be of the following format: 'Answer: $LETTER' (without quotes) where LETTER is one of ABCD. Think step by step before answering.

Question: {question}
"""


def generate_question(row: Dict[str, Any]) -> str:
    assert row["type"] == "multiple-choice", "expect only multiple-choice questions"
    question = row["question"]
    question_prompt = QUESTION_PROMPT.format(question=question)
    answer = row["answer"][0]
    return question_prompt, answer


@BENCHMARKS.register("scieval")
class SciEvalBenchmarker(Benchmarker):
    """SciEval benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        # Read data
        ds = load_dataset("OpenDFM/SciEval")["test"]

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
            function_name="get_scieval_mcq_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )
