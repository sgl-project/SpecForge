"""
GSM8K benchmark evaluation script.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from sglang.utils import download_and_cache_file, read_jsonl

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_few_shot_sgl_function

INVALID = -9999999


def get_one_example(lines: List[Dict], i: int, include_answer: bool) -> str:
    """Format a single example as plain text (legacy concat helper)."""
    ret = "Question: " + lines[i]["question"] + "\nAnswer:"
    if include_answer:
        ret += " " + lines[i]["answer"]
    return ret


def get_few_shot_messages(lines: List[Dict], k: int) -> List[Dict[str, str]]:
    """Few-shot as a list of conversation turns suitable for chat templates."""
    msgs: List[Dict[str, str]] = []
    for i in range(k):
        msgs.append({"role": "user", "content": lines[i]["question"]})
        msgs.append({"role": "assistant", "content": lines[i]["answer"]})
    return msgs


def get_answer_value(answer_str: str) -> int:
    """Extract numeric answer from model output."""
    answer_str = answer_str.replace(",", "")
    numbers = re.findall(r"\d+", answer_str)
    if len(numbers) < 1:
        return INVALID
    try:
        return ast.literal_eval(numbers[-1])
    except SyntaxError:
        return INVALID


@BENCHMARKS.register("gsm8k")
class GSM8KBenchmarker(Benchmarker):
    """GSM8K benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Load and preprocess GSM8K dataset."""
        # Read data
        url = "https://raw.githubusercontent.com/openai/grade-school-math/master/grade_school_math/data/test.jsonl"
        data_path = download_and_cache_file(url)
        lines = list(read_jsonl(data_path))

        # Few-shot examples as proper conversation turns (the first 5 records
        # become alternating user/assistant messages prepended to each query).
        self.few_shot_messages = get_few_shot_messages(lines, 5)
        # Skip the few-shot records when iterating actual queries.
        eval_lines = lines[5:]

        questions = []
        labels = []
        for i in range(len(eval_lines)):
            if self.num_samples is not None and i >= self.num_samples:
                break
            questions.append({"question": eval_lines[i]["question"]})
            labels.append(get_answer_value(eval_lines[i]["answer"]))

        assert all(l != INVALID for l in labels), "Some labels are invalid"
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[int]:
        """Extract numeric answer from model output."""
        return get_answer_value(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for GSM8K by comparing numeric answers."""
        if not labels or len(labels) == 0:
            return None
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if len(labels) > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for GSM8K with few-shot examples as message turns."""
        return create_few_shot_sgl_function(
            few_shot_messages=self.few_shot_messages,
            function_name="few_shot_gsm8k",
            answer_key="answer",
            stop=["Question", "Assistant:", "<|separator|>"],
        )
