"""
GSM1K benchmark evaluation script.
"""

import ast
import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_few_shot_sgl_function

INVALID = -9999999


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


@BENCHMARKS.register("svamp")
class SVAMPBenchmarker(Benchmarker):
    """SVAMP benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Load and preprocess SVAMP dataset."""
        # Read data
        dataset = load_dataset("ChilleD/SVAMP")["test"]

        # Construct few shot examples from SVAMP
        few_shot_examples = """
Question: Sarah bought 45 cupcakes for her birthday party. 18 were chocolate and the rest were vanilla. If 23 guests ate vanilla cupcakes, how many vanilla cupcakes are left?
Answer: First, find the total number of vanilla cupcakes by subtracting the chocolate ones from the total: 45-18=27 vanilla cupcakes. Next, subtract the number of vanilla cupcakes eaten by guests: 27-23=4. There are 4 vanilla cupcakes left.
#### 4

Question: A library has 82 books on a shelf. 29 are mystery novels and 34 are science fiction. The rest are history books. How many history books are on the shelf?
Answer: First, add the mystery and science fiction books together: 29+34=63 books. Then, subtract that total from the overall number of books: 82-63=19. There are 19 history books.
#### 19

Question: Marcus had $50. He spent $12 on a movie ticket and $9 on popcorn. His friend later gave him $15 for helping with a project. How much money does Marcus have now?
Answer: First, calculate the total amount spent: 12+9=21. Subtract the spending from his original amount: 50-21=29. Finally, add the money his friend gave him: 29+15=44. Marcus now has $44.
#### 44
        """

        questions = []
        labels = []
        for i, row in enumerate(dataset):
            if self.num_samples is not None and i >= self.num_samples:
                break

            questions.append({"question": row["question_concat"]})

            try:
                answer = ast.literal_eval(row["Answer"])
            except SyntaxError:
                answer = INVALID
            labels.append(answer)

        # Store few_shot_examples for use in create_sgl_function
        self.few_shot_examples = few_shot_examples

        assert all(l != INVALID for l in labels), "Some labels are invalid"
        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[int]:
        """Extract numeric answer from model output."""
        return get_answer_value(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for SVAMP by comparing numeric answers."""
        if not labels or len(labels) == 0:
            return None
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if len(labels) > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for SVAMP with few-shot examples."""
        return create_few_shot_sgl_function(
            few_shot_examples=self.few_shot_examples,
            function_name="few_shot_gsm8k",
            answer_key="answer",
            stop=["Question", "Assistant:", "<|separator|>"],
        )
