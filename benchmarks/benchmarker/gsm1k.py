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


@BENCHMARKS.register("gsm1k")
class GSM1KBenchmarker(Benchmarker):
    """GSM1K benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[int]]:
        """Load and preprocess GSM1K dataset."""
        # Read data
        dataset = load_dataset("ScaleAI/gsm1k")["test"]

        # Construct few shot examples from GSM8K
        few_shot_examples = """
Question: Alexis is applying for a new job and bought a new set of business clothes to wear to the interview. She went to a department store with a budget of $200 and spent $30 on a button-up shirt, $46 on suit pants, $38 on a suit coat, $11 on socks, and $18 on a belt. She also purchased a pair of shoes, but lost the receipt for them. She has $16 left from her budget. How much did Alexis pay for the shoes?
Answer: Let S be the amount Alexis paid for the shoes.
She spent S + 30 + 46 + 38 + 11 + 18 = S + <<+30+46+38+11+18=143>>143.
She used all but $16 of her budget, so S + 143 = 200 - 16 = 184.
Thus, Alexis paid S = 184 - 143 = $<<184-143=41>>41 for the shoes.
#### 41

Question: Tim rides his bike back and forth to work for each of his 5 workdays.  His work is 20 miles away.  He also goes for a weekend bike ride of 200 miles.    If he can bike at 25 mph how much time does he spend biking a week?
Answer: He bikes 20*2=<<20*2=40>>40 miles each day for work
So he bikes 40*5=<<40*5=200>>200 miles for work
That means he bikes a total of 200+200=<<200+200=400>>400 miles for work
So he bikes a total of 400/25=<<400/25=16>>16 hours
#### 16

Question: Mark buys a loaf of bread for $4.20 and some cheese for $2.05. He gives the cashier $7.00. If the cashier only has 1 quarter and 1 dime in his till, plus a bunch of nickels, how many nickels does Mark get in his change?
Answer: First subtract the cost of Mark's groceries from the amount he gives the cashier to find how much he gets in change: $7.00 - $4.20 - $2.05 = $<<7-4.2-2.05=0.75>>0.75
Then subtract the value of a quarter in cents (25) and the value of a dime in cents (10) from the change amount to find how much Mark gets paid in nickels: $0.75 - $0.25 - $0.10 = $<<0.75-0.25-0.10=0.40>>0.40
Now divide the amount Mark gets in nickels by the value per nickel in cents (5) to find how many nickels Mark gets: $0.40 / $0.05/nickel = <<0.40/0.05=8>>8 nickels
#### 8
        """

        questions = []
        labels = []
        for i, row in enumerate(dataset):
            if self.num_samples is not None and i >= self.num_samples:
                break

            questions.append({"question": row["question"]})

            try:
                answer = ast.literal_eval(row["answer"])
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
        """Compute accuracy for GSM8K by comparing numeric answers."""
        if not labels or len(labels) == 0:
            return None
        correct = sum(1 for pred, label in zip(predictions, labels) if pred == label)
        return correct / len(labels) if len(labels) > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for GSM8K with few-shot examples."""
        return create_few_shot_sgl_function(
            few_shot_examples=self.few_shot_examples,
            function_name="few_shot_gsm8k",
            answer_key="answer",
            stop=["Question", "Assistant:", "<|separator|>"],
        )
