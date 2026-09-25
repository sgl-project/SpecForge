"""
Alpaca benchmark evaluation script.
"""

from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function

SYSTEM_PROMPT_WITH_INPUT = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request."""

SYSTEM_PROMPT_NO_INPUT = """Below is an instruction that describes a task. Write a response that appropriately completes the request."""


def format_alpaca_prompt(instruction: str, input_text: Optional[str] = None) -> str:
    """Format instruction and input into Alpaca prompt format."""
    if input_text:
        return f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:"
    else:
        return f"### Instruction:\n{instruction}\n\n### Response:"


@BENCHMARKS.register("alpaca")
class AlpacaBenchmarker(Benchmarker):
    """Alpaca benchmark implementation."""

    def __init__(
        self, num_samples: Optional[int] = None, subset: Optional[List[str]] = None
    ):
        if subset is None:
            subset = ["all"]

        num_samples = 1000 if num_samples is None else num_samples
        super().__init__(num_samples, subset)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[None]]:
        """Load and preprocess Alpaca dataset."""
        dataset = load_dataset("tatsu-lab/alpaca")["train"]

        # Shuffle with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)

        if self.num_samples is not None:
            dataset = dataset.select(range(min(self.num_samples, len(dataset))))

        questions = []
        labels = []
        for row in dataset:
            input_text = row["input"] if row["input"] else None
            prompt = format_alpaca_prompt(row["instruction"], input_text)

            questions.append({"question": prompt})
            labels.append(row["output"])

        return questions, labels

    def create_sgl_function(self):
        """Create SGL function for Alpaca (single-turn)."""
        # Note: system prompt handling may need adjustment based on your needs
        # since each example could have different system prompts
        return create_simple_sgl_function(
            function_name="answer_alpaca",
            answer_key="answer",
            system_prompt=None,  # prompt is self-contained
            max_tokens=self.get_max_new_tokens(),
        )

    def get_answer_keys(self) -> List[str]:
        """Return answer keys for single-turn conversation."""
        return ["answer"]
