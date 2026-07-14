"""
MMStar benchmark evaluation script.
"""

import os
import re
import shutil
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_image_sgl_function


def extract_mmstar_answer(
    output: str, options: Optional[List[str]] = None
) -> Optional[str]:
    """Extract answer from MMStar model output.

    MMStar questions typically have multiple choice options (A, B, C, D, etc.)
    """
    output_upper = output.strip().upper()

    # Try to find answer choice (A, B, C, D, etc.)
    # Direct match for single letter
    match = re.search(r"\b([A-Z])\b", output_upper)
    if match:
        letter = match.group(1)
        if options and len(options) > 0:
            # Validate that the letter is within valid range
            max_option = chr(64 + len(options))  # 'A' + (len-1)
            if "A" <= letter <= max_option:
                return letter
        else:
            # Assume A-D are valid
            if "A" <= letter <= "D":
                return letter

    # Try to find answer in parentheses or brackets
    for pattern in [
        r"\(([A-Z])\)",
        r"\[([A-Z])\]",
        r"答案[：:]\s*([A-Z])",
        r"Answer[：:]\s*([A-Z])",
        r"选择[：:]\s*([A-Z])",
    ]:
        match = re.search(pattern, output_upper)
        if match:
            letter = match.group(1)
            if options and len(options) > 0:
                max_option = chr(64 + len(options))
                if "A" <= letter <= max_option:
                    return letter
            elif "A" <= letter <= "D":
                return letter

    return None


@BENCHMARKS.register("mmstar")
class MMStarBenchmarker(Benchmarker):
    """MMStar benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)
        """Initialize benchmark and set up cache directory."""
        self.cache_dir = None
        self.options_list = []  # Store options for each question

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[str]]]:
        """Load and preprocess MMStar dataset."""
        self.cache_dir = os.path.join(".cache", "mmstar_specforge")
        image_dir = os.path.join(self.cache_dir, "images")
        os.makedirs(self.cache_dir, exist_ok=True)
        os.makedirs(image_dir, exist_ok=True)
        print(f"Created temporary image directory: {self.cache_dir}")

        dataset = load_dataset("Lin-Chen/MMStar")["val"]
        questions = []
        labels = []
        self.options_list = []

        for idx, q in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            image = q["image"]
            image_path = os.path.join(self.cache_dir, q["meta_info"]["image_path"])
            image.convert("RGB").save(image_path, "JPEG")

            # Extract question and options
            question_full = q["question"]
            if "Options:" in question_full:
                question_text, options_text = question_full.split("Options:", 1)
                question_text = question_text.strip()
                # Parse options (typically A. option1 B. option2 etc.)
                options = []
                for line in options_text.strip().split("\n"):
                    line = line.strip()
                    if line and re.match(r"^[A-Z]\.", line):
                        option_text = re.sub(r"^[A-Z]\.\s*", "", line).strip()
                        options.append(option_text)
                self.options_list.append(options)
            else:
                question_text = question_full.strip()
                self.options_list.append([])

            item = {
                "image_path": image_path,
                "question": question_text,
            }
            questions.append(item)

            # Extract ground truth answer
            answer = None
            if "answer" in q:
                answer = str(q["answer"]).strip().upper()
            elif "correct_answer" in q:
                answer = str(q["correct_answer"]).strip().upper()
            elif "ground_truth" in q:
                answer = str(q["ground_truth"]).strip().upper()

            # Validate answer is a valid option letter
            if answer and len(answer) == 1 and "A" <= answer <= "Z":
                if self.options_list[-1]:
                    max_option = chr(64 + len(self.options_list[-1]))
                    if answer <= max_option:
                        labels.append(answer)
                    else:
                        labels.append(None)
                else:
                    labels.append(answer)
            else:
                labels.append(None)

        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Extract answer from model output."""
        # Use the options for the current question if available
        # Note: We can't easily get the question index here, so we'll use a simpler approach
        return extract_mmstar_answer(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for MMStar by comparing answer choices."""
        if not labels or len(labels) == 0:
            return None
        if all(label is None for label in labels):
            return None

        correct = 0
        valid_count = 0
        for pred, label in zip(predictions, labels):
            if label is not None:
                valid_count += 1
                if pred is not None:
                    # Normalize to uppercase for comparison
                    pred_normalized = str(pred).strip().upper()
                    label_normalized = str(label).strip().upper()
                    if pred_normalized == label_normalized:
                        correct += 1

        return correct / valid_count if valid_count > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for MMStar (image-based Q&A)."""
        return create_image_sgl_function(
            function_name="get_mmstar_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def run(self, *args, **kwargs):
        """Run benchmark and clean up cache directory."""
        try:
            return super().run(*args, **kwargs)
        finally:
            # Clean up cache directory
            if self.cache_dir and os.path.exists(self.cache_dir):
                shutil.rmtree(self.cache_dir)
                print(f"Deleted temporary directory: {self.cache_dir}")
