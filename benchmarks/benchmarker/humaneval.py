"""
HumanEval benchmark evaluation script.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def extract_code_from_output(output: str) -> Optional[str]:
    """Extract Python code from model output.

    Tries to extract code blocks or function definitions.
    """
    # Try to find code in markdown code blocks
    code_block_pattern = r"```(?:python)?\n(.*?)```"
    match = re.search(code_block_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Try to find function definition (common in HumanEval)
    # Look for "def " followed by code until the next def or end of string
    def_pattern = r"(def\s+\w+\([^)]*\):.*?)(?=\n\ndef\s+|\Z)"
    match = re.search(def_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()

    # Fallback: return the output as-is (might already be code)
    return output.strip() if output.strip() else None


def check_code_passes_tests(code: str, test_code: str, entry_point: str) -> bool:
    """Check if generated code passes the test cases.

    This is a simplified version. For full evaluation, use the official
    HumanEval evaluation framework.

    HumanEval test code typically contains assertions that will raise
    AssertionError if the code doesn't pass. If execution completes without
    exceptions, the tests pass.
    """
    try:
        # Create a safe execution environment
        namespace = {}
        # Execute the code (function definition)
        exec(code, namespace)
        # Execute the test code (which contains assertions)
        # If no exception is raised, the tests pass
        exec(test_code, namespace)
        return True
    except AssertionError:
        # Assertion failed - test didn't pass
        return False
    except Exception:
        # Any other exception (syntax error, runtime error, etc.) means test failed
        return False


@BENCHMARKS.register("humaneval")
class HumanEvalBenchmarker(Benchmarker):
    """HumanEval benchmark implementation."""

    def __init__(self, num_samples: Optional[int] = None):
        """Initialize benchmark and store test cases."""
        super().__init__(num_samples, None)
        self.test_cases = []
        self.entry_points = []

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[Dict[str, str]]]]:
        """Load and preprocess HumanEval dataset."""
        dataset = load_dataset("openai/openai_humaneval")["test"]
        questions = []
        labels = []
        self.test_cases = []
        self.entry_points = []

        for idx, q in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            questions.append({"question": q["prompt"]})

            # Store test case and entry point for evaluation
            test_code = q.get("test", "")
            entry_point = q.get("entry_point", "")
            self.test_cases.append(test_code)
            self.entry_points.append(entry_point)

            # Store canonical solution as reference (optional, for comparison)
            canonical_solution = q.get("canonical_solution", "")
            labels.append(
                {
                    "test": test_code,
                    "entry_point": entry_point,
                    "canonical_solution": canonical_solution,
                }
            )

        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        """Extract code from model output."""
        return extract_code_from_output(output)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        """Compute accuracy for HumanEval by checking if code passes tests.

        Note: This is a simplified evaluation. For official pass@k metrics,
        use the HumanEval evaluation framework.
        """
        if not labels or len(labels) == 0:
            return None
        if all(label is None for label in labels):
            return None

        correct = 0
        valid_count = 0

        for i, (pred, label) in enumerate(zip(predictions, labels)):
            if label is not None and isinstance(label, dict):
                valid_count += 1
                if pred is not None:
                    try:
                        # Get the prompt (function signature and docstring)
                        prompt = self.questions[i]["question"]
                        entry_point = label.get("entry_point", "")

                        # The prompt contains the function signature (e.g., "def function_name(...):")
                        # The generated code might be:
                        # 1. Just the function body (what we want) - need to combine with prompt
                        # 2. The complete function including signature - use as-is
                        # 3. Code in markdown blocks - already extracted by extract_code_from_output

                        pred_str = str(pred).strip()

                        # Check if pred already contains a complete function definition
                        # (starts with "def " and contains the entry_point function name)
                        if pred_str.startswith("def ") and entry_point:
                            # Check if this is the same function (by name)
                            func_name_match = re.match(r"def\s+(\w+)\s*\(", pred_str)
                            if (
                                func_name_match
                                and func_name_match.group(1) == entry_point
                            ):
                                # Generated code includes complete function, use it as-is
                                full_code = pred_str
                            else:
                                # Different function or no match, combine with prompt
                                full_code = prompt + "\n" + pred_str
                        elif pred_str.startswith("def "):
                            # Has function definition but we can't verify entry_point, use as-is
                            full_code = pred_str
                        else:
                            # Generated code is just the body, combine with prompt
                            full_code = prompt + "\n" + pred_str

                        # Check if code passes tests
                        test_code = label.get("test", "")

                        if test_code and check_code_passes_tests(
                            full_code, test_code, entry_point
                        ):
                            correct += 1
                    except Exception as e:
                        # If evaluation fails, consider it incorrect
                        # Uncomment for debugging: print(f"Error evaluating code {i}: {e}")
                        pass

        return correct / valid_count if valid_count > 0 else 0.0

    def create_sgl_function(self):
        """Create SGL function for HumanEval."""
        return create_simple_sgl_function(
            function_name="get_humaneval_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
        )

    def get_max_new_tokens(self) -> int:
        """HumanEval code generation requires more tokens."""
        return 1024
