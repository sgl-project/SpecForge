"""
MBPP benchmark evaluation script.
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from datasets import load_dataset

from .base import Benchmarker
from .registry import BENCHMARKS
from .utils import create_simple_sgl_function


def extract_code_from_output(output: str) -> Optional[str]:
    """Extract Python code from model output (markdown block or `def ...:`)."""
    code_block_pattern = r"```(?:python)?\s*(.*?)\s*```"
    match = re.search(code_block_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    def_pattern = r"(def\s+\w+\([^)]*\):.*?)(?=\n\ndef\s+|\Z)"
    match = re.search(def_pattern, output, re.DOTALL)
    if match:
        return match.group(1).strip()
    return output.strip() if output.strip() else None


def check_code_passes_tests(code: str, test_code: str) -> bool:
    """Run `code` then `test_code` (which contains assertions) in a fresh namespace.

    Returns True iff no exception is raised. Simplified vs. the official MBPP
    evaluation framework — we just want a pass/fail signal.
    """
    try:
        namespace: Dict[str, Any] = {}
        exec(code, namespace)
        exec(test_code, namespace)
        return True
    except Exception:
        return False


def build_mbpp_prompt(text: str, test_list: List[str]) -> str:
    """Standard MBPP prompt format used in the original paper."""
    tests = "\n".join(test_list)
    return (
        "You are an expert Python programmer, and here is your task: "
        f"{text} Your code should pass these tests:\n\n{tests}\n\n[BEGIN]\n"
    )


@BENCHMARKS.register("mbpp")
class MBPPBenchmarker(Benchmarker):
    """MBPP benchmark implementation (sanitized split)."""

    def __init__(self, num_samples: Optional[int] = None):
        super().__init__(num_samples, None)

    def load_data(self) -> Tuple[List[Dict[str, Any]], List[Optional[Dict[str, Any]]]]:
        # Sanitized split is the standard one quoted in DFlash and most
        # other speculative-decoding benchmarks.
        dataset = load_dataset("google-research-datasets/mbpp", "sanitized")["test"]
        questions: List[Dict[str, Any]] = []
        labels: List[Optional[Dict[str, Any]]] = []

        for idx, q in enumerate(dataset):
            if self.num_samples is not None and idx >= self.num_samples:
                break

            # Sanitized split uses `prompt`; full split uses `text`.
            text = q.get("prompt") or q.get("text") or ""
            test_list = q.get("test_list", []) or []
            # Sanitized split exposes `test_imports` (List[str]); full split
            # exposes `test_setup_code` (single str). Combine both into one
            # block so accuracy checks can run imports the tests rely on.
            test_imports = q.get("test_imports", []) or []
            test_setup_code = q.get("test_setup_code", "") or ""
            test_setup = "\n".join([*test_imports, test_setup_code]).strip()

            prompt = build_mbpp_prompt(text, test_list)
            questions.append({"question": prompt})
            labels.append(
                {
                    "test_list": test_list,
                    "test_setup": test_setup,
                    "canonical_solution": q.get("code", ""),
                }
            )

        return questions, labels

    def extract_answer(self, output: str, label: Optional[Any] = None) -> Optional[str]:
        # MBPP responses sometimes wrap in [DONE]; strip that and any leading [BEGIN].
        if output is None:
            return None
        cleaned = output.strip()
        cleaned = cleaned.split("[DONE]")[0].strip()
        if cleaned.startswith("[BEGIN]"):
            cleaned = cleaned[len("[BEGIN]") :].strip()
        return extract_code_from_output(cleaned)

    def compute_accuracy(
        self, predictions: List[Any], labels: List[Any]
    ) -> Optional[float]:
        if not labels:
            return None
        if all(label is None for label in labels):
            return None

        correct = 0
        valid = 0
        for pred, label in zip(predictions, labels):
            if label is None or not isinstance(label, dict):
                continue
            valid += 1
            if pred is None:
                continue
            test_setup = label.get("test_setup", "") or ""
            test_list = label.get("test_list", []) or []
            test_code = test_setup + "\n" + "\n".join(test_list)
            if check_code_passes_tests(str(pred), test_code):
                correct += 1
        return correct / valid if valid > 0 else 0.0

    def create_sgl_function(self):
        return create_simple_sgl_function(
            function_name="get_mbpp_answer",
            answer_key="answer",
            max_tokens=self.get_max_new_tokens(),
            stop=["[DONE]"],
        )

    def get_max_new_tokens(self) -> int:
        return 1024
