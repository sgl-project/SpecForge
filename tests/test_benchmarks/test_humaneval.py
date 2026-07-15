from __future__ import annotations

import importlib.util
import sys
import types
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[2]
HUMANEVAL_PATH = REPO_ROOT / "benchmarks" / "benchmarker" / "humaneval.py"


def _load_humaneval_module(dataset):
    """Load HumanEval with lightweight stand-ins for optional benchmark deps."""

    package_name = "_dependency_light_benchmarker"
    module_name = f"{package_name}.humaneval"

    package = types.ModuleType(package_name)
    package.__path__ = []

    base = types.ModuleType(f"{package_name}.base")

    class Benchmarker:
        def __init__(self, num_samples=None, subset=None):
            self.num_samples = num_samples
            self.subset = subset

    base.Benchmarker = Benchmarker

    registry = types.ModuleType(f"{package_name}.registry")

    class BenchmarkRegistry:
        def register(self, _name):
            return lambda cls: cls

    registry.BENCHMARKS = BenchmarkRegistry()

    utils = types.ModuleType(f"{package_name}.utils")
    utils.create_simple_sgl_function = lambda **kwargs: kwargs

    datasets = types.ModuleType("datasets")
    datasets.load_dataset = lambda _name: {"test": dataset}

    spec = importlib.util.spec_from_file_location(module_name, HUMANEVAL_PATH)
    module = importlib.util.module_from_spec(spec)
    modules = {
        package_name: package,
        f"{package_name}.base": base,
        f"{package_name}.registry": registry,
        f"{package_name}.utils": utils,
        "datasets": datasets,
        module_name: module,
    }
    with mock.patch.dict(sys.modules, modules):
        spec.loader.exec_module(module)
    return module


class HumanEvalBenchmarkerTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.prompt = "def add(left, right):"
        cls.test_code = (
            "def check(candidate):\n"
            "    assert candidate(2, 3) == 5\n"
            "    assert candidate(-1, 1) == 0\n"
        )
        cls.module = _load_humaneval_module(
            [
                {
                    "prompt": cls.prompt,
                    "test": cls.test_code,
                    "entry_point": "add",
                    "canonical_solution": "    return left + right",
                }
            ]
        )

    def test_load_data_keeps_prompt_with_its_accuracy_label(self):
        benchmarker = self.module.HumanEvalBenchmarker()

        questions, labels = benchmarker.load_data()

        self.assertEqual(questions, [{"question": self.prompt}])
        self.assertEqual(labels[0]["prompt"], self.prompt)

    def test_accuracy_uses_the_label_without_hidden_question_state(self):
        benchmarker = self.module.HumanEvalBenchmarker()
        _, labels = benchmarker.load_data()

        correct = "def add(left, right):\n    return left + right"
        wrong = "def add(left, right):\n    return left - right"

        self.assertEqual(benchmarker.compute_accuracy([correct], labels), 1.0)
        self.assertEqual(benchmarker.compute_accuracy([wrong], labels), 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
