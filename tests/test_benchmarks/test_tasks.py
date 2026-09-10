import unittest
from unittest import mock

from specforge.benchmarks.tasks import (
    TASKS,
    BenchmarkTask,
    Sample,
    TaskRegistry,
    answers,
    humaneval,
    mbpp,
)
from specforge.benchmarks.tasks.gsm8k import GSM8KTask
from specforge.benchmarks.tasks.mmstar import count_options
from specforge.benchmarks.tasks.mtbench import MTBenchTask


class AnswerHelpersTest(unittest.TestCase):
    def test_boxed_handles_nested_braces_and_takes_the_last_box(self):
        text = r"first \boxed{1} then \boxed{\frac{a}{b}}"
        self.assertEqual(answers.extract_boxed(text), r"\frac{a}{b}")
        self.assertEqual(answers.extract_boxed(r"\boxed 42 end"), "42")
        self.assertIsNone(answers.extract_boxed("no box"))

    def test_final_number_prefers_boxed_and_strips_separators(self):
        self.assertEqual(answers.extract_final_number(r"1 2 \boxed{1,234}"), "1234")
        self.assertEqual(answers.extract_final_number("so 3 then 4.5"), "4.5")
        self.assertIsNone(answers.extract_final_number("none"))

    def test_numbers_equal_tolerates_formatting(self):
        self.assertTrue(answers.numbers_equal("1,000", "1000"))
        self.assertTrue(answers.numbers_equal("2.0", "2"))
        self.assertFalse(answers.numbers_equal("2", "3"))
        self.assertFalse(answers.numbers_equal(None, "3"))

    def test_math_answers_equal_normalizes_latex(self):
        self.assertTrue(answers.math_answers_equal(r"\dfrac{1}{2}", r"\frac{1}{2}"))
        self.assertTrue(answers.math_answers_equal(r"\text{cm}", "cm"))
        self.assertFalse(answers.math_answers_equal("x", "y"))

    def test_choice_prefers_explicit_markers(self):
        self.assertEqual(answers.extract_choice("A and B... Answer: C"), "C")
        self.assertEqual(answers.extract_choice("答案：D"), "D")
        self.assertEqual(answers.extract_choice("I pick (B)."), "B")
        self.assertEqual(answers.extract_choice("Answer: E", "ABCDE"), "E")
        self.assertIsNone(answers.extract_choice("no idea"))

    def test_python_code_extraction(self):
        fenced = "text\n```python\ndef f():\n    return 1\n```\nmore"
        self.assertEqual(answers.extract_python_code(fenced), "def f():\n    return 1")
        bare = "Here:\ndef g(x):\n    return x\nDone."
        self.assertEqual(answers.extract_python_code(bare), "def g(x):\n    return x")
        self.assertIsNone(answers.extract_python_code("   "))

    def test_run_python_tests_reports_pass_fail_and_timeout(self):
        self.assertTrue(answers.run_python_tests("assert 1 + 1 == 2\n"))
        self.assertFalse(answers.run_python_tests("assert 1 + 1 == 3\n"))
        self.assertFalse(
            answers.run_python_tests("while True:\n    pass\n", timeout_seconds=0.5)
        )


class CodeTaskScoringTest(unittest.TestCase):
    def test_humaneval_scores_body_only_and_full_function_outputs(self):
        task = humaneval.HumanEvalTask()
        label = {
            "prompt": "def add(a, b):\n",
            "test": "def check(candidate):\n    assert candidate(2, 3) == 5\n",
            "entry_point": "add",
        }
        self.assertTrue(task.score("    return a + b", label))
        self.assertTrue(
            task.score("```python\ndef add(a, b):\n    return a + b\n```", label)
        )
        self.assertFalse(task.score("def add(a, b):\n    return a - b", label))

    def test_mbpp_runs_setup_then_asserts(self):
        task = mbpp.MBPPTask()
        label = {"setup": "import math", "tests": ["assert sq(3) == 9"]}
        self.assertTrue(task.score("def sq(x):\n    return x * x", label))
        self.assertFalse(task.score("def sq(x):\n    return x", label))
        self.assertFalse(task.score("", label))


class DatasetTaskTest(unittest.TestCase):
    def test_gsm8k_extracts_reference_answers_and_limits_rows(self):
        rows = [
            {"question": "Q1", "answer": "steps #### 1,200"},
            {"question": "Q2", "answer": "steps #### 7"},
        ]
        with mock.patch("datasets.load_dataset", return_value=rows):
            samples = GSM8KTask(num_samples=1).samples()
        self.assertEqual(len(samples), 1)
        self.assertTrue(samples[0].turns[0].startswith("Q1"))
        self.assertIn("boxed", samples[0].turns[0])
        self.assertEqual(samples[0].label, "1200")
        self.assertTrue(GSM8KTask().score(r"so \boxed{1200}", "1200"))

    def test_mtbench_keeps_both_turns_and_is_unscored(self):
        rows = [{"prompt": ["first", "second"]}]
        with mock.patch("datasets.load_dataset", return_value=rows):
            samples = MTBenchTask().samples()
        self.assertEqual(samples, [Sample(turns=["first", "second"])])
        self.assertFalse(MTBenchTask().scored)
        self.assertTrue(GSM8KTask().scored)

    def test_mmstar_option_counting(self):
        question = "What?\nOptions: A. x\nB. y\nC. z\nD. w\nE. v"
        self.assertEqual(count_options(question), 5)
        self.assertEqual(count_options("no options"), 0)


class RegistryTest(unittest.TestCase):
    def test_builtin_tasks_are_registered(self):
        self.assertTrue(
            {"gsm8k", "math500", "humaneval", "mbpp", "mtbench", "ceval", "mmlu"}
            <= set(TASKS.names())
        )

    def test_unknown_task_lists_alternatives(self):
        with self.assertRaisesRegex(KeyError, "unknown task 'nope'; available: aime"):
            TASKS.get("nope")

    def test_duplicate_names_are_rejected(self):
        registry = TaskRegistry()

        @registry.register
        class First(BenchmarkTask):
            name = "dup"

            def load(self):
                return []

        with self.assertRaisesRegex(ValueError, "already registered"):

            @registry.register
            class Second(BenchmarkTask):
                name = "dup"

                def load(self):
                    return []

        registry.register(First)  # re-registering the same class is fine
        self.assertEqual(registry.names(), ["dup"])

    def test_subsets_are_rejected_unless_supported(self):
        with self.assertRaisesRegex(ValueError, "does not support subsets"):
            GSM8KTask(subset=["x"])


if __name__ == "__main__":
    unittest.main()
