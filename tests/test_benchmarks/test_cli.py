import contextlib
import io
import unittest
from unittest import mock

from specforge.benchmarks.config import BenchmarkConfig
from specforge.cli import main


class BenchmarkCliTest(unittest.TestCase):
    def test_help_describes_the_server_not_an_algorithm(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["benchmark", "--help"]), 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("SGLang server", help_text)
        for algorithm in ("EAGLE3", "DFlash", "DSpark"):
            self.assertNotIn(algorithm, help_text)

    def test_list_tasks_prints_the_registry(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(main(["benchmark", "--list-tasks"]), 0)
        self.assertIn("gsm8k", output.getvalue())
        self.assertIn("mtbench", output.getvalue())

    def test_flags_and_overrides_build_the_config(self):
        with mock.patch("specforge.benchmarks.runner.main", return_value=0) as run:
            status = main(
                [
                    "benchmark",
                    "--model",
                    "thinkingmachines/Inkling",
                    "--task",
                    "gsm8k:5",
                    "--task",
                    "ceval:3:accountant",
                    "--base-url",
                    "http://localhost:1",
                    "--trust-remote-code",
                    "sampling.temperature=0.5",
                ]
            )
        self.assertEqual(status, 0)
        config = run.call_args.args[0]
        self.assertIsInstance(config, BenchmarkConfig)
        self.assertEqual(config.model, "thinkingmachines/Inkling")
        self.assertEqual([task.name for task in config.tasks], ["gsm8k", "ceval"])
        self.assertEqual(config.tasks[1].subset, ["accountant"])
        self.assertEqual(config.server.base_url, "http://localhost:1")
        self.assertTrue(config.trust_remote_code)
        self.assertFalse(config.enable_thinking)
        self.assertEqual(config.sampling.temperature, 0.5)

    def test_invalid_config_is_a_usage_error(self):
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            status = main(
                ["benchmark", "--model", "m", "--task", "gsm8k", "bogus.field=1"]
            )
        self.assertEqual(status, 2)
        self.assertIn("does not exist", stderr.getvalue())

        with contextlib.redirect_stderr(stderr):
            status = main(["benchmark", "--task", "gsm8k"])
        self.assertEqual(status, 2)
        self.assertIn("model", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
