import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest import mock

from specforge.benchmarks import sglang


class SGLangBenchmarkTest(unittest.TestCase):
    def test_mt_bench_prompts_preserve_turns(self):
        rows = [{"prompt": ["first", "second"]}]
        with mock.patch("datasets.load_dataset", return_value=rows):
            prompts = sglang._load_prompts("mt-bench", max_samples=None)
        self.assertEqual(prompts, [["first", "second"]])

    def test_prompt_loader_rejects_empty_inputs(self):
        with mock.patch("datasets.load_dataset", return_value=[]):
            with self.assertRaisesRegex(ValueError, "did not contain any prompts"):
                sglang._load_prompts("gsm8k", max_samples=None)
        with self.assertRaisesRegex(ValueError, "--max-samples must be positive"):
            sglang._load_prompts("gsm8k", max_samples=0)

    def test_sglang_path_excludes_warmup_from_reported_totals(self):
        args = SimpleNamespace(
            model="thinkingmachines/Inkling",
            trust_remote_code=False,
            dataset="gsm8k",
            max_samples=None,
            num_prompts=3,
            concurrency=2,
            enable_thinking=False,
            base_url="http://127.0.0.1:30000",
            timeout_seconds=30,
        )
        response = {
            "meta_info": {
                "completion_tokens": 2,
                "spec_verify_ct": 1,
                "spec_accept_length": 3.0,
            }
        }
        flush_response = mock.Mock()
        flush_response.raise_for_status.return_value = None
        with (
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(),
            ),
            mock.patch.object(sglang, "_load_prompts", return_value=[["prompt"]]),
            mock.patch.object(sglang, "_apply_chat_template", return_value="rendered"),
            mock.patch.object(sglang, "_send_sglang", return_value=response) as send,
            mock.patch("requests.get", return_value=flush_response) as flush,
        ):
            result = sglang._run_sglang(args)

        self.assertEqual(send.call_count, args.num_prompts + args.concurrency)
        self.assertEqual(result.samples, 3)
        self.assertEqual(result.output_tokens, 6)
        self.assertEqual(result.spec_verify_count, 3)
        self.assertEqual(result.average_acceptance_length, 3.0)
        flush.assert_called_once_with(
            "http://127.0.0.1:30000/flush_cache",
            timeout=30,
        )

    def test_shared_cli_dispatches_sglang_benchmark(self):
        from specforge.cli import main

        with mock.patch.object(sglang, "run", return_value=7) as run:
            status = main(
                [
                    "benchmark",
                    "sglang",
                    "--model",
                    "thinkingmachines/Inkling",
                    "--dataset",
                    "gsm8k",
                ]
            )

        self.assertEqual(status, 7)
        self.assertEqual(run.call_args.args[0].model, "thinkingmachines/Inkling")

    def test_cli_help_describes_the_backend_not_an_algorithm(self):
        from specforge.cli import main

        output = StringIO()
        with redirect_stdout(output), self.assertRaises(SystemExit) as exited:
            main(["benchmark", "sglang", "--help"])

        self.assertEqual(exited.exception.code, 0)
        help_text = " ".join(output.getvalue().split())
        self.assertIn("a running SGLang server", help_text)
        self.assertNotIn("DSpark", help_text)
        self.assertNotIn("DFlash", help_text)


if __name__ == "__main__":
    unittest.main()
