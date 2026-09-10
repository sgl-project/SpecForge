import unittest
from typing import Optional
from unittest import mock

from specforge.benchmarks.client import Generation, SGLangClient
from specforge.benchmarks.config import BenchmarkConfig
from specforge.benchmarks.report import (
    BenchmarkReport,
    RunRecord,
    TaskMetrics,
    format_summary,
)
from specforge.benchmarks.runner import TaskRunner, run_benchmark, sampling_params_for
from specforge.benchmarks.server import build_server_args
from specforge.benchmarks.tasks import BenchmarkTask, Sample


class FakeRenderer:
    def render(self, messages):
        return " | ".join(f"{m['role']}:{m['content']}" for m in messages)


class FakeClient:
    """Echoes a canned reply and counts requests."""

    def __init__(self, reply="reply 4", tokens=2, verify=1):
        self.reply = reply
        self.tokens = tokens
        self.verify = verify
        self.prompts = []
        self.flushes = 0

    def flush_cache(self):
        self.flushes += 1
        return True

    def generate(self, prompt, sampling_params, image_path=None):
        self.prompts.append(prompt)
        return Generation(
            text=self.reply,
            completion_tokens=self.tokens,
            spec_verify_count=self.verify,
        )


class NumberTask(BenchmarkTask):
    name = "numbers"

    def __init__(self, samples, **kwargs):
        super().__init__(**kwargs)
        self._samples = samples

    def load(self):
        return self._samples

    def score(self, output: str, label) -> Optional[bool]:
        if label is None:
            return None
        return output.endswith(label)


class ChatTask(BenchmarkTask):
    name = "chat"
    system_prompt = "be brief"

    def load(self):
        return [Sample(turns=["hi", "again"])]


def _runner(task, client, concurrency=2, warmup=True):
    return TaskRunner(
        task=task,
        client=client,
        renderer=FakeRenderer(),
        sampling_params={"max_new_tokens": 8},
        concurrency=concurrency,
        warmup=warmup,
        progress=False,
    )


class TaskRunnerTest(unittest.TestCase):
    def test_warmup_is_excluded_from_totals_and_flushes_cache(self):
        samples = [Sample(turns=["q"], label="4") for _ in range(3)]
        client = FakeClient()
        metrics = _runner(NumberTask(samples), client).run()

        # 2 warmup requests + 3 timed requests.
        self.assertEqual(len(client.prompts), 5)
        self.assertEqual(client.flushes, 1)
        self.assertEqual(metrics.num_samples, 3)
        self.assertEqual(metrics.num_requests, 3)
        self.assertEqual(metrics.output_tokens, 6)
        self.assertEqual(metrics.spec_verify_count, 3)
        self.assertEqual(metrics.accept_length, 2.0)
        self.assertGreater(metrics.throughput_tokens_per_second, 0)

    def test_accuracy_counts_only_scored_samples(self):
        samples = [
            Sample(turns=["q"], label="4"),
            Sample(turns=["q"], label="5"),
            Sample(turns=["q"], label=None),
        ]
        metrics = _runner(NumberTask(samples), FakeClient(), warmup=False).run()
        self.assertEqual(metrics.num_scored, 2)
        self.assertEqual(metrics.accuracy, 0.5)

    def test_unscored_tasks_report_no_accuracy(self):
        metrics = _runner(ChatTask(), FakeClient(), warmup=False).run()
        self.assertIsNone(metrics.accuracy)
        self.assertEqual(metrics.num_scored, 0)

    def test_multi_turn_replays_history_with_system_prompt(self):
        client = FakeClient(reply="ok")
        metrics = _runner(ChatTask(), client, warmup=False).run()
        self.assertEqual(metrics.num_samples, 1)
        self.assertEqual(metrics.num_requests, 2)
        self.assertEqual(
            client.prompts,
            [
                "system:be brief | user:hi",
                "system:be brief | user:hi | assistant:ok | user:again",
            ],
        )

    def test_no_speculation_leaves_accept_length_unset(self):
        client = FakeClient(verify=None)
        metrics = _runner(ChatTask(), client, warmup=False).run()
        self.assertIsNone(metrics.accept_length)
        self.assertIsNone(metrics.spec_verify_count)

    def test_empty_task_is_an_error(self):
        with self.assertRaisesRegex(ValueError, "no samples"):
            _runner(NumberTask([]), FakeClient()).run()


class SamplingParamsTest(unittest.TestCase):
    def test_precedence_task_override_then_global_then_task_default(self):
        class Capped(BenchmarkTask):
            name = "capped"
            max_new_tokens = 111
            stop = ["END"]

            def load(self):
                return []

        base = {"model": "m", "tasks": [{"name": "gsm8k"}]}
        config = BenchmarkConfig.model_validate(base)
        params = sampling_params_for(config, config.tasks[0], Capped())
        self.assertEqual(params["max_new_tokens"], 111)
        self.assertEqual(params["stop"], ["END"])
        self.assertEqual(params["temperature"], 0.0)

        config = BenchmarkConfig.model_validate(
            {**base, "sampling": {"max_new_tokens": 222}}
        )
        self.assertEqual(
            sampling_params_for(config, config.tasks[0], Capped())["max_new_tokens"],
            222,
        )

        config = BenchmarkConfig.model_validate(
            {**base, "tasks": [{"name": "gsm8k", "max_new_tokens": 333}]}
        )
        self.assertEqual(
            sampling_params_for(config, config.tasks[0], Capped())["max_new_tokens"],
            333,
        )


class ServerArgsTest(unittest.TestCase):
    def test_baseline_and_speculative_entries(self):
        config = BenchmarkConfig.model_validate(
            {
                "model": "target",
                "draft_model": "draft",
                "tasks": [{"name": "gsm8k"}],
                "concurrency": 4,
                "trust_remote_code": True,
                "server": {
                    "launch": True,
                    "base_url": "http://0.0.0.0:31000",
                    "args": ["--tp-size", "2"],
                },
                "matrix": [
                    {},
                    {"steps": 3, "topk": 1, "draft_tokens": 4, "batch_size": 8},
                ],
            }
        )
        baseline, spec = config.matrix
        args = build_server_args(config, baseline)
        self.assertEqual(
            args[:6], ["--model-path", "target", "--host", "0.0.0.0", "--port", "31000"]
        )
        self.assertIn("--max-running-requests", args)
        self.assertEqual(args[args.index("--cuda-graph-max-bs-decode") + 1], "4")
        self.assertEqual(args[args.index("--max-running-requests") + 1], "4")
        self.assertNotIn("--speculative-algorithm", args)
        self.assertIn("--trust-remote-code", args)
        self.assertEqual(args[-2:], ["--tp-size", "2"])

        args = build_server_args(config, spec)
        self.assertEqual(args[args.index("--max-running-requests") + 1], "8")
        self.assertEqual(args[args.index("--speculative-algorithm") + 1], "EAGLE3")
        self.assertEqual(
            args[args.index("--speculative-draft-model-path") + 1], "draft"
        )
        self.assertEqual(args[args.index("--speculative-num-steps") + 1], "3")
        self.assertEqual(args[args.index("--speculative-eagle-topk") + 1], "1")
        self.assertEqual(args[args.index("--speculative-num-draft-tokens") + 1], "4")


class RunBenchmarkTest(unittest.TestCase):
    def test_requires_a_healthy_server_when_not_launching(self):
        config = BenchmarkConfig.model_validate(
            {"model": "m", "tasks": [{"name": "gsm8k"}]}
        )
        with (
            mock.patch("specforge.benchmarks.runner.PromptRenderer"),
            mock.patch.object(SGLangClient, "is_ready", return_value=False),
        ):
            with self.assertRaisesRegex(RuntimeError, "no healthy SGLang server"):
                run_benchmark(config, progress=False)

    def test_unknown_task_fails_before_any_server_work(self):
        config = BenchmarkConfig.model_validate(
            {"model": "m", "tasks": [{"name": "nope"}]}
        )
        with mock.patch("specforge.benchmarks.runner.PromptRenderer") as renderer:
            with self.assertRaises(KeyError):
                run_benchmark(config, progress=False)
        renderer.assert_not_called()

    def test_matrix_run_launches_one_server_per_entry(self):
        config = BenchmarkConfig.model_validate(
            {
                "model": "m",
                "draft_model": "d",
                "tasks": [{"name": "gsm8k", "num_samples": 1}],
                "server": {"launch": True},
                "matrix": [{}, {"steps": 3, "topk": 1, "draft_tokens": 4}],
            }
        )
        rows = [{"question": "q", "answer": "#### 4"}]
        with (
            mock.patch(
                "specforge.benchmarks.runner.PromptRenderer",
                return_value=FakeRenderer(),
            ),
            mock.patch(
                "specforge.benchmarks.runner.SGLangClient", return_value=FakeClient()
            ),
            mock.patch("specforge.benchmarks.runner.ManagedServer") as server,
            mock.patch("datasets.load_dataset", return_value=rows),
        ):
            report = run_benchmark(config, progress=False)

        self.assertEqual(server.call_count, 2)
        self.assertEqual(
            [run.config_label for run in report.runs], ["baseline", "eagle3-s3-k1-d4"]
        )
        self.assertEqual(report.runs[1].config["steps"], 3)
        self.assertEqual(report.runs[0].metrics.accuracy, 1.0)


class ReportTest(unittest.TestCase):
    def test_summary_table_and_json_round_trip(self):
        report = BenchmarkReport(
            model="m", draft_model=None, sampling={}, concurrency=1
        )
        report.runs.append(
            RunRecord(
                config=None,
                task={"name": "gsm8k"},
                metrics=TaskMetrics(
                    num_samples=2,
                    num_requests=2,
                    output_tokens=10,
                    latency_seconds=2.0,
                    throughput_tokens_per_second=5.0,
                    accept_length=None,
                    accuracy=0.5,
                    num_scored=2,
                ),
            )
        )
        table = format_summary(report)
        self.assertIn("server", table)
        self.assertIn("50.0%", table)
        self.assertEqual(report.to_dict()["runs"][0]["metrics"]["output_tokens"], 10)
        self.assertEqual(report.to_dict()["runs"][0]["config"], None)


if __name__ == "__main__":
    unittest.main()
