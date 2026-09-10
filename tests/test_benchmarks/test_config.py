import json
import os
import tempfile
import unittest

from specforge.benchmarks.config import (
    BenchmarkConfig,
    SpeculativeConfig,
    TaskConfig,
    build_config,
)


class TaskSpecParsingTest(unittest.TestCase):
    def test_name_only(self):
        self.assertEqual(TaskConfig.parse("gsm8k"), TaskConfig(name="gsm8k"))

    def test_name_and_count(self):
        self.assertEqual(
            TaskConfig.parse("gsm8k:200"), TaskConfig(name="gsm8k", num_samples=200)
        )

    def test_name_count_and_subsets(self):
        self.assertEqual(
            TaskConfig.parse("ceval:5:accountant,law"),
            TaskConfig(name="ceval", num_samples=5, subset=["accountant", "law"]),
        )

    def test_subsets_without_count(self):
        self.assertEqual(
            TaskConfig.parse("mmlu::abstract_algebra"),
            TaskConfig(name="mmlu", subset=["abstract_algebra"]),
        )

    def test_rejects_malformed_specs(self):
        for spec in ("", "gsm8k:many", "a:1:b:c"):
            with self.assertRaises(ValueError, msg=spec):
                TaskConfig.parse(spec)


class SpeculativeConfigTest(unittest.TestCase):
    def test_labels_are_derived_from_the_tree_shape(self):
        self.assertEqual(SpeculativeConfig().describe(), "baseline")
        self.assertEqual(
            SpeculativeConfig(steps=3, topk=1, draft_tokens=4).describe(),
            "eagle3-s3-k1-d4",
        )
        self.assertEqual(
            SpeculativeConfig(label="fast", steps=3, topk=1, draft_tokens=4).describe(),
            "fast",
        )

    def test_enabled_entries_need_a_full_tree_shape(self):
        with self.assertRaisesRegex(ValueError, "positive topk and draft_tokens"):
            SpeculativeConfig(steps=3)


class BenchmarkConfigValidationTest(unittest.TestCase):
    def _config(self, **fields):
        base = {"model": "m", "tasks": [{"name": "gsm8k"}]}
        base.update(fields)
        return BenchmarkConfig.model_validate(base)

    def test_matrix_requires_launch(self):
        with self.assertRaisesRegex(ValueError, "server.launch"):
            self._config(matrix=[{}])

    def test_launch_defaults_to_a_baseline_matrix(self):
        config = self._config(server={"launch": True})
        self.assertEqual([entry.describe() for entry in config.matrix], ["baseline"])

    def test_speculative_launch_requires_a_draft_model(self):
        with self.assertRaisesRegex(ValueError, "draft_model is required"):
            self._config(
                server={"launch": True},
                matrix=[{"steps": 3, "topk": 1, "draft_tokens": 4}],
            )

    def test_matrix_labels_must_be_unique(self):
        with self.assertRaisesRegex(ValueError, "unique"):
            self._config(
                draft_model="d",
                server={"launch": True},
                matrix=[
                    {"steps": 3, "topk": 1, "draft_tokens": 4},
                    {"steps": 3, "topk": 1, "draft_tokens": 4},
                ],
            )

    def test_unknown_fields_are_rejected(self):
        with self.assertRaises(ValueError):
            self._config(num_prompts=3)

    def test_at_least_one_task(self):
        with self.assertRaises(ValueError):
            BenchmarkConfig.model_validate({"model": "m", "tasks": []})


class BuildConfigTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = os.path.join(self.tmp.name, "bench.json")
        with open(self.path, "w") as handle:
            json.dump(
                {
                    "model": "file-model",
                    "tasks": [{"name": "mtbench"}],
                    "server": {"base_url": "http://file:1"},
                    "concurrency": 4,
                },
                handle,
            )

    def test_flags_override_file_and_tasks_replace_them(self):
        config = build_config(
            self.path,
            {"model": "flag-model", "concurrency": None, "trust_remote_code": True},
            ["gsm8k:10"],
            [],
        )
        self.assertEqual(config.model, "flag-model")
        self.assertEqual(config.concurrency, 4)
        self.assertEqual(config.server.base_url, "http://file:1")
        self.assertTrue(config.trust_remote_code)
        self.assertEqual(config.tasks, [TaskConfig(name="gsm8k", num_samples=10)])

    def test_overrides_win_over_flags(self):
        config = build_config(
            self.path, {"concurrency": 8}, [], ["concurrency=16", "sampling.top_p=0.9"]
        )
        self.assertEqual(config.concurrency, 16)
        self.assertEqual(config.sampling.top_p, 0.9)

    def test_overrides_must_name_existing_fields(self):
        with self.assertRaisesRegex(ValueError, "does not exist"):
            build_config(self.path, {}, [], ["sampling.temp=1"])

    def test_works_without_a_file(self):
        config = build_config(
            None, {"model": "m", "base_url": "http://x:2"}, ["gsm8k"], []
        )
        self.assertEqual(config.model, "m")
        self.assertEqual(config.server.base_url, "http://x:2")


if __name__ == "__main__":
    unittest.main()
