"""CPU-safe checks of the benchmark's comparison boundaries and statistics."""

import importlib.util
import unittest
from pathlib import Path

SCRIPT = (
    Path(__file__).resolve().parents[2] / "scripts" / "benchmark_training_backends.py"
)
spec = importlib.util.spec_from_file_location("training_backend_benchmark", SCRIPT)
benchmark = importlib.util.module_from_spec(spec)
spec.loader.exec_module(benchmark)


class BackendBenchmarkContractTest(unittest.TestCase):
    def test_default_uses_one_model_run_per_process(self):
        self.assertEqual(benchmark.build_parser().parse_args([]).repeats, 1)

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch is not installed")
    def test_exact_resume_comparison_accepts_optimizer_metadata(self):
        import torch

        state = {
            "lr_scheduler_type": "constant",
            "offload": False,
            "optional": None,
            "masters": [torch.ones(2)],
        }
        benchmark._assert_state_equal(state, state)
        with self.assertRaises(AssertionError):
            benchmark._assert_state_equal(
                {**state, "lr_scheduler_type": "cosine"}, state
            )
        with self.assertRaises(AssertionError):
            benchmark._assert_state_equal({**state, "masters": [torch.zeros(2)]}, state)

    def test_controlled_recipes_keep_same_full_target_and_backbone_geometry(self):
        recipes = [
            benchmark.resolve_config(name)[0] for name in benchmark.ARCHITECTURES
        ]
        for key in (
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "vocab_size",
            "num_hidden_layers",
            "block_size",
        ):
            self.assertEqual(len({config[key] for config in recipes}), 1, key)
        self.assertEqual(recipes[0]["hidden_size"], 2560)
        self.assertEqual(recipes[0]["vocab_size"], 151936)
        self.assertEqual(recipes[0]["num_hidden_layers"], 5)
        self.assertEqual(recipes[1]["architectures"], ["DFlash2DraftModel"])
        self.assertEqual(recipes[1]["dflash_config"]["selector_rank"], 256)
        self.assertEqual(recipes[2]["dflash_config"]["markov_rank"], 256)
        self.assertTrue(recipes[2]["dflash_config"]["enable_confidence_head"])

    def test_stock_preserves_real_dflash2_vocabulary_and_dspark_block_size(self):
        dflash2, _ = benchmark.resolve_config("dflash2", "stock")
        dspark, _ = benchmark.resolve_config("dspark", "stock")
        self.assertEqual(dflash2["vocab_size"], 248320)
        self.assertEqual(dspark["block_size"], 7)
        self.assertNotEqual(dflash2["vocab_size"], dspark["vocab_size"])

    def test_tiny_preserves_actual_architecture_and_head_types(self):
        for algorithm, architecture in benchmark.ARCHITECTURES.items():
            config, _ = benchmark.resolve_config(algorithm, tiny=True)
            self.assertEqual(config["architectures"], [architecture])
            self.assertEqual(config["num_hidden_layers"], 2)
            self.assertEqual(config["vocab_size"], 128)
            self.assertEqual(config["dflash_config"]["target_layer_ids"], [1, 2])
        self.assertEqual(
            benchmark.resolve_config("dflash2", tiny=True)[0]["dflash_config"][
                "selector_rank"
            ],
            8,
        )
        self.assertEqual(
            benchmark.resolve_config("dspark", tiny=True)[0]["dflash_config"][
                "markov_rank"
            ],
            8,
        )

    def test_throughput_uses_total_time_not_mean_of_step_rates(self):
        result = benchmark.summarize_times([1.0, 3.0], 100)
        self.assertEqual(result["input_context_tokens_per_second"], 50.0)
        self.assertEqual(result["optimizer_step_seconds_median"], 2.0)
        self.assertEqual(result["optimizer_step_seconds_p95"], 3.0)
        for durations in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                benchmark.summarize_times(durations, 100)

    def test_validation_cannot_accidentally_gather_full_size_checkpoints(self):
        args = benchmark.build_parser().parse_args(
            ["--backend", "fsdp", "--output", "report.json", "--check-resume"]
        )
        with self.assertRaisesRegex(ValueError, "require --tiny"):
            benchmark.validate_args(args)
        args.tiny = True
        benchmark.validate_args(args)

    def test_rejects_incomplete_or_empty_measurements(self):
        args = benchmark.build_parser().parse_args(
            ["--backend", "fsdp", "--output", "report.json", "--steps", "0"]
        )
        with self.assertRaisesRegex(ValueError, "steps.*positive"):
            benchmark.validate_args(args)

    def test_rejects_unsupported_torchtitan_precision_before_model_allocation(self):
        args = benchmark.build_parser().parse_args(
            [
                "--backend",
                "torchtitan",
                "--output",
                "report.json",
                "--precision",
                "fp32",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires --precision bf16"):
            benchmark.validate_args(args)


if __name__ == "__main__":
    unittest.main()
