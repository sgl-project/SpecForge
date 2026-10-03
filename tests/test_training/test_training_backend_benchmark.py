"""CPU-safe tests of the native benchmark workload and measurement contract."""

import contextlib
import io
import sys
import unittest
from pathlib import Path

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import benchmark_training_backends as benchmark
    import training_benchmark_recipes as recipes
finally:
    sys.path.pop(0)


class BackendBenchmarkContractTest(unittest.TestCase):
    def args(self, *extra):
        return benchmark.parse_args(
            [
                "--specforge-root",
                ".",
                "--backend",
                "torchtitan",
                "--algorithm",
                "dflash2",
                "--output",
                "result.json",
                *extra,
            ]
        )

    def test_default_workload_has_full_length_objective_and_independent_processes(self):
        args = self.args()
        self.assertEqual(
            (args.seq_length, args.num_anchors, args.objective_chunk_blocks),
            (4096, 512, 128),
        )
        self.assertEqual((args.batch_size, args.accumulation_steps), (1, 2))
        self.assertEqual((args.warmup_steps, args.steps), (10, 20))
        self.assertEqual(args.attention, "flex_attention")
        self.assertEqual(args.sharding, "SHARD_GRAD_OP")
        self.assertFalse(hasattr(args, "repeats"))
        self.assertFalse(args.compile)

    def test_rejects_empty_measurements_and_unsupported_baseline_features(self):
        cases = (
            ("--steps", "0"),
            ("--warmup-steps", "0"),
            ("--learning-rate", "nan"),
            ("--learning-rate", "-1"),
            ("--backend", "fsdp", "--compile"),
            ("--backend", "fsdp", "--tp-size", "2"),
        )
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*args)

    def test_controlled_recipes_keep_same_full_target_and_backbone_geometry(self):
        configs = [recipes.resolve_config(name)[0] for name in recipes.ARCHITECTURES]
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
            self.assertEqual(len({config[key] for config in configs}), 1, key)
        self.assertEqual(configs[0]["hidden_size"], 2560)
        self.assertEqual(configs[0]["vocab_size"], 151936)
        self.assertEqual(configs[0]["num_hidden_layers"], 5)
        self.assertEqual(configs[1]["architectures"], ["DFlash2DraftModel"])
        self.assertEqual(configs[1]["dflash_config"]["selector_rank"], 256)
        self.assertEqual(configs[2]["dflash_config"]["markov_rank"], 256)
        self.assertTrue(configs[2]["dflash_config"]["enable_confidence_head"])

    def test_stock_recipe_differences_are_not_hidden(self):
        dflash2, _ = recipes.resolve_config("dflash2", "stock")
        dspark, _ = recipes.resolve_config("dspark", "stock")
        self.assertEqual(dflash2["vocab_size"], 248320)
        self.assertEqual(dspark["block_size"], 7)
        self.assertNotEqual(dflash2["vocab_size"], dspark["vocab_size"])

    def test_tiny_preserves_real_architectures_and_auxiliary_heads(self):
        for algorithm, architecture in recipes.ARCHITECTURES.items():
            config, _ = recipes.resolve_config(algorithm, tiny=True)
            self.assertEqual(config["architectures"], [architecture])
            self.assertEqual(config["num_hidden_layers"], 2)
            self.assertEqual(config["vocab_size"], 128)
            self.assertEqual(config["dflash_config"]["target_layer_ids"], [1, 2])
        self.assertEqual(
            recipes.resolve_config("dflash2", tiny=True)[0]["dflash_config"][
                "selector_rank"
            ],
            8,
        )
        self.assertEqual(
            recipes.resolve_config("dspark", tiny=True)[0]["dflash_config"][
                "markov_rank"
            ],
            8,
        )

    def test_throughput_divides_total_tokens_by_total_time(self):
        result = recipes.summarize_times([1.0, 3.0], 100)
        self.assertEqual(result["input_context_tokens_per_second"], 50.0)
        self.assertEqual(result["optimizer_step_seconds_median"], 2.0)
        self.assertEqual(result["optimizer_step_seconds_p95"], 3.0)
        for durations in ([], [0], [-1], [float("nan")]):
            with self.assertRaises(ValueError):
                recipes.summarize_times(durations, 100)


if __name__ == "__main__":
    unittest.main()
