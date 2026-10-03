"""CPU-safe tests of the native benchmark workload and measurement contract."""

import contextlib
import copy
import io
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
sys.path.insert(0, str(SCRIPTS))
try:
    import benchmark_training_backends as benchmark
    import training_backend_matrix as matrix
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
            ("--backend", "fsdp", "--cuda-graphs"),
            ("--titan-engine", "graph"),
            ("--titan-engine", "graph", "--compile", "--tp-size", "2"),
        )
        for args in cases:
            with self.subTest(args=args), contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    self.args(*args)

    def test_acceleration_flags_keep_explicit_recipe_and_engine(self):
        args = self.args("--compile", "--cuda-graphs")
        self.assertTrue(args.cuda_graphs)
        self.assertEqual(args.titan_engine, "trainer")
        args = self.args(
            "--compile", "--titan-engine", "graph", "--graph-inductor", "full"
        )
        self.assertEqual(args.graph_inductor, "full")

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

    def test_source_identity_includes_direct_custom_kernel_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            kernel = root / "specforge/modeling/layers/fused_conv.py"
            kernel.parent.mkdir(parents=True)
            kernel.write_text("first version")
            before = benchmark.source_hashes(root)
            kernel.write_text("second version")
            self.assertNotEqual(before, benchmark.source_hashes(root))
            self.assertEqual(list(before), ["specforge/modeling/layers/fused_conv.py"])

    def test_original_fsdp_environment_need_not_install_torchtitan(self):
        def version(name):
            if name == "torchtitan":
                raise benchmark.importlib.metadata.PackageNotFoundError(name)
            return "present"

        with patch.object(benchmark.importlib.metadata, "version", side_effect=version):
            self.assertIsNone(benchmark.package_versions()["torchtitan"])


class MatchedBackendMatrixTest(unittest.TestCase):
    def records(self):
        records = {}
        for algorithm in matrix.ALGORITHMS:
            for case in matrix.CASES:
                native = case not in ("fsdp213", "fsdp214")
                graph = case == "graph-full"
                records[(algorithm, case, 1)] = {
                    "backend": "torchtitan" if native else "fsdp",
                    "source_sha256": {"kernel.py": "same"},
                    "benchmark_sha256": "driver",
                    "recipe_helpers_sha256": "recipe",
                    "versions": {
                        "torch": "2.13.0" if case == "fsdp213" else "2.14.0",
                        "transformers": "5.12.1",
                    },
                    "cuda": "13",
                    "device": "H200",
                    "recipe_source": "/frozen/configs/qwen3-4b-dflash.json",
                    "comparison_contract": {
                        "config": {"architectures": [recipes.ARCHITECTURES[algorithm]]},
                        "warmup_steps": 1,
                        "steps": 2,
                    },
                    "runtime": {
                        "titan_engine": "graph" if graph else "trainer",
                        "compile": native,
                        "cuda_graphs": native,
                        "graph_inductor": "full" if graph else None,
                        "kernel_environment": {"fused": "1"},
                        "compute_dtype": "bfloat16",
                        "adam_state_dtype": "float32",
                        "float32_matmul_precision": "highest",
                        "allow_tf32_matmul": False,
                        "allow_tf32_cudnn": True,
                        "deterministic_algorithms": False,
                        "visible_devices": "2,3",
                    },
                    "losses_including_warmup": [5.0, 4.0, 3.0],
                    "step_seconds_max_rank": [2.0, 4.0] if not native else [1.0, 2.0],
                    "peak_allocated_bytes_max_rank": 2**30,
                }
        return records

    def tolerances(self):
        return dict(first_atol=1e-5, trajectory_atol=1e-5, rtol=1e-4)

    def test_plan_is_36_fresh_processes_with_rotated_case_order(self):
        trials = matrix.plan_trials(
            Path("python"),
            Path("source"),
            Path("driver"),
            Path("output"),
            3,
            10,
            20,
            "2,3",
            python213=Path("python213"),
        )
        self.assertEqual(len(trials), 36)
        self.assertEqual(len({trial["name"] for trial in trials}), 36)
        for algorithm in matrix.ALGORITHMS:
            orders = [
                [
                    trial["case"]
                    for trial in trials
                    if trial["algorithm"] == algorithm and trial["repeat"] == repeat
                ]
                for repeat in range(1, 4)
            ]
            for position in range(4):
                self.assertEqual(len({order[position] for order in orders}), 3)
        for trial in trials:
            command = trial["command"]
            self.assertEqual(
                "--compile" in command, trial["case"] not in ("fsdp213", "fsdp214")
            )
            self.assertEqual(
                "--cuda-graphs" in command, trial["case"] not in ("fsdp213", "fsdp214")
            )
            self.assertIn("--nproc-per-node=2", command)
            self.assertEqual(
                command[0], "python213" if trial["case"] == "fsdp213" else "python"
            )

    def test_identical_objectives_yield_paired_trial_speedups(self):
        report = matrix.validate_and_summarize(
            self.records(), self.tolerances(), repeats=1
        )
        self.assertTrue(report["passed"])
        self.assertEqual(len(report["performance_rows"]), 12)
        self.assertEqual(
            report["performance_rows"][2]["median_paired_speedup_vs_fsdp213"], 2.0
        )

    def test_renaming_an_entire_algorithm_family_cannot_relabel_its_results(self):
        records = self.records()
        for case in matrix.CASES:
            records[("dflash", case, 1)] = copy.deepcopy(records[("dflash2", case, 1)])
        with self.assertRaisesRegex(ValueError, "recorded architecture"):
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)

    def test_algorithm_identity_requires_a_single_matching_architecture(self):
        for architectures in (None, [], ["UnknownDraftModel"], ["DFlashDraftModel"]):
            records = self.records()
            records[("dspark", "graph-full", 1)]["comparison_contract"]["config"][
                "architectures"
            ] = architectures
            with self.subTest(architectures=architectures):
                with self.assertRaisesRegex(ValueError, "recorded architecture"):
                    matrix.validate_and_summarize(records, self.tolerances(), repeats=1)

    def test_controlled_and_stock_sources_are_checked_without_inferring_algorithm(self):
        records = self.records()
        # The shared controlled DFlash source is valid for all three algorithms;
        # the actual recorded architecture, not its source filename, identifies it.
        self.assertTrue(
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)[
                "passed"
            ]
        )
        for algorithm in matrix.ALGORITHMS:
            for case in matrix.CASES:
                records[(algorithm, case, 1)][
                    "recipe_source"
                ] = f"/frozen/configs/{recipes.STOCK_CONFIGS[algorithm]}"
        self.assertTrue(
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)[
                "passed"
            ]
        )
        for invalid in (None, "/frozen/configs/qwen3-4b-dspark.json"):
            changed = copy.deepcopy(records)
            changed[("dflash", "titan-cuda", 1)]["recipe_source"] = invalid
            with self.subTest(source=invalid):
                with self.assertRaisesRegex(ValueError, "recipe source"):
                    matrix.validate_and_summarize(changed, self.tolerances(), repeats=1)
        records[("dflash2", "graph-full", 1)][
            "recipe_source"
        ] = "/frozen/configs/qwen3-4b-dflash.json"
        with self.assertRaisesRegex(ValueError, "Mixed recipe sources"):
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)

    def test_graph_failure_retains_separate_fsdp_native_table(self):
        records = self.records()
        records[("dflash2", "graph-full", 1)]["losses_including_warmup"][-1] += 0.517
        report = matrix.validate_and_summarize(records, self.tolerances(), repeats=1)
        self.assertFalse(report["passed"])
        self.assertFalse(report["graph_gate_passed"])
        self.assertEqual(len(report["performance_rows"]), 9)
        self.assertFalse(
            any(row["case"] == "graph-full" for row in report["performance_rows"])
        )
        self.assertEqual(len(report["graph_diagnostic_rows"]), 3)
        failures = [gate for gate in report["gates"] if not gate["passed"]]
        self.assertEqual(failures[0]["failing_steps"], [3])

    def test_fsdp_drift_is_observed_separately_from_same_policy_graph_gate(self):
        records = self.records()
        records[("dflash", "fsdp214", 1)]["losses_including_warmup"][1] += 0.1
        report = matrix.validate_and_summarize(records, self.tolerances(), repeats=1)
        self.assertTrue(report["passed"])
        self.assertAlmostEqual(
            report["fsdp_precision_observations"][0]["maximum_absolute_error"], 0.1
        )
        self.assertIsNone(report["fsdp_precision_observations"][0]["passed"])

    def test_rejects_mixed_snapshot_data_runtime_or_missing_measurements(self):
        changes = (
            lambda value: value["source_sha256"].update({"kernel.py": "older"}),
            lambda value: value["comparison_contract"].update({"inputs": "different"}),
            lambda value: value["runtime"].update(allow_tf32_matmul=True),
            lambda value: value["runtime"].update(cuda_graphs=False),
            lambda value: value["step_seconds_max_rank"].pop(),
            lambda value: value["losses_including_warmup"].pop(),
            lambda value: value["losses_including_warmup"].__setitem__(0, float("nan")),
        )
        for change in changes:
            records = self.records()
            change(records[("dspark", "graph-full", 1)])
            with self.subTest(change=change), self.assertRaises(ValueError):
                matrix.validate_and_summarize(records, self.tolerances(), repeats=1)
        records = self.records()
        records.pop(("dspark", "fsdp213", 1))
        with self.assertRaises(ValueError):
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)

    def test_first_step_uses_its_own_stricter_tolerance(self):
        tolerances = dict(first_atol=1e-4, trajectory_atol=1e-2, rtol=0.0)
        result = matrix.loss_difference([5.0, 4.0], [5.001, 4.001], **tolerances)
        self.assertEqual(result["failing_steps"], [1])

    def test_correctness_collector_timings_cannot_enter_performance_table(self):
        records = self.records()
        records[("dflash", "titan-cuda", 1)]["correctness_collector"] = {
            "timings_are_not_performance_results": True
        }
        with self.assertRaisesRegex(ValueError, "not performance trials"):
            matrix.validate_and_summarize(records, self.tolerances(), repeats=1)

    def test_reports_median_of_paired_speedups_not_ratio_of_medians(self):
        records = self.records()
        for repeat in (2, 3):
            for (algorithm, case, _), payload in list(records.items())[:12]:
                records[(algorithm, case, repeat)] = copy.deepcopy(payload)
        for repeat, baseline, candidate in (
            (1, 1.0, 1.0),
            (2, 10.0, 2.0),
            (3, 2.0, 10.0),
        ):
            records[("dflash", "fsdp213", repeat)]["step_seconds_max_rank"] = [
                baseline
            ] * 2
            records[("dflash", "titan-cuda", repeat)]["step_seconds_max_rank"] = [
                candidate
            ] * 2
        report = matrix.validate_and_summarize(records, self.tolerances(), repeats=3)
        self.assertEqual(
            report["performance_rows"][2]["paired_speedups_vs_fsdp213"], [1.0, 5.0, 0.2]
        )
        self.assertEqual(
            report["performance_rows"][2]["median_paired_speedup_vs_fsdp213"], 1.0
        )


if __name__ == "__main__":
    unittest.main()
