"""The public recipe selects a runtime without changing FSDP's mesh contract."""

import unittest

from specforge.application import resolve_run
from specforge.config import Config, apply_overrides


def recipe(**training):
    return Config.model_validate(
        {
            "model": {
                "target_model_path": "target",
                "draft_model_config": "draft.json",
            },
            "data": {"hidden_states_path": "/features"},
            "training": {"strategy": "dflash", **training},
        }
    )


class TorchTitanRecipeTest(unittest.TestCase):
    def test_default_fsdp_and_legacy_tp_restriction(self):
        self.assertEqual(recipe().training.backend, "fsdp")
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            recipe(tp_size=2)

    def test_titan_tp_reaches_application_composition(self):
        cfg = recipe(backend="torchtitan", tp_size=2)
        self.assertIs(resolve_run(cfg).config, cfg)
        cfg.validate_world_size(4)
        with self.assertRaisesRegex(ValueError, "divisible"):
            cfg.validate_world_size(3)

    def test_composed_mesh_degrees_are_validated_together(self):
        cfg = recipe(
            backend="torchtitan",
            tp_size=2,
            torchtitan={"cp_size": 2, "pp_size": 2, "dp_shard": 2},
        )
        cfg.validate_world_size(16)
        with self.assertRaisesRegex(ValueError, "multiply"):
            cfg.validate_world_size(8)

    def test_titan_options_cannot_be_silently_ignored_by_fsdp(self):
        with self.assertRaisesRegex(ValueError, "require backend=torchtitan"):
            recipe(torchtitan={"compile": True})

    def test_unsupported_algorithm_and_optimizer_are_explicit(self):
        with self.assertRaisesRegex(ValueError, "supports DFlash"):
            recipe(backend="torchtitan", strategy="eagle3")
        with self.assertRaisesRegex(ValueError, "CPU-offload"):
            recipe(backend="torchtitan", optimizer_cpu_offload=True)

    def test_pipeline_graph_conflict_is_rejected_before_gpu_allocation(self):
        with self.assertRaisesRegex(
            ValueError, "pipeline parallelism with CUDA graphs"
        ):
            recipe(
                backend="torchtitan",
                torchtitan={"pp_size": 2, "disable_cuda_graphs": False},
            )

    def test_cli_overrides_validate_the_result(self):
        cfg = apply_overrides(
            recipe(), ["training.backend=torchtitan", "training.tp_size=2"]
        )
        self.assertEqual(cfg.training.tp_size, 2)
        self.assertTrue(cfg.training.torchtitan.disable_cuda_graphs)


if __name__ == "__main__":
    unittest.main()
