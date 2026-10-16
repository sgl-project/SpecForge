import json
import unittest
from pathlib import Path

from specforge.config.schema import load_config

ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    ROOT
    / "examples/configs/online/disaggregated/managed-local"
    / "qwen3-8b-dspark-kv-conv-dpard-managed.yaml"
)


class DSparkKVManagedRecipeTest(unittest.TestCase):
    def test_objective_architecture_and_retention(self):
        config = load_config(str(RECIPE))
        self.assertEqual(config.training.strategy, "dspark_kv")
        self.assertEqual(config.training.dspark_objective, "dpard")
        self.assertEqual(config.training.dspark_dpard_rho, 0.5)
        self.assertEqual(config.training.dspark_confidence_head_alpha, 0)
        self.assertEqual(config.training.max_checkpoints, 0)
        self.assertNotIn("max_checkpoints:", RECIPE.read_text())
        draft = json.loads((ROOT / config.model.draft_model_config).read_text())
        self.assertTrue(draft["is_causal"])
        self.assertEqual(draft["dflash_config"]["conv_kernel_size"], 2)

    def test_managed_capture_and_trainer_are_disjoint(self):
        config = load_config(str(RECIPE))
        managed = config.deployment.disaggregated.managed_local
        self.assertEqual(config.deployment.trainer.nproc_per_node, 1)
        self.assertEqual(managed.trainer_cuda_visible_devices, ["1"])
        self.assertEqual(managed.capture_servers[0].cuda_visible_devices, ["0"])
        self.assertEqual(managed.capture_servers[0].tp_size, 1)
        self.assertTrue(config.model.sglang_disable_cuda_graph)
        self.assertTrue(config.model.sglang_disable_radix_cache)
        self.assertGreaterEqual(
            config.model.sglang_context_length, config.data.max_length + 7
        )


if __name__ == "__main__":
    unittest.main()
