"""Validate the stage-1 recipes without a target model or external mapping file."""

import json
import unittest
from pathlib import Path

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.application.planning import validate_resolved_run
from specforge.config import Config

ROOT = Path(__file__).resolve().parents[2]


class Eagle4RecipeTest(unittest.TestCase):
    def test_recipes_resolve_with_full_vocabulary_and_no_mapping_file(self):
        algorithm = builtin_algorithm_registry().resolve("eagle4")
        self.assertFalse(algorithm.spec.capabilities.supports_vocab_mapping)
        self.assertFalse(algorithm.providers.vocab_mapping_modes)
        for relative in (
            "examples/configs/offline/colocated/qwen3-4b-eagle4.yaml",
            "examples/configs/offline/colocated/qwen3.8-27b-eagle4.yaml",
            "examples/configs/online/disaggregated/managed-local/qwen3-4b-eagle4.yaml",
        ):
            with self.subTest(recipe=relative):
                cfg = Config.from_file(str(ROOT / relative))
                validate_resolved_run(cfg, algorithm)
                self.assertFalse(cfg.model.vocab_mapping_path)
                draft = json.loads((ROOT / cfg.model.draft_model_config).read_text())
                self.assertEqual(draft["draft_vocab_size"], draft["vocab_size"])
                self.assertEqual(
                    draft["efficient_lm_head_config"],
                    {"type": "low_rank", "rank": 128, "num_candidates": 128},
                )
                self.assertEqual(cfg.training.ttt_length, 7)
                self.assertEqual(cfg.training.attention_backend, "flex_attention")
                if cfg.deployment.mode == "disaggregated":
                    self.assertEqual(cfg.deployment.trainer.nproc_per_node, 7)
                    managed = cfg.deployment.disaggregated.managed_local
                    self.assertEqual(len(managed.trainer_cuda_visible_devices), 7)
                    self.assertEqual(len(managed.capture_servers), 1)
                else:
                    self.assertEqual(cfg.training.max_steps, 20)


if __name__ == "__main__":
    unittest.main()
