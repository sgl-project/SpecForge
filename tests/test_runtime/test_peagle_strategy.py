"""P-EAGLE uses the shared strategy for both capture and checkpoint state."""

import unittest
from pathlib import Path

from specforge.application import resolve_run
from specforge.runtime.contracts import TrainBatch
from tests.test_algorithms.test_objective_contracts import build_runtime
from tests.test_algorithms.test_objectives import training_batch
from tests.test_runtime._fixtures import build_strategy
from tests.test_runtime.test_canonical_training import run_config


class PEagleStrategyTest(unittest.TestCase):
    def test_capture_and_shared_execution(self):
        config, runtime = build_runtime("PEagleDraftModel")
        resolved = resolve_run(run_config(Path("target"), config))
        self.assertEqual(
            resolved.algorithm.providers.server_streaming_for("text").capture_method,
            "eagle3",
        )
        self.assertIsNotNone(resolved.algorithm.providers.offline_for("text"))
        strategy = build_strategy(runtime)
        result = strategy.forward_loss(
            TrainBatch(["0"], "peagle", training_batch(config))
        )
        result.loss.backward()
        self.assertIsNotNone(runtime.draft_model.mask_hidden.grad)
        state = strategy.checkpoint_state_filter(runtime.state_dict())
        self.assertIn("mask_hidden", state)
        self.assertTrue(all(not name.startswith("draft_model.") for name in state))

    def test_missing_target_features_fail_before_execution(self):
        config, runtime = build_runtime("PEagleDraftModel")
        batch = training_batch(config)
        del batch["target_last_hidden_states"]
        with self.assertRaises(ValueError):
            build_strategy(runtime).forward_loss(TrainBatch(["0"], "peagle", batch))
