# coding=utf-8
"""Config and application validation of ``training.cost_aware_*``."""

import copy
import unittest

from pydantic import ValidationError

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.application import resolve_run
from specforge.config import Config
from specforge.training.cost_aware import resolve_scheduling_policy

MINIMAL = {
    "model": {
        "target_model_path": "some/target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "/mapping.pt",
    },
    "data": {"hidden_states_path": "/features"},
}

ONLINE_DEPLOYMENT = {
    "mode": "disaggregated",
    "disaggregated": {
        "control_dir": "/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
    },
}


def _offline(training: dict) -> Config:
    payload = copy.deepcopy(MINIMAL)
    payload["training"] = training
    return Config.model_validate(payload)


def _online(training: dict) -> Config:
    payload = copy.deepcopy(MINIMAL)
    payload["model"]["target_backend"] = "sglang"
    payload["data"] = {"train_data_path": "/train.jsonl"}
    payload["training"] = {"max_steps": 1, **training}
    payload["deployment"] = copy.deepcopy(ONLINE_DEPLOYMENT)
    return Config.model_validate(payload)


class CostAwareSchemaTest(unittest.TestCase):
    def test_defaults_are_off(self):
        cfg = _offline({})
        self.assertFalse(cfg.training.cost_aware_scheduling)
        self.assertEqual(cfg.training.cost_aware_key, "auto")
        self.assertEqual(cfg.training.cost_aware_window, 0)
        with self.assertRaises(ValidationError):
            _offline({"cost_aware_scheduling": True, "cost_aware_key": "cost"})

    def test_key_and_window_require_the_switch(self):
        with self.assertRaisesRegex(
            ValidationError, "requires training.cost_aware_scheduling"
        ):
            _offline({"cost_aware_key": "anchors"})
        with self.assertRaisesRegex(
            ValidationError, "requires training.cost_aware_scheduling"
        ):
            _offline({"cost_aware_window": 8})
        cfg = _offline(
            {
                "cost_aware_scheduling": True,
                "cost_aware_key": "length",
                "cost_aware_window": 8,
            }
        )
        self.assertEqual(cfg.training.cost_aware_key, "length")

    def test_online_rejects_a_bucket_window(self):
        with self.assertRaisesRegex(ValidationError, "offline features"):
            _online({"cost_aware_scheduling": True, "cost_aware_window": 4})
        cfg = _online({"cost_aware_scheduling": True})
        self.assertTrue(cfg.training.cost_aware_scheduling)


class CostAwarePlanningTest(unittest.TestCase):
    def test_eagle3_keeps_token_weights_only_at_batch_size_one(self):
        with self.assertRaisesRegex(ValueError, "per-token weights unchanged"):
            resolve_run(
                _offline(
                    {
                        "strategy": "eagle3",
                        "batch_size": 2,
                        "cost_aware_scheduling": True,
                    }
                )
            )
        resolve_run(
            _offline(
                {"strategy": "eagle3", "batch_size": 1, "cost_aware_scheduling": True}
            )
        )

    def test_anchor_keys_need_an_anchor_objective(self):
        with self.assertRaisesRegex(ValueError, "anchor-sampling objective"):
            resolve_run(
                _offline(
                    {
                        "strategy": "eagle3",
                        "cost_aware_scheduling": True,
                        "cost_aware_key": "anchors",
                    }
                )
            )
        resolve_run(
            _offline(
                {"strategy": "dspark", "batch_size": 4, "cost_aware_scheduling": True}
            )
        )
        resolve_run(
            _offline(
                {
                    "strategy": "dspark",
                    "cost_aware_scheduling": True,
                    "cost_aware_key": "length",
                }
            )
        )

    def test_policy_resolution(self):
        registry = builtin_algorithm_registry()
        dspark = registry.resolve("dspark")
        eagle3 = registry.resolve("eagle3")
        self.assertIsNone(resolve_scheduling_policy(_offline({}).training, dspark))
        policy = resolve_scheduling_policy(
            _offline(
                {
                    "strategy": "dspark",
                    "cost_aware_scheduling": True,
                    "num_anchors": 256,
                }
            ).training,
            dspark,
        )
        self.assertEqual((policy.key, policy.num_anchors), ("anchors", 256))
        self.assertEqual(
            resolve_scheduling_policy(
                _offline(
                    {"strategy": "eagle3", "cost_aware_scheduling": True}
                ).training,
                eagle3,
            ).key,
            "length",
        )
        explicit = resolve_scheduling_policy(
            _offline(
                {
                    "strategy": "dspark",
                    "cost_aware_scheduling": True,
                    "cost_aware_key": "length",
                    "cost_aware_window": 64,
                }
            ).training,
            dspark,
        )
        self.assertEqual(
            explicit.contract(),
            {"key": "length", "num_anchors": 512, "window_rounds": 64},
        )

    def test_every_builtin_algorithm_declares_a_key_or_rejects_the_switch(self):
        registry = builtin_algorithm_registry()
        for name in ("dflash", "dspark", "domino"):
            self.assertEqual(
                registry.resolve(name).spec.capabilities.scheduling_key, "anchors"
            )
        for name in ("eagle3", "peagle", "mtp"):
            self.assertEqual(
                registry.resolve(name).spec.capabilities.scheduling_key, "length"
            )


class DSparkNormalizationFollowsSchedulingTest(unittest.TestCase):
    def test_step_only_under_cost_aware_scheduling(self):
        from specforge.algorithms.model_providers import (
            resolve_dspark_loss_normalization,
        )

        self.assertEqual(
            resolve_dspark_loss_normalization(_offline({}).training), "round"
        )
        self.assertEqual(
            resolve_dspark_loss_normalization(
                _offline({"strategy": "dspark", "cost_aware_scheduling": True}).training
            ),
            "step",
        )


if __name__ == "__main__":
    unittest.main()
