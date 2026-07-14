# coding=utf-8
"""Typed contract for draft config resolution and weights-only warm starts."""

import unittest

from pydantic import ValidationError

from specforge.config import Config


def _payload(strategy: str, **model_overrides):
    model = {
        "target_model_path": "target/model",
        "target_backend": "hf",
        **model_overrides,
    }
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    if strategy == "eagle3":
        model["vocab_mapping_path"] = "/mapping.pt"
    return {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }


class ModelLoadingSchemaTest(unittest.TestCase):
    def test_working_legacy_auto_config_strategies_can_omit_source(self):
        for strategy in ("eagle3", "peagle", "dflash"):
            with self.subTest(strategy=strategy):
                cfg = Config.model_validate(_payload(strategy))
                self.assertIsNone(cfg.model.draft_model_config)

    def test_specialized_dflash_architectures_still_require_a_config(self):
        with self.assertRaisesRegex(
            ValidationError, "requires model.draft_model_config"
        ):
            Config.model_validate(_payload("domino"))
        with self.assertRaisesRegex(
            ValidationError, "requires model.draft_model_config"
        ):
            Config.model_validate(
                {
                    **_payload("dflash"),
                    "training": {
                        "strategy": "dspark",
                        "deployment_mode": "disaggregated",
                        "role": "consumer",
                        "total_steps": 1,
                    },
                    "data": {"train_data_path": "/train.jsonl"},
                }
            )

        # A warm-start HF repository or local directory can provide config.json.
        cfg = Config.model_validate(
            _payload("domino", draft_checkpoint_path="org/domino-draft")
        )
        self.assertEqual(cfg.model.draft_checkpoint_path, "org/domino-draft")

    def test_typed_architecture_overrides_are_strategy_scoped(self):
        dflash = Config.model_validate(
            _payload(
                "dflash",
                draft_num_hidden_layers=3,
                draft_block_size=8,
            )
        )
        self.assertEqual(dflash.model.draft_num_hidden_layers, 3)
        self.assertEqual(dflash.model.draft_block_size, 8)

        with self.assertRaisesRegex(ValidationError, "EAGLE3 has one"):
            Config.model_validate(_payload("eagle3", draft_num_hidden_layers=2))
        with self.assertRaisesRegex(ValidationError, "DFlash only"):
            Config.model_validate(_payload("peagle", draft_block_size=8))

    def test_warm_start_and_full_resume_are_mutually_exclusive(self):
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            Config.model_validate(
                {
                    **_payload("eagle3", draft_checkpoint_path="/base/checkpoint"),
                    "training": {
                        "strategy": "eagle3",
                        "resume_from": "/run/checkpoint",
                    },
                }
            )

    def test_disaggregated_producer_can_resolve_warm_start_metadata(self):
        payload = _payload(
            "dflash",
            draft_checkpoint_path="org/dflash-draft",
        )
        payload["model"]["target_backend"] = "sglang"
        payload["data"] = {"train_data_path": "/train.jsonl"}
        payload["training"] = {
            "strategy": "dflash",
            "deployment_mode": "disaggregated",
            "role": "producer",
            "total_steps": 1,
        }
        cfg = Config.model_validate(payload)
        self.assertEqual(cfg.model.draft_checkpoint_path, "org/dflash-draft")


if __name__ == "__main__":
    unittest.main(verbosity=2)
