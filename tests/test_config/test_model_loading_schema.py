# coding=utf-8
"""Typed contract for draft config resolution and weights-only warm starts."""

import unittest

from pydantic import ValidationError

from specforge.application import resolve_run
from specforge.config import Config

ONLINE_DEPLOYMENT = {
    "mode": "disaggregated",
    "disaggregated": {
        "control_dir": "/control",
        "backend": "mooncake",
        "server_urls": ["http://127.0.0.1:30000"],
    },
}


def _payload(strategy: str, **model_overrides):
    model = {
        "target_model_path": "target/model",
        **model_overrides,
    }
    data = (
        {"train_data_path": "/train.jsonl"}
        if strategy == "peagle"
        else {"hidden_states_path": "/features"}
    )
    if strategy in ("eagle3", "peagle"):
        model["vocab_mapping_path"] = "/mapping.pt"
    payload = {
        "model": model,
        "data": data,
        "training": {"strategy": strategy},
    }
    if strategy == "peagle":
        model["target_backend"] = "sglang"
        payload["training"]["max_steps"] = 1
        payload["deployment"] = ONLINE_DEPLOYMENT
    return payload


class ModelLoadingSchemaTest(unittest.TestCase):
    def test_working_legacy_auto_config_strategies_can_omit_source(self):
        for strategy in ("eagle3", "peagle", "dflash"):
            with self.subTest(strategy=strategy):
                cfg = Config.model_validate(_payload(strategy))
                resolve_run(cfg)
                self.assertIsNone(cfg.model.draft_model_config)

    def test_specialized_dflash_architectures_still_require_a_config(self):
        with self.assertRaisesRegex(ValueError, "requires model.draft_model_config"):
            resolve_run(Config.model_validate(_payload("domino")))
        with self.assertRaisesRegex(ValueError, "requires model.draft_model_config"):
            resolve_run(
                Config.model_validate(
                    {
                        **_payload("dflash"),
                        "model": {
                            **_payload("dflash")["model"],
                            "target_backend": "sglang",
                        },
                        "training": {
                            "strategy": "dspark",
                            "role": "consumer",
                            "total_steps": 1,
                        },
                        "data": {"train_data_path": "/train.jsonl"},
                        "deployment": ONLINE_DEPLOYMENT,
                    }
                )
            )

        # A warm-start HF repository or local directory can provide config.json.
        cfg = Config.model_validate(
            _payload("domino", draft_checkpoint_path="org/domino-draft")
        )
        resolve_run(cfg)
        self.assertEqual(cfg.model.draft_checkpoint_path, "org/domino-draft")

    def test_typed_architecture_overrides_are_strategy_scoped(self):
        dflash = Config.model_validate(
            _payload(
                "dflash",
                draft_num_hidden_layers=3,
                draft_block_size=8,
            )
        )
        resolve_run(dflash)
        self.assertEqual(dflash.model.draft_num_hidden_layers, 3)
        self.assertEqual(dflash.model.draft_block_size, 8)

        with self.assertRaisesRegex(
            ValueError,
            "requires model.draft_num_hidden_layers=1",
        ):
            resolve_run(
                Config.model_validate(_payload("eagle3", draft_num_hidden_layers=2))
            )
        with self.assertRaisesRegex(
            ValueError,
            "does not support model.draft_block_size",
        ):
            resolve_run(Config.model_validate(_payload("peagle", draft_block_size=8)))

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
            "role": "producer",
            "total_steps": 1,
        }
        payload["deployment"] = ONLINE_DEPLOYMENT
        cfg = Config.model_validate(payload)
        resolve_run(cfg)
        self.assertEqual(cfg.model.draft_checkpoint_path, "org/dflash-draft")


if __name__ == "__main__":
    unittest.main(verbosity=2)
