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
    def test_all_algorithms_require_inline_config_even_with_checkpoint(self):
        for strategy in (
            "eagle3",
            "peagle",
            "dflash",
            "dspark",
            "domino",
            "dspine",
            "mtp",
        ):
            for source in (None, "/draft.json", "org/model", {}):
                with self.subTest(strategy=strategy, source=source):
                    payload = _payload(strategy, draft_checkpoint_path="org/draft")
                    if source is not None:
                        payload["model"]["draft_model_config"] = source
                    with self.assertRaisesRegex(ValidationError, "draft_model_config"):
                        Config.model_validate(payload)

    def test_task_structural_overrides_are_rejected(self):
        from tests.draft_config_utils import draft_settings

        for key, value in (
            ("draft_num_hidden_layers", 2),
            ("draft_block_size", 8),
            ("mask_token_id", 3),
            ("aux_hidden_state_layer_ids", [1, 5, 8]),
        ):
            with self.subTest(key=key), self.assertRaisesRegex(ValidationError, key):
                Config.model_validate(
                    _payload(
                        "dflash",
                        draft_model_config=draft_settings("dflash"),
                        **{key: value},
                    )
                )

    def test_inline_settings_accept_nested_cli_overrides(self):
        from specforge.config import apply_overrides
        from tests.draft_config_utils import draft_settings

        cfg = Config.model_validate(
            _payload("dflash", draft_model_config=draft_settings("dflash"))
        )
        updated = apply_overrides(cfg, ["model.draft_model_config.block_size=8"])
        self.assertEqual(updated.model.draft_model_config["block_size"], 8)
        self.assertEqual(cfg.model.draft_model_config["block_size"], 4)
        resolve_run(updated)

    def test_warm_start_and_full_resume_are_mutually_exclusive(self):
        from tests.draft_config_utils import draft_settings

        payload = _payload(
            "eagle3",
            draft_checkpoint_path="/base/checkpoint",
            draft_model_config=draft_settings(),
        )
        payload["training"]["resume_from"] = "/run/checkpoint"
        with self.assertRaisesRegex(ValidationError, "mutually exclusive"):
            Config.model_validate(payload)
