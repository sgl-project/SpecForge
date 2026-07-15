# coding=utf-8
"""Config schema mechanics: parsing, validation, dotted overrides."""

import copy
import json
import os
import tempfile
import unittest

from pydantic import ValidationError

from specforge.config import Config, apply_overrides, load_config

MINIMAL = {
    "model": {
        "target_model_path": "some/target",
        "draft_model_config": "draft.json",
        "vocab_mapping_path": "/mapping.pt",
    },
    "data": {"hidden_states_path": "/features"},
}


def _write(payload: dict, suffix: str) -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    with os.fdopen(fd, "w") as f:
        if suffix == ".yaml":
            import yaml

            yaml.safe_dump(payload, f)
        else:
            json.dump(payload, f)
    return path


class ConfigSchemaTest(unittest.TestCase):
    def test_from_file_yaml_and_json(self):
        for suffix in (".yaml", ".json"):
            path = _write(MINIMAL, suffix)
            self.addCleanup(os.unlink, path)
            cfg = Config.from_file(path)
            self.assertEqual(cfg.model.target_model_path, "some/target")
            self.assertEqual(cfg.mode, "offline")
            self.assertEqual(cfg.training.strategy, "eagle3")  # defaults applied

    def test_exactly_one_data_source(self):
        with self.assertRaises(ValidationError):
            Config.model_validate({**MINIMAL, "data": {}})
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"hidden_states_path": "/a", "prompts_path": "/b"},
                }
            )
        online = Config.model_validate(
            {**MINIMAL, "data": {"prompts_path": "/prompts.jsonl"}}
        )
        self.assertEqual(online.mode, "online")
        raw = Config.model_validate(
            {**MINIMAL, "data": {"train_data_path": "/conversations.jsonl"}}
        )
        self.assertEqual(raw.mode, "online")

    def test_offline_eagle3_requires_explicit_vocab_mapping(self):
        missing = copy.deepcopy(MINIMAL)
        missing["model"].pop("vocab_mapping_path")
        with self.assertRaisesRegex(ValidationError, "vocab_mapping_path"):
            Config.model_validate(missing)
        self.assertEqual(Config.model_validate(MINIMAL).mode, "offline")

    def test_unknown_backend_rejected(self):
        bad = {**MINIMAL, "model": {**MINIMAL["model"], "target_backend": "vllm"}}
        with self.assertRaises(ValidationError):
            Config.model_validate(bad)

    def test_unknown_fields_and_unsupported_modes_fail_early(self):
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "data": {"train_data_path": "/data.jsonl", "is_vlm": True},
                }
            )
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"attention_backend": "usp"},
                }
            )
        parallel = Config.model_validate(
            {
                **MINIMAL,
                "training": {"tp_size": 2, "sp_ulysses_size": 2},
            }
        )
        self.assertEqual(parallel.training.tp_size, 2)
        self.assertEqual(parallel.training.sp_ulysses_size, 2)
        with self.assertRaises(ValidationError):
            Config.model_validate(
                {
                    **MINIMAL,
                    "training": {"strategy": "dflash"},
                }
            )
        dspark = Config.model_validate(
            {
                **MINIMAL,
                "data": {"train_data_path": "/data.jsonl"},
                "training": {
                    "strategy": "dspark",
                    "deployment_mode": "disaggregated",
                    "role": "producer",
                    "total_steps": 10,
                },
            }
        )
        self.assertEqual(dspark.training.strategy, "dspark")
        with self.assertRaisesRegex(ValidationError, "target_backend=sglang"):
            Config.model_validate(
                {
                    **MINIMAL,
                    "model": {**MINIMAL["model"], "target_backend": "hf"},
                    "data": {"train_data_path": "/data.jsonl"},
                    "training": {
                        "strategy": "dflash",
                        "deployment_mode": "disaggregated",
                        "role": "producer",
                        "total_steps": 10,
                    },
                }
            )

    def test_strategy_specific_capture_and_attention_are_validated(self):
        online = {**MINIMAL, "data": {"train_data_path": "/data.jsonl"}}
        with self.assertRaisesRegex(ValidationError, "exactly three"):
            Config.model_validate(
                {
                    **online,
                    "model": {
                        **online["model"],
                        "aux_hidden_state_layer_ids": [1, 2],
                    },
                }
            )
        with self.assertRaisesRegex(ValidationError, "would be ignored"):
            Config.model_validate(
                {
                    **online,
                    "model": {
                        **online["model"],
                        "aux_hidden_state_layer_ids": [1, 2, 3],
                    },
                    "training": {"strategy": "dflash"},
                }
            )
        with self.assertRaisesRegex(ValidationError, "P-EAGLE"):
            Config.model_validate(
                {
                    **online,
                    "training": {
                        "strategy": "peagle",
                        "attention_backend": "sdpa",
                    },
                }
            )
        with self.assertRaisesRegex(ValidationError, "EAGLE3 attention_backend"):
            Config.model_validate(
                {**online, "training": {"attention_backend": "eager"}}
            )
        with self.assertRaisesRegex(ValidationError, "DFlash-family"):
            Config.model_validate(
                {
                    **online,
                    "training": {
                        "strategy": "dflash",
                        "attention_backend": "fa",
                    },
                }
            )

    def test_invalid_core_bounds_fail_during_config_validation(self):
        cases = (
            ("data.max_length", 0),
            ("data.build_dataset_num_proc", 0),
            ("training.num_epochs", 0),
            ("training.max_steps", 0),
            ("training.total_steps", 0),
            ("training.batch_size", 0),
            ("training.accumulation_steps", 0),
            ("training.save_interval", -1),
            ("training.log_interval", 0),
            ("training.max_checkpoints", -1),
        )
        for path, value in cases:
            with self.subTest(path=path):
                raw = copy.deepcopy(MINIMAL)
                section, field = path.split(".")
                raw.setdefault(section, {})[field] = value
                with self.assertRaises(ValidationError):
                    Config.model_validate(raw)

    def test_overrides_coerce_and_revalidate(self):
        cfg = Config.model_validate(MINIMAL)
        out = apply_overrides(
            cfg, ["training.learning_rate=1e-3", "training.max_steps=7", "run_id=r2"]
        )
        self.assertEqual(out.training.learning_rate, 1e-3)
        self.assertEqual(out.training.max_steps, 7)
        self.assertEqual(out.run_id, "r2")
        # original untouched
        self.assertIsNone(cfg.training.max_steps)

    def test_override_bad_path_or_form_raises(self):
        cfg = Config.model_validate(MINIMAL)
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ["training.no_such_field=1"])
        with self.assertRaises(ValueError):
            apply_overrides(cfg, ["not-an-assignment"])

    def test_load_config_applies_overrides(self):
        path = _write(MINIMAL, ".json")
        self.addCleanup(os.unlink, path)
        cfg = load_config(path, ["training.batch_size=4"])
        self.assertEqual(cfg.training.batch_size, 4)


if __name__ == "__main__":
    unittest.main(verbosity=2)
