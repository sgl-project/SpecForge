# coding=utf-8
"""Config schema mechanics: parsing, validation, dotted overrides."""

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

    def test_unknown_backend_rejected(self):
        bad = {**MINIMAL, "model": {**MINIMAL["model"], "target_backend": "vllm"}}
        with self.assertRaises(ValidationError):
            Config.model_validate(bad)

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
