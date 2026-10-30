"""Checked-in settings and embedded recipes use the versioned format."""

import json
import unittest
from pathlib import Path

import yaml

from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.contract import draft_config_contract


class DraftSettingsTest(unittest.TestCase):
    def test_all_settings_use_layer_specs_and_roundtrip_to_legacy(self):
        settings = sorted(Path("configs").glob("*.json"))
        self.assertGreater(len(settings), 0)
        for path in settings:
            with self.subTest(path=path):
                data = json.loads(path.read_text())
                self.assertEqual(data["schema_version"], 1)
                config = AutoDraftConfig.from_inline(data)
                legacy = AutoDraftConfig.resolve(config.to_hf_config())
                self.assertEqual(
                    draft_config_contract(legacy), draft_config_contract(config)
                )
                self.assertEqual(len(config.layers), config.num_hidden_layers)

    def test_all_embedded_examples_use_version_one(self):
        count = 0
        for path in Path("examples/configs").rglob("*.yaml"):
            data = yaml.safe_load(path.read_text())
            inline = (data or {}).get("model", {}).get("draft_model_config")
            if inline is None:
                continue
            with self.subTest(path=path):
                self.assertIsInstance(inline, dict)
                self.assertEqual(inline["schema_version"], 1)
                config = AutoDraftConfig.from_inline(inline)
                AutoDraftConfig.validate_training_config(config)
            count += 1
        self.assertGreater(count, 0)

    def test_native_setting_name_identifies_native_architecture(self):
        path = Path("configs/qwen3.5-4b-mtp-native.json")
        self.assertTrue(path.is_file())
        self.assertFalse(Path("configs/qwen3.5-4b-mtp.json").exists())
        self.assertEqual(
            json.loads(path.read_text())["architectures"], ["Qwen3_5MTPDraftModel"]
        )
