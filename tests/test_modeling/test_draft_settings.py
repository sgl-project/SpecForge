"""Embedded training recipes own the versioned draft settings."""

import unittest
from pathlib import Path

import yaml

from specforge.config.draft import AutoDraftConfig
from specforge.config.draft.serialization.contract import draft_config_contract


class DraftSettingsTest(unittest.TestCase):
    def test_embedded_recipes_validate_and_roundtrip(self):
        root = Path(__file__).resolve().parents[2]
        recipes = sorted((root / "examples/configs").rglob("*.yaml"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=str(path.relative_to(root))):
                inline = yaml.safe_load(path.read_text())["model"]["draft_model_config"]
                self.assertIsInstance(inline, dict)
                self.assertEqual(inline["schema_version"], 1)
                config = AutoDraftConfig.from_inline(inline)
                AutoDraftConfig.validate_training_config(config)
                self.assertIs(AutoDraftConfig.resolve(config), config)
                self.assertEqual(len(config.layers), config.num_hidden_layers)
                legacy = AutoDraftConfig.resolve(config.to_hf_config())
                self.assertEqual(
                    draft_config_contract(legacy), draft_config_contract(config)
                )
                if config.architectures[0] == "DSparkDraftModel":
                    self.assertEqual(config.model_type, "qwen3")
