"""Every checked-in training recipe owns a complete inline draft configuration."""

import unittest
from pathlib import Path

import yaml

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.config import Config
from specforge.training.model_loading import resolve_draft_config

ROOT = Path(__file__).resolve().parents[2]


class ExampleDraftConfigWiringTest(unittest.TestCase):
    def test_all_recipes_resolve_inline_settings_without_model_allocation(self):
        registry = builtin_algorithm_registry()
        recipes = list((ROOT / "examples/configs").rglob("*.yaml"))
        self.assertTrue(recipes)
        for path in recipes:
            with self.subTest(recipe=str(path.relative_to(ROOT))):
                payload = yaml.safe_load(path.read_text())
                self.assertIsInstance(payload["model"]["draft_model_config"], dict)
                cfg = Config.model_validate(payload)
                provider = registry.resolve(
                    cfg.training.strategy
                ).providers.model.draft_config
                resolved = resolve_draft_config(cfg, provider=provider)
                self.assertIn(
                    resolved.architectures[0], provider.compatible_architectures
                )
