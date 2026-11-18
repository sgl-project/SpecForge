"""Bundled recipes use canonical drafts independently of their filenames."""

import json
import unittest
from pathlib import Path

import yaml

from specforge.config.draft.codec import encode_config, parse_draft
from specforge.config.schema import load_config

ROOT = Path(__file__).resolve().parents[2]


class ExampleDraftConfigWiringTest(unittest.TestCase):
    def test_all_presets_round_trip_yaml_and_json(self):
        paths = list((ROOT / "specforge/config/draft/presets").rglob("*.yaml"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(preset=path.name):
                config = parse_draft(
                    yaml.safe_load(path.read_text()), target_config=None
                )
                payload = encode_config(config)
                self.assertEqual(
                    parse_draft(json.loads(json.dumps(payload)), target_config=None),
                    config,
                )

    def test_recipes_reference_canonical_presets(self):
        count = 0
        for path in (ROOT / "examples/configs").rglob("*.yaml"):
            raw = yaml.safe_load(path.read_text())
            if not isinstance(raw, dict) or "model" not in raw:
                continue
            with self.subTest(recipe=str(path)):
                config = load_config(str(path))
                self.assertIsNotNone(config.draft)
                source = config.draft
                payload = (
                    yaml.safe_load((ROOT / source).read_text())
                    if isinstance(source, str)
                    else source
                )
                parse_draft(payload, target_config=None)
                self.assertNotIn("strategy", raw.get("training", {}))
                self.assertNotIn("draft_model_config", raw["model"])
                count += 1
        self.assertGreater(count, 50)
