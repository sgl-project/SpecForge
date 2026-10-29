"""Checked-in recipes use the canonical draft setting format."""

import json
import unittest
from pathlib import Path

from specforge.modeling.draft.config import AutoDraftConfig


class DraftSettingsTest(unittest.TestCase):
    def test_checked_in_settings_use_the_public_layout(self):
        paths = sorted((Path(__file__).resolve().parents[2] / "configs").glob("*.json"))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(setting=path.name):
                raw = json.loads(path.read_text())
                self.assertIn("spec_config", raw)
                self.assertNotIn("dflash_config", raw)
                # Serving-only legacy metadata (e.g. use_aux_hidden_state)
                # remains intact, but interpreted fields have one location.
                self.assertNotIn(
                    "eagle_aux_hidden_state_layer_ids", raw.get("eagle_config", {})
                )
                self.assertNotIn("mtp_config", raw)
                config = AutoDraftConfig.finalize(AutoDraftConfig.from_dict(raw))
                public = AutoDraftConfig.to_dict(config)
                self.assertEqual(public["spec_config"], raw["spec_config"])
