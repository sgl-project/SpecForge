"""Migration matches the existing DSpine model and its saved options."""

import unittest

from specforge.config.draft.codec.legacy.dspine import extract_dspine_training_settings
from tests.test_config.test_draft_legacy_dspine import dspine_payload


class DSpineCodecParityTest(unittest.TestCase):
    def test_training_settings_match_existing_defaults_and_overrides(self):
        for options in (
            {},
            {
                "top_k": 4,
                "ce_weight": 0.7,
                "alignment_weight": 2,
                "replacement_layers": 0,
                "alignment_warmup_steps": 0,
            },
        ):
            payload = dspine_payload() | {"dspine_config": options}
            expected = (
                dict(
                    top_k=16,
                    ce_weight=0.1,
                    alignment_weight=0.5,
                    alignment_warmup_steps=146,
                    replacement_layers=3,
                    replacement_probability=0.5,
                    replacement_hold_ratio=1 / 6,
                    replacement_end_ratio=1 / 3,
                    candidate_mass_threshold=1e-4,
                )
                | options
            )
            self.assertEqual(extract_dspine_training_settings(payload), expected)


if __name__ == "__main__":
    unittest.main()
