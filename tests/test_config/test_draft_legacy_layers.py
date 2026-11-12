"""Historical feature defaults belong only to legacy migration."""

import unittest

from specforge.config.draft.codec.legacy.rules import default_target_layer_ids


class LegacyLayerDefaultsTest(unittest.TestCase):
    def test_default_feature_layers_preserve_historical_rounding(self):
        self.assertEqual(default_target_layer_ids(36, 5), (1, 9, 17, 25, 33))
        self.assertEqual(default_target_layer_ids(28, 5), (1, 7, 13, 19, 25))
        self.assertEqual(default_target_layer_ids(36, 1), (18,))
        self.assertEqual(default_target_layer_ids(7, 3), (1, 2, 4))
        with self.assertRaisesRegex(ValueError, "duplicates.*explicit distinct"):
            default_target_layer_ids(4, 3)
        for value in (True, 0, -1, 2.5):
            with self.assertRaisesRegex(ValueError, "num_target_layers"):
                default_target_layer_ids(value, 2)
            with self.assertRaisesRegex(ValueError, "num_draft_layers"):
                default_target_layer_ids(8, value)
        with self.assertRaisesRegex(ValueError, "out of range"):
            default_target_layer_ids(2, 3)
