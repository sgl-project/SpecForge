"""Target-derived settings are resolved once and checked on canonical reload."""

import copy
import unittest

from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.codec.target import (
    default_target_layer_ids,
    resolve_target_logits,
)
from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from tests.test_config.draft_fixtures import draft_config

TARGET = {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8}


class TargetSettingsTest(unittest.TestCase):
    def test_yaml_derives_bound_head_setting_without_mutating_inputs(self):
        payload = encode_config(draft_config())
        payload.pop("target_logit_transform")
        target = {**TARGET, "logits_mup_width_multiplier": 8}
        original = copy.deepcopy((payload, target))
        node = parse_draft(payload, target_config=target)
        self.assertEqual(
            node.target_logit_transform, MuPLogitTransformSpec(multiplier=8)
        )
        saved = encode_config(node)
        self.assertEqual(decode_config(saved, target_config=target), node)
        self.assertEqual(decode_config(saved), node)
        self.assertEqual((payload, target), original)
        self.assertIsInstance(
            parse_draft(payload, target_config=TARGET).target_logit_transform,
            NoLogitTransformSpec,
        )
        with self.assertRaisesRegex(ValueError, "target_logit_transform"):
            parse_draft(payload)
        with self.assertRaisesRegex(ValueError, "target_logit_transform"):
            decode_config(payload, target_config=target)

    def test_explicit_or_saved_settings_are_checked_not_overwritten(self):
        payload = encode_config(draft_config())
        for reader in (parse_draft, decode_config):
            with self.assertRaisesRegex(ValueError, "does not match"):
                reader(
                    payload, target_config={**TARGET, "logits_mup_width_multiplier": 8}
                )
        self.assertEqual(payload["target_logit_transform"], {"type": "none"})

    def test_closed_logit_table_rejects_unsupported_values(self):
        for name in (
            "final_logit_softcapping",
            "logit_scale",
            "logits_scale",
            "logits_scaling",
        ):
            for value in (True, "1", float("nan"), float("inf"), 2):
                with self.assertRaisesRegex(ValueError, name):
                    resolve_target_logits({name: value})
        for value in (0, -1, True, "8", float("nan")):
            with self.assertRaisesRegex(ValueError, "multiplier"):
                resolve_target_logits({"logits_mup_width_multiplier": value})
        self.assertEqual(
            resolve_target_logits(
                {"logits_scaling": 1, "final_logit_softcapping": None}
            ),
            NoLogitTransformSpec(),
        )
        with self.assertRaisesRegex(ValueError, "unwrapped"):
            resolve_target_logits({"text_config": TARGET})
        with self.assertRaisesRegex(ValueError, "mapping"):
            resolve_target_logits(None)

    def test_binding_dimensions_and_feature_indices_are_validated(self):
        payload = encode_config(draft_config())
        for name in ("hidden_size", "vocab_size", "num_hidden_layers"):
            for value in (None, True, 0, "32"):
                with self.assertRaisesRegex(ValueError, name):
                    decode_config(payload, target_config={**TARGET, name: value})
        for name in ("hidden_size", "vocab_size"):
            with self.assertRaisesRegex(ValueError, name):
                decode_config(payload, target_config={**TARGET, name: 128})
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            decode_config(payload, target_config={**TARGET, "num_hidden_layers": 3})

    def test_unbound_head_does_not_inherit_target_transform(self):
        node = draft_config(
            target_bindings=("embed_tokens",), target_logit_transform=None
        )
        restored = parse_draft(
            encode_config(node), target_config={**TARGET, "final_logit_softcapping": 30}
        )
        self.assertIsNone(restored.target_logit_transform)

    def test_default_feature_layers_preserve_historical_rounding(self):
        self.assertEqual(default_target_layer_ids(36, 5), (1, 9, 17, 25, 33))
        self.assertEqual(default_target_layer_ids(28, 5), (1, 7, 13, 19, 25))
        self.assertEqual(default_target_layer_ids(36, 1), (18,))
        self.assertEqual(default_target_layer_ids(7, 3), (1, 2, 4))
        self.assertEqual(default_target_layer_ids(4, 3), (1, 1, 1))
        for value in (True, 0, -1, 2.5):
            with self.assertRaisesRegex(ValueError, "num_target_layers"):
                default_target_layer_ids(value, 2)
            with self.assertRaisesRegex(ValueError, "num_draft_layers"):
                default_target_layer_ids(8, value)
        with self.assertRaisesRegex(ValueError, "out of range"):
            default_target_layer_ids(2, 3)


if __name__ == "__main__":
    unittest.main()
