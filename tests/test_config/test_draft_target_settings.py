"""Target-derived settings are resolved once and checked on canonical reload."""

import copy
import unittest
from dataclasses import dataclass, replace
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.architecture.base import ArchitectureSpec
from specforge.config.draft.architecture.features import TargetFeatureSpec
from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.codec.target import resolve_target_logits
from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from specforge.config.draft.registry import register_spec
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_canonical_codec import shorthand

TARGET = {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8}


@dataclass(frozen=True, kw_only=True)
class RenamedFeaturesSpec(ArchitectureSpec):
    inputs: tuple[TargetFeatureSpec, ...]

    @property
    def protocol(self):
        return None

    def validate_target(self, target):
        for node in self.inputs:
            node.validate_target(target)


class TargetSettingsTest(unittest.TestCase):
    def test_custom_architecture_validates_its_own_nested_features(self):
        features = draft_config().spec_config.target_features
        with patch.dict(registry._SPECS):
            register_spec(
                "RenamedFeaturesDraftModel", RenamedFeaturesSpec, lambda _: None
            )
            config = draft_config(
                architecture="RenamedFeaturesDraftModel",
                spec_config=RenamedFeaturesSpec(inputs=(features, features)),
            )
            self.assertEqual(
                decode_config(encode_config(config), target_config=TARGET), config
            )
            invalid = replace(
                config,
                spec_config=RenamedFeaturesSpec(
                    inputs=(features, replace(features, target_layer_ids=(99,)))
                ),
            )
            with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                decode_config(encode_config(invalid), target_config=TARGET)

    def test_target_width_is_independent_of_unbound_draft_width(self):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        payload = encode_config(config)
        payload["spec_config"]["target_features"]["hidden_size"] = 64
        target = {**TARGET, "hidden_size": 64}
        node = parse_draft(payload, target_config=target)
        self.assertEqual(node.hidden_size, 32)
        self.assertEqual(node.spec_config.target_features.input_width(), 128)
        self.assertEqual(decode_config(encode_config(node)), node)
        with self.assertRaisesRegex(ValueError, "target_features.hidden_size"):
            parse_draft(payload, target_config=TARGET)

    def test_registration_requires_an_explicit_target_validator(self):
        @dataclass(frozen=True)
        class MissingTargetCheck(ArchitectureSpec):
            @property
            def protocol(self):
                return None

        with self.assertRaisesRegex(TypeError, "concrete frozen dataclass"):
            register_spec("MissingTargetCheck", MissingTargetCheck, lambda _: None)

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
            parse_draft(payload, target_config=None)
        with self.assertRaisesRegex(ValueError, "target_logit_transform"):
            decode_config(payload, target_config=target)

    def test_yaml_requires_feature_layers_and_mask_token(self):
        for source in (encode_config(draft_config()), shorthand()):
            for field in ("hidden_size", "target_layer_ids", "mask_token_id"):
                payload = copy.deepcopy(source)
                del payload["spec_config"]["target_features"][field]
                for target in (TARGET, None):
                    with self.subTest(field=field, target=target):
                        with self.assertRaisesRegex(ValueError, field):
                            parse_draft(payload, target_config=target)

    def test_explicit_feature_layers_follow_the_capture_order(self):
        payload = encode_config(draft_config())
        features = payload["spec_config"]["target_features"]
        for reader in (parse_draft, decode_config):
            features["target_layer_ids"] = [1, 6]
            node = reader(payload, target_config=TARGET)
            self.assertEqual(node.spec_config.target_features.target_layer_ids, (1, 6))
            for ids in (None, [], [6, 1], [1, 6, 6]):
                features["target_layer_ids"] = ids
                with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                    reader(payload, target_config=TARGET)

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

    def test_final_target_layer_is_not_a_tap(self):
        payload = encode_config(draft_config())
        payload["spec_config"]["target_features"]["target_layer_ids"] = [1, 7]
        with self.assertRaisesRegex(ValueError, "final target layer"):
            decode_config(payload, target_config=TARGET)
        payload["spec_config"]["target_features"]["target_layer_ids"] = [1, 6]
        decode_config(payload, target_config=TARGET)

    def test_bound_vocab_uses_the_padded_target_size(self):
        payload = encode_config(draft_config(vocab_size=128, draft_vocab_size=128))
        padded = {**TARGET, "vocab_size": 100, "padded_vocab_size": 128}
        self.assertEqual(decode_config(payload, target_config=padded).vocab_size, 128)
        with self.assertRaisesRegex(ValueError, "vocab_size"):
            decode_config(payload, target_config={**TARGET, "vocab_size": 100})

    def test_known_head_changing_fields_fail_closed(self):
        for target in (
            {"lm_head_multiplier": 0.0390625},
            {"norm_head": True},
            {"dim_model_base": 256, "hidden_size": 2560},
        ):
            with self.assertRaisesRegex(ValueError, "unsupported target logit setting"):
                resolve_target_logits(target)
        for target in (
            {"lm_head_multiplier": 1},
            {"norm_head": False},
            {"dim_model_base": 2560, "hidden_size": 2560},
        ):
            self.assertEqual(resolve_target_logits(target), NoLogitTransformSpec())


if __name__ == "__main__":
    unittest.main()
