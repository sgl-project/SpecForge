"""EAGLE3 fusion dimensions and feature semantics are explicit and immutable."""

import dataclasses
import json
import unittest

from specforge.config.draft.architecture import Eagle3Spec
from specforge.config.draft.architecture.eagle_fusion import AuxiliaryFeatureSpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config


def eagle_config(**overrides):
    base = draft_config()
    return dataclasses.replace(
        base,
        **(
            dict(
                architecture="Eagle3DraftModel",
                layer_types=("full_dense",),
                attention_configs={
                    "full": dataclasses.replace(
                        base.attention_configs["full"], is_causal=True
                    )
                },
                draft_vocab_size=32,
                target_bindings=("embed_tokens",),
                target_logit_transform=None,
                spec_config=Eagle3Spec(
                    target_features=AuxiliaryFeatureSpec(
                        target_layer_ids=(1, 3, 4), hidden_size=32
                    ),
                    normalize_features=False,
                    norm_output=False,
                ),
            )
            | overrides
        ),
    )


class Eagle3SpecTest(unittest.TestCase):
    def test_fusion_and_output_norm_choices_round_trip(self):
        for feature_norm in (False, True):
            for output_norm in (False, True):
                base = eagle_config()
                spec = dataclasses.replace(
                    base.spec_config,
                    normalize_features=feature_norm,
                    norm_output=output_norm,
                )
                node = dataclasses.replace(base, spec_config=spec)
                self.assertIsNone(spec.protocol)
                self.assertEqual(spec.target_features.input_width, 96)
                self.assertEqual(spec.attention_input_width(32, 0), 64)
                restored = decode_config(json.loads(json.dumps(encode_config(node))))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))

    def test_target_feature_width_and_layers_are_checked_on_reload(self):
        payload = encode_config(eagle_config(target_bindings=()))
        target = {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8}
        decode_config(payload, target_config=target)
        for field, value, message in (
            ("hidden_size", 64, "target_features.hidden_size"),
            ("num_hidden_layers", 4, "target_layer_ids"),
        ):
            with self.assertRaisesRegex(ValueError, message):
                decode_config(payload, target_config=target | {field: value})
        spec = dataclasses.replace(
            eagle_config().spec_config,
            target_features=AuxiliaryFeatureSpec(
                target_layer_ids=(1, 3, 4), hidden_size=64
            ),
        )
        node = eagle_config(spec_config=spec, target_bindings=())
        decode_config(encode_config(node), target_config=target | {"hidden_size": 64})

    def test_single_layer_causality_and_bound_head_constraints(self):
        with self.assertRaisesRegex(ValueError, "exactly one"):
            eagle_config(layer_types=("full_dense", "full_dense"))
        with self.assertRaisesRegex(ValueError, "causal"):
            eagle_config(attention_configs=draft_config().attention_configs)
        with self.assertRaisesRegex(ValueError, "bound lm_head"):
            eagle_config(
                target_bindings=("lm_head",),
                target_logit_transform=draft_config().target_logit_transform,
            )
        for index in (-1, True, 1):
            with self.assertRaises(ValueError):
                eagle_config().spec_config.attention_input_width(32, index)

    def test_feature_nodes_and_boolean_settings_are_strict(self):
        for layers in ([], (1, 2), (1, 2, 3, 4), (1, True, 3), (1, -1, 3)):
            with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                AuxiliaryFeatureSpec(target_layer_ids=layers, hidden_size=32)
        for width in (0, True, 1.5):
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                AuxiliaryFeatureSpec(target_layer_ids=(1, 2, 3), hidden_size=width)
        for field, value in (
            ("target_features", {}),
            ("normalize_features", 1),
            ("norm_output", None),
        ):
            with self.assertRaisesRegex(ValueError, field):
                dataclasses.replace(eagle_config().spec_config, **{field: value})
        with self.assertRaises(dataclasses.FrozenInstanceError):
            eagle_config().spec_config.norm_output = True

    def test_runtime_and_mapping_tensors_do_not_enter_structure(self):
        node = eagle_config()
        for field in ("ttt_length", "num_steps", "d2t", "loss_weight", "mask_token_id"):
            payload = encode_config(node)
            payload["spec_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        for field in ("target_features", "normalize_features", "norm_output"):
            payload = encode_config(node)
            payload["spec_config"].pop(field)
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)


if __name__ == "__main__":
    unittest.main()
