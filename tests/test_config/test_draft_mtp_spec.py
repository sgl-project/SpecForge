"""MTP fusion and alignment are independent of native checkpoint layout."""

import dataclasses
import json
import unittest

from specforge.config.draft.architecture import MTPSpec
from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config


def mtp_config(**overrides):
    base = draft_config()
    return dataclasses.replace(
        base,
        **(
            dict(
                architecture="MTPDraftModel",
                layer_types=("full_dense",),
                attention_configs={
                    "full": dataclasses.replace(
                        base.attention_configs["full"], is_causal=True
                    )
                },
                spec_config=MTPSpec(
                    target_features=FinalHiddenFeatureSpec(hidden_size=32)
                ),
            )
            | overrides
        ),
    )


class MTPSpecTest(unittest.TestCase):
    def test_fusion_alignment_and_canonical_round_trip(self):
        for depth in (1, 3):
            node = mtp_config(layer_types=("full_dense",) * depth)
            spec = node.spec_config
            self.assertIsNone(spec.protocol)
            self.assertEqual(spec.feature_token_offset, -1)
            self.assertEqual(spec.prediction_token_offset, 1)
            restored = decode_config(json.loads(json.dumps(encode_config(node))))
            self.assertEqual(restored, node)
            self.assertEqual(hash(restored), hash(node))

    def test_final_features_validate_width_without_selecting_a_decoder_layer(self):
        node = mtp_config(target_bindings=(), target_logit_transform=None)
        payload = encode_config(node)
        # A final output is not an indexed intermediate decoder feature.
        decode_config(payload, target_config={"hidden_size": 32})
        with self.assertRaisesRegex(ValueError, "target_features.hidden_size"):
            decode_config(payload, target_config={"hidden_size": 64})
        spec = MTPSpec(target_features=FinalHiddenFeatureSpec(hidden_size=64))
        node = dataclasses.replace(node, spec_config=spec)
        decode_config(encode_config(node), target_config={"hidden_size": 64})

    def test_causality_vocabulary_and_feature_types_are_checked(self):
        with self.assertRaisesRegex(ValueError, "causal"):
            mtp_config(attention_configs=draft_config().attention_configs)
        with self.assertRaisesRegex(ValueError, "MTP requires draft_vocab_size"):
            mtp_config(
                draft_vocab_size=32,
                target_bindings=(),
                target_logit_transform=None,
            )
        with self.assertRaisesRegex(ValueError, "FinalHiddenFeatureSpec"):
            MTPSpec(target_features={"hidden_size": 32})
        for value in (0, True, 1.5):
            with self.assertRaises(ValueError):
                FinalHiddenFeatureSpec(hidden_size=value)

    def test_checkpoint_and_training_policy_are_not_architecture_fields(self):
        for field in ("native_prefix", "share_lm_head", "loss_weight", "num_steps"):
            payload = encode_config(mtp_config())
            payload["spec_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            mtp_config().spec_config.target_features.hidden_size = 64
        payload = encode_config(mtp_config())
        payload["spec_config"].clear()
        with self.assertRaisesRegex(ValueError, "target_features"):
            decode_config(payload)


if __name__ == "__main__":
    unittest.main()
