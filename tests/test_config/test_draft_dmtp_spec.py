"""DMTP separates model slots, external candidates and feature alignment."""

import json
import unittest
from dataclasses import FrozenInstanceError, replace

from specforge.config.draft.architecture import DMTPSpec
from specforge.config.draft.architecture.block import (
    AnchoredBlockSpec,
    ExternalFirstBlockSpec,
)
from specforge.config.draft.architecture.dmtp import MTPFeatureSpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config


def dmtp_config(**overrides):
    return draft_config(
        architecture="DMTPDraftModel",
        spec_config=DMTPSpec(
            block=ExternalFirstBlockSpec(block_size=7),
            target_features=draft_config().spec_config.target_features,
            mtp_features=MTPFeatureSpec(hidden_size=48),
        ),
        **overrides,
    )


class DMTPSpecTest(unittest.TestCase):
    def test_round_trip_counts_and_alignment(self):
        config = dmtp_config()
        spec = config.spec_config
        self.assertEqual(spec.protocol.num_model_slots, 7)
        self.assertEqual(spec.protocol.num_draft_tokens, 6)
        self.assertEqual(spec.protocol.num_external_candidates, 1)
        self.assertEqual(spec.protocol.verify_window, 8)
        self.assertEqual(spec.protocol.prediction_positions, (1, 2, 3, 4, 5, 6))
        self.assertEqual(spec.context_end_offset, -1)
        self.assertEqual(spec.mtp_features.token_offset, -1)
        self.assertEqual(spec.mtp_features.fusion_input_width(32), 80)
        payload = encode_config(config)
        restored = decode_config(json.loads(json.dumps(payload)))
        self.assertEqual(restored, config)
        self.assertEqual(hash(restored), hash(config))
        self.assertNotIn("context_end_offset", payload["spec_config"])
        self.assertNotIn("token_offset", payload["spec_config"]["mtp_features"])
        with self.assertRaises(FrozenInstanceError):
            spec.mtp_features.hidden_size = 32

    def test_structural_fields_are_required_and_policy_fields_rejected(self):
        for name in ("block", "target_features", "mtp_features"):
            payload = encode_config(dmtp_config())
            del payload["spec_config"][name]
            with self.assertRaisesRegex(ValueError, name):
                decode_config(payload)
        for name in ("context_end_offset", "loss_weight", "mtp_checkpoint"):
            payload = encode_config(dmtp_config())
            payload["spec_config"][name] = 1
            with self.assertRaisesRegex(ValueError, name):
                decode_config(payload)
        payload = encode_config(dmtp_config())
        del payload["spec_config"]["mtp_features"]["hidden_size"]
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            decode_config(payload)

    def test_invalid_nodes_and_widths(self):
        for width in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "hidden_size"):
                MTPFeatureSpec(hidden_size=width)
        spec = dmtp_config().spec_config
        for field, value in (
            ("block", AnchoredBlockSpec(block_size=7)),
            ("target_features", {}),
            ("mtp_features", None),
        ):
            with self.assertRaisesRegex(ValueError, field):
                replace(spec, **{field: value})
        with self.assertRaises(ValueError):
            replace(spec, block=ExternalFirstBlockSpec(block_size=1))

    def test_context_constraints_and_explicit_predictor_width(self):
        config = dmtp_config()
        for change in (
            {
                "draft_vocab_size": 32,
                "target_bindings": (),
                "target_logit_transform": None,
            },
            {
                "spec_config": replace(
                    config.spec_config,
                    target_features=replace(
                        config.spec_config.target_features, mask_token_id=64
                    ),
                )
            },
        ):
            with self.assertRaises(ValueError):
                replace(config, **change)
        target = {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8}
        restored = decode_config(encode_config(config), target_config=target)
        # Predictor width is independently declared; it is not guessed from target width.
        self.assertEqual(restored.spec_config.mtp_features.hidden_size, 48)
        with self.assertRaisesRegex(ValueError, "target_layer_ids"):
            decode_config(
                encode_config(config), target_config=target | {"num_hidden_layers": 2}
            )


if __name__ == "__main__":
    unittest.main()
