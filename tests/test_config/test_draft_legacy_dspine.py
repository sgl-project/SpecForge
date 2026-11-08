"""DSpine migration separates injection structure from training policy."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from specforge.config.draft.codec.legacy.dspine import extract_dspine_training_settings
from tests.test_config.test_draft_legacy_dflash import legacy_payload, target_for


def dspine_payload():
    return legacy_payload() | {
        "architectures": ["DSpineDraftModel"],
        "dspine_config": {"transfer_rank": 8, "message_size": 4},
    }


class LegacyDSpineTest(unittest.TestCase):
    def test_causality_and_round_trip_preserve_effective_structure(self):
        payload = dspine_payload() | {
            "num_hidden_layers": 2,
            "layer_types": ["sliding_attention", "full_attention"],
            "use_sliding_window": True,
            "sliding_window": 128,
        }
        for causal in (None, True):
            source = payload | {"is_causal": causal}
            before = copy.deepcopy(source)
            node = decode_legacy_config(source, target_config=target_for(source))
            self.assertTrue(
                all(attn.is_causal for attn in node.attention_configs.values())
            )
            restored = decode_config(json.loads(json.dumps(encode_config(node))))
            self.assertEqual(restored, node)
            self.assertEqual(hash(restored), hash(node))
            self.assertEqual(source, before)
        with self.assertRaisesRegex(ValueError, "causal"):
            decode_legacy_config(
                payload | {"is_causal": False}, target_config=target_for(payload)
            )

    def test_training_policy_does_not_change_model_structure(self):
        payload = dspine_payload()
        node = decode_legacy_config(payload, target_config=target_for(payload))
        overrides = {
            "top_k": 4,
            "ce_weight": 0.3,
            "replacement_layers": 1,
            "alignment_weight": 2.0,
            "replacement_probability": 0.8,
        }
        payload["dspine_config"].update(overrides)
        self.assertEqual(
            decode_legacy_config(payload, target_config=target_for(payload)), node
        )
        settings = extract_dspine_training_settings(payload)
        for name, value in overrides.items():
            self.assertEqual(settings[name], value)
            self.assertNotIn(name, encode_config(node)["spec_config"])
        self.assertNotIn("transfer_rank", settings)
        self.assertNotIn("message_size", settings)
        settings["top_k"] = 1
        self.assertEqual(extract_dspine_training_settings(payload)["top_k"], 4)

    def test_missing_options_resolve_historical_defaults(self):
        payload = dspine_payload()
        payload.pop("dspine_config")
        node = decode_legacy_config(payload, target_config=target_for(payload))
        self.assertEqual(node.spec_config.injection.transfer_rank, 1024)
        self.assertEqual(node.spec_config.injection.message_size, 512)
        settings = extract_dspine_training_settings(payload)
        self.assertEqual(settings["top_k"], 16)
        self.assertEqual(settings["alignment_warmup_steps"], 146)
        self.assertEqual(settings["replacement_hold_ratio"], 1 / 6)
        self.assertEqual(
            decode_legacy_config(
                payload | {"dspine_config": None}, target_config=target_for(payload)
            ),
            node,
        )

    def test_invalid_structure_and_training_settings_fail(self):
        for name, value in (
            ("unknown", 1),
            ("transfer_rank", True),
            ("transfer_rank", 1000000),
            ("message_size", 0),
            ("top_k", 0),
            ("top_k", 1000000),
            ("alignment_warmup_steps", 1.5),
            ("replacement_layers", -1),
            ("alignment_weight", -1),
            ("ce_weight", 2),
            ("replacement_probability", True),
            ("replacement_probability", -0.1),
            ("replacement_hold_ratio", 0.5),
            ("replacement_end_ratio", 2),
            ("candidate_mass_threshold", 1),
            ("alignment_weight", float("nan")),
        ):
            with self.subTest(name=name, value=value):
                payload = dspine_payload()
                payload["dspine_config"][name] = value
                with self.assertRaises(ValueError):
                    decode_legacy_config(payload, target_config=target_for(payload))
        payload = dspine_payload()
        payload["dflash_config"]["projector_type"] = "dspark"
        with self.assertRaisesRegex(ValueError, "projector_type"):
            decode_legacy_config(payload, target_config=target_for(payload))
        with self.assertRaisesRegex(ValueError, "DSpineDraftModel"):
            extract_dspine_training_settings(legacy_payload())


if __name__ == "__main__":
    unittest.main()
