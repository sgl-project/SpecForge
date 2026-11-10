"""P-EAGLE conversion records effective computation, not ignored HF options."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from specforge.config.draft.immutable import FrozenMap
from specforge.config.draft.rope import DefaultRoPESpec
from tests.test_config.test_draft_legacy_dflash import CONFIGS


def peagle_payload():
    return json.loads((CONFIGS / "qwen3-8b-peagle.json").read_text())


def target_for_peagle(payload):
    return {
        "hidden_size": payload["hidden_size"],
        "vocab_size": payload["vocab_size"],
        "num_hidden_layers": 36,
    }


class LegacyPEagleTest(unittest.TestCase):
    def test_recorded_mask_and_residual_settings(self):
        payload = peagle_payload()
        payload.pop("norm_before_residual", None)
        target = target_for_peagle(payload)
        for mask in (None, -1, True, payload["vocab_size"]):
            with (
                self.subTest(mask=mask),
                self.assertRaisesRegex(ValueError, "mask_token_id"),
            ):
                decode_legacy_config(payload, target_config=target, mask_token_id=mask)
        node = decode_legacy_config(
            payload, target_config=target, mask_token_id=42, norm_before_residual=True
        )
        self.assertEqual(node.spec_config.mask_token_id, 42)
        self.assertTrue(node.spec_config.norm_before_residual)
        self.assertEqual(decode_config(encode_config(node)), node)
        with self.assertRaisesRegex(ValueError, "conflicting.*norm_before_residual"):
            decode_legacy_config(
                payload | {"norm_before_residual": False},
                target_config=target,
                mask_token_id=42,
                norm_before_residual=True,
            )

    def test_padding_index_is_preserved_and_validated(self):
        payload = peagle_payload()
        nodes = [
            decode_legacy_config(
                payload | {"pad_token_id": pad},
                mask_token_id=1,
                target_config=target_for_peagle(payload),
            )
            for pad in (None, 0, 1)
        ]
        self.assertEqual(len(set(nodes)), 3)
        for pad in (-1, True, payload["vocab_size"]):
            with self.assertRaisesRegex(ValueError, "pad_token_id"):
                decode_legacy_config(
                    payload | {"pad_token_id": pad},
                    mask_token_id=1,
                    target_config=target_for_peagle(payload),
                )

    def test_bundled_recipe_expands_target_features_and_round_trips(self):
        payload = peagle_payload()
        before = copy.deepcopy(payload)
        node = decode_legacy_config(
            payload, mask_token_id=1, target_config=target_for_peagle(payload)
        )
        self.assertEqual(node.num_hidden_layers, 4)
        self.assertEqual(node.spec_config.target_features.target_layer_ids, (1, 17, 32))
        self.assertEqual(node.spec_config.mask_hidden_shape, (1, 1, 12288))
        self.assertEqual(node.target_bindings, ())
        self.assertEqual(node.spec_config.pad_token_id, payload.get("pad_token_id"))
        self.assertIsNone(node.target_logit_transform)
        self.assertEqual(node.draft_vocab_size, 32000)
        restored = decode_config(json.loads(json.dumps(encode_config(node))))
        self.assertEqual(restored, node)
        self.assertEqual(hash(restored), hash(node))
        self.assertEqual(payload, before)

    def test_explicit_feature_layers_and_llama_defaults(self):
        payload = {
            "architectures": ["PEagleDraftModel"],
            "hidden_size": 32,
            "num_hidden_layers": 2,
            "intermediate_size": 64,
            "vocab_size": 64,
            "num_attention_heads": 4,
            "eagle_config": {"eagle_aux_hidden_state_layer_ids": [3, 1, 2]},
            "norm_before_residual": True,
        }
        node = decode_legacy_config(
            payload, mask_token_id=1, target_config=target_for_peagle(payload)
        )
        self.assertEqual(node.num_hidden_layers, 2)
        self.assertEqual(node.draft_vocab_size, 64)
        self.assertEqual(node.attention_configs["default"].head_dim, 8)
        self.assertEqual(node.attention_configs["default"].num_key_value_heads, 4)
        self.assertEqual(node.spec_config.target_features.target_layer_ids, (1, 2, 3))
        self.assertTrue(node.spec_config.norm_before_residual)
        # The public codec accepts Mapping, not only concrete dictionaries.
        mapping = FrozenMap(
            {key: value for key, value in payload.items() if key != "eagle_config"}
            | {"architectures": tuple(payload["architectures"])}
        )
        from_mapping = decode_legacy_config(
            mapping, mask_token_id=1, target_config=target_for_peagle(payload)
        )
        self.assertEqual(
            from_mapping.spec_config.target_features.target_layer_ids, (1, 17, 32)
        )

    def test_ignored_legacy_options_do_not_change_effective_structure(self):
        payload = peagle_payload()
        expected = decode_legacy_config(
            payload, mask_token_id=1, target_config=target_for_peagle(payload)
        )
        changed = payload | {
            "sliding_window": 128,
            "tie_word_embeddings": True,
            "rope_scaling": {"rope_type": "linear", "factor": 4},
        }
        self.assertEqual(
            decode_legacy_config(
                changed, mask_token_id=1, target_config=target_for_peagle(payload)
            ),
            expected,
        )
        self.assertIsInstance(expected.rope_parameters, DefaultRoPESpec)
        changed["rope_parameters"] = {"rope_type": "default", "rope_theta": 1234}
        with self.assertRaisesRegex(ValueError, "conflicting.*rope_theta"):
            decode_legacy_config(
                changed, mask_token_id=1, target_config=target_for_peagle(payload)
            )

    def test_invalid_settings_and_unknown_fields_are_rejected(self):
        for field, value in (
            ("draft_vocab_size", 0),
            ("target_hidden_size", 64),
            ("norm_before_residual", 1),
            ("pretraining_tp", 2),
            ("num_hidden_layers", 0),
            ("eagle_config", []),
            ("rope_scaling", []),
            ("training", {}),
        ):
            with self.subTest(field=field):
                payload = peagle_payload() | {field: value}
                with self.assertRaises(ValueError):
                    decode_legacy_config(
                        payload,
                        mask_token_id=1,
                        target_config=target_for_peagle(payload),
                    )
        for layers in ([1, 1, 2], [1, 2], [1, True, 3], [1, 2, 36]):
            payload = peagle_payload() | {
                "eagle_config": {"eagle_aux_hidden_state_layer_ids": layers}
            }
            with self.assertRaisesRegex(ValueError, "layer_ids"):
                decode_legacy_config(
                    payload, mask_token_id=1, target_config=target_for_peagle(payload)
                )


if __name__ == "__main__":
    unittest.main()
