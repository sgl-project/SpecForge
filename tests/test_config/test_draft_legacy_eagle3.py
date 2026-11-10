"""All bundled EAGLE3 configurations become explicit single-layer structures."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from tests.test_config.test_draft_legacy_dflash import CONFIGS


def eagle_payload():
    return json.loads((CONFIGS / "qwen3-8b-eagle3.json").read_text())


def target_for_eagle(payload):
    return {
        "hidden_size": payload["hidden_size"],
        "vocab_size": payload["vocab_size"],
        "num_hidden_layers": 128,
    }


class LegacyEagle3Test(unittest.TestCase):
    def test_unsupported_structural_settings_are_rejected(self):
        from tests.test_config.test_draft_legacy_mtp import mtp_payload
        from tests.test_config.test_draft_legacy_peagle import peagle_payload

        for payload in (eagle_payload(), peagle_payload(), mtp_payload()):
            architecture = payload["architectures"][0]
            settings = dict(
                mlp_bias=True,
                use_sliding_window=True,
                layer_types=["sliding_attention"],
            )
            if architecture == "Qwen3_5MTPDraftModel":
                settings.update(num_hidden_layers=2, is_causal=False)
            else:
                settings.update(
                    attention_bias=True,
                    attention_dropout=0.2,
                    partial_rotary_factor=0.5,
                )
            options = {"mask_token_id": 1} if architecture == "PEagleDraftModel" else {}
            for field, value in settings.items():
                with self.subTest(architecture=architecture, field=field):
                    with self.assertRaisesRegex(ValueError, field):
                        decode_legacy_config(
                            payload | {field: value},
                            target_config=target_for_eagle(payload),
                            **options,
                        )

    def test_every_bundled_eagle_recipe_round_trips(self):
        count = 0
        for path in CONFIGS.glob("*.json"):
            payload = json.loads(path.read_text())
            if payload.get("architectures") != ["LlamaForCausalLMEagle3"]:
                continue
            count += 1
            with self.subTest(path=path.name):
                before = copy.deepcopy(payload)
                node = decode_legacy_config(
                    payload, target_config=target_for_eagle(payload)
                )
                self.assertEqual(node.architecture, "Eagle3DraftModel")
                self.assertEqual(node.num_hidden_layers, 1)
                self.assertEqual(node.target_bindings, ("embed_tokens",))
                self.assertIsNone(node.target_logit_transform)
                self.assertEqual(
                    node.spec_config.normalize_features, payload.get("fc_norm", False)
                )
                self.assertEqual(
                    node.spec_config.norm_output, payload.get("norm_output", False)
                )
                restored = decode_config(json.loads(json.dumps(encode_config(node))))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(payload, before)
        self.assertGreaterEqual(count, 27)

    def test_legacy_depth_and_target_family_fields_do_not_select_layers(self):
        payload = eagle_payload()
        expected = decode_legacy_config(
            payload, target_config=target_for_eagle(payload)
        )
        changed = payload | {
            "num_hidden_layers": 8,
            "model_type": "qwen3",
            "target_model_type": "gemma3_text",
            "decoder_sparse_step": 2,
            "sliding_window": 128,
        }
        self.assertEqual(
            decode_legacy_config(changed, target_config=target_for_eagle(payload)),
            expected,
        )

    def test_feature_selection_and_norm_choices_are_explicit(self):
        payload = eagle_payload() | {
            "eagle_config": {
                "use_aux_hidden_state": False,
                "eagle_aux_hidden_state_layer_ids": [3, 1, 2],
            },
            "fc_norm": True,
            "norm_output": True,
        }
        node = decode_legacy_config(payload, target_config=target_for_eagle(payload))
        self.assertEqual(node.spec_config.target_features.target_layer_ids, (1, 2, 3))
        self.assertTrue(node.spec_config.normalize_features)
        self.assertTrue(node.spec_config.norm_output)
        self.assertEqual(
            node.spec_config.target_features.input_width, 3 * payload["hidden_size"]
        )

    def test_invalid_parameters_and_other_architecture_fields_fail(self):
        for field, value in (
            ("fc_norm", 1),
            ("norm_output", "false"),
            ("pretraining_tp", 2),
            ("draft_vocab_size", 0),
            ("norm_before_residual", True),
            ("target_hidden_size", 1),
            ("num_hidden_layers", 0),
            ("dflash_config", {}),
        ):
            payload = eagle_payload() | {field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                decode_legacy_config(payload, target_config=target_for_eagle(payload))
        for method in (
            {"use_aux_hidden_state": 1},
            {"unknown": True},
            {"eagle_aux_hidden_state_layer_ids": [1, 2, 128]},
        ):
            payload = eagle_payload() | {"eagle_config": method}
            with self.assertRaises(ValueError):
                decode_legacy_config(payload, target_config=target_for_eagle(payload))


if __name__ == "__main__":
    unittest.main()
