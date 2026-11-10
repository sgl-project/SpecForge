"""Legacy MTP migration preserves effective structure and target bindings."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from specforge.config.draft.logits import MuPLogitTransformSpec, NoLogitTransformSpec
from tests.test_config.test_draft_legacy_dflash import CONFIGS


def mtp_payload():
    return json.loads((CONFIGS / "qwen3.5-4b-mtp.json").read_text())


def target_for_mtp(payload):
    return {name: payload[name] for name in ("hidden_size", "vocab_size")}


class LegacyMTPTest(unittest.TestCase):
    def test_mup_shared_head_and_tied_target_guard(self):
        payload = mtp_payload()
        target = target_for_mtp(payload) | {"logits_mup_width_multiplier": 2}
        node = decode_legacy_config(payload, target_config=target)
        self.assertEqual(
            node.target_logit_transform, MuPLogitTransformSpec(multiplier=2)
        )
        with self.assertRaisesRegex(ValueError, "tied"):
            decode_legacy_config(
                payload, target_config=target | {"tie_word_embeddings": True}
            )
        payload["mtp_config"] = {"share_lm_head": False}
        self.assertIsNone(
            decode_legacy_config(payload, target_config=target).target_logit_transform
        )

    def test_rotary_aliases_conflicts_and_native_sections(self):
        payload = mtp_payload()
        target = target_for_mtp(payload)
        expected = decode_legacy_config(payload, target_config=target)
        modern = payload.copy()
        modern["rope_parameters"] = {"rope_theta": modern.pop("rope_theta")}
        self.assertEqual(decode_legacy_config(modern, target_config=target), expected)
        for params in (
            {"rope_theta": 5000},
            {"rope_type": "linear", "factor": 2},
            {"partial_rotary_factor": 0.5},
        ):
            with self.subTest(params=params), self.assertRaises(ValueError):
                decode_legacy_config(
                    payload | {"rope_parameters": params}, target_config=target
                )
        target["rope_parameters"] = dict(
            rope_theta=10_000_000,
            partial_rotary_factor=0.25,
            mrope_section=[11, 11, 10],
            mrope_interleaved=True,
        )
        node = decode_legacy_config(payload, target_config=target)
        self.assertEqual(node.rope_parameters.mrope_section, (11, 11, 10))
        self.assertTrue(node.rope_parameters.mrope_interleaved)
        self.assertEqual(decode_config(encode_config(node), target_config=target), node)
        target["rope_parameters"]["rope_theta"] = 5000
        self.assertEqual(decode_legacy_config(payload, target_config=target), expected)

    def test_bundled_recipe_round_trip_and_binding_choices(self):
        for method in (None, {}, {"share_lm_head": True}, {"share_lm_head": False}):
            payload = mtp_payload() | {"mtp_config": method}
            before = copy.deepcopy(payload)
            node = decode_legacy_config(payload, target_config=target_for_mtp(payload))
            with self.subTest(method=method):
                self.assertEqual(node.architecture, "MTPDraftModel")
                self.assertEqual(node.num_hidden_layers, 1)
                self.assertEqual(node.norm_config.weight_offset, 1)
                self.assertTrue(node.attention_configs["default"].output_gate)
                self.assertEqual(node.rope_parameters.partial_rotary_factor, 0.25)
                self.assertEqual(node.rope_parameters.rope_theta, 10_000_000)
                self.assertFalse(node.tie_word_embeddings)
                shared = method is None or method.get("share_lm_head", True)
                self.assertEqual("lm_head" in node.target_bindings, shared)
                self.assertEqual(
                    node.target_logit_transform,
                    NoLogitTransformSpec() if shared else None,
                )
                restored = decode_config(json.loads(json.dumps(encode_config(node))))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(payload, before)

    def test_unused_hf_settings_do_not_change_the_network(self):
        payload = mtp_payload()
        target = target_for_mtp(payload)
        expected = decode_legacy_config(payload, target_config=target)
        changes = {
            "sliding_window": 128,
            "draft_vocab_size": 64,
        }
        self.assertEqual(
            decode_legacy_config(payload | changes, target_config=target), expected
        )

    def test_defaults_and_explicit_attention_controls(self):
        payload = mtp_payload()
        for name in (
            "num_key_value_heads",
            "head_dim",
            "attn_output_gate",
            "partial_rotary_factor",
            "attention_bias",
            "attention_dropout",
        ):
            payload.pop(name)
        payload["num_attention_heads"] = 32
        node = decode_legacy_config(payload, target_config=target_for_mtp(payload))
        attention = node.attention_configs["default"]
        # The Qwen default KV head count is 32, so use a matching query count.
        self.assertEqual(attention.head_dim, 128)
        self.assertFalse(attention.output_gate)
        self.assertEqual(node.rope_parameters.partial_rotary_factor, 1.0)
        payload.update(
            num_key_value_heads=None, attention_bias=True, attention_dropout=0.1
        )
        attention = decode_legacy_config(
            payload, target_config=target_for_mtp(payload)
        ).attention_configs["default"]
        self.assertEqual(attention.num_key_value_heads, 32)
        self.assertTrue(attention.attention_bias)
        self.assertEqual(attention.attention_dropout, 0.1)

    def test_invalid_structure_and_incompatible_target_fail(self):
        payload = mtp_payload()
        target = target_for_mtp(payload)
        for changes in (
            {"draft_vocab_size": 0},
            {"num_hidden_layers": 0},
            {"head_dim": None},
            {"attn_output_gate": "true"},
            {"mtp_config": {"share_lm_head": 1}},
            {"mtp_config": {"native_prefix": "mtp."}},
            {"mtp_config": []},
            {"dflash_config": {}},
            {"partial_rotary_factor": 0.0},
            {"rope_parameters": []},
            {"rope_parameters": {"unknown": 1}},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                decode_legacy_config(payload | changes, target_config=target)
        for changes in (
            {"hidden_size": 1},
            {"vocab_size": 1},
        ):
            with self.subTest(target=changes), self.assertRaises(ValueError):
                decode_legacy_config(payload, target_config=target | changes)


if __name__ == "__main__":
    unittest.main()
