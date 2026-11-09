"""DSpark migration preserves effective heads, layout, and draft-token counts."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from tests.test_config.test_draft_legacy_dflash import CONFIGS, target_for


def dspark_payload():
    return json.loads((CONFIGS / "qwen3-8b-dspark.json").read_text())


class LegacyDSparkTest(unittest.TestCase):
    def test_all_bundled_recipes_round_trip_including_yarn(self):
        paths = sorted(CONFIGS.glob("*dspark*.json"))
        self.assertGreaterEqual(len(paths), 9)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                before = copy.deepcopy(payload)
                node = decode_legacy_config(payload, target_config=target_for(payload))
                restored = decode_config(json.loads(json.dumps(encode_config(node))))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(node.architecture, "DSparkDraftModel")
                self.assertEqual(
                    node.spec_config.protocol.num_draft_tokens, payload["block_size"]
                )
                self.assertEqual(
                    node.spec_config.protocol.verify_window, payload["block_size"] + 1
                )
                self.assertEqual(payload, before)

    def test_head_presence_is_resolved_from_legacy_options(self):
        for kind in ("vanilla", "gated", "rnn"):
            payload = dspark_payload()
            method = payload["dflash_config"]
            method["markov_head_type"] = kind
            method.pop("enable_confidence_head")
            spec = decode_legacy_config(
                payload, target_config=target_for(payload)
            ).spec_config
            self.assertEqual(spec.markov.type, kind)
            self.assertTrue(spec.confidence.with_markov)
            method["enable_confidence_head"] = False
            node = decode_legacy_config(payload, target_config=target_for(payload))
            self.assertIsNone(node.spec_config.confidence)
            self.assertNotIn(
                "confidence_head_alpha", encode_config(node)["spec_config"]
            )
        payload = dspark_payload()
        method = payload["dflash_config"]
        method.update(
            markov_rank=None,
            confidence_head_with_markov=False,
            enable_confidence_head=True,
        )
        spec = decode_legacy_config(
            payload, target_config=target_for(payload)
        ).spec_config
        self.assertIsNone(spec.markov)
        self.assertEqual(spec.confidence.input_width(32, spec.markov), 32)

    def test_redundant_exports_cannot_override_effective_head_settings(self):
        payload = dspark_payload()
        payload.update(
            markov_rank=256, markov_head_type="vanilla", target_hidden_size=None
        )
        before = decode_legacy_config(payload, target_config=target_for(payload))
        for field, value in (("markov_rank", 128), ("markov_head_type", "gated")):
            with self.assertRaisesRegex(ValueError, "conflicting"):
                decode_legacy_config(
                    payload | {field: value}, target_config=target_for(payload)
                )
        payload["dflash_config"]["shift_label"] = True
        self.assertEqual(
            decode_legacy_config(payload, target_config=target_for(payload)), before
        )
        with self.assertRaisesRegex(ValueError, "target_hidden_size"):
            decode_legacy_config(
                payload | {"target_hidden_size": 32}, target_config=target_for(payload)
            )

    def test_top_level_heads_match_nested_settings_and_reject_conflicts(self):
        payload = json.loads((CONFIGS / "kimi-k3-dspark.json").read_text())
        target = target_for(payload)
        expected = decode_legacy_config(payload, target_config=target)
        alternatives = {
            "markov_rank": 128,
            "markov_head_type": "gated",
            "enable_confidence_head": False,
            "confidence_head_with_markov": False,
            "confidence_head_alpha": 0.0,
        }
        for name, value in alternatives.items():
            payload[name] = payload["dflash_config"][name]
            with (
                self.subTest(field=name),
                self.assertRaisesRegex(ValueError, f"conflicting.*{name}"),
            ):
                decode_legacy_config(payload | {name: value}, target_config=target)
        self.assertEqual(decode_legacy_config(payload, target_config=target), expected)
        for name in alternatives:
            del payload["dflash_config"][name]
        actual = decode_legacy_config(payload, target_config=target)
        self.assertEqual(actual, expected)
        self.assertEqual(actual.architecture, "DSparkDraftModel")
        self.assertEqual(actual.spec_config.markov.rank, 256)
        self.assertTrue(actual.spec_config.confidence.with_markov)
        del payload["enable_confidence_head"]
        self.assertEqual(decode_legacy_config(payload, target_config=target), expected)
        for name in alternatives:
            payload.pop(name, None)
        spec = decode_legacy_config(payload, target_config=target).spec_config
        self.assertIsNone(spec.markov)
        self.assertIsNone(spec.confidence)

    def test_invalid_heads_and_component_contracts_fail(self):
        for name, value in (
            ("markov_rank", True),
            ("markov_rank", -1),
            ("markov_head_type", "other"),
            ("enable_confidence_head", 1),
            ("confidence_head_alpha", float("nan")),
            ("projector_type", "domino"),
            ("selector_rank", 8),
        ):
            payload = dspark_payload()
            payload["dflash_config"][name] = value
            with self.assertRaisesRegex(
                ValueError, "markov|confidence|projector|selector"
            ):
                decode_legacy_config(payload, target_config=target_for(payload))
        payload = dspark_payload()
        payload["dflash_config"].update(markov_rank=0, enable_confidence_head=False)
        with self.assertRaisesRegex(ValueError, "requires a Markov head"):
            decode_legacy_config(payload, target_config=target_for(payload))
        payload = dspark_payload()
        payload["num_key_value_heads"] = payload["num_attention_heads"]
        with self.assertRaisesRegex(ValueError, "GQA requires fewer"):
            decode_legacy_config(payload, target_config=target_for(payload))
        payload["dflash_config"]["attention_mode"] = "mha"
        self.assertEqual(
            decode_legacy_config(payload, target_config=target_for(payload))
            .attention_configs["full_attention"]
            .type,
            "mha",
        )


if __name__ == "__main__":
    unittest.main()
