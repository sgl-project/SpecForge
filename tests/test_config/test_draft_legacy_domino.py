"""Domino legacy conversion preserves explicit GRU and prediction semantics."""

import copy
import json
import unittest

from specforge.config.draft.codec import (
    decode_config,
    decode_legacy_config,
    encode_config,
)
from tests.test_config.test_draft_legacy_dflash import CONFIGS, target_for


def domino_payload():
    return json.loads((CONFIGS / "qwen3-8b-domino.json").read_text())


class LegacyDominoTest(unittest.TestCase):
    def test_all_bundled_recipes_round_trip(self):
        paths = sorted(CONFIGS.glob("*domino*.json"))
        self.assertGreaterEqual(len(paths), 3)
        for path in paths:
            with self.subTest(path=path.name):
                payload = json.loads(path.read_text())
                before = copy.deepcopy(payload)
                node = decode_legacy_config(payload, target_config=target_for(payload))
                method = payload["dflash_config"]
                self.assertEqual(
                    node.spec_config.gru.hidden_size, method["gru_hidden_dim"]
                )
                self.assertEqual(
                    node.spec_config.gru.projection_size, method["emb_dim"]
                )
                self.assertEqual(
                    node.spec_config.block.shift_label, method["shift_label"]
                )
                self.assertFalse(node.tie_word_embeddings)
                self.assertEqual(node.target_bindings, ("embed_tokens", "lm_head"))
                restored = decode_config(json.loads(json.dumps(encode_config(node))))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertEqual(payload, before)

    def test_missing_alignment_fields_resolve_historical_defaults(self):
        payload = domino_payload()
        for field in ("shift_label", "pure_draft_prefix_len", "projector_type"):
            payload["dflash_config"].pop(field)
        spec = decode_legacy_config(
            payload, target_config=target_for(payload)
        ).spec_config
        self.assertFalse(spec.block.shift_label)
        self.assertEqual(spec.pure_draft_prefix_len, 0)
        self.assertEqual(spec.suffix_start, 1)

    def test_top_level_dimension_cannot_override_or_fill_effective_setting(self):
        payload = domino_payload()
        payload["emb_dim"] += 1
        with self.assertRaisesRegex(ValueError, "conflicting.*emb_dim"):
            decode_legacy_config(payload, target_config=target_for(payload))
        payload = domino_payload()
        payload["dflash_config"].pop("emb_dim")
        with self.assertRaisesRegex(ValueError, "missing emb_dim"):
            decode_legacy_config(payload, target_config=target_for(payload))
        payload = domino_payload()
        payload["dflash_config"].pop("gru_hidden_dim")
        with self.assertRaisesRegex(ValueError, "missing gru_hidden_dim"):
            decode_legacy_config(payload, target_config=target_for(payload))

    def test_invalid_or_other_algorithm_settings_fail(self):
        for name, value in (
            ("emb_dim", None),
            ("gru_hidden_dim", 0),
            ("shift_label", 1),
            ("pure_draft_prefix_len", -1),
            ("pure_draft_prefix_len", 16),
            ("projector_type", "dspark"),
            ("lambda_base", 0.1),
            ("markov_rank", 8),
        ):
            with self.subTest(name=name, value=value):
                payload = domino_payload()
                payload["dflash_config"][name] = value
                with self.assertRaises(ValueError):
                    decode_legacy_config(payload, target_config=target_for(payload))
        for value in (None, [], "domino"):
            payload = domino_payload() | {"dflash_config": value}
            with self.assertRaisesRegex(ValueError, "dflash_config"):
                decode_legacy_config(payload, target_config=target_for(payload))


if __name__ == "__main__":
    unittest.main()
