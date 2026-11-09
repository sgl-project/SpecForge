"""DSpark structural heads and block semantics, independent of loss settings."""

import dataclasses
import json
import unittest

from specforge.config.draft.architecture import DSparkSpec
from specforge.config.draft.architecture.block import (
    AnchoredBlockSpec,
    DraftOnlyBlockSpec,
)
from specforge.config.draft.architecture.dspark import (
    ConfidenceHeadSpec,
    MarkovHeadSpec,
)
from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from tests.test_config.draft_fixtures import draft_config


def dspark_spec(**overrides):
    values = dict(
        block=DraftOnlyBlockSpec(block_size=7),
        target_features=draft_config().spec_config.target_features,
        markov=MarkovHeadSpec(type="vanilla", rank=8),
        confidence=ConfidenceHeadSpec(with_markov=True),
    )
    return DSparkSpec(**(values | overrides))


class DSparkSpecTest(unittest.TestCase):
    def test_draft_only_counts_and_typed_heads_round_trip(self):
        for kind in ("vanilla", "gated", "rnn"):
            spec = dspark_spec(markov=MarkovHeadSpec(type=kind, rank=8))
            node = draft_config(architecture="DSparkDraftModel", spec_config=spec)
            self.assertEqual(spec.protocol.num_model_slots, 7)
            self.assertEqual(spec.protocol.num_draft_tokens, 7)
            self.assertEqual(spec.protocol.verify_window, 8)
            self.assertEqual(spec.protocol.prediction_positions, tuple(range(7)))
            self.assertEqual(spec.confidence.input_width(32, spec.markov), 40)
            restored = decode_config(json.loads(json.dumps(encode_config(node))))
            self.assertEqual(restored, node)
            self.assertEqual(hash(restored), hash(node))

    def test_single_slot_is_rejected_when_loading_config(self):
        payload = encode_config(
            draft_config(architecture="DSparkDraftModel", spec_config=dspark_spec())
        )
        for reader in (parse_draft, decode_config):
            payload["spec_config"]["block"]["block_size"] = 1
            with self.assertRaisesRegex(ValueError, "DSpark block_size.*at least 2"):
                reader(payload, target_config=None)
            payload["spec_config"]["block"]["block_size"] = 2
            self.assertEqual(
                reader(
                    payload, target_config=None
                ).spec_config.protocol.num_draft_tokens,
                2,
            )

    def test_heads_are_explicitly_optional(self):
        for markov in (None, MarkovHeadSpec(type="gated", rank=4)):
            for confidence in (None, ConfidenceHeadSpec(with_markov=False)):
                spec = dspark_spec(markov=markov, confidence=confidence)
                node = draft_config(architecture="DSparkDraftModel", spec_config=spec)
                self.assertEqual(decode_config(encode_config(node)), node)
                if confidence is not None:
                    self.assertEqual(confidence.input_width(32, markov), 32)
        with self.assertRaisesRegex(ValueError, "requires a Markov head"):
            dspark_spec(markov=None)
        with self.assertRaisesRegex(ValueError, "requires a Markov head"):
            ConfidenceHeadSpec(with_markov=True).input_width(32, None)

    def test_invalid_nodes_fail_without_coercion(self):
        for kind in (None, "unknown", [], True):
            with self.assertRaisesRegex(ValueError, "markov.type"):
                MarkovHeadSpec(type=kind, rank=4)
        for rank in (0, -1, True, 2.5, "4"):
            with self.assertRaisesRegex(ValueError, "markov.rank"):
                MarkovHeadSpec(type="vanilla", rank=rank)
        for value in (1, "true", None):
            with self.assertRaisesRegex(ValueError, "boolean"):
                ConfidenceHeadSpec(with_markov=value)
        for name, value in (
            ("block", AnchoredBlockSpec(block_size=8)),
            ("target_features", {}),
            ("markov", {}),
            ("confidence", {}),
        ):
            with self.assertRaisesRegex(ValueError, name):
                dspark_spec(**{name: value})

    def test_heads_do_not_derive_structure_from_training_settings(self):
        node = draft_config(architecture="DSparkDraftModel", spec_config=dspark_spec())
        for field in ("confidence_head_alpha", "loss_weight", "projector_type"):
            payload = encode_config(node)
            payload["spec_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        for field in ("markov", "confidence"):
            payload = encode_config(node)
            payload["spec_config"].pop(field)
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.spec_config.markov.rank = 16

    def test_target_context_constraints_are_shared_without_algorithm_inheritance(self):
        spec = dspark_spec()
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            draft_config(
                architecture="DSparkDraftModel", spec_config=spec, draft_vocab_size=32
            )
        with self.assertRaisesRegex(ValueError, "mask_token_id"):
            draft_config(
                architecture="DSparkDraftModel",
                spec_config=dataclasses.replace(
                    spec,
                    target_features=dataclasses.replace(
                        spec.target_features, mask_token_id=64
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
