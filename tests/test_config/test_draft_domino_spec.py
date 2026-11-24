"""Domino exposes training alignment and serving candidates separately."""

import dataclasses
import json
import unittest

from specforge.config.draft.architecture import DominoSpec
from specforge.config.draft.architecture.block import AnchoredBlockSpec
from specforge.config.draft.architecture.domino import DominoBlockSpec, GRUHeadSpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.draft_fixtures import draft_config


def domino_spec(**overrides):
    return DominoSpec(
        **(
            dict(
                block=DominoBlockSpec(block_size=4, shift_label=False),
                target_features=draft_config().spec_config.target_features,
                gru=GRUHeadSpec(hidden_size=16, projection_size=8),
                pure_draft_prefix_len=1,
            )
            | overrides
        )
    )


class DominoSpecTest(unittest.TestCase):
    def test_shifted_and_unshifted_protocols_round_trip(self):
        for shifted in (False, True):
            block = DominoBlockSpec(block_size=4, shift_label=shifted)
            spec = domino_spec(block=block)
            node = draft_config(architecture="DominoDraftModel", spec_config=spec)
            self.assertEqual(block.num_model_slots, 4)
            self.assertEqual(block.num_draft_tokens, 3)
            self.assertEqual(block.verify_window, 4)
            self.assertEqual(block.num_external_candidates, 0)
            self.assertEqual(
                block.prediction_positions, (0, 1, 2, 3) if shifted else (1, 2, 3)
            )
            self.assertEqual(
                block.candidate_positions, (0, 1, 2) if shifted else (1, 2, 3)
            )
            self.assertEqual(spec.suffix_start, 1 if shifted else 2)
            self.assertEqual(spec.gru.input_width(node.hidden_size), 48)
            restored = decode_config(json.loads(json.dumps(encode_config(node))))
            self.assertEqual(restored, node)
            self.assertEqual(hash(restored), hash(node))

    def test_prefix_boundaries_leave_a_nonempty_correction(self):
        for shifted in (False, True):
            block = DominoBlockSpec(block_size=4, shift_label=shifted)
            largest_prefix = 3 if shifted else 2
            spec = domino_spec(block=block, pure_draft_prefix_len=largest_prefix)
            self.assertEqual(spec.suffix_start, 3)
            with self.assertRaisesRegex(ValueError, "nonempty"):
                domino_spec(block=block, pure_draft_prefix_len=largest_prefix + 1)
        for value in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "pure_draft_prefix_len"):
                domino_spec(pure_draft_prefix_len=value)

    def test_invalid_nodes_and_dimensions_fail_without_coercion(self):
        for size in (0, 1, True, 2.5):
            with self.assertRaises(ValueError):
                DominoBlockSpec(block_size=size, shift_label=False)
        for shifted in (None, 1, "false"):
            with self.assertRaisesRegex(ValueError, "boolean"):
                DominoBlockSpec(block_size=4, shift_label=shifted)
        for field in ("hidden_size", "projection_size"):
            for value in (0, -1, True, 2.5):
                with self.assertRaisesRegex(ValueError, field):
                    GRUHeadSpec(
                        **({"hidden_size": 8, "projection_size": 4} | {field: value})
                    )
        for field, value in (
            ("block", AnchoredBlockSpec(block_size=4)),
            ("target_features", {}),
            ("gru", None),
        ):
            with self.assertRaisesRegex(ValueError, field):
                domino_spec(**{field: value})

    def test_structural_fields_are_required_and_training_fields_rejected(self):
        node = draft_config(architecture="DominoDraftModel", spec_config=domino_spec())
        for field in ("block", "target_features", "gru", "pure_draft_prefix_len"):
            payload = encode_config(node)
            payload["spec_config"].pop(field)
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        payload = encode_config(node)
        payload["spec_config"]["lambda_base"] = 0.1
        with self.assertRaisesRegex(ValueError, "lambda_base"):
            decode_config(payload)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            node.spec_config.gru.hidden_size = 32
        with self.assertRaisesRegex(ValueError, "draft_vocab_size"):
            dataclasses.replace(node, draft_vocab_size=32)


if __name__ == "__main__":
    unittest.main()
