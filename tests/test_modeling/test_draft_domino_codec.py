"""Legacy Domino construction and sampling agree with the migrated protocol."""

import unittest
from unittest.mock import patch

import torch

from specforge.config.draft.codec import decode_legacy_config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_config.test_draft_legacy_dflash import target_for
from tests.test_config.test_draft_legacy_domino import domino_payload
from tests.test_modeling.test_domino_model import _TinyTarget


class DominoCodecParityTest(unittest.TestCase):
    def test_head_shapes_alignment_and_saved_config_match_existing_model(self):
        for shifted in (False, True):
            for prefix in (0, 1):
                with self.subTest(shifted=shifted, prefix=prefix):
                    payload = domino_payload() | {
                        "hidden_size": 32,
                        "intermediate_size": 64,
                        "num_hidden_layers": 1,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 8,
                        "vocab_size": 64,
                        "num_target_layers": 8,
                        "block_size": 4,
                        "layer_types": ["full_attention"],
                        "emb_dim": 8,
                        "bos_token_id": None,
                        "eos_token_id": None,
                    }
                    payload["dflash_config"].update(
                        mask_token_id=63,
                        target_layer_ids=[1, 3],
                        emb_dim=8,
                        gru_hidden_dim=16,
                        pure_draft_prefix_len=prefix,
                        shift_label=shifted,
                    )
                    node = decode_legacy_config(
                        payload, target_config=target_for(payload)
                    )
                    model = AutoDraftModel.from_config(
                        AutoDraftConfig.from_dict(payload)
                    )
                    spec = node.spec_config
                    self.assertEqual(model.prefix_gru.hidden_size, spec.gru.hidden_size)
                    self.assertEqual(
                        model.embed_proj[0].in_features,
                        spec.gru.input_width(node.hidden_size),
                    )
                    self.assertEqual(
                        model.embed_proj[0].out_features, spec.gru.projection_size
                    )
                    self.assertEqual(model.suffix_start, spec.suffix_start)
                    hidden = torch.randn(1, 1, 4, 32)
                    correction = model.compute_correction_logits(
                        prev_token_embeddings=hidden, hidden_states=hidden
                    )
                    self.assertEqual(
                        tuple(correction.shape), (1, 1, 4 - spec.suffix_start, 64)
                    )
                    self.assertEqual(
                        decode_legacy_config(
                            model.config.to_dict(), target_config=target_for(payload)
                        ),
                        node,
                    )

                    # Give each output slot a distinct argmax token, then ask the
                    # existing sampler which slots it consumes for this alignment.
                    logits = torch.zeros(1, 1, 4, 64)
                    for slot in range(4):
                        logits[0, 0, slot, slot + 1] = 10
                    with patch.object(model, "apply_logits_head", return_value=logits):
                        candidates = model._sample_draft_tokens(
                            _TinyTarget(64, 32),
                            hidden[:, 0],
                            torch.zeros(1, 4, dtype=torch.long),
                        )
                    self.assertEqual(
                        candidates.tolist(),
                        [[slot + 1 for slot in spec.protocol.candidate_positions]],
                    )


if __name__ == "__main__":
    unittest.main()
