"""Legacy DSpark head decisions match the currently constructed model."""

import unittest

from specforge.config.draft.codec import decode_legacy_config
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from tests.test_config.test_draft_legacy_dflash import target_for
from tests.test_config.test_draft_legacy_dspark import dspark_payload


class DSparkCodecParityTest(unittest.TestCase):
    def test_migrated_head_shapes_and_presence_match_existing_models(self):
        for kind, rank, enabled, with_markov in (
            ("vanilla", 8, None, True),
            ("gated", 8, True, True),
            ("rnn", 8, True, False),
            ("vanilla", 0, True, False),
            ("vanilla", 8, False, True),
            ("vanilla", 0, None, False),
        ):
            with self.subTest(
                kind=kind, rank=rank, enabled=enabled, with_markov=with_markov
            ):
                payload = dspark_payload()
                payload.update(
                    hidden_size=32,
                    intermediate_size=64,
                    num_hidden_layers=1,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    vocab_size=64,
                    num_target_layers=8,
                    layer_types=["full_attention"],
                )
                method = payload["dflash_config"]
                method.update(
                    mask_token_id=63,
                    target_layer_ids=[1, 3],
                    markov_rank=rank,
                    markov_head_type=kind,
                    confidence_head_with_markov=with_markov,
                    confidence_head_alpha=1 if rank else 0,
                )
                if enabled is None:
                    method.pop("enable_confidence_head")
                else:
                    method["enable_confidence_head"] = enabled
                node = decode_legacy_config(payload, target_config=target_for(payload))
                model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
                spec = node.spec_config
                self.assertEqual(spec.markov is None, model.markov_head is None)
                self.assertEqual(spec.confidence is None, model.confidence_head is None)
                if spec.markov is not None:
                    self.assertEqual(spec.markov.rank, model.markov_head.markov_rank)
                    self.assertEqual(
                        spec.markov.type, model.markov_head.markov_head_type
                    )
                if spec.confidence is not None:
                    self.assertEqual(
                        spec.confidence.input_width(node.hidden_size, spec.markov),
                        model.confidence_head.proj.in_features,
                    )
                self.assertEqual(
                    decode_legacy_config(
                        model.config.to_dict(), target_config=target_for(payload)
                    ),
                    node,
                )


if __name__ == "__main__":
    unittest.main()
