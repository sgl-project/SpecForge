"""Migration matches the existing DSpine model and its saved options."""

import unittest
from dataclasses import asdict

from specforge.config.draft.codec import decode_legacy_config
from specforge.config.draft.codec.legacy.dspine import extract_dspine_training_settings
from specforge.modeling.auto import AutoDraftModel
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.dspine import DSpineConfig
from tests.test_config.test_draft_legacy_dflash import target_for
from tests.test_config.test_draft_legacy_dspine import dspine_payload


class DSpineCodecParityTest(unittest.TestCase):
    def test_training_settings_match_existing_defaults_and_overrides(self):
        for options in (
            {},
            {
                "top_k": 4,
                "ce_weight": 0.7,
                "alignment_weight": 2,
                "replacement_layers": 0,
                "alignment_warmup_steps": 0,
            },
        ):
            payload = dspine_payload() | {"dspine_config": options}
            expected = asdict(DSpineConfig(**options))
            expected.pop("transfer_rank")
            expected.pop("message_size")
            self.assertEqual(extract_dspine_training_settings(payload), expected)

    def test_injection_shapes_and_saved_config_match_existing_model(self):
        payload = dspine_payload() | {
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "vocab_size": 64,
            "num_target_layers": 8,
            "layer_types": ["full_attention"] * 2,
            "bos_token_id": None,
            "eos_token_id": None,
        }
        payload["dflash_config"].update(mask_token_id=63, target_layer_ids=[1, 3])
        node = decode_legacy_config(payload, target_config=target_for(payload))
        model = AutoDraftModel.from_config(AutoDraftConfig.from_dict(payload))
        injection = node.spec_config.injection
        self.assertTrue(model.config.is_causal)
        self.assertEqual(
            tuple(model.transfer_codes.shape),
            (node.vocab_size, injection.transfer_rank),
        )
        self.assertEqual(len(model.injection.readouts), node.num_hidden_layers)
        self.assertEqual(
            model.injection.readouts[0].out_features, injection.transfer_rank
        )
        self.assertEqual(model.injection.writes[0].in_features, injection.message_size)
        saved = model.config.to_dict()
        self.assertEqual(
            decode_legacy_config(saved, target_config=target_for(payload)), node
        )
        self.assertEqual(
            extract_dspine_training_settings(saved),
            extract_dspine_training_settings(payload),
        )


if __name__ == "__main__":
    unittest.main()
