"""Migration rules match the existing HF loader's effective configuration."""

import copy
import json
import unittest

from specforge.config.draft.codec import decode_legacy_config
from specforge.modeling.draft.config import AutoDraftConfig
from specforge.modeling.draft.dflash import (
    build_target_layer_ids,
    resolve_dflash_attention_layout,
)
from specforge.modeling.draft.dflash_mask import resolve_dflash_is_causal
from tests.test_config.test_draft_legacy_dflash import (
    CONFIGS,
    legacy_payload,
    target_for,
)


class LegacyHFParityTest(unittest.TestCase):
    def assert_effective_config_matches(self, payload):
        target = target_for(payload)
        old = AutoDraftConfig.from_dict(payload)
        new = decode_legacy_config(payload, target_config=target)
        self.assertEqual(new.hidden_size, old.hidden_size)
        self.assertEqual(new.vocab_size, old.vocab_size)
        self.assertEqual(new.num_hidden_layers, old.num_hidden_layers)
        self.assertEqual(new.max_position_embeddings, old.max_position_embeddings)
        self.assertEqual(new.norm_config.eps, old.rms_norm_eps)
        self.assertEqual(new.initializer_range, old.initializer_range)
        self.assertEqual(
            new.rope_parameters.rope_theta, old.rope_parameters["rope_theta"]
        )
        self.assertEqual(
            new.rope_parameters.rope_type, old.rope_parameters["rope_type"]
        )
        layers, window = resolve_dflash_attention_layout(old)
        self.assertEqual(new.layer_types, layers)
        for kind in layers:
            attention = new.attention_configs[kind]
            self.assertEqual(attention.head_dim, old.head_dim)
            self.assertEqual(attention.num_attention_heads, old.num_attention_heads)
            self.assertEqual(attention.num_key_value_heads, old.num_key_value_heads)
            self.assertEqual(attention.attention_bias, old.attention_bias)
            self.assertEqual(attention.attention_dropout, old.attention_dropout)
            self.assertEqual(
                attention.is_causal,
                resolve_dflash_is_causal(getattr(old, "is_causal", None), kind),
            )
            self.assertEqual(
                attention.sliding_window,
                window if kind == "sliding_attention" else None,
            )
        self.assertEqual(new, decode_legacy_config(old.to_dict(), target_config=target))

    def test_bundled_configs_and_hf_resaves_have_the_same_effective_structure(self):
        for path in sorted(CONFIGS.glob("*dflash*.json")):
            with self.subTest(path=path.name):
                self.assert_effective_config_matches(json.loads(path.read_text()))

    def test_omitted_qwen_defaults_and_derived_layout_match(self):
        original = legacy_payload()
        original["num_hidden_layers"] = 3
        original.pop("layer_types")
        for name in (
            "num_key_value_heads",
            "head_dim",
            "attention_dropout",
            "attention_bias",
            "rms_norm_eps",
            "initializer_range",
            "max_position_embeddings",
            "rope_theta",
        ):
            original.pop(name)
        for sliding in (False, True):
            payload = copy.deepcopy(original)
            payload.update(
                use_sliding_window=sliding, sliding_window=128, max_window_layers=1
            )
            for causal in (None, True, False):
                self.assert_effective_config_matches(payload | {"is_causal": causal})

    def test_feature_derivation_matches_existing_implementation(self):
        for count in (1, 2, 3, 5, 8):
            payload = legacy_payload()
            payload["num_hidden_layers"] = count
            payload.pop("layer_types")
            payload["dflash_config"].pop("target_layer_ids")
            new = decode_legacy_config(payload, target_config=target_for(payload))
            self.assertEqual(
                new.spec_config.target_features.target_layer_ids,
                tuple(build_target_layer_ids(payload["num_target_layers"], count)),
            )


if __name__ == "__main__":
    unittest.main()
