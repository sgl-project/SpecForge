"""Independent serving constraints and field expectations."""

import unittest
from dataclasses import replace

from specforge.checkpoint.formats.sglang import convert_config
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.config.draft.rope import MultiAxisRoPESpec
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dflash2 import dflash2_config
from tests.test_modeling.test_canonical_domino import model_config as domino_config
from tests.test_modeling.test_canonical_dspark import model_config as dspark_config
from tests.test_modeling.test_canonical_eagle3 import model_config as eagle_config
from tests.test_modeling.test_draft_mla import mla_node


def multi_axis_config(config, *, interleaved=False):
    rope = config.rope_parameters
    return replace(
        config,
        rope_parameters=MultiAxisRoPESpec(
            interleaved=False,
            rope_theta=rope.rope_theta,
            partial_rotary_factor=rope.partial_rotary_factor,
            mrope_section=(2, 1, 1),
            mrope_interleaved=interleaved,
        ),
    )


class ServingBoundariesTest(unittest.TestCase):
    def test_sliding_window_and_token_contract(self):
        base = dflash2_config(bound=True)
        for window in (1, 7, 8, 2048):
            config = replace(
                base,
                attention_configs={
                    "full": replace(
                        base.attention_configs["full"], sliding_window=window
                    )
                },
            )
            if window < 8:
                with self.assertRaisesRegex(ValueError, "sliding window"):
                    convert_config(config)
                continue
            fields = convert_config(config)
            self.assertEqual(fields["sliding_window"], window)
            self.assertFalse(fields["is_causal"])
            self.assertEqual(fields["layer_types"], ["sliding_attention"] * 2)
            self.assertEqual(fields["dflash_config"]["target_layer_ids"], [1, 3])
            for node in (fields, fields["dflash_config"]):
                self.assertEqual(node["mask_token"], "<|MASK|>")
                self.assertEqual(node["mask_token_id"], 63)

    def test_mup_and_offset_norm_limits(self):
        for config in (dflash2_config(bound=True), domino_config(bound=True)):
            with self.assertRaisesRegex(ValueError, "muP multiplier"):
                convert_config(
                    replace(
                        config,
                        target_logit_transform=MuPLogitTransformSpec(multiplier=24),
                    )
                )
        for config in (draft_config(), dspark_config(bound=True)):
            convert_config(
                replace(
                    config, target_logit_transform=MuPLogitTransformSpec(multiplier=24)
                )
            )
        config = dspark_config(bound=True)
        with self.assertRaisesRegex(ValueError, "offset"):
            convert_config(
                replace(
                    config, norm_config=replace(config.norm_config, weight_offset=1)
                )
            )

    def test_mixed_conv_and_feature_width_are_rejected(self):
        config = dflash2_config(bound=True)
        mixed = replace(
            config,
            layer_configs={
                **config.layer_configs,
                "plain": LayerSpec(attention="full", mlp="dense"),
            },
            layer_types=("full_dense", "plain"),
        )
        with self.assertRaisesRegex(ValueError, "convolutions differ"):
            convert_config(mixed)
        spec = config.spec_config
        with self.assertRaisesRegex(ValueError, "feature width"):
            convert_config(
                replace(
                    config,
                    target_bindings=(),
                    target_logit_transform=None,
                    spec_config=replace(
                        spec,
                        target_features=replace(spec.target_features, hidden_size=64),
                    ),
                )
            )

    def test_rotary_pairing_comes_from_rope_node(self):
        config = draft_config()
        rope = replace(config.rope_parameters, interleaved=True)
        fields = convert_config(replace(config, rope_parameters=rope))
        self.assertFalse(fields["rope_is_neox_style"])
        config = replace(
            config,
            rope_parameters=rope,
            attention_configs={
                "full": mla_node(
                    attention_bias=False, attention_dropout=0, sliding_window=None
                )
            },
        )
        fields = convert_config(config)
        self.assertTrue(fields["rope_interleave"])
        self.assertEqual(
            fields["qk_rope_head_dim"],
            config.attention_configs["full"].qk_rope_head_dim,
        )

    def test_multi_axis_and_static_dspark(self):
        eagle = multi_axis_config(eagle_config(bound=True))
        native = multi_axis_config(mtp_config(), interleaved=True)
        native = replace(
            native, norm_config=replace(native.norm_config, weight_offset=1)
        )
        for config in (eagle, native):
            fields = convert_config(config)
            self.assertEqual(fields["rope_parameters"]["rope_type"], "default")
            self.assertEqual(fields["rope_parameters"]["mrope_section"], [2, 1, 1])
            self.assertEqual(
                fields["rope_parameters"]["mrope_interleaved"],
                config.rope_parameters.mrope_interleaved,
            )
        fields = convert_config(dspark_config(bound=True, confidence=False))
        self.assertFalse(fields["enable_confidence_head"])
        self.assertFalse(fields["confidence_head_with_markov"])
        with self.assertRaisesRegex(ValueError, "one layer"):
            convert_config(replace(native, layer_types=native.layer_types * 2))
        attention = next(iter(eagle.attention_configs.values()))
        with self.assertRaisesRegex(ValueError, "divisible"):
            convert_config(
                replace(
                    eagle,
                    attention_configs={
                        name: replace(
                            attention, num_attention_heads=3, num_key_value_heads=1
                        )
                        for name in eagle.attention_configs
                    },
                )
            )
