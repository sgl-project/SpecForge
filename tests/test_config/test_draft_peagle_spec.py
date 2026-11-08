"""P-EAGLE reuses feature semantics without inheriting EAGLE3's single-layer contract."""

import dataclasses
import unittest

from specforge.config.draft.architecture import Eagle3Spec, PEagleSpec
from specforge.config.draft.architecture.eagle_fusion import AuxiliaryFeatureSpec
from specforge.config.draft.attention import MLASpec
from specforge.config.draft.codec import decode_config, encode_config
from tests.test_config.test_draft_eagle3_spec import eagle_config


def peagle_config(**overrides):
    base = eagle_config()
    return dataclasses.replace(
        base,
        **(
            dict(
                architecture="PEagleDraftModel",
                layer_types=("full_dense",) * 4,
                spec_config=PEagleSpec(
                    target_features=base.spec_config.target_features,
                    norm_before_residual=False,
                ),
            )
            | overrides
        ),
    )


class PEagleSpecTest(unittest.TestCase):
    def test_first_and_later_projection_widths_and_mask_round_trip(self):
        for depth in (1, 4):
            for normalized_residual in (False, True):
                base = peagle_config()
                spec = dataclasses.replace(
                    base.spec_config, norm_before_residual=normalized_residual
                )
                node = dataclasses.replace(
                    base, spec_config=spec, layer_types=("full_dense",) * depth
                )
                self.assertIsNone(spec.protocol)
                self.assertEqual(spec.mask_hidden_shape, (1, 1, 96))
                self.assertEqual(
                    [spec.attention_input_width(32, i) for i in range(depth)],
                    [64] + [32] * (depth - 1),
                )
                restored = decode_config(encode_config(node))
                self.assertEqual(restored, node)
                self.assertEqual(hash(restored), hash(node))
                self.assertNotIsInstance(spec, Eagle3Spec)

    def test_features_can_have_a_different_width_when_embeddings_are_unbound(self):
        spec = PEagleSpec(
            target_features=AuxiliaryFeatureSpec(
                target_layer_ids=(1, 2, 3), hidden_size=48
            ),
            norm_before_residual=True,
        )
        node = peagle_config(spec_config=spec, target_bindings=())
        self.assertEqual(spec.mask_hidden_shape, (1, 1, 144))
        self.assertEqual(spec.attention_input_width(node.hidden_size, 0), 64)
        self.assertEqual(
            decode_config(
                encode_config(node),
                target_config={"hidden_size": 48, "num_hidden_layers": 8},
            ),
            node,
        )

    def test_first_attention_must_declare_fusion_width_support(self):
        mla = MLASpec(
            rope_parameters=None,
            num_attention_heads=4,
            q_lora_rank=None,
            kv_lora_rank=8,
            qk_nope_head_dim=4,
            qk_rope_head_dim=4,
            v_head_dim=8,
            attention_bias=False,
            is_causal=True,
            sliding_window=None,
            attention_dropout=0.0,
            rope_interleave=True,
        )
        with self.assertRaisesRegex(ValueError, "variable_input_width"):
            peagle_config(attention_configs={"full": mla})
        for index in (-1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "layer_index"):
                peagle_config().spec_config.attention_input_width(32, index)

    def test_required_fields_and_runtime_settings_are_strict(self):
        node = peagle_config()
        for field in ("target_features", "norm_before_residual"):
            payload = encode_config(node)
            payload["spec_config"].pop(field)
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        for field in (
            "block_size",
            "ttt_length",
            "norm_output",
            "normalize_features",
            "d2t",
        ):
            payload = encode_config(node)
            payload["spec_config"][field] = 1
            with self.assertRaisesRegex(ValueError, field):
                decode_config(payload)
        for value in (None, 1, "false"):
            with self.assertRaisesRegex(ValueError, "norm_before_residual"):
                dataclasses.replace(node.spec_config, norm_before_residual=value)
        with self.assertRaisesRegex(ValueError, "target_features"):
            dataclasses.replace(node.spec_config, target_features={})


if __name__ == "__main__":
    unittest.main()
