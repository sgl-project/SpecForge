"""Each contract DraftConfig enforces fails on the input it exists to reject."""

import dataclasses
import unittest
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.architecture import DFlashSpec
from specforge.config.draft.attention import AttentionSpec
from specforge.config.draft.codec import decode_config, encode_config, parse_draft
from specforge.config.draft.conv import ConvSpec, GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import NoLogitTransformSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.registry import register_component_schema, register_spec
from specforge.config.draft.rope import DefaultRoPESpec
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_canonical_fixtures import EXPECTED, dflash2_config
from tests.test_config.test_draft_dspark_spec import dspark_spec
from tests.test_config.test_draft_dspine_spec import dspine_config

TARGET = {"hidden_size": 32, "vocab_size": 64, "num_hidden_layers": 8}


def with_features(config, **changes):
    features = dataclasses.replace(config.spec_config.target_features, **changes)
    spec = dataclasses.replace(config.spec_config, target_features=features)
    return dataclasses.replace(config, spec_config=spec)


def with_conv(config, conv):
    return dataclasses.replace(
        config,
        conv_configs={"conv": conv},
        layer_configs={
            "full_dense": LayerSpec(attention="full", mlp="dense", sublayer_conv="conv")
        },
    )


class ContractsTest(unittest.TestCase):
    def test_every_architecture_checks_its_target_context_and_layers(self):
        for name, build in EXPECTED.items():
            config = build()
            with self.subTest(architecture=name):
                with self.assertRaisesRegex(ValueError, "mask_token_id"):
                    with_features(config, mask_token_id=config.vocab_size)
                with self.assertRaisesRegex(
                    ValueError, "draft_vocab_size == vocab_size"
                ):
                    dataclasses.replace(
                        config,
                        draft_vocab_size=config.vocab_size - 1,
                        target_bindings=("embed_tokens",),
                        target_logit_transform=None,
                    )
                with self.assertRaisesRegex(ValueError, "target_layer_ids"):
                    decode_config(
                        encode_config(config),
                        target_config={**TARGET, "num_hidden_layers": 3},
                    )

    def test_unknown_binding_names_are_rejected_on_their_own(self):
        with self.assertRaisesRegex(ValueError, "only embed_tokens and lm_head"):
            draft_config(
                target_bindings=("embed_tokens", "lm_heads"),
                target_logit_transform=None,
            )

    def test_an_embedding_binding_alone_checks_target_dimensions(self):
        node = draft_config(
            target_bindings=("embed_tokens",), target_logit_transform=None
        )
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            decode_config(
                encode_config(node), target_config={**TARGET, "hidden_size": 64}
            )

    def test_components_declare_no_capability_by_default(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class PlainAttention(AttentionSpec):
            type: ClassVar[str] = "test_plain_attention"
            num_heads: int

            @property
            def rotary_input_dim(self):
                return None

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class PlainConv(ConvSpec):
            type: ClassVar[str] = "test_plain_conv"
            width: int

        with patch.dict(registry._SCHEMAS):
            register_component_schema("attention", PlainAttention)
            register_component_schema("conv", PlainConv)
            with self.assertRaisesRegex(ValueError, "requires target_context_kv"):
                draft_config(
                    attention_configs={
                        "full": PlainAttention(rope_parameters=None, num_heads=4)
                    }
                )
            with self.assertRaisesRegex(ValueError, "block_sublayer_pair"):
                with_conv(dflash2_config(), PlainConv(width=4))

    def test_dspine_rejects_an_attention_without_a_causal_field(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class ContextAttention(AttentionSpec):
            type: ClassVar[str] = "test_context_attention"

            def capabilities(self) -> frozenset[str]:
                return frozenset({"target_context_kv"})

            num_heads: int

            @property
            def rotary_input_dim(self):
                return None

        with patch.dict(registry._SCHEMAS):
            register_component_schema("attention", ContextAttention)
            with self.assertRaisesRegex(ValueError, "DSpine requires causal attention"):
                dataclasses.replace(
                    dspine_config(),
                    attention_configs={
                        "full": ContextAttention(rope_parameters=None, num_heads=4)
                    },
                )

    def test_conv_kernel_is_bounded_by_the_protocol_slots(self):
        with_conv(dflash2_config(), GroupedCausalConvSpec(kernel_size=8, group_size=8))
        with self.assertRaisesRegex(ValueError, "kernel_size exceeds"):
            with_conv(
                dflash2_config(), GroupedCausalConvSpec(kernel_size=9, group_size=8)
            )
        dspark = draft_config(
            architecture="DSparkDraftModel", spec_config=dspark_spec()
        )
        with_conv(dspark, GroupedCausalConvSpec(kernel_size=7, group_size=8))
        with self.assertRaisesRegex(ValueError, "kernel_size exceeds"):
            with_conv(dspark, GroupedCausalConvSpec(kernel_size=8, group_size=8))

    def test_model_level_nodes_must_be_the_registered_class(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class ShadowNorm(RMSNormSpec):
            pass

        @dataclasses.dataclass(frozen=True, kw_only=True)
        class ShadowLogits(NoLogitTransformSpec):
            pass

        with self.assertRaisesRegex(ValueError, "norm_config must use its registered"):
            draft_config(norm_config=ShadowNorm(eps=1e-6, weight_offset=0))
        with self.assertRaisesRegex(ValueError, "target_logit_transform must use"):
            draft_config(target_logit_transform=ShadowLogits())

    def test_a_node_holding_a_mutable_value_is_rejected(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class BandRoPE(DefaultRoPESpec):
            rope_type: ClassVar[str] = "test_band"
            bands: list

        with patch.dict(registry._SCHEMAS):
            register_component_schema("rope", BandRoPE)
            rope = BandRoPE(rope_theta=10000, partial_rotary_factor=1.0, bands=[1.0])
            with self.assertRaisesRegex(ValueError, "immutable and hashable"):
                draft_config(rope_parameters=rope)

    def test_architecture_identity_is_exact(self):
        @dataclasses.dataclass(frozen=True, kw_only=True)
        class ExtendedSpec(DFlashSpec):
            extra_rank: int

        spec = draft_config().spec_config
        with self.assertRaisesRegex(ValueError, "must be DFlashSpec"):
            draft_config(
                spec_config=ExtendedSpec(
                    block=spec.block, target_features=spec.target_features, extra_rank=1
                )
            )
        with patch.dict(registry._SPECS):
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_spec("DFlashDraftModel", DFlashSpec, lambda _: None)

    def test_parse_draft_needs_an_explicit_target_decision(self):
        with self.assertRaises(TypeError):
            parse_draft(encode_config(draft_config()))


if __name__ == "__main__":
    unittest.main()
