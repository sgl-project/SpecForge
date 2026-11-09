"""Layer validation delegates convolution rules with explicit model context."""

import unittest
from dataclasses import dataclass
from typing import ClassVar
from unittest.mock import patch

from specforge.config.draft import registry
from specforge.config.draft.attention import GQASpec
from specforge.config.draft.conv import ConvSpec, GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.config.draft.logits import NoLogitTransformSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.registry import register_component_schema
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec
from specforge.config.draft.validate import validate_layers
from tests.test_config.test_draft_layer_definitions import definitions


@dataclass(frozen=True, kw_only=True)
class CustomConv(ConvSpec):
    type: ClassVar[str] = "test_context_conv"
    width_multiple: int


def check_layers(conv, *, slots=8):
    base = definitions()
    validate_layers(
        hidden_size=32,
        attention_configs={"full": base["attention_configs"]["full"]},
        mlp_configs=base["mlp_configs"],
        conv_configs={"conv": conv},
        layer_configs={
            "with_conv": LayerSpec(attention="full", mlp="dense", sublayer_conv="conv")
        },
        layer_types=("with_conv",),
        num_model_slots=slots,
    )


class ConvValidationTest(unittest.TestCase):
    def test_grouped_rules_keep_layer_error_paths(self):
        check_layers(GroupedCausalConvSpec(kernel_size=8, group_size=8))
        for kernel, group, slots, message in (
            (3, 3, 8, "group_size must divide"),
            (3, 8, None, "requires a block protocol"),
            (9, 8, 8, "kernel_size exceeds"),
        ):
            with (
                self.subTest(message=message),
                self.assertRaisesRegex(
                    ValueError, "layer_configs.with_conv.sublayer_conv: .*" + message
                ),
            ):
                check_layers(
                    GroupedCausalConvSpec(kernel_size=kernel, group_size=group),
                    slots=slots,
                )

    def test_custom_conv_registers_its_own_context_rules(self):
        seen = []

        def validate(node, *, hidden_size, num_model_slots):
            seen.append((hidden_size, num_model_slots))
            if hidden_size % node.width_multiple:
                raise ValueError("custom width mismatch")

        with patch.dict(registry._SCHEMAS), patch.dict(registry._VALIDATORS):
            with self.assertRaisesRegex(TypeError, "callable"):
                register_component_schema("conv", CustomConv, validate=1)
            register_component_schema("conv", CustomConv, validate=validate)
            register_component_schema("conv", CustomConv, validate=validate)
            register_component_schema("conv", CustomConv)
            with self.assertRaisesRegex(ValueError, "already registered"):
                register_component_schema("conv", CustomConv, validate=lambda *a: None)
            check_layers(CustomConv(width_multiple=8), slots=None)
            self.assertEqual(seen, [(32, None)])
            with self.assertRaisesRegex(
                ValueError, "sublayer_conv: custom width mismatch"
            ):
                check_layers(CustomConv(width_multiple=3))

    def test_attention_validator_receives_layer_context(self):
        seen = []
        with patch.dict(registry._VALIDATORS):
            register_component_schema(
                "attention",
                GQASpec,
                validate=lambda node, **context: seen.append(context),
            )
            check_layers(GroupedCausalConvSpec(kernel_size=8, group_size=8))
        self.assertEqual(seen, [{"hidden_size": 32, "num_model_slots": 8}])

    def test_model_level_kinds_reject_validators_that_would_never_run(self):
        for kind, schema in (
            ("norm", RMSNormSpec),
            ("residual", StandardResidualSpec),
            ("rope", DefaultRoPESpec),
            ("logits", NoLogitTransformSpec),
        ):
            with (
                self.subTest(kind=kind),
                patch.dict(registry._VALIDATORS),
                self.assertRaisesRegex(ValueError, "no layer context"),
            ):
                register_component_schema(kind, schema, validate=lambda node, **c: None)


if __name__ == "__main__":
    unittest.main()
