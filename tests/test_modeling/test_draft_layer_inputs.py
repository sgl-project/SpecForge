"""The backbone preserves explicit layout/context inputs when preparing RoPE."""

import dataclasses
import unittest

import torch
from torch import nn

from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec
from specforge.modeling.draft.backbone import DraftBackbone
from specforge.modeling.draft.backbone import LayerInputs as BackboneLayerInputs
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.residual import StandardResidual
from specforge.modeling.draft.layers.rope import build_rotary
from tests.test_modeling.test_draft_backbone import TestLayer


def make_layout():
    positions = torch.arange(3).unsqueeze(0)
    return AttentionLayout(
        query_positions=positions,
        key_positions=positions,
        visibility=torch.ones(1, 1, 3, 3, dtype=torch.bool),
    )


class LayerInputsTest(unittest.TestCase):
    def test_layout_survives_backbone_preparation_and_layer_execution(self):
        self.assertIs(BackboneLayerInputs, LayerInputs)
        layout = make_layout()
        context = torch.randn(1, 3, 4, requires_grad=True)
        entry = LayerInputs(
            attention_layout=layout,
            target_context=context,
            position_ids=layout.key_positions,
        )
        layer = TestLayer(width=4)
        backbone = DraftBackbone(
            [layer],
            final_norm=nn.Identity(),
            rotary=[
                build_rotary(
                    DefaultRoPESpec(
                        interleaved=False,
                        rope_theta=10000.0,
                        partial_rotary_factor=1.0,
                    ),
                    head_dim=4,
                )
            ],
            residual=StandardResidual(StandardResidualSpec()),
        )
        hidden = torch.randn(1, 3, 4, requires_grad=True)
        output = backbone(hidden, [entry])
        prepared = layer.seen_inputs[-1]
        self.assertIs(prepared.attention_layout, layout)
        self.assertIs(prepared.target_context, context)
        self.assertIs(prepared.position_ids, layout.key_positions)
        self.assertIsNotNone(prepared.position_embeddings)
        self.assertIsNone(entry.position_embeddings)
        output.sum().backward()
        self.assertIsNotNone(hidden.grad)
        self.assertIsNotNone(context.grad)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            entry.attention_layout = None

    def test_layout_cannot_silently_override_a_prepared_mask(self):
        layout = make_layout()
        with self.assertRaisesRegex(ValueError, "not both"):
            LayerInputs(attention_layout=layout, attention_mask=layout.visibility)
        with self.assertRaises(TypeError):
            LayerInputs(attention_layout=layout.visibility)
        legacy = LayerInputs(attention_mask=layout.visibility)
        self.assertIs(legacy.attention_mask, layout.visibility)
        self.assertIsNone(legacy.attention_layout)


if __name__ == "__main__":
    unittest.main()
