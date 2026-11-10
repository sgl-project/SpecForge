"""Sequential and model-owned execution share the same decoder path."""

import copy
import dataclasses
import unittest

import torch
from torch import nn

from specforge.config.draft.residual import StandardResidualSpec
from specforge.modeling.draft.backbone import DraftBackbone, LayerInputs
from specforge.modeling.draft.layers.inputs import LayerOutput
from specforge.modeling.draft.layers.residual import ResidualStreams, StandardResidual


class TestLayer(nn.Module):
    def __init__(self, width=4):
        super().__init__()
        self.proj = nn.Linear(width, width)
        self.seen_inputs = []

    def forward(self, hidden, inputs):
        self.seen_inputs.append(inputs)
        if inputs.target_context is not None:
            hidden = hidden + inputs.target_context
        if inputs.position_embeddings is not None:
            cos, sin = inputs.position_embeddings
            hidden = hidden + cos + sin
        return torch.tanh(self.proj(hidden))


class TestRotary(nn.Module):
    def __init__(self, scale):
        super().__init__()
        self.scale = scale
        self.calls = 0

    def forward(self, hidden, positions):
        self.calls += 1
        angle = positions.unsqueeze(-1).to(hidden.dtype) * self.scale
        return angle.cos(), angle.sin()


def make_backbone():
    return DraftBackbone(
        [TestLayer(), TestLayer()],
        final_norm=nn.LayerNorm(4),
        rotary=[TestRotary(0.5), TestRotary(0.25)],
        residual=StandardResidual(StandardResidualSpec()),
    )


def make_inputs():
    positions = torch.arange(3).unsqueeze(0)
    return [
        LayerInputs(
            target_context=torch.randn(1, 3, 4),
            position_ids=positions + index,
        )
        for index in range(2)
    ]


class BackboneTest(unittest.TestCase):
    def test_sequential_manual_and_direct_execution_are_identical(self):
        backbone = make_backbone()
        hidden, inputs = torch.randn(1, 3, 4), make_inputs()
        expected = hidden
        for layer, rotary, entry in zip(backbone.layers, backbone.rotary, inputs):
            prepared = dataclasses.replace(
                entry, position_embeddings=rotary(hidden, entry.position_ids)
            )
            expected = layer(expected, prepared)
        expected = backbone.final_norm(expected)
        state = backbone.begin(hidden, inputs)
        for _ in range(backbone.num_layers):
            state = backbone.run_next_layer(state)
        for actual in (backbone.finalize(state), backbone(hidden, inputs)):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for layer, entry in zip(backbone.layers, inputs):
            self.assertIs(layer.seen_inputs[-1].target_context, entry.target_context)
            self.assertIsNone(entry.position_embeddings)

    def test_typed_layer_results_share_execution_and_checkpointing(self):
        @dataclasses.dataclass(frozen=True, eq=False)
        class Output(LayerOutput):
            feature: torch.Tensor

        class Layer(TestLayer):
            def forward(self, hidden, inputs, *, debug=False):
                hidden = super().forward(hidden, inputs)
                return Output(hidden, hidden * (2 if debug else 1))

        backbone = make_backbone()
        backbone.layers = nn.ModuleList([Layer(), Layer()])
        hidden, inputs = torch.randn(1, 3, 4), make_inputs()
        expected = backbone(hidden, inputs)
        reference_grads = None
        for checkpointing in (False, True):
            backbone.zero_grad(set_to_none=True)
            backbone.gradient_checkpointing = checkpointing
            state = backbone.begin(hidden, inputs)
            for _ in backbone.layers:
                state = backbone.run_next_layer(state, debug=True)
            torch.testing.assert_close(
                backbone.finalize(state), expected, rtol=0, atol=0
            )
            torch.testing.assert_close(state.layer_output.feature, state.hidden * 2)
            state.layer_output.feature.sum().backward()
            grads = [
                p.grad.clone() for layer in backbone.layers for p in layer.parameters()
            ]
            if reference_grads is not None:
                torch.testing.assert_close(grads, reference_grads, rtol=0, atol=0)
            reference_grads = grads

    def test_model_owned_injection_keeps_intermediate_gradients(self):
        backbone = make_backbone()
        reference = copy.deepcopy(backbone)
        injection = nn.Linear(4, 4)
        expected_injection = copy.deepcopy(injection)
        hidden, inputs = torch.randn(1, 3, 4), make_inputs()
        state = backbone.begin(hidden, inputs)
        features = []
        for _ in range(backbone.num_layers):
            state = backbone.run_next_layer(state)
            pre_injection = state.hidden
            state = state.with_hidden(injection(state.hidden))
            features.append(state.hidden)
        actual = backbone.finalize(state)
        expected = hidden
        reference_features = []
        for layer, rotary, entry in zip(reference.layers, reference.rotary, inputs):
            expected = layer(
                expected,
                dataclasses.replace(
                    entry, position_embeddings=rotary(hidden, entry.position_ids)
                ),
            )
            expected_pre = expected
            expected = expected_injection(expected)
            reference_features.append(expected)
        expected = reference.final_norm(expected)
        torch.testing.assert_close(
            (actual, pre_injection, *features),
            (expected, expected_pre, *reference_features),
            rtol=0,
            atol=0,
        )
        (
            actual.square().sum() + pre_injection.sum() + sum(x.sum() for x in features)
        ).backward()
        (
            expected.square().sum()
            + expected_pre.sum()
            + sum(x.sum() for x in reference_features)
        ).backward()
        for left, right in zip(
            list(backbone.parameters()) + list(injection.parameters()),
            list(reference.parameters()) + list(expected_injection.parameters()),
        ):
            self.assertIsNotNone(left.grad)
            torch.testing.assert_close(left.grad, right.grad, rtol=0, atol=0)

    def test_state_order_ownership_and_layout_checks(self):
        backbone = make_backbone()
        state = backbone.begin(torch.randn(1, 3, 4), make_inputs())
        with self.assertRaisesRegex(ValueError, "every layer"):
            backbone.finalize(state)
        with self.assertRaisesRegex(ValueError, "another backbone"):
            make_backbone().run_next_layer(state)
        with self.assertRaisesRegex(ValueError, "another backbone"):
            make_backbone().finalize(state)
        with self.assertRaisesRegex(ValueError, "residual layout"):
            state.with_hidden(torch.randn(1, 3, 8))
        with self.assertRaises(dataclasses.FrozenInstanceError):
            state.next_layer = 1
        finished = backbone.run_next_layer(backbone.run_next_layer(state))
        self.assertEqual(state.next_layer, 0)
        with self.assertRaisesRegex(ValueError, "no next layer"):
            backbone.run_next_layer(finished)

    def test_input_validation_and_precomputed_rotary(self):
        backbone = make_backbone()
        hidden = torch.randn(1, 3, 4)
        with self.assertRaisesRegex(ValueError, "one entry"):
            backbone.begin(hidden, [LayerInputs()])
        with self.assertRaisesRegex(ValueError, "position_ids"):
            backbone.begin(hidden, [LayerInputs(), LayerInputs()])
        with self.assertRaises(TypeError):
            backbone.begin(hidden, [{}, {}])
        embeddings = (torch.ones(1, 3, 1), torch.zeros(1, 3, 1))
        inputs = [LayerInputs(position_embeddings=embeddings)] * 2
        state = backbone.begin(hidden, inputs)
        inputs.clear()
        self.assertEqual(len(state.layer_inputs), 2)
        self.assertEqual([r.calls for r in backbone.rotary], [0, 0])
        self.assertIs(state.layer_inputs[0].position_embeddings, embeddings)

    def test_residual_expansion_and_reduction_surround_the_layers(self):
        class TwoStreams(ResidualStreams):
            def build_connection(
                self, *, hidden_size, layer_index, sublayer, device=None, dtype=None
            ):
                raise NotImplementedError("this fixture tests only stream boundaries")

            def expand(self, hidden):
                return torch.stack((hidden, hidden * 2), dim=-2)

            def reduce(self, hidden):
                return hidden.sum(dim=-2)

        backbone = DraftBackbone(
            [TestLayer()],
            final_norm=nn.LayerNorm(4),
            rotary=[None],
            residual=TwoStreams(),
        )
        hidden = torch.randn(1, 3, 4)
        state = backbone.begin(hidden, [LayerInputs()])
        self.assertEqual(state.hidden.shape, (1, 3, 2, 4))
        expected = backbone.final_norm(
            backbone.layers[0](state.hidden, LayerInputs()).sum(-2)
        )
        torch.testing.assert_close(
            backbone(hidden, [LayerInputs()]), expected, rtol=0, atol=0
        )
        residual = StandardResidual(StandardResidualSpec())
        self.assertIs(residual.reduce(residual.expand(hidden)), hidden)


if __name__ == "__main__":
    unittest.main()
