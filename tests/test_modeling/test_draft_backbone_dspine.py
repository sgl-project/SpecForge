"""The existing DSpine computation can own the shared backbone's layer loop."""

import copy
import unittest

import torch
from torch import nn

from specforge.config.draft.residual import StandardResidualSpec
from specforge.modeling.draft.backbone import DraftBackbone, LayerInputs
from specforge.modeling.draft.dspine import DSpineOutput
from specforge.modeling.draft.layers.residual import StandardResidual
from tests.test_modeling.test_dspine import forward_inputs, initialized_model


class LegacyLayerAdapter(nn.Module):
    """Test-only signature adapter; production models are not switched yet."""

    def __init__(self, layer, mask):
        super().__init__()
        self.layer = layer
        self.mask = mask

    def forward(self, hidden, inputs):
        return self.layer(
            hidden_states=hidden,
            target_hidden=inputs.target_context,
            attention_mask=self.mask,
            position_ids=inputs.position_ids,
            position_embeddings=inputs.position_embeddings,
        )


def model_owned_forward(model, inputs, checkpointing):
    backbone = DraftBackbone(
        [LegacyLayerAdapter(layer, inputs["attention_mask"]) for layer in model.layers],
        final_norm=model.norm,
        rotary=[model.rotary_emb] * len(model.layers),
        residual=StandardResidual(StandardResidualSpec()),
    )
    backbone.gradient_checkpointing = checkpointing
    context = model.hidden_norm(model.fc(inputs["target_hidden"]))
    layer_inputs = [
        LayerInputs(
            target_context=context,
            position_ids=inputs["position_ids"],
        )
    ] * backbone.num_layers
    batch, blocks, size = inputs["block_token_ids"].shape
    state = backbone.begin(inputs["noise_embedding"], layer_inputs)
    features = []
    for index in range(backbone.num_layers):
        state = backbone.run_next_layer(state)
        pre_injection = state.hidden.reshape(batch, blocks, size, -1)
        hidden = model.injection(
            pre_injection,
            index,
            inputs["block_token_ids"],
            model.transfer_codes,
            inputs["replace_mask"],
        )
        features.append(model.injection.feature(hidden, index))
        state = state.with_hidden(hidden.reshape(batch, blocks * size, -1))
    return DSpineOutput(
        backbone.finalize(state).reshape(batch, blocks, size, -1),
        pre_injection,
        torch.stack(features),
    )


class DSpineBackboneTest(unittest.TestCase):
    def test_reference_forward_refinement_and_all_gradients_match(self):
        original, _ = initialized_model(replacement_layers=2)
        with torch.no_grad():
            # Nonzero writes exercise message, gate, and readout gradients.
            for write in original.injection.writes:
                write.weight.normal_(std=0.1)
        original_inputs = forward_inputs(original)
        original_inputs["replace_mask"] = torch.tensor([[True, False], [False, True]])
        results = []
        for mode in ("reference", "manual", "checkpointed"):
            model = copy.deepcopy(original)
            inputs = {key: value.clone() for key, value in original_inputs.items()}
            for name in ("noise_embedding", "target_hidden"):
                inputs[name].requires_grad_()
            output = (
                model(**inputs)
                if mode == "reference"
                else model_owned_forward(model, inputs, mode == "checkpointed")
            )
            refined = model.refine(
                output.pre_injection[..., 1:, :], inputs["block_token_ids"][..., :-1]
            )
            values = (*output, refined)
            sum(x.square().sum() for x in values).backward()
            gradients = {name: p.grad for name, p in model.named_parameters()}
            gradients.update(
                {
                    name: inputs[name].grad
                    for name in ("noise_embedding", "target_hidden")
                }
            )
            results.append((values, gradients))
        expected, expected_grads = results[0]
        for values, gradients in results[1:]:
            torch.testing.assert_close(values, expected, rtol=0, atol=0)
            self.assertEqual(gradients.keys(), expected_grads.keys())
            for name, reference in expected_grads.items():
                with self.subTest(parameter=name):
                    if reference is None:
                        self.assertIsNone(gradients[name])
                    else:
                        torch.testing.assert_close(
                            gradients[name], reference, rtol=0, atol=0
                        )


if __name__ == "__main__":
    unittest.main()
