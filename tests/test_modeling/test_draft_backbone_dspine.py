"""Compare the backbone step API with an explicit DSpine decoder/injection loop."""

import copy
import unittest
from dataclasses import replace

import torch

from specforge.modeling.draft.layers.attention.backends import AttentionPreparationScope
from tests.test_modeling.test_canonical_dspine import initialized_model, model_inputs


class DSpineBackboneTest(unittest.TestCase):
    def test_reference_forward_refinement_and_all_gradients_match(self):
        original = initialized_model()
        args, tokens, mask = model_inputs(original.config.draft)
        results = []
        for manual, checkpointing in ((True, False), (False, False), (False, True)):
            model = copy.deepcopy(original)
            model.backbone.gradient_checkpointing = checkpointing
            ids, source, entries = args
            source = source.detach().clone().requires_grad_()
            masks = (mask,) + (None,) * (model.backbone.num_layers - 1)
            if manual:
                hidden = model.embed_tokens(ids)
                context = model.fusion(source)
                features = []
                with AttentionPreparationScope() as scope:
                    attention = [
                        layer.self_attn.prepare(
                            entry.attention_layout, scope, batch_size=ids.shape[0]
                        )
                        for layer, entry in zip(model.backbone.layers, entries)
                    ]
                for index, layer in enumerate(model.backbone.layers):
                    entry = replace(
                        entries[index],
                        target_context=context,
                        attention=attention[index],
                        attention_layout=None,
                        position_embeddings=model.backbone.rotary[index](
                            hidden, entries[index].position_ids
                        ),
                    )
                    hidden = layer(hidden, entry)
                    pre = hidden.reshape(2, 2, 4, -1)
                    injected = model.injection(
                        pre, index, tokens, model.transfer_codes, masks[index]
                    )
                    features.append(model.injection.feature(injected, index))
                    hidden = injected.reshape_as(hidden)
                values = (
                    model.backbone.final_norm(hidden).reshape_as(pre),
                    pre,
                    torch.stack(features),
                )
            else:
                output = model(
                    ids,
                    source,
                    entries,
                    block_token_ids=tokens,
                    replace_masks=masks,
                    output_logits=False,
                )
                values = (
                    output.hidden_states,
                    output.pre_injection,
                    output.layer_features,
                )
            refined = model.refine(values[1][..., 1:, :], tokens[..., :-1])
            sum(value.square().sum() for value in (*values, refined)).backward()
            results.append(
                (
                    values,
                    refined,
                    source.grad,
                    {name: p.grad for name, p in model.named_parameters()},
                )
            )
        for actual in results[1:]:
            torch.testing.assert_close(actual, results[0])
