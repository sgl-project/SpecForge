"""Full DSpark composition preserves head math and recurrent block boundaries."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

import torch

from specforge.config.draft.architecture.block import DraftOnlyBlockSpec
from specforge.config.draft.architecture.dspark import (
    ConfidenceHeadSpec,
    MarkovHeadSpec,
)
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DSparkDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_dspark_spec import dspark_spec
from tests.test_modeling.test_draft_gqa import layout_for


def model_config(kind="rnn", *, confidence=True, with_markov=True, bound=False):
    return draft_config(
        architecture="DSparkDraftModel",
        spec_config=dspark_spec(
            block=DraftOnlyBlockSpec(block_size=4),
            markov=None if kind is None else MarkovHeadSpec(type=kind, rank=4),
            confidence=(
                ConfidenceHeadSpec(with_markov=with_markov) if confidence else None
            ),
        ),
        **({} if bound else {"target_bindings": (), "target_logit_transform": None}),
    )


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.arange(16, device=device).reshape(2, 8)
    features = torch.randn(2, 3, 64, device=device, dtype=dtype, requires_grad=True)
    layout = layout_for(torch.empty(2, 8, 32, device=device), context_length=3)
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    return (ids, features, entries), torch.randint(64, (2, 8), device=device)


class CanonicalDSparkTest(unittest.TestCase):

    def test_recurrence_resets_at_block_boundaries(self):
        actual = build_draft_model(model_config())
        args, previous = model_inputs(actual.config.draft)
        output = actual(*args, predecessor_ids=previous)
        changed = previous.clone()
        changed[:, :4] = (changed[:, :4] + 1) % 64
        other = actual(*args, predecessor_ids=changed)
        torch.testing.assert_close(
            other.logits[:, 4:], output.logits[:, 4:], rtol=0, atol=0
        )
        torch.testing.assert_close(
            other.confidence_scores[:, 4:],
            output.confidence_scores[:, 4:],
            rtol=0,
            atol=0,
        )
        self.assertFalse(torch.equal(other.logits[:, :4], output.logits[:, :4]))

    def test_binding_checkpoint_and_hf_restore(self):
        config = model_config(bound=True)
        actual = build_draft_model(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, value in weights.items():
            getattr(actual, name).load_target_weight(value)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = actual(*args, predecessor_ids=previous)
        expected = checkpointed(*args, predecessor_ids=previous)
        torch.testing.assert_close(
            (output.logits, output.confidence_scores),
            (expected.logits, expected.confidence_scores),
            rtol=0,
            atol=0,
        )
        (output.logits.sum() + output.confidence_scores.sum()).backward()
        (expected.logits.sum() + expected.confidence_scores.sum()).backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = DSparkDraftModel.from_pretrained(path)
        for name, value in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(value)
        result = restored(*args, predecessor_ids=previous)
        torch.testing.assert_close(result.logits, output.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            result.confidence_scores, output.confidence_scores, rtol=0, atol=0
        )
        self.assertNotIn("lm_head.weight", actual.state_dict())

    def test_optional_heads_hidden_only_meta_and_input_contract(self):
        config = model_config(None, confidence=False)
        actual = build_draft_model(config)
        args, _ = model_inputs(config)
        output = actual(*args)
        self.assertIsNone(actual.markov_head)
        self.assertIsNone(actual.confidence_head)
        self.assertIsNone(output.confidence_scores)
        torch.testing.assert_close(
            output.logits, actual.lm_head(output.hidden_states), rtol=0, atol=0
        )
        headed = build_draft_model(model_config())
        hidden = headed(*args, output_logits=False)
        self.assertIsNone(hidden.logits)
        self.assertIsNone(hidden.confidence_scores)
        with self.assertRaisesRegex(ValueError, "requires predecessor_ids"):
            headed(*args)
        with self.assertRaisesRegex(ValueError, "whole draft blocks"):
            headed(args[0][:, :-1], args[1], args[2])
        with self.assertRaisesRegex(ValueError, "integer IDs"):
            headed(*args, predecessor_ids=args[0].float())
        meta = build_draft_model(model_config(), device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        tied = build_draft_model(replace(config, tie_word_embeddings=True))
        self.assertIs(tied.embed_tokens.weight, tied.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
