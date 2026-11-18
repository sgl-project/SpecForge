"""Domino composition preserves shifted alignment and independent block state."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

import torch

from specforge.config.draft.architecture.domino import DominoBlockSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DominoDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_domino_spec import domino_spec
from tests.test_modeling.test_draft_gqa import layout_for


def model_config(*, shifted=False, prefix=1, bound=False):
    return draft_config(
        architecture="DominoDraftModel",
        spec_config=domino_spec(
            block=DominoBlockSpec(block_size=4, shift_label=shifted),
            pure_draft_prefix_len=prefix,
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


class CanonicalDominoTest(unittest.TestCase):

    def test_gru_resets_at_block_boundaries(self):
        for shifted in (False, True):
            config = model_config(shifted=shifted)
            actual = build_draft_model(config)
            args, previous = model_inputs(config)
            output = actual(
                *args, predecessor_ids=previous, output_correction_logits=True
            )
            changed = previous.clone()
            changed[:, :4] = (changed[:, :4] + 1) % 64
            other = actual(
                *args, predecessor_ids=changed, output_correction_logits=True
            )
            torch.testing.assert_close(
                other.logits[:, 4:], output.logits[:, 4:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                other.correction_logits[:, 1],
                output.correction_logits[:, 1],
                rtol=0,
                atol=0,
            )
            self.assertFalse(
                torch.equal(
                    other.correction_logits[:, 0], output.correction_logits[:, 0]
                )
            )

    def test_predecessors_do_not_leak_into_earlier_slots(self):
        for shifted in (False, True):
            model = build_draft_model(model_config(shifted=shifted, prefix=0))
            args, previous = model_inputs(model.config.draft)
            expected = model(*args, predecessor_ids=previous)
            self.assertIsNone(expected.correction_logits)
            for slot in range(4):
                with self.subTest(shifted=shifted, slot=slot):
                    changed = previous.clone()
                    changed[:, slot] = (changed[:, slot] + 1) % 64
                    output = model(*args, predecessor_ids=changed)
                    torch.testing.assert_close(
                        output.logits[:, :slot],
                        expected.logits[:, :slot],
                        rtol=0,
                        atol=0,
                    )
                    torch.testing.assert_close(
                        output.logits[:, 4:], expected.logits[:, 4:], rtol=0, atol=0
                    )
                    if slot == 0 and not shifted:
                        torch.testing.assert_close(
                            output.logits, expected.logits, rtol=0, atol=0
                        )
                    else:
                        self.assertFalse(
                            torch.equal(
                                output.logits[:, slot:4], expected.logits[:, slot:4]
                            )
                        )

    def test_binding_checkpoint_and_hf_restore(self):
        config = model_config(bound=True)
        actual = build_draft_model(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, value in weights.items():
            getattr(actual, name).load_target_weight(value)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = actual(*args, predecessor_ids=previous, output_correction_logits=True)
        expected = checkpointed(*args, predecessor_ids=previous)
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        output.logits.sum().backward()
        expected.logits.sum().backward()
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
            restored = DominoDraftModel.from_pretrained(path)
        for name, value in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(value)
        result = restored(
            *args, predecessor_ids=previous, output_correction_logits=True
        )
        torch.testing.assert_close(result.logits, output.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            result.correction_logits, output.correction_logits, rtol=0, atol=0
        )
        self.assertNotIn("lm_head.weight", actual.state_dict())

    def test_hidden_only_meta_tying_and_input_contract(self):
        config = model_config()
        actual = build_draft_model(config)
        args, previous = model_inputs(config)
        hidden = actual(*args, output_logits=False)
        self.assertIsNone(hidden.logits)
        self.assertIsNone(hidden.correction_logits)
        with self.assertRaisesRegex(ValueError, "requires predecessor_ids"):
            actual(*args)
        with self.assertRaisesRegex(ValueError, "whole draft blocks"):
            actual(args[0][:, :-1], args[1], args[2], predecessor_ids=previous)
        with self.assertRaisesRegex(ValueError, "integer IDs"):
            actual(*args, predecessor_ids=previous.float())
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        tied = build_draft_model(replace(config, tie_word_embeddings=True))
        self.assertIs(tied.embed_tokens.weight, tied.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
