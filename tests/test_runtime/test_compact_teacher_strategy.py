# coding=utf-8
"""Compact EAGLE3 teacher stays behind the canonical strategy seam."""

import types
import unittest

import torch

from specforge.core.compact_teacher import (
    compute_target_from_hidden,
    compute_target_p_padded_from_hidden,
    tiled_logsumexp_argmax,
)
from specforge.core.eagle3_adapters import SdpaLikeAdapter
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import Eagle3TrainStrategy
from specforge.training.strategies.registry import resolve_strategy
from specforge.utils import padding


def _mapping(vocab=8, draft_vocab=3):
    selected = torch.tensor([0, 3, 7])
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[selected] = True
    d2t = selected - torch.arange(draft_vocab)
    return t2d, d2t


def _reference_teacher(target, t2d, loss_mask):
    target = target.float()
    target_token_ids = target.argmax(dim=-1)
    draft_target = target[..., t2d]
    return (
        torch.softmax(draft_target, dim=-1),
        torch.exp(draft_target - torch.logsumexp(target, dim=-1, keepdim=True)),
        target_token_ids,
        t2d[target_token_ids][..., None].int() * loss_mask,
    )


def _reference_padded_teacher(target, t2d, loss_mask, length):
    target_p, target_p_on_draft, token_ids, position_mask = _reference_teacher(
        target, t2d, loss_mask
    )
    return (
        torch.nn.functional.pad(
            target_p, (0, 0, 0, length), value=1 / target_p.shape[-1]
        ),
        torch.nn.functional.pad(target_p_on_draft, (0, 0, 0, length), value=0.0),
        torch.nn.functional.pad(token_ids, (0, length), value=0),
        position_mask,
    )


def _numerical_inputs(dtype=torch.float32):
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(2, 6, 8, generator=generator).to(dtype)
    weight = torch.randn(64, 8, generator=generator).to(dtype)
    selected = torch.sort(torch.randperm(64, generator=generator)[:16]).values
    t2d = torch.zeros(64, dtype=torch.bool)
    t2d[selected] = True
    loss_mask = torch.randint(0, 2, (2, 6, 1), generator=generator).int()
    return hidden, weight, t2d, loss_mask


def _assert_teacher_close(test_case, actual, expected):
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-5)
    test_case.assertTrue(torch.equal(actual[2], expected[2]))
    test_case.assertTrue(torch.equal(actual[3], expected[3]))


class _TargetHead(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 8, bias=False)
        self.forward_calls = 0

    def preprocess(self, input_ids, target, loss_mask):
        return input_ids[:, 1:], target[:, 1:], loss_mask[:, 1:, None]

    def forward(self, hidden):
        self.forward_calls += 1
        return self.fc(hidden)


class _Draft(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.config = types.SimpleNamespace(draft_vocab_size=3)
        t2d, d2t = _mapping()
        self.register_buffer("t2d", t2d)
        self.register_buffer("d2t", d2t)
        self.weight = torch.nn.Parameter(torch.zeros(1))


class _Eagle3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.draft_model = _Draft()
        self.kwargs = None

    def forward(self, **kwargs):
        self.kwargs = kwargs
        value = self.draft_model.weight.sum() + 1.0
        one = [value]
        return one, one, one, one, one, one, one


def _batch(target_repr="hidden_state"):
    return TrainBatch(
        tensors={
            "input_ids": torch.tensor([[1, 2, 3, 4]]),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
            "loss_mask": torch.ones(1, 4, dtype=torch.long),
            "hidden_state": torch.randn(1, 4, 4),
            "target": torch.randn(1, 4, 4 if target_repr == "hidden_state" else 8),
        },
        sample_ids=["sample-0"],
        strategy="eagle3",
        metadata={"target_repr": target_repr},
    )


class CompactTeacherStrategyTest(unittest.TestCase):
    def test_chunked_teacher_matches_full_vocab_reference(self):
        generator = torch.Generator().manual_seed(0)
        hidden = torch.randn(2, 3, 4, generator=generator)
        weight = torch.randn(8, 4, generator=generator)
        t2d, _ = _mapping()
        loss_mask = torch.ones(2, 3, 1, dtype=torch.long)
        logits = torch.nn.functional.linear(hidden, weight).float()
        draft_logits = logits[..., t2d]
        target_ids = logits.argmax(dim=-1)
        expected = (
            torch.softmax(draft_logits, dim=-1),
            torch.exp(draft_logits - torch.logsumexp(logits, dim=-1, keepdim=True)),
            target_ids,
            t2d[target_ids][..., None].long() * loss_mask,
        )

        actual = compute_target_from_hidden(
            hidden, weight, t2d, loss_mask, chunk_size=2
        )

        torch.testing.assert_close(actual[0], expected[0])
        torch.testing.assert_close(actual[1], expected[1])
        self.assertTrue(torch.equal(actual[2], expected[2]))
        self.assertTrue(torch.equal(actual[3], expected[3]))

    def test_bf16_chunked_teacher_matches_full_vocab_reference(self):
        hidden, weight, t2d, loss_mask = _numerical_inputs(torch.bfloat16)
        full_logits = torch.nn.functional.linear(hidden, weight)

        actual = compute_target_from_hidden(
            hidden, weight, t2d, loss_mask, chunk_size=7
        )

        _assert_teacher_close(
            self, actual, _reference_teacher(full_logits, t2d, loss_mask)
        )

    def test_argmax_tie_across_chunks_uses_lowest_vocab_id(self):
        hidden = torch.ones(1, 1, 4)
        weight = torch.zeros(8, 4)
        weight[1] = 10.0
        weight[5] = 10.0
        full_logits = torch.nn.functional.linear(hidden, weight)

        log_z, token_ids = tiled_logsumexp_argmax(hidden, weight, chunk_size=3)

        self.assertEqual(token_ids.item(), 1)
        self.assertTrue(torch.equal(token_ids, full_logits.argmax(dim=-1)))
        torch.testing.assert_close(
            log_z,
            torch.logsumexp(full_logits.float(), dim=-1, keepdim=True),
            rtol=1e-5,
            atol=1e-6,
        )

    def test_bf16_padded_teacher_matches_full_vocab_reference(self):
        hidden, weight, t2d, loss_mask = _numerical_inputs(torch.bfloat16)
        full_logits = torch.nn.functional.linear(hidden, weight)
        length = 3

        actual = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, length, chunk_size=7
        )
        expected = _reference_padded_teacher(full_logits, t2d, loss_mask, length)

        _assert_teacher_close(self, actual, expected)
        self.assertTrue(torch.all(actual[0][:, -length:] == 1 / 16))
        self.assertTrue(torch.all(actual[1][:, -length:] == 0.0))
        self.assertTrue(torch.all(actual[2][:, -length:] == 0))

    def test_multistep_teacher_views_match_full_vocab_reference(self):
        hidden, weight, t2d, loss_mask = _numerical_inputs(torch.bfloat16)
        full_logits = torch.nn.functional.linear(hidden, weight)
        ttt_length = 3
        expected = _reference_padded_teacher(full_logits, t2d, loss_mask, ttt_length)
        actual = compute_target_p_padded_from_hidden(
            hidden, weight, t2d, loss_mask, ttt_length, chunk_size=7
        )
        adapter = SdpaLikeAdapter(model=None)
        input_ids = torch.zeros(hidden.shape[:2], dtype=torch.long)
        expected_position_mask = expected[3]
        actual_position_mask = actual[3]
        expected_loss_mask = loss_mask.clone()
        actual_loss_mask = loss_mask.clone()

        for step in range(ttt_length):
            common = {
                "idx": step,
                "ttt_length": ttt_length,
                "global_input_ids": input_ids,
                "attention_mask": None,
                "position_ids": None,
                "hidden_states": hidden,
                "seq_length": hidden.shape[1],
            }
            expected_view = adapter.step_view(
                loss_mask=expected_loss_mask,
                target_p_padded=expected[0],
                target_p_on_draft_padded=expected[1],
                target_token_ids_padded=expected[2],
                position_mask=expected_position_mask,
                **common,
            )
            actual_view = adapter.step_view(
                loss_mask=actual_loss_mask,
                target_p_padded=actual[0],
                target_p_on_draft_padded=actual[1],
                target_token_ids_padded=actual[2],
                position_mask=actual_position_mask,
                **common,
            )
            _assert_teacher_close(
                self,
                (
                    actual_view.target_p,
                    actual_view.target_p_on_draft,
                    actual_view.target_token_ids,
                    actual_view.position_mask,
                ),
                (
                    expected_view.target_p,
                    expected_view.target_p_on_draft,
                    expected_view.target_token_ids,
                    expected_view.position_mask,
                ),
            )
            self.assertTrue(torch.equal(actual_view.loss_mask, expected_view.loss_mask))
            if step != ttt_length - 1:
                expected_position_mask = padding(expected_position_mask, left=False)
                actual_position_mask = padding(actual_position_mask, left=False)
                expected_loss_mask = padding(expected_loss_mask, left=False)
                actual_loss_mask = padding(actual_loss_mask, left=False)

    def test_numerical_inputs_are_validated(self):
        hidden, weight, t2d, loss_mask = _numerical_inputs()
        cases = (
            (
                "bool mask",
                lambda: compute_target_from_hidden(
                    hidden, weight, t2d.int(), loss_mask
                ),
            ),
            (
                "must equal vocab_size",
                lambda: compute_target_from_hidden(hidden, weight, t2d[:-1], loss_mask),
            ),
            (
                "hidden size",
                lambda: compute_target_from_hidden(
                    hidden[..., :-1], weight, t2d, loss_mask
                ),
            ),
            (
                "chunk_size must be positive",
                lambda: tiled_logsumexp_argmax(hidden, weight, chunk_size=0),
            ),
        )
        for message, operation in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    operation()

    def test_compact_path_skips_full_vocab_target_projection(self):
        model = _Eagle3()
        head = _TargetHead()
        strategy = Eagle3TrainStrategy(
            model,
            target_head=head,
            compact_teacher=True,
            compact_teacher_chunk_size=2,
        )

        strategy.forward_loss(_batch())

        self.assertEqual(head.forward_calls, 0)
        self.assertIsNone(model.kwargs["target"])
        self.assertEqual(model.kwargs["target_hidden_for_compact"].shape, (1, 3, 4))
        self.assertIs(model.kwargs["target_head_weight"], head.fc.weight)
        self.assertEqual(model.kwargs["compact_teacher_chunk_size"], 2)
        self.assertEqual(model.kwargs["loss_mask"].shape, (1, 3, 1))

    def test_default_path_remains_full_vocab_projection(self):
        model = _Eagle3()
        head = _TargetHead()

        Eagle3TrainStrategy(model, target_head=head).forward_loss(_batch())

        self.assertEqual(head.forward_calls, 1)
        self.assertEqual(model.kwargs["target"].shape, (1, 3, 8))
        self.assertNotIn("target_hidden_for_compact", model.kwargs)

    def test_compact_path_rejects_online_target_repr(self):
        strategy = Eagle3TrainStrategy(
            _Eagle3(), target_head=_TargetHead(), compact_teacher=True
        )
        with self.assertRaisesRegex(ValueError, "offline-only"):
            strategy.forward_loss(_batch(target_repr="logits"))

    def test_registry_forwards_compact_strategy_kwargs(self):
        strategy = resolve_strategy("eagle3").make_strategy(
            _Eagle3(),
            target_head=_TargetHead(),
            compact_teacher=True,
            compact_teacher_chunk_size=4,
        )
        self.assertTrue(strategy.compact_teacher)
        self.assertEqual(strategy.compact_teacher_chunk_size, 4)

    def test_invalid_mapping_fails_during_assembly(self):
        model = _Eagle3()
        model.draft_model.d2t[0] += 1
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            Eagle3TrainStrategy(model, target_head=_TargetHead(), compact_teacher=True)

    def test_strategy_rejects_invalid_compact_configuration(self):
        cases = (
            ({"target_head": None}, "requires the offline target_head"),
            ({"compact_teacher_chunk_size": 0}, "positive integer"),
        )
        for kwargs, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    parameters = {
                        "target_head": _TargetHead(),
                        "compact_teacher": True,
                        **kwargs,
                    }
                    Eagle3TrainStrategy(_Eagle3(), **parameters)


if __name__ == "__main__":
    unittest.main(verbosity=2)
