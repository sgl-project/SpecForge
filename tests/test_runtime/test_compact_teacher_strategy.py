# coding=utf-8
"""Compact EAGLE3 teacher stays behind the canonical strategy seam."""

import unittest
from unittest import mock

import torch

from specforge.core.compact_teacher import (
    compute_target_from_hidden,
    compute_target_p_padded_from_hidden,
    tiled_logsumexp_argmax,
)
from tests.test_algorithms.fixtures import builtin_algorithm_registry

EAGLE3 = builtin_algorithm_registry().resolve("eagle3")


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

    def test_compact_teacher_projects_each_vocab_row_once(self):
        hidden, weight, t2d, loss_mask = _numerical_inputs()
        linear = torch.nn.functional.linear

        with mock.patch(
            "specforge.core.compact_teacher.F.linear", wraps=linear
        ) as mock_linear:
            compute_target_from_hidden(hidden, weight, t2d, loss_mask, chunk_size=7)

        projected_rows = sum(
            call.args[1].shape[0] for call in mock_linear.call_args_list
        )
        self.assertEqual(projected_rows, weight.shape[0])

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


if __name__ == "__main__":
    unittest.main()
