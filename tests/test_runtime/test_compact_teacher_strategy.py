# coding=utf-8
"""Compact EAGLE3 teacher stays behind the canonical strategy seam."""

import types
import unittest

import torch

from specforge.core.compact_teacher import compute_target_from_hidden
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import Eagle3TrainStrategy
from specforge.training.strategies.registry import resolve_strategy


def _mapping(vocab=8, draft_vocab=3):
    selected = torch.tensor([0, 3, 7])
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[selected] = True
    d2t = selected - torch.arange(draft_vocab)
    return t2d, d2t


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
            Eagle3TrainStrategy(
                model, target_head=_TargetHead(), compact_teacher=True
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
