"""Canonical compact teacher matches full-vocabulary projection."""

import unittest
from unittest.mock import patch

import torch

from specforge.algorithms.objectives.predictions import Teacher
from specforge.algorithms.objectives.teacher import tiled_logsumexp_argmax


class CompactTeacherStrategyTest(unittest.TestCase):
    def test_values_and_vocab_projection_count(self):
        for dtype in (torch.float32, torch.bfloat16):
            with self.subTest(dtype=dtype):
                torch.manual_seed(0)
                hidden = torch.randn(2, 6, 8, dtype=dtype)
                head = torch.nn.Linear(8, 64, bias=False, dtype=dtype)
                mask = torch.zeros(64, dtype=torch.bool)
                mask[torch.randperm(64)[:16]] = True
                expected = head(hidden).float()
                with patch(
                    "specforge.algorithms.objectives.teacher.F.linear",
                    wraps=torch.nn.functional.linear,
                ) as linear:
                    logits, ids, mass = Teacher(hidden, head).compact(mask, 7)
                torch.testing.assert_close(logits, expected[..., mask])
                torch.testing.assert_close(ids, expected.argmax(-1))
                torch.testing.assert_close(
                    mass,
                    expected.softmax(-1)[..., mask].sum(-1),
                    rtol=1e-4,
                    atol=1e-5,
                )
                self.assertFalse(logits.requires_grad)
                self.assertEqual(
                    sum(c.args[1].shape[0] for c in linear.call_args_list), 64
                )

    def test_argmax_ties_keep_first_token(self):
        hidden = torch.ones(1, 1, 4)
        weight = torch.zeros(8, 4)
        weight[2] = weight[6] = 1
        log_z, ids, _ = tiled_logsumexp_argmax(hidden, weight, chunk_size=3)
        logits = hidden @ weight.T
        torch.testing.assert_close(ids, logits.argmax(-1))
        torch.testing.assert_close(log_z, logits.logsumexp(-1, keepdim=True))

    def test_invalid_teacher_inputs(self):
        head = torch.nn.Linear(4, 8, bias=False)
        teacher = Teacher(torch.ones(1, 2, 4), head)
        with self.assertRaisesRegex(ValueError, "vocabulary mask"):
            teacher.compact(torch.ones(7, dtype=torch.bool), 2)
        with self.assertRaisesRegex(ValueError, "chunk_size"):
            teacher.compact(torch.ones(8, dtype=torch.bool), 0)
        with self.assertRaisesRegex(ValueError, "linear teacher"):
            Teacher(teacher.values).compact(torch.ones(8, dtype=torch.bool), 2)
