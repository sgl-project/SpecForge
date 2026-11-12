"""CE/teacher-L1 parity and safe composition with ordinary projection autograd."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from specforge.core import fused_loss
from specforge.core.fused_loss import fused_ce_l1_loss

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False


class CEL1InterfaceTest(unittest.TestCase):
    def test_eager_uses_explicit_labels_and_stops_teacher_gradients(self):
        logits = torch.tensor([[[1.0, 2.0, 0.0]]], requires_grad=True)
        teacher = torch.tensor([[[0.0, 4.0, 1.0]]], requires_grad=True)
        labels = torch.tensor([[0]])  # Deliberately differs from teacher argmax=1.
        with patch.dict("sys.modules", {"triton": None}):
            terms = fused_loss.ce_l1_loss(
                logits, labels, teacher_logits=teacher, top_k=2
            )
        self.assertIsInstance(terms, fused_loss.CEL1Loss)
        torch.testing.assert_close(
            terms.cross_entropy,
            F.cross_entropy(
                logits.flatten(0, 1), labels.flatten(), reduction="none"
            ).view(1, 1),
        )
        self.assertEqual(terms.topk_values.shape, (1, 1, 2))
        self.assertFalse(terms.top1_probability.requires_grad)
        (terms.cross_entropy + 0.5 * terms.teacher_l1).sum().backward()
        self.assertIsNotNone(logits.grad)
        self.assertIsNone(teacher.grad)
        self.assertIsNone(fused_loss.ce_l1_loss(logits, labels).teacher_l1)

    def test_fused_rejects_invalid_labels_before_kernel_launch(self):
        logits = torch.zeros(1, 2, 3)
        for labels in (torch.zeros(1, 2), torch.zeros(2, dtype=torch.long)):
            with self.subTest(shape=labels.shape, dtype=labels.dtype):
                with self.assertRaisesRegex(ValueError, "target_ids"):
                    fused_loss.fused_ce_l1_loss(logits, labels)


@unittest.skipUnless(
    torch.cuda.is_available() and TRITON_AVAILABLE, "CUDA/Triton required"
)
class LogitsLossTest(unittest.TestCase):
    def test_weighted_ce_and_teacher_l1(self):
        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            for vocab in (777, 151936):
                for ce_scale, teacher_scale in ((1, 0.0), (1, 0.5), (1, 1.0), (0, 1.0)):
                    with self.subTest(
                        dtype=dtype,
                        vocab=vocab,
                        ce_scale=ce_scale,
                        teacher_scale=teacher_scale,
                    ):
                        torch.manual_seed(7)
                        logits = torch.randn(
                            2, 5, vocab, device="cuda", dtype=dtype, requires_grad=True
                        )
                        teacher = torch.randn_like(logits, requires_grad=True)
                        reference = logits.detach().clone().requires_grad_(True)
                        targets = torch.randint(vocab, (2, 5), device="cuda")
                        weights = torch.rand(2, 5, device="cuda")
                        weights[:, 0] = 0
                        terms = fused_ce_l1_loss(
                            logits,
                            targets,
                            teacher_logits=teacher if teacher_scale else None,
                        )
                        reference_fp32 = reference.float()
                        ce = F.cross_entropy(
                            reference_fp32.flatten(0, 1),
                            targets.flatten(),
                            reduction="none",
                        ).view(2, 5)
                        actual, expected = terms.cross_entropy, ce
                        if teacher_scale:
                            l1 = (
                                (
                                    reference_fp32.softmax(-1)
                                    - teacher.detach().float().softmax(-1)
                                )
                                .abs()
                                .sum(-1)
                            )
                            torch.testing.assert_close(
                                terms.teacher_l1, l1, atol=2e-6, rtol=2e-5
                            )
                            actual = (
                                actual + teacher_scale * terms.teacher_l1
                                if ce_scale
                                else terms.teacher_l1
                            )
                            expected = expected + teacher_scale * l1 if ce_scale else l1
                        self.assertEqual(terms.cross_entropy.dtype, torch.float32)
                        torch.testing.assert_close(
                            actual, expected, atol=2e-5, rtol=2e-5
                        )
                        (actual * weights).sum().backward()
                        (expected * weights).sum().backward()
                        error = (
                            logits.grad.float() - reference.grad.float()
                        ).norm() / reference.grad.float().norm()
                        self.assertLess(
                            error.item(), 0.002 if dtype != torch.float32 else 1e-5
                        )
                        self.assertTrue(torch.all(logits.grad[:, 0] == 0))
                        self.assertIsNone(teacher.grad)

    def test_projection_and_other_consumers_keep_their_gradients(self):
        torch.manual_seed(8)
        hidden = torch.randn(9, 32, device="cuda", requires_grad=True)
        weight = torch.randn(101, 32, device="cuda", requires_grad=True)
        logits = F.linear(hidden, weight)
        saved = logits.detach().clone()
        targets = torch.randint(101, (9,), device="cuda")
        # Equal distributions exercise the zero subgradient of abs at zero.
        terms = fused_ce_l1_loss(
            logits, targets, teacher_logits=logits.detach(), top_k=3
        )
        loss = (
            (terms.cross_entropy + terms.teacher_l1).sum()
            + terms.topk_values.sum()
            + logits.square().mean()
        )
        actual = torch.autograd.grad(loss, (hidden, weight), retain_graph=True)
        repeated = torch.autograd.grad(loss, (hidden, weight), retain_graph=True)
        expected = (
            F.cross_entropy(logits, targets, reduction="sum")
            + logits.gather(-1, terms.topk_ids).sum()
            + logits.square().mean()
        )
        reference = torch.autograd.grad(expected, (hidden, weight))
        for a, b, c in zip(actual, repeated, reference):
            torch.testing.assert_close(a, b)
            torch.testing.assert_close(a, c, rtol=1e-4, atol=2e-5)
        torch.testing.assert_close(logits, saved, rtol=0, atol=0)
        torch.testing.assert_close(
            terms.teacher_l1, torch.zeros_like(terms.teacher_l1), rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
