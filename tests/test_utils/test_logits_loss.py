"""CE/teacher-L1 parity and safe composition with ordinary projection autograd."""

import unittest
from unittest.mock import patch

import torch
import torch.nn.functional as F

from specforge.losses import fused as fused_loss
from specforge.losses.fused import fused_ce_l1_loss


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

    def test_eager_detached_teacher_l1_saves_fewer_tensors(self):
        logits = torch.randn(2, 3, 17, requires_grad=True)
        teacher = torch.randn(2, 3, 17)
        labels = torch.randint(0, 17, (2, 3))

        def run(teacher_l1_grad):
            saved = []
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: (saved.append(t), t)[1], lambda t: t
            ):
                out = fused_loss.ce_l1_loss(
                    logits,
                    labels,
                    teacher_logits=teacher,
                    top_k=4,
                    teacher_l1_grad=teacher_l1_grad,
                )
            return out, len(saved)

        grad_out, grad_saved = run(True)
        detached_out, detached_saved = run(False)
        self.assertTrue(grad_out.teacher_l1.requires_grad)
        self.assertFalse(detached_out.teacher_l1.requires_grad)
        self.assertTrue(detached_out.cross_entropy.requires_grad)
        torch.testing.assert_close(detached_out.teacher_l1, grad_out.teacher_l1)
        self.assertLess(detached_saved, grad_saved)
        detached_out.cross_entropy.sum().backward()
        self.assertIsNotNone(logits.grad)


@unittest.skipUnless(
    torch.cuda.is_available() and fused_loss.triton_available("cuda"),
    "CUDA/Triton required",
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

    def test_leading_neg_inf_tile(self):
        logits = torch.randn(1, 1024, device="cuda", dtype=torch.float32)
        logits[:, :512] = float("-inf")
        logits.requires_grad_(True)
        targets = torch.tensor([600], device="cuda", dtype=torch.long)
        terms = fused_ce_l1_loss(logits, targets, top_k=16)
        expected_ce = F.cross_entropy(logits.detach(), targets, reduction="none")
        self.assertTrue(torch.isfinite(terms.cross_entropy).all())
        torch.testing.assert_close(terms.cross_entropy, expected_ce)
        terms.cross_entropy.sum().backward()
        self.assertTrue(torch.isfinite(logits.grad).all())

    def test_identical_logits_with_topk_have_zero_l1(self):
        identical = torch.full((1, 8192), -1000.0, device="cuda", dtype=torch.float32)
        identical[:, 0] = 0.0
        identical[:, 512::512] = -17.5
        identical.requires_grad_(True)
        terms = fused_ce_l1_loss(
            identical,
            torch.zeros(1, device="cuda", dtype=torch.long),
            teacher_logits=identical.detach().clone(),
            top_k=16,
        )
        torch.testing.assert_close(
            terms.teacher_l1, torch.zeros_like(terms.teacher_l1), rtol=0, atol=0
        )
        terms.teacher_l1.sum().backward()
        torch.testing.assert_close(
            identical.grad, torch.zeros_like(identical.grad), rtol=0, atol=0
        )

    def test_detached_teacher_l1_does_not_save_teacher_storage(self):
        draft = torch.randn(
            2, 1024, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        teacher = torch.randn(2, 1024, device="cuda", dtype=torch.bfloat16)
        targets = torch.zeros(2, device="cuda", dtype=torch.long)
        teacher_ptr = teacher.untyped_storage().data_ptr()

        def run(teacher_l1_grad):
            saved_ptrs = set()
            with torch.autograd.graph.saved_tensors_hooks(
                lambda t: (saved_ptrs.add(t.untyped_storage().data_ptr()), t)[1],
                lambda t: t,
            ):
                out = fused_ce_l1_loss(
                    draft,
                    targets,
                    teacher_logits=teacher,
                    top_k=16,
                    teacher_l1_grad=teacher_l1_grad,
                )
            return out, saved_ptrs

        grad_out, grad_ptrs = run(True)
        self.assertTrue(grad_out.teacher_l1.requires_grad)
        self.assertIn(teacher_ptr, grad_ptrs)

        detached_out, detached_ptrs = run(False)
        self.assertFalse(detached_out.teacher_l1.requires_grad)
        self.assertNotIn(teacher_ptr, detached_ptrs)
        torch.testing.assert_close(detached_out.teacher_l1, grad_out.teacher_l1)
        detached_out.cross_entropy.sum().backward()
        self.assertIsNotNone(draft.grad)


if __name__ == "__main__":
    unittest.main()
