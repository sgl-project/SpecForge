# coding=utf-8
"""Tests for the fused Triton DFlash2 unary head."""

import unittest

import torch
import torch.nn.functional as F

try:
    import triton  # noqa: F401

    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

CUDA_TRITON = torch.cuda.is_available() and TRITON_AVAILABLE


def _project_loss(hidden, weight, targets, top_k):
    from specforge.core.fused_loss import fused_ce_l1_loss

    terms = fused_ce_l1_loss(F.linear(hidden, weight), targets, top_k=top_k)
    return terms.cross_entropy, terms.topk_values, terms.topk_ids, terms.argmax_ids


def _reference_head(hidden, weight, targets, topk_ids, grad_nlq, grad_topv):
    """Reference neg-log-q and input gradients from the same BF16 logits.

    Returns the eager path (autograd rounds the FP32 logit gradient to BF16
    before the BF16 input-gradient GEMM, as the reference objective does) and a
    full-FP32 backward (FP32 logit gradient and FP32 GEMM).
    """
    eager_hidden = hidden.detach().clone().requires_grad_(True)
    eager_logits = F.linear(eager_hidden, weight).float()
    neg_log_q = F.cross_entropy(eager_logits, targets, reduction="none")
    torch.autograd.backward(
        (neg_log_q, eager_logits.gather(-1, topk_ids)), (grad_nlq, grad_topv)
    )

    fp32_logits = eager_logits.detach().clone().requires_grad_(True)
    fp32_neg_log_q = F.cross_entropy(fp32_logits, targets, reduction="none")
    torch.autograd.backward(
        (fp32_neg_log_q, fp32_logits.gather(-1, topk_ids)), (grad_nlq, grad_topv)
    )
    fp32_grad = fp32_logits.grad @ weight.float()
    return neg_log_q.detach(), eager_logits.detach(), eager_hidden.grad, fp32_grad


@unittest.skipUnless(CUDA_TRITON, "fused DFlash2 head requires CUDA and Triton")
class FusedHeadKernelTest(unittest.TestCase):
    def _compare(self, vocab_size, top_k=16, rows=37, hidden_size=64):

        torch.manual_seed(vocab_size)
        device = torch.device("cuda")
        hidden = torch.randn(rows, hidden_size, device=device, dtype=torch.bfloat16)
        weight = 0.25 * torch.randn(
            vocab_size, hidden_size, device=device, dtype=torch.bfloat16
        )
        # Duplicate head rows create exact logit ties for argmax and top-k,
        # and a zero hidden row makes every logit of that row equal.
        weight[5] = weight[5] * 4
        weight[vocab_size - 3] = weight[5]
        hidden[3] = 0
        targets = torch.randint(vocab_size, (rows,), device=device)
        targets[0] = 5
        targets[1] = vocab_size - 1
        grad_nlq = torch.rand(rows, device=device)
        grad_topv = torch.randn(rows, top_k, device=device)
        grad_nlq[::4] = 0
        grad_topv[::4] = 0
        grad_topv[1::4] = 0

        actual_hidden = hidden.clone().requires_grad_(True)
        neg_log_q, topv, topi, argmax = _project_loss(
            actual_hidden, weight, targets, top_k
        )
        torch.autograd.backward((neg_log_q, topv), (grad_nlq, grad_topv))
        eager_nlq, eager_logits, eager_grad, fp32_grad = _reference_head(
            hidden, weight, targets, topi, grad_nlq, grad_topv
        )

        torch.testing.assert_close(neg_log_q, eager_nlq, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(argmax, eager_logits.argmax(dim=-1)))
        self.assertEqual(argmax[3].item(), 0)
        # BF16 -> FP32 is exact, so the candidate values match bit for bit.
        sorted_values = eager_logits.sort(dim=-1, descending=True).values
        boundary_tie = sorted_values[:, top_k - 1] == sorted_values[:, top_k]
        self.assertTrue(boundary_tie[3].item())
        _assert_same_topk(self, topv, topi, eager_logits, top_k)
        self.assertEqual(topi[3].tolist(), list(range(top_k)))

        zero_rows = (grad_nlq == 0) & (grad_topv == 0).all(dim=-1)
        self.assertTrue(zero_rows.any())
        self.assertTrue(torch.all(actual_hidden.grad[zero_rows] == 0))
        # The BF16 logit gradient and BF16 GEMM round like the eager path; the
        # remaining error against the FP32 backward is that shared rounding.
        scale = fp32_grad.abs().max()
        torch.testing.assert_close(
            actual_hidden.grad.float(), fp32_grad, rtol=2e-2, atol=5e-3 * scale
        )
        self.assertLess(
            ((actual_hidden.grad.float() - fp32_grad).norm() / fp32_grad.norm()).item(),
            5e-3,
        )
        torch.testing.assert_close(actual_hidden.grad, eager_grad, rtol=1e-2, atol=1e-3)

    def test_matches_reference_for_odd_vocabularies(self):
        for vocab_size in (1000, 50003):
            with self.subTest(vocab_size=vocab_size):
                self._compare(vocab_size)
        # Single and non-power-of-two candidate lists pad the selection slots.
        for vocab_size, top_k in ((1000, 3), (50003, 1)):
            with self.subTest(vocab_size=vocab_size, top_k=top_k):
                self._compare(vocab_size, top_k=top_k)

    def test_wide_candidate_sets_and_retained_graphs(self):
        from specforge.core._fused_logits_loss_triton import MAX_FUSED_TOP_K

        torch.manual_seed(4)
        device = torch.device("cuda")
        weight = torch.randn(3001, 32, device=device, dtype=torch.bfloat16)
        targets = torch.randint(3001, (19,), device=device)
        for top_k in (MAX_FUSED_TOP_K, MAX_FUSED_TOP_K + 36):
            with self.subTest(top_k=top_k):
                hidden = torch.randn(
                    19, 32, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                neg_log_q, topv, topi, _ = _project_loss(hidden, weight, targets, top_k)
                reference_values, reference_ids = (
                    F.linear(hidden.detach(), weight).float().topk(top_k, dim=-1)
                )
                self.assertTrue(torch.equal(topv, reference_values))
                self.assertTrue(
                    torch.equal(
                        topi.sort(dim=-1).values, reference_ids.sort(dim=-1).values
                    )
                )
                loss = neg_log_q.sum() + topv.sum()
                loss.backward(retain_graph=True)
                first_gradient = hidden.grad.clone()
                hidden.grad = None
                loss.backward()
                torch.testing.assert_close(hidden.grad, first_gradient)

    def test_argmax_and_neg_log_q_without_topk(self):

        torch.manual_seed(3)
        device = torch.device("cuda")
        hidden = torch.randn(
            2, 5, 32, device=device, dtype=torch.bfloat16, requires_grad=True
        )
        weight = torch.randn(777, 32, device=device, dtype=torch.bfloat16)
        targets = torch.randint(777, (2, 5), device=device)

        neg_log_q, topv, topi, argmax = _project_loss(hidden, weight, targets, 0)
        neg_log_q.sum().backward()

        reference = hidden.detach().clone().requires_grad_(True)
        logits = F.linear(reference, weight).float()
        expected = F.cross_entropy(
            logits.reshape(-1, 777), targets.reshape(-1), reduction="none"
        ).reshape(2, 5)
        expected.sum().backward()
        self.assertEqual(tuple(topv.shape), (2, 5, 0))
        self.assertEqual(tuple(topi.shape), (2, 5, 0))
        torch.testing.assert_close(neg_log_q, expected, rtol=1e-5, atol=1e-5)
        self.assertTrue(torch.equal(argmax, logits.argmax(dim=-1)))
        torch.testing.assert_close(hidden.grad, reference.grad, rtol=1e-2, atol=1e-3)

    def test_topk_matches_fp32_topk_at_full_vocabulary(self):
        """Random and peaked BF16 rows over the 248k Qwen3.8 vocabulary."""
        from specforge.core._fused_logits_loss_triton import _launch_stats

        torch.manual_seed(5)
        device = torch.device("cuda")
        rows, vocab_size, top_k = 96, 248320, 16
        random_logits = 1.5 * torch.randn(rows, vocab_size, device=device)
        # A few dominant tokens over a long flat tail, as a trained head emits.
        peaked_logits = 2.0 * torch.randn(rows, vocab_size, device=device)
        hot = torch.randint(vocab_size, (rows, 8), device=device)
        peaked_logits.scatter_(1, hot, 15.0 + 5 * torch.rand(rows, 8, device=device))
        for name, logits in (("random", random_logits), ("peaked", peaked_logits)):
            with self.subTest(logits=name):
                logits = logits.to(torch.bfloat16)
                targets = torch.randint(vocab_size, (rows,), device=device)
                lse = torch.empty(rows, device=device)
                target_logit = torch.empty_like(lse)
                argmax = torch.empty(rows, device=device, dtype=torch.long)
                topv = torch.empty(rows, top_k, device=device)
                topi = torch.empty(rows, top_k, device=device, dtype=torch.long)

                _launch_stats(logits, targets, lse, target_logit, argmax, topv, topi)

                reference = logits.float()
                torch.testing.assert_close(
                    lse, torch.logsumexp(reference, dim=-1), rtol=1e-6, atol=1e-5
                )
                self.assertTrue(torch.equal(argmax, reference.argmax(dim=-1)))
                self.assertTrue(
                    torch.equal(
                        target_logit, reference.gather(1, targets[:, None])[:, 0]
                    )
                )
                _assert_same_topk(self, topv, topi, reference, top_k)


def _assert_same_topk(test, values, ids, reference_logits, top_k):
    """Assert ``torch.topk``'s values and candidate sets, lowest index on ties.

    The fused kernel orders equal values by ascending index; ``torch.topk``'s
    sort leaves equal values in an unspecified order, so ids compare as sets.
    """
    reference_values, reference_ids = reference_logits.topk(top_k, dim=-1)
    test.assertTrue(torch.equal(values, reference_values))
    test.assertTrue(
        torch.equal(ids.sort(dim=-1).values, reference_ids.sort(dim=-1).values)
    )
    test.assertTrue(torch.equal(reference_logits.gather(-1, ids), values))
    equal_neighbors = values[:, 1:] == values[:, :-1]
    test.assertTrue(
        torch.all(ids[:, 1:][equal_neighbors] > ids[:, :-1][equal_neighbors])
    )


if __name__ == "__main__":
    unittest.main()
