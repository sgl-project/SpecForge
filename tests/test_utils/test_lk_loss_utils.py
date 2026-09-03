import unittest

import torch

from specforge.core.lk_loss import (
    compute_acceptance_rate,
    compute_lk_loss,
    compute_lk_loss_per_token,
    expected_acceptance_rate,
)


class TestLKLossUtils(unittest.TestCase):

    def test_expected_acceptance_rate(self):
        target = torch.tensor(
            [[[0.7, 0.3], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        draft = torch.tensor(
            [[[0.6, 0.4], [0.2, 0.8]]],
            dtype=torch.float32,
        )
        acceptance = expected_acceptance_rate(target, draft)
        expected = torch.tensor([[0.9, 0.9]], dtype=torch.float32)
        torch.testing.assert_close(acceptance, expected)

    def test_expected_acceptance_rate_shape_mismatch(self):
        target = torch.rand(1, 2, 3)
        draft = torch.rand(1, 2, 4)
        with self.assertRaises(ValueError):
            _ = expected_acceptance_rate(target, draft)

    def test_compute_acceptance_rate(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        target_probs = torch.tensor(
            [[[0.8, 0.2], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        position_mask = torch.ones((1, 2, 1), dtype=torch.bool)
        acceptance, log_acceptance = compute_acceptance_rate(
            logits=logits,
            target_probs=target_probs,
            position_mask=position_mask,
        )
        self.assertGreaterEqual(acceptance.item(), 0.0)
        self.assertLessEqual(acceptance.item(), 1.0)
        self.assertTrue(torch.isfinite(log_acceptance))

    def test_compute_acceptance_rate_with_reduce(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        target_probs = torch.tensor(
            [[[0.8, 0.2], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        position_mask = torch.ones((1, 2, 1), dtype=torch.bool)

        def reduce_fn(local_correct, local_denom):
            return local_correct * 2, local_denom * 2

        acceptance, log_acceptance = compute_acceptance_rate(
            logits=logits,
            target_probs=target_probs,
            position_mask=position_mask,
            reduce_fn=reduce_fn,
        )
        self.assertGreaterEqual(acceptance.item(), 0.0)
        self.assertLessEqual(acceptance.item(), 1.0)
        self.assertTrue(torch.isfinite(log_acceptance))

    def test_compute_acceptance_rate_log_before_mean(self):
        logits = torch.tensor(
            [[[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]]], dtype=torch.float32
        )
        target_probs = torch.tensor(
            [[[0.8, 0.2], [0.1, 0.9], [0.5, 0.5]]],
            dtype=torch.float32,
        )
        position_mask = torch.tensor([[[1], [1], [0]]], dtype=torch.bool)

        acceptance_rate, log_acceptance_rate = compute_acceptance_rate(
            logits=logits,
            target_probs=target_probs,
            position_mask=position_mask,
        )

        draft_p = torch.softmax(logits, dim=-1)
        acc_per_tok = torch.minimum(target_probs, draft_p).sum(dim=-1)
        mask = position_mask.squeeze(-1).float()

        expected_acc = (acc_per_tok * mask).sum() / mask.sum()
        expected_log_acc = (
            torch.where(acc_per_tok > 0, torch.log(acc_per_tok), 0.0) * mask
        ).sum() / mask.sum()

        torch.testing.assert_close(acceptance_rate, expected_acc)
        torch.testing.assert_close(log_acceptance_rate, expected_log_acc)

    def test_compute_lk_loss_lambda(self):
        kl_loss = torch.tensor(1.2, dtype=torch.float32)
        acceptance_rate = torch.tensor(0.7, dtype=torch.float32)
        log_acceptance_rate = torch.log(acceptance_rate)
        combined = compute_lk_loss(
            kl_loss=kl_loss,
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=log_acceptance_rate,
            lk_loss_type="lambda",
            kl_scale=1.0,
            kl_decay=1.0,
        )
        self.assertTrue(torch.isfinite(combined))

    def test_compute_lk_loss_alpha(self):
        acceptance_rate = torch.tensor(0.7, dtype=torch.float32)
        log_acceptance_rate = torch.log(acceptance_rate)
        loss = compute_lk_loss(
            kl_loss=torch.tensor(1.2, dtype=torch.float32),
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=log_acceptance_rate,
            lk_loss_type="alpha",
            kl_scale=1.0,
            kl_decay=1.0,
        )
        torch.testing.assert_close(loss, -log_acceptance_rate)

    def test_compute_lk_loss_tv(self):
        acceptance_rate = torch.tensor(0.7, dtype=torch.float32)
        loss = compute_lk_loss(
            kl_loss=torch.tensor(1.2, dtype=torch.float32),
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=torch.log(acceptance_rate),
            lk_loss_type="tv",
            kl_scale=1.0,
            kl_decay=1.0,
        )
        torch.testing.assert_close(loss, 1.0 - acceptance_rate)

    def test_compute_lk_loss_unknown_type_raises(self):
        with self.assertRaises(ValueError):
            _ = compute_lk_loss(
                kl_loss=torch.tensor(1.2, dtype=torch.float32),
                acceptance_rate=torch.tensor(0.7, dtype=torch.float32),
                log_acceptance_rate=torch.tensor(-0.3, dtype=torch.float32),
                lk_loss_type="invalid",
                kl_scale=1.0,
                kl_decay=1.0,
            )

    def test_compute_lk_loss_per_token_matches_angelspec_lambda(self):
        logits = torch.tensor(
            [[[2.0, -1.0, 0.5], [-0.5, 1.5, 0.0]]],
            dtype=torch.float32,
        )
        target_probs = torch.tensor(
            [[[0.2, 0.7, 0.1], [0.6, 0.1, 0.3]]],
            dtype=torch.float32,
        )

        loss, acceptance, kl_weight = compute_lk_loss_per_token(
            logits=logits,
            target_probs=target_probs,
            lk_loss_type="lambda",
            kl_scale=1.0,
            kl_decay=3.0,
        )

        draft_log_probs = torch.log_softmax(logits, dim=-1)
        draft_probs = draft_log_probs.exp()
        tv = 0.5 * (target_probs - draft_probs).abs().sum(dim=-1)
        expected_acceptance = 1.0 - tv
        expected_kl = (target_probs * (target_probs.log() - draft_log_probs)).sum(
            dim=-1
        )
        expected_weight = torch.exp(-3.0 * expected_acceptance)
        expected_loss = expected_weight * expected_kl + (1.0 - expected_weight) * tv
        torch.testing.assert_close(acceptance, expected_acceptance)
        torch.testing.assert_close(kl_weight, expected_weight)
        torch.testing.assert_close(loss, expected_loss)

    def test_compute_lk_loss_per_token_alpha_and_tv(self):
        logits = torch.tensor([[0.0, 1.0]], dtype=torch.float32)
        target_probs = torch.tensor([[0.75, 0.25]], dtype=torch.float32)
        acceptance = expected_acceptance_rate(
            target_probs,
            torch.softmax(logits, dim=-1),
        )

        alpha, alpha_acceptance, alpha_weight = compute_lk_loss_per_token(
            logits=logits,
            target_probs=target_probs,
            lk_loss_type="alpha",
            kl_scale=1.0,
            kl_decay=1.0,
        )
        tv, tv_acceptance, tv_weight = compute_lk_loss_per_token(
            logits=logits,
            target_probs=target_probs,
            lk_loss_type="tv",
            kl_scale=1.0,
            kl_decay=1.0,
        )

        torch.testing.assert_close(alpha, -acceptance.log())
        torch.testing.assert_close(tv, 1.0 - acceptance)
        torch.testing.assert_close(alpha_acceptance, acceptance)
        torch.testing.assert_close(tv_acceptance, acceptance)
        self.assertIsNone(alpha_weight)
        self.assertIsNone(tv_weight)

    def test_compute_lk_loss_per_token_detaches_teacher_distribution(self):
        logits = torch.tensor([[0.0, 1.0]], requires_grad=True)
        teacher_logits = torch.tensor([[1.0, 0.0]], requires_grad=True)

        loss, _acceptance, _weight = compute_lk_loss_per_token(
            logits=logits,
            target_probs=torch.softmax(teacher_logits, dim=-1),
            lk_loss_type="lambda",
            kl_scale=1.0,
            kl_decay=3.0,
        )
        loss.sum().backward()

        self.assertIsNotNone(logits.grad)
        self.assertIsNone(teacher_logits.grad)


if __name__ == "__main__":
    unittest.main(verbosity=2)
