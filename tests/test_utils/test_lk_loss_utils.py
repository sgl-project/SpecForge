import unittest

import torch

import_error = None
try:
    from specforge.core.lk_loss import (
        combine_kl_and_lk_loss,
        compute_acceptance_rate,
        expected_acceptance_rate_torch,
        masked_mean,
    )
except ModuleNotFoundError as exc:
    import_error = exc
    combine_kl_and_lk_loss = None
    compute_acceptance_rate = None
    expected_acceptance_rate_torch = None
    masked_mean = None


@unittest.skipIf(
    import_error is not None,
    f"specforge import dependencies are missing: {import_error}",
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
        acceptance = expected_acceptance_rate_torch(target, draft)
        expected = torch.tensor([[0.9, 0.9]], dtype=torch.float32)
        torch.testing.assert_close(acceptance, expected)

    def test_expected_acceptance_rate_shape_mismatch(self):
        target = torch.rand(1, 2, 3)
        draft = torch.rand(1, 2, 4)
        with self.assertRaises(ValueError):
            _ = expected_acceptance_rate_torch(target, draft)

    def test_masked_mean_with_bool_mask(self):
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        mask = torch.tensor([True, False, True])
        output = masked_mean(values, mask)
        self.assertAlmostEqual(output.item(), 2.0, places=6)

    def test_masked_mean_with_float_mask_and_reduce(self):
        values = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        mask = torch.tensor([1.0, 0.0, 1.0], dtype=torch.float32)

        def reduce_fn(local_correct, local_denom):
            return local_correct * 2, local_denom * 2

        output = masked_mean(values, mask, reduce_fn=reduce_fn)
        self.assertAlmostEqual(output.item(), 2.0, places=6)

    def test_compute_acceptance_rate(self):
        logits = torch.tensor([[[2.0, 0.0], [0.0, 2.0]]], dtype=torch.float32)
        target_probs = torch.tensor(
            [[[0.8, 0.2], [0.1, 0.9]]],
            dtype=torch.float32,
        )
        position_mask = torch.ones((1, 2, 1), dtype=torch.bool)
        acceptance = compute_acceptance_rate(
            logits=logits,
            target_probs=target_probs,
            position_mask=position_mask,
        )
        self.assertGreaterEqual(acceptance.item(), 0.0)
        self.assertLessEqual(acceptance.item(), 1.0)

    def test_combine_kl_and_lk_loss_lambda(self):
        kl_loss = torch.tensor(1.2, dtype=torch.float32)
        acceptance_rate = torch.tensor(0.7, dtype=torch.float32)
        combined = combine_kl_and_lk_loss(
            kl_loss=kl_loss,
            acceptance_rate=acceptance_rate,
            lk_loss_type="lambda",
            kl_scale=1.0,
            kl_decay=1.0,
            lk_eps=1e-8,
        )
        self.assertTrue(torch.isfinite(combined))


if __name__ == "__main__":
    unittest.main(verbosity=2)
