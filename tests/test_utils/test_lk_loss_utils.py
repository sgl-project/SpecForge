import unittest

import torch

import_error = None
try:
    from specforge.core.lk_loss import expected_acceptance_rate_torch, masked_mean
except ModuleNotFoundError as exc:
    import_error = exc
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
