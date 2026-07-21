import unittest

import torch
import torch.nn.functional as F

from specforge.core.domino_loss import domino_weighted_cross_entropy


def _reference(base, correction, targets, weights, block_size, suffix_start):
    """Materialize corrected logits using the original PyTorch path."""
    num_blocks = base.shape[0] // block_size
    base_3d = base.reshape(num_blocks, block_size, -1)
    suffix_size = block_size - suffix_start
    final = torch.cat(
        (
            base_3d[:, :suffix_start],
            base_3d[:, suffix_start:]
            + correction.reshape(num_blocks, suffix_size, -1),
        ),
        dim=1,
    ).reshape_as(base)
    final_loss_sum = (
        F.cross_entropy(final, targets, reduction="none") * weights
    ).sum()
    base_loss_sum = (
        F.cross_entropy(base, targets, reduction="none") * weights
    ).sum()
    return (
        final_loss_sum,
        base_loss_sum,
        final.argmax(dim=-1),
        base.argmax(dim=-1),
    )


class DominoCrossEntropyTest(unittest.TestCase):
    def _compare(
        self, device, *, block_size=4, suffix_start=1,
        num_blocks=3, vocab_size=19,
    ):
        torch.manual_seed(7)
        rows = num_blocks * block_size
        correction_rows = num_blocks * (block_size - suffix_start)
        targets = torch.randint(vocab_size, (rows,), device=device)
        weights = torch.rand(rows, device=device)
        weights[::5] = 0

        actual_base = torch.randn(rows, vocab_size, device=device, requires_grad=True)
        actual_correction = torch.randn(
            correction_rows, vocab_size, device=device, requires_grad=True
        )
        expected_base = actual_base.detach().clone().requires_grad_(True)
        expected_correction = actual_correction.detach().clone().requires_grad_(True)

        actual = domino_weighted_cross_entropy(
            actual_base, actual_correction, targets, weights,
            block_size, suffix_start,
            use_fused=device.type == "cuda",
        )
        expected = _reference(
            expected_base, expected_correction, targets, weights,
            block_size, suffix_start,
        )
        torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[1], expected[1], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])

        (0.7 * actual[0] + 0.3 * actual[1]).backward()
        (0.7 * expected[0] + 0.3 * expected[1]).backward()
        torch.testing.assert_close(
            actual_base.grad, expected_base.grad, rtol=1e-4, atol=1e-4
        )
        torch.testing.assert_close(
            actual_correction.grad,
            expected_correction.grad,
            rtol=1e-4,
            atol=1e-4,
        )

    def test_cpu_fallback_matches_pytorch(self):
        self._compare(torch.device("cpu"))

    def test_invalid_correction_shape_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "correction logits"):
            domino_weighted_cross_entropy(
                torch.randn(4, 8),
                torch.randn(4, 8),
                torch.zeros(4, dtype=torch.long),
                torch.ones(4),
                block_size=4,
                suffix_start=1,
                use_fused=False,
            )

    def test_fused_path_requires_cuda(self):
        with self.assertRaisesRegex(ValueError, "requires CUDA"):
            domino_weighted_cross_entropy(
                torch.randn(4, 8),
                torch.randn(3, 8),
                torch.zeros(4, dtype=torch.long),
                torch.ones(4),
                block_size=4,
                suffix_start=1,
                use_fused=True,
            )

    @unittest.skipUnless(torch.cuda.is_available(), "Triton loss requires CUDA")
    def test_triton_matches_pytorch(self):
        device = torch.device("cuda")
        self._compare(device)
        self._compare(
            device, block_size=4, suffix_start=0,
            num_blocks=2, vocab_size=2053,
        )
        self._compare(
            device, block_size=4, suffix_start=3,
            num_blocks=2, vocab_size=2053,
        )

        # Both reductions must preserve PyTorch's leftmost tie break.
        base = torch.zeros(8, 2053, device=device)
        correction = torch.zeros(6, 2053, device=device)
        targets = torch.arange(8, device=device)
        weights = torch.ones(8, device=device)
        actual = domino_weighted_cross_entropy(
            base,
            correction,
            targets,
            weights,
            block_size=4,
            suffix_start=1,
            use_fused=True,
        )
        torch.testing.assert_close(actual[2], torch.zeros_like(actual[2]))
        torch.testing.assert_close(actual[3], torch.zeros_like(actual[3]))


if __name__ == "__main__":
    unittest.main()
