"""Transfer initialization preserves the legacy eigenspace and tensor ownership."""

import unittest

import torch
from torch import nn
from torch.nn import functional as F

from specforge.modeling.draft.heads.transfer_space import initialize_transfer_space


class TransferSpaceTest(unittest.TestCase):
    def _check(self, device, dtype):
        # Centered, normalized rows yield diag(6, 4, 2); the leading axes are exact.
        rows = torch.eye(3, device=device).repeat_interleave(
            torch.tensor([3, 2, 1], device=device), dim=0
        )
        head = torch.cat((rows, -rows)).to(dtype).requires_grad_()
        original = head.detach().clone()
        expected_basis = torch.tensor(
            [[0, 1, 0], [1, 0, 0]], device=device, dtype=dtype
        )
        expected_codes = head.detach() @ expected_basis.T
        for chunk in (1, 5, 32):
            with self.subTest(device=device, dtype=dtype, chunk=chunk):
                codes = torch.empty(12, 2, device=device, dtype=dtype)
                readouts = [
                    nn.Linear(3, 2, bias=False, device=device, dtype=dtype)
                    for _ in range(2)
                ]
                identities = [id(codes), *(id(r.weight) for r in readouts)]
                initialize_transfer_space(head, codes, readouts, chunk_size=chunk)
                signs = (readouts[0].weight * expected_basis).sum(dim=-1).sign()
                torch.testing.assert_close(
                    codes * signs, expected_codes, rtol=0, atol=0
                )
                for readout in readouts:
                    torch.testing.assert_close(
                        readout.weight * signs[:, None], expected_basis, rtol=0, atol=0
                    )
                    self.assertTrue(readout.weight.requires_grad)
                self.assertEqual(
                    identities, [id(codes), *(id(r.weight) for r in readouts)]
                )
                self.assertFalse(codes.requires_grad)
                torch.testing.assert_close(head, original, rtol=0, atol=0)
                self.assertIsNone(head.grad)

    def test_cpu_fp32_and_bf16(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._check("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_placement_and_low_precision(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._check("cuda", dtype)

    def test_nontrivial_centering_whitening_and_row_normalization(self):
        torch.manual_seed(17)
        head = torch.randn(19, 5) * torch.arange(1, 6) + 2
        # Independent dense form of main's transfer math (118e4e39).
        rows = F.normalize(head, dim=-1)
        centered = rows - rows.mean(dim=0)
        values, vectors = torch.linalg.eigh(centered.T @ centered)
        basis = vectors[:, -3:]
        expected = F.normalize(centered @ basis / values[-3:].sqrt(), dim=-1)
        for chunk in (1, 7, 32):
            with self.subTest(chunk=chunk):
                codes = torch.empty(19, 3)
                readout = nn.Linear(5, 3, bias=False)
                initialize_transfer_space(head, codes, [readout], chunk_size=chunk)
                signs = (readout.weight * basis.T).sum(dim=-1).sign()
                torch.testing.assert_close(
                    readout.weight * signs[:, None], basis.T, atol=2e-5, rtol=2e-5
                )
                torch.testing.assert_close(
                    codes * signs, expected, atol=2e-5, rtol=2e-5
                )

    def test_invalid_sources_do_not_modify_destinations(self):
        codes = torch.full((12, 2), 7.0)
        readouts = [nn.Linear(3, 2, bias=False)]
        original = readouts[0].weight.detach().clone()
        for head, chunk in (
            (torch.zeros(12, 3), 5),
            (torch.full((12, 3), float("nan")), 5),
            (torch.randn(12, 3), 0),
            (torch.randn(12, 3), True),
            (torch.randn(12, 4), 5),
            (torch.randn(11, 3), 5),
        ):
            with self.subTest(shape=head.shape, chunk=chunk):
                with self.assertRaises(ValueError):
                    initialize_transfer_space(head, codes, readouts, chunk_size=chunk)
                torch.testing.assert_close(
                    codes, torch.full_like(codes, 7), rtol=0, atol=0
                )
                torch.testing.assert_close(readouts[0].weight, original, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
