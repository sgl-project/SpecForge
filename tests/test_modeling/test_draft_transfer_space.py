"""Transfer initialization preserves the legacy eigenspace and tensor ownership."""

import unittest

import torch
from torch import nn

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
                torch.testing.assert_close(codes, expected_codes, rtol=0, atol=0)
                for readout in readouts:
                    torch.testing.assert_close(
                        readout.weight, expected_basis, rtol=0, atol=0
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
