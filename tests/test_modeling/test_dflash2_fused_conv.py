"""Grouped convolution preserves outputs and gradients across execution choices."""

import copy
import unittest
from unittest.mock import patch

import torch

from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.modeling.draft.layers.conv import GroupedCausalConv
from specforge.modeling.draft.layers.kernels import configure_kernels


class FusedConvTest(unittest.TestCase):
    def make_conv(self, device="cpu", dtype=torch.float32):
        conv = GroupedCausalConv(
            GroupedCausalConvSpec(kernel_size=3, group_size=8),
            hidden_size=32,
            block_size=4,
        ).to(device=device, dtype=dtype)
        with torch.no_grad():
            conv.base_kernel.normal_()
            conv.kernel_projection.weight.normal_(std=0.1)
        return conv

    def test_disabled_is_lazy_and_cpu_explicit_fusion_fails(self):
        conv = self.make_conv()
        with patch.dict("sys.modules", {"triton": None}):
            configure_kernels(conv)
            hidden, kernel = conv.prepare(torch.randn(2, 8, 32))
            conv.finish(hidden, kernel).sum().backward()
        with self.assertRaisesRegex(ValueError, "CUDA/ROCm"):
            configure_kernels(conv, use_fused_conv=True)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_fused_outputs_and_gradients(self):
        for dtype in (torch.float32, torch.bfloat16):
            torch.manual_seed(3)
            eager = self.make_conv("cuda", dtype)
            fused = copy.deepcopy(eager)
            configure_kernels(fused, use_fused_conv=True)
            inputs = torch.randn(2, 8, 32, device="cuda", dtype=dtype)
            results = []
            from specforge.modeling.draft.layers.conv import triton

            with patch.object(
                triton,
                "dflash2_grouped_conv_fused",
                wraps=triton.dflash2_grouped_conv_fused,
            ) as kernel:
                for conv in (eager, fused):
                    x = inputs.detach().clone().requires_grad_()
                    hidden, coefficients = conv.prepare(x)
                    output = conv.finish(hidden.tanh(), coefficients)
                    output.float().square().mean().backward()
                    results.append(
                        (output, x.grad, [p.grad for p in conv.parameters()])
                    )
                self.assertGreater(kernel.call_count, 0)
            tolerance = 0.03 if dtype == torch.bfloat16 else 3e-5
            torch.testing.assert_close(
                results[0], results[1], atol=tolerance, rtol=tolerance
            )
