"""Fused-kernel routing of the shared teacher TV/L1 distance."""

import os
import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from specforge.core import fused_loss, token_objective
from specforge.core.token_objective import (
    FUSED_TEACHER_TV_ENV,
    LEGACY_FUSED_TEACHER_TV_ENV,
    _use_fused_teacher_tv,
    teacher_tv_distance,
)


def _triton_available() -> bool:
    try:
        import triton
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return hasattr(triton, "jit")


def _eager_tv(draft_logits, teacher_logits):
    return 0.5 * (
        torch.softmax(draft_logits.float(), -1)
        - torch.softmax(teacher_logits.float(), -1)
    ).abs().sum(-1)


class FusedTeacherTVRoutingTest(unittest.TestCase):
    """CPU: the default route depends on dtype and device; the env overrides it."""

    def test_default_route(self):
        for dtype, is_cuda, fused in (
            (torch.bfloat16, True, True),
            (torch.bfloat16, False, True),
            (torch.float32, True, True),  # DFlash2's FP32 unary logits
            (torch.float32, False, False),
            (torch.float16, True, False),
        ):
            with (
                self.subTest(dtype=dtype, is_cuda=is_cuda),
                mock.patch.dict(os.environ),
            ):
                os.environ.pop(FUSED_TEACHER_TV_ENV, None)
                os.environ.pop(LEGACY_FUSED_TEACHER_TV_ENV, None)
                logits = SimpleNamespace(dtype=dtype, is_cuda=is_cuda)
                self.assertEqual(fused, _use_fused_teacher_tv(logits))

    def test_env_overrides_the_default(self):
        logits = SimpleNamespace(dtype=torch.float32, is_cuda=True)
        for env, value, fused in (
            (FUSED_TEACHER_TV_ENV, "0", False),
            (FUSED_TEACHER_TV_ENV, "1", True),
            (LEGACY_FUSED_TEACHER_TV_ENV, "0", False),
        ):
            with (
                self.subTest(env=env, value=value),
                mock.patch.dict(os.environ, {env: value}),
            ):
                os.environ.pop(
                    (
                        LEGACY_FUSED_TEACHER_TV_ENV
                        if env == FUSED_TEACHER_TV_ENV
                        else FUSED_TEACHER_TV_ENV
                    ),
                    None,
                )
                self.assertEqual(fused, _use_fused_teacher_tv(logits))


@unittest.skipUnless(
    torch.cuda.is_available() and _triton_available(), "requires CUDA and Triton"
)
class FusedTeacherTVKernelTest(unittest.TestCase):
    """CUDA: BF16 and FP32 logits take the kernel by default and match eager."""

    def test_default_kernel_route_matches_eager(self):
        for dtype in (torch.bfloat16, torch.float32):
            with (
                self.subTest(dtype=dtype),
                mock.patch.dict(os.environ),
                mock.patch.object(
                    fused_loss, "fused_tv_loss", wraps=fused_loss.fused_tv_loss
                ) as kernel,
            ):
                os.environ.pop(FUSED_TEACHER_TV_ENV, None)
                os.environ.pop(LEGACY_FUSED_TEACHER_TV_ENV, None)
                torch.manual_seed(0)
                # A non-power-of-two vocabulary exercises the kernel's masked tail.
                draft = (torch.randn(2, 3, 4099, device="cuda") * 3).to(dtype)
                draft.requires_grad_()
                teacher = (torch.randn(2, 3, 4099, device="cuda") * 3).to(dtype)

                tv, l1 = teacher_tv_distance(draft, teacher)
                kernel.assert_called_once()
                self.assertEqual(tv.shape, (2, 3))
                torch.testing.assert_close(l1, 2.0 * tv)

                reference = draft.detach().float().requires_grad_()
                expected = _eager_tv(reference, teacher)
                torch.testing.assert_close(tv.float(), expected, atol=1e-4, rtol=1e-3)
                grad_output = torch.randn(2, 3, device="cuda")
                tv.backward(grad_output)
                expected.backward(grad_output)
                torch.testing.assert_close(
                    draft.grad.float(), reference.grad, atol=1e-3, rtol=1e-2
                )


if __name__ == "__main__":
    unittest.main()
