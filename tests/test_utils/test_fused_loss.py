"""Fused Triton losses vs their eager references (specforge.core.fused_loss).

One test per loss, comparing loss value and logits-gradient against eager in
the three regimes that catch distinct bugs: fp32 with saturated point-mass
rows (gradient formula, log-space underflow), bf16 (the training dtype), and
the 151936-wide vocab (multi-block streaming, non-power-of-2 tail). The
upstream gradient contains exact zeros, so masked rows take the backward
kernel's early-out and must return exact zeros from an uninitialized buffer.

The accelerator is auto-detected: CUDA when present, otherwise Ascend NPU
(via triton-ascend). Without Triton or an accelerator the fused legs skip
gracefully, while the device-cap and eager legs still run on CPU. The
151936-wide leg also exercises the smaller NPU BLOCK_SIZE cap
(MAX_FUSED_SIZE_NPU = 4096), which forces the tighter multi-block streaming
loop.
"""

import math
import unittest
from types import SimpleNamespace

import torch

from specforge.core import fused_loss


def _triton_available() -> bool:
    """True only for a real Triton install; a stray ``triton/`` namespace
    directory on sys.path (no ``jit``, no ``language``) must not count."""
    try:
        import triton
        import triton.language  # noqa: F401
    except ImportError:
        return False
    return hasattr(triton, "jit")


def _npu_available() -> bool:
    # torch.npu is only bound once torch_npu has been imported; nothing on the
    # test collection path imports it, so probe explicitly here.
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    npu = getattr(torch, "npu", None)
    return npu is not None and npu.is_available()


def _accelerator_device() -> str | None:
    """Pick the accelerator the fused kernels can run on, or None to skip."""
    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu"
    return None


DEVICE = _accelerator_device()
requires_fused = unittest.skipUnless(
    DEVICE is not None and _triton_available(),
    "fused Triton losses require Triton plus a CUDA or Ascend NPU accelerator",
)

# (name, eager fn name, fused fn name); both resolved lazily from the module
CASES = [
    ("kl_div", "kl_div_loss", "fused_kl_div_loss"),
    ("rkl", "reverse_kl_div_loss", "fused_reverse_kl_div_loss"),
    ("jsd", "js_div_loss", "fused_js_div_loss"),
    ("ce", "ce_loss", "fused_ce_loss"),
    ("tv", "tv_loss", "fused_tv_loss"),
    ("nla", "neg_log_acceptance_loss", "fused_nla_loss"),
    ("lk_hybrid", "lk_hybrid_loss", "fused_lk_hybrid_loss"),
]

# Loss values are fp32 on both paths; gradients allow bf16 1-ulp rounding
# (both paths quantize at the leaf). A wrong gradient formula errs by orders
# of magnitude more than either bound.
LOSS_TOL = {"atol": 1e-4, "rtol": 1e-3}
GRAD_TOL = {"atol": 1e-3, "rtol": 1e-2}
# eager ce computes cross_entropy in bf16, so its bf16 legs are bounded by
# eager's own rounding, not by the (fp32) fused kernel
CE_BF16_LOSS_TOL = {"atol": 0.2, "rtol": 1e-2}
CE_BF16_GRAD_TOL = {"atol": 1e-2, "rtol": 2e-2}


def _assert_fused_matches_eager(
    eager_fn, fused_fn, logits, targets, loss_tol, grad_tol
):
    le = logits.detach().clone().requires_grad_(True)
    lf = logits.detach().clone().requires_grad_(True)
    out_e = eager_fn(le, targets)
    out_f = fused_fn(lf, targets)
    assert out_f.dtype == torch.float32  # like the eager fp32 softmax (#788)
    torch.testing.assert_close(out_f, out_e.float(), **loss_tol)

    go = torch.randn_like(out_e, dtype=torch.float32)
    go[:, ::3] = 0.0  # rows taking the go == 0 early-out
    (out_e.float() * go).sum().backward()
    (out_f * go).sum().backward()
    assert le.grad is not None
    assert lf.grad is not None
    torch.testing.assert_close(lf.grad.float(), le.grad.float(), **grad_tol)


class TestFusedLoss(unittest.TestCase):
    @requires_fused
    def test_fused_matches_eager(self):
        """Fused == eager (value + gradient) across the three failure-mode regimes."""
        for name, eager_name, fused_name in CASES:
            with self.subTest(name=name):
                eager_fn = getattr(fused_loss, eager_name)
                fused_fn = getattr(fused_loss, fused_name)

                # fp32, with saturated +-30 point-mass rows (one agreeing, one disagreeing)
                torch.manual_seed(0)
                logits = torch.randn(1, 32, 512, device=DEVICE) * 3
                targets = torch.randn(1, 32, 512, device=DEVICE) * 3
                logits[0, -2:] = -30.0
                targets[0, -2:] = -30.0
                logits[0, -2:, 0] = 30.0
                targets[0, -2, 0] = 30.0  # last two rows: draft==target, then disagree
                targets[0, -1, 7] = 30.0
                _assert_fused_matches_eager(
                    eager_fn, fused_fn, logits, targets, LOSS_TOL, GRAD_TOL
                )

                # bf16, the training dtype
                torch.manual_seed(1)
                logits = (torch.randn(1, 64, 512, device=DEVICE) * 3).bfloat16()
                targets = (torch.randn(1, 64, 512, device=DEVICE) * 3).bfloat16()
                _assert_fused_matches_eager(
                    eager_fn,
                    fused_fn,
                    logits,
                    targets,
                    CE_BF16_LOSS_TOL if name == "ce" else LOSS_TOL,
                    CE_BF16_GRAD_TOL if name == "ce" else GRAD_TOL,
                )

                # Qwen3's 151936 vocab exceeds MAX_FUSED_SIZE (and
                # MAX_FUSED_SIZE_NPU): multi-block streaming plus a
                # non-power-of-2 masked tail. On NPU the per-block cap is 4096,
                # so this leg also covers the tighter block loop.
                torch.manual_seed(2)
                logits = torch.randn(1, 3, 151936, device=DEVICE) * 3
                targets = torch.randn(1, 3, 151936, device=DEVICE) * 3
                _assert_fused_matches_eager(
                    eager_fn, fused_fn, logits, targets, LOSS_TOL, GRAD_TOL
                )

    @requires_fused
    def test_compiles_fullgraph(self):
        """torch.compile(fullgraph=True) must trace the fused losses.

        The OP selector crosses the autograd.Function boundary, and Dynamo cannot
        represent a tl.constexpr object there -- passing one graph-breaks (or
        fails under fullgraph). Model forwards are wrapped in torch.compile, so
        this guards the compiled training path.
        """
        for name, _eager_name, fused_name in CASES:
            with self.subTest(name=name):
                fused_fn = getattr(fused_loss, fused_name)
                logits = torch.randn(1, 8, 512, device=DEVICE, requires_grad=True)
                targets = torch.randn(1, 8, 512, device=DEVICE)

                # Eager warmup first: the lazy Triton assembly in
                # _require_triton() is not Dynamo-traceable, so it must happen
                # before the compiled region runs.
                fused_fn(logits.detach(), targets)

                compiled = torch.compile(
                    lambda a, b: fused_fn(a, b).sum(), fullgraph=True
                )
                compiled(logits, targets).backward()
                assert logits.grad is not None
                assert torch.isfinite(logits.grad).all()

    def test_calculate_settings_respects_device_cap(self):
        """`_calculate_settings` picks the right BLOCK_SIZE for each device without
        needing NPU or CUDA hardware -- the helper only reads ``device.type`` and
        is intentionally Triton-free so this test runs on CPU-only hosts.

        Exercises the NPU cap (MAX_FUSED_SIZE_NPU = 4096) and the CUDA cap
        (MAX_FUSED_SIZE = 131072) at vocab sizes that span the boundaries, so
        upstream CI (which typically has no NPU) still covers the NPU branch.
        """
        npu_cases = (
            (512, 512),
            (4096, 4096),
            (8192, 4096),
            (32768, 4096),
            (131072, 4096),
            (151936, 4096),
        )
        for vocab, expected in npu_cases:
            block, _ = fused_loss._calculate_settings(
                vocab, SimpleNamespace(type="npu")
            )
            assert (
                block == expected
            ), f"NPU cap: vocab={vocab} -> BLOCK_SIZE={block}, expected {expected}"

        cuda_cases = (
            (512, 512),
            (4096, 4096),
            (8192, 8192),
            (131072, 131072),
            (151936, 131072),
        )
        for vocab, expected in cuda_cases:
            block, _ = fused_loss._calculate_settings(
                vocab, SimpleNamespace(type="cuda")
            )
            assert (
                block == expected
            ), f"CUDA cap: vocab={vocab} -> BLOCK_SIZE={block}, expected {expected}"

    def test_eager_implementation_supports_differentiable_targets(self):
        """The explicit eager implementation preserves target gradients."""
        torch.manual_seed(3)
        for eager_name in (
            "kl_div_loss",
            "reverse_kl_div_loss",
            "js_div_loss",
            "tv_loss",
            "neg_log_acceptance_loss",
            "lk_hybrid_loss",
        ):
            with self.subTest(eager_name=eager_name):
                logits = torch.randn(1, 4, 64, requires_grad=True)
                targets = torch.randn(1, 4, 64, requires_grad=True)
                loss_fn = getattr(fused_loss, eager_name)
                loss_fn(logits, targets).sum().backward()
                assert logits.grad is not None, eager_name
                assert targets.grad is not None, eager_name

    def test_eager_losses_self_consistent(self):
        """CPU-checkable identities among the eager references, including a
        saturated point-mass row mirroring the fused parity regimes."""
        torch.manual_seed(4)
        logits = torch.randn(1, 8, 128) * 3
        targets = torch.randn(1, 8, 128) * 3
        logits[0, -1] = -30.0
        targets[0, -1] = -30.0
        logits[0, -1, 0] = 30.0
        targets[0, -1, 7] = 30.0  # disagreeing point masses

        kl = fused_loss.kl_div_loss(logits, targets)
        rkl = fused_loss.reverse_kl_div_loss(logits, targets)
        jsd = fused_loss.js_div_loss(logits, targets)
        tv = fused_loss.tv_loss(logits, targets)
        nla = fused_loss.neg_log_acceptance_loss(logits, targets)
        hybrid = fused_loss.lk_hybrid_loss(logits, targets)

        assert torch.isfinite(kl).all() and (kl >= -1e-6).all()
        assert torch.isfinite(rkl).all() and (rkl >= -1e-6).all()
        assert (jsd >= -1e-6).all() and (jsd <= math.log(2.0) + 1e-6).all()
        assert (tv >= 0.0).all() and (tv <= 1.0).all()
        # nla = -log(alpha) with alpha = 1 - tv; hybrid blends kl/tv with the
        # detached overlap weight exp(-eta * alpha)
        torch.testing.assert_close(nla, -torch.log((1.0 - tv).clamp_min(1e-5)))
        weight = torch.exp(-3.0 * (1.0 - tv))
        torch.testing.assert_close(hybrid, weight * kl + (1.0 - weight) * tv)

        # ce uses argmax(targets) as hard labels
        ce = fused_loss.ce_loss(logits, targets)
        ref = torch.nn.functional.cross_entropy(
            logits.reshape(-1, 128),
            targets.argmax(dim=-1).reshape(-1),
            reduction="none",
        ).reshape(1, 8)
        torch.testing.assert_close(ce, ref)


if __name__ == "__main__":
    unittest.main()
