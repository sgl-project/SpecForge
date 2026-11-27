"""Ring flash attention backward dtype and FP32 dq accumulation tests."""

import importlib
import math
import sys
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.modeling.draft.layers.attention.backends.cached_flash import (
    _USPRingFlashCachedMergeFunc,
)


class _FakeRingComm:
    def __init__(self, process_group, *, rank=0, world_size=1):
        self.rank = rank
        self.world_size = world_size

    def send_recv(self, to_send, recv_tensor=None):
        return to_send.clone()

    def commit(self):
        pass

    def wait(self):
        pass


class RingFlashAttnBackwardDtypeTest(unittest.TestCase):
    def _load_module(self):
        try:
            return importlib.import_module("specforge.layers.ring.ring_flash_attn")
        except ImportError:
            fake_kernels = types.ModuleType("yunchang.kernels")
            fake_kernels.AttnType = SimpleNamespace(FA="FA", SPARSE_SAGE="SPARSE_SAGE")
            fake_kernels.select_flash_attn_impl = lambda *args, **kwargs: None
            fake_yunchang = types.ModuleType("yunchang")
            fake_yunchang.kernels = fake_kernels
            self.enterContext(
                patch.dict(
                    sys.modules,
                    {"yunchang": fake_yunchang, "yunchang.kernels": fake_kernels},
                )
            )
            self.addCleanup(
                sys.modules.pop, "specforge.layers.ring.ring_flash_attn", None
            )
            return importlib.import_module("specforge.layers.ring.ring_flash_attn")

    def test_backward_preserves_query_dtype_and_fp16_precision(self):
        ring_mod = self._load_module()
        value = 1.0 + 2**-10  # exact in fp16, rounds to 1.0 in bf16

        def fake_bwd(
            dout, q, k, v, out, softmax_lse, dq_buf, dk_buf, dv_buf, *args, **kwargs
        ):
            dq_buf.fill_(value)
            dk_buf.fill_(value)
            dv_buf.fill_(value)

        for dtype in (torch.float16, torch.bfloat16, torch.float32):
            with self.subTest(dtype=dtype):
                q = torch.ones(1, 2, 2, 4, dtype=dtype)
                k = torch.ones_like(q)
                v = torch.ones_like(q)
                out = torch.ones_like(q)
                dout = torch.ones_like(q)
                lse = torch.zeros(1, 2, 2, dtype=torch.float32)
                with (
                    patch.object(ring_mod, "RingComm", _FakeRingComm),
                    patch.object(
                        ring_mod, "select_flash_attn_impl", return_value=fake_bwd
                    ),
                ):
                    dq, dk, dv = ring_mod.ring_flash_attn_backward(
                        None, dout, q, k, v, out, lse, softmax_scale=0.5
                    )
                    dq_fp32, _, _ = ring_mod.ring_flash_attn_backward(
                        None,
                        dout,
                        q,
                        k,
                        v,
                        out,
                        lse,
                        softmax_scale=0.5,
                        return_fp32_dq=True,
                    )
                expected = torch.full_like(q, value)
                self.assertEqual(dq.dtype, dtype)
                self.assertEqual(dk.dtype, dtype)
                self.assertEqual(dv.dtype, dtype)
                self.assertTrue(torch.equal(dq, expected))
                self.assertTrue(torch.equal(dk, expected))
                self.assertTrue(torch.equal(dv, expected))
                self.assertEqual(dq_fp32.dtype, torch.float32)
                self.assertTrue(torch.equal(dq_fp32, expected.float()))

    def test_cached_usp_backward_keeps_fp32_dq_subtotal_to_avoid_fp16_overflow(self):
        ring_mod = self._load_module()

        def fake_two_rank_comm(process_group):
            return _FakeRingComm(process_group, rank=1, world_size=2)

        def fake_fwd(*args, **kwargs):
            # Two equal keys (k=2048, v=1) on ranks 0 and 1 with q=0: out=1, lse=log(2).
            out = torch.ones(1, 1, 1, 1, dtype=torch.float16)
            lse = torch.full((1, 1, 1), math.log(2.0), dtype=torch.float32)
            return out, lse

        def fake_bwd(
            dout, q, k, v, out, softmax_lse, dq_buf, dk_buf, dv_buf, *args, **kwargs
        ):
            # With merged_lse=log(3), out=0, v=1, dout=240, scale=0.25, k=2048:
            # each ring rank contributes dq = (1/3) * 240 * 1 * 0.25 * 2048 = 40960.
            dq_buf.fill_(40960.0)
            dk_buf.zero_()
            dv_buf.zero_()

        q = torch.zeros(1, 1, 1, 1, dtype=torch.float16, requires_grad=True)
        cache_k = torch.full((1, 2, 1, 1, 1), 2048.0, dtype=torch.float16)
        cache_v = torch.tensor([[[[[1.0]]], [[[-2.0]]]]], dtype=torch.float16)
        grad_out = torch.full((1, 1, 1, 1), 240.0, dtype=torch.float16)

        with (
            patch.object(ring_mod, "RingComm", fake_two_rank_comm),
            patch.object(ring_mod, "ring_flash_attn_forward", side_effect=fake_fwd),
            patch.object(ring_mod, "select_flash_attn_impl", return_value=fake_bwd),
        ):
            out = _USPRingFlashCachedMergeFunc.apply(q, cache_k, cache_v, None, 0.25)
            out.backward(grad_out)

        self.assertEqual(q.grad.dtype, torch.float16)
        self.assertTrue(torch.isfinite(q.grad).all())
        torch.testing.assert_close(q.grad, torch.zeros_like(q.grad), rtol=0, atol=1e-2)


if __name__ == "__main__":
    unittest.main()
