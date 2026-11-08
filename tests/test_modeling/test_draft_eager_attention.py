"""The eager backend preserves grouped attention, masks, dropout, and gradients."""

import unittest
from itertools import product
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers.models.qwen3.modeling_qwen3 import eager_attention_forward

from specforge.modeling.draft.layers.attention.backends import eager_attention


class EagerAttentionTest(unittest.TestCase):
    def _assert_hf_parity(self, device, dtypes):
        for dtype in dtypes:
            for kv_heads in (4, 2, 1):
                for mask_kind in ("none", "boolean", "bias"):
                    for training in (False, True):
                        with self.subTest(
                            device=device,
                            dtype=dtype,
                            kv_heads=kv_heads,
                            mask=mask_kind,
                            training=training,
                        ):
                            # Noncontiguous head-major views, unequal Q/K lengths,
                            # and unequal query/value widths also cover MLA-shaped input.
                            inputs = [
                                torch.randn(
                                    2, length, heads, width, device=device, dtype=dtype
                                )
                                .transpose(1, 2)
                                .requires_grad_()
                                for heads, length, width in (
                                    (4, 3, 8),
                                    (kv_heads, 5, 8),
                                    (kv_heads, 5, 6),
                                )
                            ]
                            copies = [
                                value.detach().clone().requires_grad_()
                                for value in inputs
                            ]
                            mask = expected_mask = None
                            if mask_kind == "boolean":
                                mask = torch.rand(2, 1, 3, 5, device=device) > 0.5
                                mask[..., 0] = True
                                expected_mask = torch.zeros_like(
                                    mask, dtype=dtype
                                ).masked_fill(~mask, -torch.inf)
                            elif mask_kind == "bias":
                                mask = torch.randn(
                                    1, 4, 3, 5, device=device, requires_grad=True
                                )
                                expected_mask = mask.detach().clone().requires_grad_()
                            torch.manual_seed(11)
                            actual = eager_attention(
                                *inputs,
                                attention_mask=mask,
                                scale=0.3,
                                dropout_p=0.25,
                                training=training,
                            )
                            torch.manual_seed(11)
                            reference, _ = eager_attention_forward(
                                SimpleNamespace(
                                    num_key_value_groups=4 // kv_heads,
                                    training=training,
                                ),
                                *copies,
                                expected_mask,
                                scaling=0.3,
                                dropout=0.25,
                            )
                            reference = reference.transpose(1, 2)
                            torch.testing.assert_close(
                                actual, reference, rtol=0, atol=0
                            )
                            upstream = torch.randn_like(actual)
                            actual.backward(upstream)
                            reference.backward(upstream)
                            for left, right in zip(inputs, copies):
                                torch.testing.assert_close(
                                    left.grad, right.grad, rtol=0, atol=0
                                )
                            if mask_kind == "bias":
                                torch.testing.assert_close(
                                    mask.grad, expected_mask.grad, rtol=0, atol=0
                                )

    def test_cpu_hf_outputs_dropout_and_gradients(self):
        self._assert_hf_parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_hf_outputs_dropout_and_gradients(self):
        self._assert_hf_parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_fully_blocked_rows_match_sdpa_and_have_zero_gradients(self):
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for device, additive in product(devices, (False, True)):
            q, k, v = [
                torch.randn(1, 2, 3, 8, device=device, requires_grad=True)
                for _ in range(3)
            ]
            mask = torch.ones(1, 1, 3, 3, device=device, dtype=torch.bool)
            mask[..., 1, :] = False
            if additive:
                mask = torch.zeros_like(mask, dtype=torch.float32).masked_fill(
                    ~mask, -torch.inf
                )
            actual = eager_attention(
                q, k, v, attention_mask=mask, scale=0.3, dropout_p=0.0, training=True
            )
            with sdpa_kernel(SDPBackend.MATH):
                reference = F.scaled_dot_product_attention(
                    q, k, v, attn_mask=mask, scale=0.3
                )
            torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
            torch.testing.assert_close(
                actual[:, :, 1], torch.zeros_like(actual[:, :, 1]), rtol=0, atol=0
            )
            actual[:, :, 1].sum().backward()
            for value in (q, k, v):
                torch.testing.assert_close(
                    value.grad, torch.zeros_like(value), rtol=0, atol=0
                )

    def test_bidirectional_blocks_cannot_bypass_context_isolation(self):
        q = torch.randn(1, 4, 4, 8, requires_grad=True)
        k, v = [torch.randn(1, 2, 6, 8, requires_grad=True) for _ in range(2)]
        mask = torch.zeros(1, 1, 4, 6, dtype=torch.bool)
        mask[:, :, :2, [0, 2, 3]] = True
        mask[:, :, 2:, [0, 1, 4, 5]] = True
        kwargs = dict(attention_mask=mask, scale=0.3, dropout_p=0.0, training=False)
        actual = eager_attention(q, k, v, **kwargs)
        changed_k, changed_v = k.detach().clone(), v.detach().clone()
        changed_k[:, :, [1, 4, 5]] = 1000
        changed_v[:, :, [1, 4, 5]] = -1000
        changed = eager_attention(q, changed_k, changed_v, **kwargs)
        torch.testing.assert_close(actual[:, :, :2], changed[:, :, :2], rtol=0, atol=0)
        actual[:, :, :2].sum().backward()
        for value in (k, v):
            forbidden = value.grad[:, :, [1, 4, 5]]
            torch.testing.assert_close(
                forbidden, torch.zeros_like(forbidden), rtol=0, atol=0
            )

    def test_finite_negative_bias_is_not_a_hard_mask(self):
        q = k = torch.zeros(1, 1, 1, 2)
        value = torch.ones(1, 1, 1, 2)
        actual = eager_attention(
            q,
            k,
            value,
            attention_mask=torch.full((1, 1, 1, 1), -10000.0),
            scale=1.0,
            dropout_p=0.0,
            training=False,
        )
        torch.testing.assert_close(actual, value, rtol=0, atol=0)

    def test_empty_queries_or_keys_have_defined_outputs(self):
        for queries, keys in ((0, 3), (3, 0)):
            q = torch.randn(1, 2, queries, 4, requires_grad=True)
            k = torch.randn(1, 1, keys, 4, requires_grad=True)
            v = torch.randn(1, 1, keys, 6, requires_grad=True)
            output = eager_attention(
                q, k, v, attention_mask=None, scale=0.5, dropout_p=0.0, training=False
            )
            torch.testing.assert_close(
                output, torch.zeros(1, 2, queries, 6), rtol=0, atol=0
            )
            output.sum().backward()
            for value in (q, k, v):
                torch.testing.assert_close(
                    value.grad, torch.zeros_like(value), rtol=0, atol=0
                )

    def test_invalid_inputs_fail_before_attention_computation(self):
        q = torch.randn(2, 4, 3, 8)
        k = torch.randn(2, 2, 5, 8)
        v = torch.randn(2, 2, 5, 6)
        kwargs = dict(attention_mask=None, scale=0.3, dropout_p=0.0, training=False)
        with patch("torch.matmul") as compute:
            for changed in (
                {"scale": 0},
                {"scale": float("nan")},
                {"dropout_p": 1},
                {"training": 1},
                {"attention_mask": torch.zeros(3, 5)},
                {"attention_mask": torch.zeros(2, 3, 3, 5)},
                {"attention_mask": torch.zeros(2, 1, 3, 5, dtype=torch.long)},
                {"attention_mask": torch.zeros(2, 1, 3, 5, device="meta")},
            ):
                with self.assertRaises(ValueError):
                    eager_attention(q, k, v, **(kwargs | changed))
            with self.assertRaises(TypeError):
                eager_attention(q, k, v, **(kwargs | {"attention_mask": object()}))
            for inputs in (
                (q[0], k, v),
                (q, k[:, :1], v),
                (q[:, :3], k, v),
                (q, k[..., :4], v),
                (q, k.double(), v),
                (q[:1], k, v),
            ):
                with self.assertRaises(ValueError):
                    eager_attention(*inputs, **kwargs)
            compute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
