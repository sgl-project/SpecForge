"""Visibility must have the same meaning for every dense execution backend."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from torch.nn.attention.flex_attention import create_block_mask
from torch.testing import assert_close
from transformers.cache_utils import DynamicCache

from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers.config import AttentionConfig
from specforge.modeling.draft.layers.factory import build_attention
from tests.test_modeling.test_layers.test_composition import config, context


class VisibilityTest(unittest.TestCase):
    def make_attention(self, backend, *, causal=True, window=None, kind="gqa"):
        cfg = config(kind)
        cfg.attention_bias = True
        cfg._attn_implementation = backend
        return build_attention(
            cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind, causal, window)
        )

    def test_causal_prefix_and_cache_match_explicit_mask(self):
        for backend in ("eager", "sdpa"):
            for kind in ("gqa", "mla"):
                with self.subTest(backend=backend, kind=kind):
                    attention = self.make_attention(backend, kind=kind)
                    hidden = torch.randn(2, 3, 16)
                    prefix = torch.randn(2, 4, 16)
                    ctx = replace(context(hidden, prefix), attention_mask=None)
                    expected_mask = (
                        torch.arange(7)[None, :] <= (4 + torch.arange(3))[:, None]
                    )
                    expected = attention(
                        hidden,
                        replace(ctx, attention_mask=expected_mask[None, None]),
                        is_causal=False,
                    ).hidden_states
                    assert_close(
                        attention(hidden, ctx).hidden_states, expected, atol=0, rtol=0
                    )
                    cache = DynamicCache()
                    attention(
                        prefix, replace(context(prefix), attention_mask=None), cache
                    )
                    cached = attention(
                        hidden, replace(context(hidden), attention_mask=None), cache
                    ).hidden_states
                    assert_close(cached, expected, atol=1e-6, rtol=1e-5)

    def test_causal_intersects_explicit_allow_mask_and_override_disables_default(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                attention = self.make_attention(backend)
                hidden = torch.randn(1, 4, 16)
                ctx = context(hidden)  # all keys allowed by explicit mask
                changed = hidden.clone()
                changed[:, -1] += torch.randn(16) * 10
                assert_close(
                    attention(hidden, ctx).hidden_states[:, 0],
                    attention(changed, ctx).hidden_states[:, 0],
                    atol=0,
                    rtol=0,
                )
                self.assertGreater(
                    (
                        attention(hidden, ctx, is_causal=False).hidden_states[:, 0]
                        - attention(changed, ctx, is_causal=False).hidden_states[:, 0]
                    )
                    .abs()
                    .max()
                    .item(),
                    1e-5,
                )

    def test_logical_positions_and_window_are_independent_of_storage_order(self):
        for backend in ("eager", "sdpa"):
            with self.subTest(backend=backend):
                attention = self.make_attention(backend, window=3)
                hidden = torch.randn(2, 2, 16)
                prefix = torch.randn(2, 5, 16)
                positions = torch.tensor([[0, 1, 2, 3, 4, 1, 2]])
                ctx = replace(
                    context(hidden, prefix),
                    query_positions=positions[:, -2:],
                    key_positions=positions,
                )
                distance = positions[:, -2:, None] - positions[:, None, :]
                allow = ((distance >= 0) & (distance < 3))[:, None]
                reference = self.make_attention(backend, causal=False)
                reference.load_state_dict(attention.state_dict())
                expected = reference(
                    hidden, replace(ctx, attention_mask=allow)
                ).hidden_states
                assert_close(
                    attention(hidden, ctx).hidden_states, expected, atol=0, rtol=0
                )

    def test_empty_queries_zero_bias_and_have_finite_zero_gradients(self):
        for backend in ("eager", "sdpa"):
            for additive in (False, True):
                with self.subTest(backend=backend, additive=additive):
                    attention = self.make_attention(backend, causal=False)
                    with torch.no_grad():
                        attention.o_proj.bias.fill_(1)
                    hidden = torch.randn(2, 3, 16, requires_grad=True)
                    mask = (
                        torch.full((2, 1, 3, 3), float("-inf"))
                        if additive
                        else torch.zeros(2, 1, 3, 3, dtype=torch.bool)
                    )
                    output = attention(
                        hidden, replace(context(hidden), attention_mask=mask)
                    ).hidden_states
                    assert_close(output, torch.zeros_like(output), atol=0, rtol=0)
                    output.sum().backward()
                    for p in (hidden, *attention.parameters()):
                        self.assertTrue(torch.isfinite(p.grad).all())
                        assert_close(p.grad, torch.zeros_like(p.grad), atol=0, rtol=0)

    def test_one_empty_head_cannot_leak_forbidden_values(self):
        eager = self.make_attention("eager", causal=False)
        sdpa = self.make_attention("sdpa", causal=False)
        sdpa.load_state_dict(eager.state_dict())
        hidden = torch.randn(2, 3, 16)
        mask = torch.ones(2, 4, 3, 3, dtype=torch.bool)
        mask[:, 0] = False
        ctx = replace(context(hidden), attention_mask=mask)
        expected = sdpa(hidden, ctx).hidden_states
        assert_close(eager(hidden, ctx).hidden_states, expected, atol=1e-6, rtol=1e-5)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaVisibilityTest(unittest.TestCase):
    def attention(self, backend, causal=True):
        cfg = config()
        cfg.hidden_size = 64
        cfg.head_dim = 16
        cfg.attention_bias = True
        cfg._attn_implementation = backend
        return build_attention(
            cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(is_causal=causal)
        ).to(device="cuda", dtype=torch.bfloat16)

    def context(self, hidden, prefix, mask=None):
        ctx = context(hidden, prefix)
        length = hidden.shape[1] + prefix.shape[1]
        return replace(
            ctx,
            attention_mask=mask,
            position_embeddings=(
                hidden.new_ones(1, length, 16),
                hidden.new_zeros(1, length, 16),
            ),
        )

    def test_flex_and_flash_native_causal_prefix_match_sdpa(self):
        hidden = torch.randn(2, 4, 64, device="cuda", dtype=torch.bfloat16)
        prefix = torch.randn(2, 3, 64, device="cuda", dtype=torch.bfloat16)
        reference = self.attention("sdpa")
        ctx = self.context(hidden, prefix)
        expected = reference(hidden, ctx).hidden_states
        for backend in ("flex_attention", "flash_attention_2"):
            with (
                self.subTest(backend=backend),
                patch.dict(
                    "os.environ", {"SPECFORGE_FLEX_ATTENTION_BACKEND": "TRITON"}
                ),
            ):
                attention = self.attention(backend)
                attention.load_state_dict(reference.state_dict())
                actual = attention(hidden, ctx).hidden_states
                assert_close(actual, expected, atol=0.01, rtol=0.03)

    def test_flex_empty_rows_and_partial_heads_have_zero_output_and_gradients(self):
        with patch.dict("os.environ", {"SPECFORGE_FLEX_ATTENTION_BACKEND": "TRITON"}):
            attention = self.attention("flex_attention", causal=False)
            reference = self.attention("sdpa", causal=False)
            with torch.no_grad():
                attention.o_proj.bias.fill_(1)
            reference.load_state_dict(attention.state_dict())
            hidden = torch.randn(
                1, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
            )
            prefix = hidden[:, :0]

            def visible(b, h, q, k):
                return (q != 0) & (h != 0)

            mask = create_block_mask(
                visible, B=1, H=4, Q_LEN=4, KV_LEN=4, device="cuda"
            )
            output = attention(hidden, self.context(hidden, prefix, mask)).hidden_states
            dense_mask = torch.ones(1, 4, 4, 4, device="cuda", dtype=torch.bool)
            dense_mask[:, :, 0] = False
            dense_mask[:, 0] = False
            expected = reference(
                hidden, self.context(hidden, prefix, dense_mask)
            ).hidden_states
            assert_close(output, expected, atol=0.01, rtol=0.03)
            assert_close(output[:, 0], torch.zeros_like(output[:, 0]), atol=0, rtol=0)
            output[:, 0].sum().backward()
            for p in (hidden, *attention.parameters()):
                self.assertTrue(torch.isfinite(p.grad).all())
                assert_close(p.grad, torch.zeros_like(p.grad), atol=0, rtol=0)

    def test_flash_padding_cannot_revive_masked_query_with_projection_bias(self):
        attention = self.attention("flash_attention_2")
        with torch.no_grad():
            attention.o_proj.bias.fill_(1)
        hidden = torch.randn(
            1, 4, 64, device="cuda", dtype=torch.bfloat16, requires_grad=True
        )
        mask = torch.tensor([[False, True, True, True]], device="cuda")
        output = attention(
            hidden, self.context(hidden, hidden[:, :0], mask)
        ).hidden_states
        assert_close(output[:, 0], torch.zeros_like(output[:, 0]), atol=0, rtol=0)
        output[:, 0].sum().backward()
        for p in (hidden, *attention.parameters()):
            self.assertTrue(torch.isfinite(p.grad).all())
            assert_close(p.grad, torch.zeros_like(p.grad), atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
