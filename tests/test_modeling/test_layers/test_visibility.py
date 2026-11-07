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


class AlgorithmMaskTest(unittest.TestCase):
    """Masks that already encode causal/window rules are not rebuilt per layer."""

    def _dflash(self, layer_types, is_causal=None):
        from specforge.modeling.draft.dflash import DFlashDraftModel

        cfg = config()
        cfg.layer_types = list(layer_types)
        cfg.sliding_window = 3
        cfg.use_sliding_window = True
        if is_causal is not None:
            cfg.is_causal = is_causal
        torch.manual_seed(0)
        return DFlashDraftModel(cfg).eval()

    def _forward(self, model):
        from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask

        anchors = torch.tensor([[2, 5]])
        keep = torch.tensor([[True, True]])
        masks = {
            layer_type: create_dflash_sdpa_mask(
                anchors,
                keep,
                8,
                4,
                "cpu",
                sliding_window=3 if layer_type == "sliding_attention" else None,
                is_causal=getattr(model.config, "is_causal", None),
            )
            for layer_type in set(model.layer_types)
        }
        positions = torch.cat(
            (torch.arange(8)[None], (anchors[..., None] + torch.arange(4)).flatten(1)),
            dim=1,
        )
        torch.manual_seed(1)
        return model(
            position_ids=positions,
            attention_mask=masks,
            noise_embedding=torch.randn(1, 8, 16),
            target_hidden=torch.randn(1, 8, model.fc.in_features),
        )

    def test_dflash_masks_skip_rebuild_and_match_rebuilt_masks(self):
        from specforge.modeling.draft.layers.attention.backends import dense

        for layer_types, is_causal in (
            (("sliding_attention", "full_attention"), None),
            (("full_attention", "full_attention"), True),
        ):
            with self.subTest(layer_types=layer_types, is_causal=is_causal):
                model = self._dflash(layer_types, is_causal)
                with patch.object(
                    dense, "resolve_visibility", wraps=dense.resolve_visibility
                ) as resolve:
                    with torch.no_grad():
                        actual = self._forward(model)
                self.assertEqual(resolve.call_count, 0)

                # Forcing the generic path reproduces the same output.
                from specforge.modeling.draft import dflash

                original = dflash.Qwen3DFlashDecoderLayer._attention_context

                def rebuild(self, *args):
                    return replace(original(self, *args), mask_encodes_visibility=False)

                with (
                    patch.object(
                        dflash.Qwen3DFlashDecoderLayer, "_attention_context", rebuild
                    ),
                    patch.object(
                        dense, "resolve_visibility", wraps=dense.resolve_visibility
                    ) as resolve,
                    torch.no_grad(),
                ):
                    expected = self._forward(model)
                self.assertEqual(resolve.call_count, len(layer_types))
                assert_close(actual, expected, atol=0, rtol=0)

    def test_flex_requests_lse_only_when_o_proj_has_bias(self):
        from specforge.modeling.draft.layers.attention.backends import dense

        for bias in (False, True):
            with self.subTest(bias=bias):
                cfg = config()
                cfg.attention_bias = bias
                cfg._attn_implementation = "flex_attention"
                attention = build_attention(
                    cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig()
                )
                hidden = torch.randn(1, 3, 16)
                calls = []

                def fake_flex(q, k, v, **kwargs):
                    calls.append(kwargs)
                    out = torch.zeros_like(q)
                    if kwargs["return_lse"]:
                        return out, torch.zeros(q.shape[:-1])
                    return out

                block_mask = create_block_mask(
                    lambda b, h, q, k: q >= 0,
                    B=1,
                    H=None,
                    Q_LEN=3,
                    KV_LEN=3,
                    device="cpu",
                )
                with patch.object(dense, "compile_friendly_flex_attention", fake_flex):
                    attention(
                        hidden, replace(context(hidden), attention_mask=block_mask)
                    )
                self.assertEqual(calls[-1]["return_lse"], bias)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA required")
class CudaVisibilityTest(unittest.TestCase):
    def attention(self, backend, causal=True, bias=True):
        cfg = config()
        cfg.hidden_size = 64
        cfg.head_dim = 16
        cfg.attention_bias = bias
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

    def test_bias_free_flex_skips_lse_and_still_zeroes_empty_rows_and_heads(self):
        from specforge.modeling.draft.layers.attention.backends import dense

        with patch.dict("os.environ", {"SPECFORGE_FLEX_ATTENTION_BACKEND": "TRITON"}):
            attention = self.attention("flex_attention", causal=False, bias=False)
            reference = self.attention("sdpa", causal=False, bias=False)
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
            with patch.object(
                dense,
                "compile_friendly_flex_attention",
                wraps=dense.compile_friendly_flex_attention,
            ) as flex:
                output = attention(
                    hidden, self.context(hidden, prefix, mask)
                ).hidden_states
            self.assertFalse(flex.call_args.kwargs["return_lse"])
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
