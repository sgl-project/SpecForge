"""Backend parity and preparation lifetime, including checkpoint recomputation."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from torch._dynamo.testing import CompileCounter
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.layers.attention.backends import (
    AttentionPreparationScope,
    resolve_attention_backend,
)
from specforge.modeling.draft.layers.attention.backends.flex import FlexAttentionBackend
from specforge.modeling.draft.layers.attention.masks import AttentionLayout, PaddingMask
from tests.test_modeling.test_draft_gqa import build, make_node


def layout_for(device):
    positions = torch.arange(7, device=device)[None]
    active = torch.tensor([[True, True, True, True, True, True, False]], device=device)
    # Query zero has no visible keys even though marked active: exercise empty rows.
    return AttentionLayout(
        query_positions=positions,
        key_positions=positions,
        visibility=lambda b, h, q, k: (q > 0) & (k % 2 == 0) & ((h % 2 == 0) | (q > 1)),
        active_queries=active,
    )


def check_backend(backend_name):
    if backend_name == "flex_attention" and not torch.cuda.is_available():
        raise unittest.SkipTest("compiled Flex parity requires CUDA")
    device = "cuda" if backend_name == "flex_attention" else "cpu"
    torch.manual_seed(0)
    layout = layout_for(device)
    values = [torch.randn(1, heads, 7, 16, device=device) for heads in (4, 2, 2)]
    results = []
    for name in ("eager", backend_name):
        q, k, v = [value.clone().requires_grad_() for value in values]
        with AttentionPreparationScope() as scope:
            attention = scope.prepare(
                resolve_attention_backend(name),
                layout,
                batch_size=1,
                num_heads=4,
                is_causal=True,
                sliding_window=4,
            )
        result = attention.attend(
            q, k, v, scale=0.25, dropout_p=0, training=True, return_lse=name != "sdpa"
        )
        projected = result.mask_output(result.output.transpose(1, 2).flatten(2) + 1)
        projected.square().sum().backward()
        assert torch.count_nonzero(projected[:, [0, 6]]) == 0
        results.append((projected.detach(), q.grad, k.grad, v.grad, result.lse))
    for expected, actual in zip(results[0][:4], results[1][:4]):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=3e-5)
    if backend_name == "flex_attention":
        torch.testing.assert_close(results[1][4], results[0][4], atol=3e-5, rtol=3e-5)


def check_preparation():
    layers = nn.ModuleList(
        [build(make_node(attention_dropout=0), rotary_dim=0) for _ in range(2)]
    )
    hidden = torch.randn(1, 7, 16, requires_grad=True)
    layout = layout_for("cpu")
    backend = layers[0].backend
    with patch.object(backend, "prepare", wraps=backend.prepare) as prepare:
        with AttentionPreparationScope() as scope:
            inputs = [layer.prepare(layout, scope, batch_size=1) for layer in layers]
            assert inputs[0] is inputs[1]
            windowed = scope.prepare(
                backend,
                layout,
                batch_size=1,
                num_heads=2,
                is_causal=True,
                sliding_window=2,
            )
            assert windowed is not inputs[0]
        for layer, attention in zip(layers, inputs):
            hidden = checkpoint(
                layer,
                hidden,
                attention=attention,
                position_embeddings=None,
                use_reentrant=False,
            )
        hidden.sum().backward()
        assert prepare.call_count == 2
        with AttentionPreparationScope() as scope:
            fresh = layers[0].prepare(layout, scope, batch_size=1)
        assert fresh is not inputs[0]
        assert prepare.call_count == 3
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "per-forward"):
        layers[0].prepare(layout, scope, batch_size=1)


class TestAttentionBackends(unittest.TestCase):
    def test_sdpa(self):
        check_backend("sdpa")

    def test_flex(self):
        check_backend("flex_attention")

    def test_eager_reuses_empty_rows(self):
        with AttentionPreparationScope() as scope:
            attention = scope.prepare(
                resolve_attention_backend("eager"),
                layout_for("cpu"),
                batch_size=1,
                num_heads=4,
                is_causal=True,
                sliding_window=None,
            )
        q, k, v = [torch.randn(1, h, 7, 16, requires_grad=True) for h in (4, 2, 2)]
        with patch.object(
            torch.Tensor,
            "any",
            side_effect=AssertionError("mask reduced during execution"),
        ):
            for _ in range(2):
                result = attention.attend(
                    q, k, v, scale=0.25, dropout_p=0, training=True
                )
                self.assertEqual(result.output[:, 1::2, 1].count_nonzero().item(), 0)
                result.output.sum().backward()
        for tensor in (q, k, v):
            self.assertTrue(torch.isfinite(tensor.grad).all())

    def test_backend_names_and_flex_kernel_selection(self):
        for name in ("eager", "sdpa", "flex_attention", "fa"):
            self.assertIs(
                resolve_attention_backend(name), resolve_attention_backend(name)
            )
        with self.assertRaisesRegex(NotImplementedError, "USP"):
            resolve_attention_backend("usp")
        positions = torch.arange(3)[None]
        valid = torch.ones(1, 3, dtype=torch.bool)
        layout = AttentionLayout(
            query_positions=positions,
            key_positions=positions,
            visibility=PaddingMask(valid, valid),
            active_queries=valid,
        )
        for kernel, block_size in (("TRITON", 128), ("FLASH", (256, 128))):
            backend = FlexAttentionBackend(kernel_backend=kernel)
            with patch(
                "specforge.modeling.draft.layers.attention.backends.flex._compiled_block_mask"
            ) as prepare:
                backend.prepare(
                    layout,
                    batch_size=1,
                    num_heads=4,
                    is_causal=True,
                    sliding_window=None,
                )
                self.assertIsNone(prepare.call_args.args[2])
                self.assertEqual(prepare.call_args.kwargs["BLOCK_SIZE"], block_size)
            q = torch.randn(1, 4, 3, 8)
            with patch(
                "specforge.modeling.draft.layers.attention.backends.flex._compiled_attention",
                return_value=(q, SimpleNamespace(lse=None)),
            ) as attend:
                backend.attend(
                    q,
                    q,
                    q,
                    mask=object(),
                    scale=0.25,
                    dropout_p=0,
                    training=False,
                    return_lse=False,
                )
                self.assertEqual(
                    attend.call_args.kwargs["kernel_options"], {"BACKEND": kernel}
                )

    def test_preparation_compiles_without_tensor_identity_guards(self):
        backend = resolve_attention_backend("eager")

        def forward(query, positions, visible):
            layout = AttentionLayout(
                query_positions=positions, key_positions=positions, visibility=visible
            )
            with AttentionPreparationScope() as scope:
                first = scope.prepare(
                    backend,
                    layout,
                    batch_size=1,
                    num_heads=2,
                    is_causal=True,
                    sliding_window=None,
                )
                second = scope.prepare(
                    backend,
                    layout,
                    batch_size=1,
                    num_heads=2,
                    is_causal=True,
                    sliding_window=None,
                )
                assert first is second
            return first.attend(
                query, query, query, scale=0.5, dropout_p=0, training=False
            ).output

        torch._dynamo.reset()
        counter = CompileCounter()
        compiled = torch.compile(forward, backend=counter, fullgraph=True)
        for _ in range(6):
            args = (
                torch.randn(1, 2, 3, 4),
                torch.arange(3)[None],
                torch.ones(1, 1, 3, 3, dtype=torch.bool),
            )
            torch.testing.assert_close(compiled(*args), forward(*args))
        self.assertEqual(counter.frame_count, 1)

    def test_head_specific_inactive_rows(self):
        positions = torch.arange(3)[None]
        visible = torch.ones(1, 2, 3, 3, dtype=torch.bool)
        visible[:, 0, 0] = False
        visible[:, :, 1] = False
        q = torch.randn(1, 2, 3, 4)
        for name in ("eager", "sdpa"):
            with AttentionPreparationScope() as scope:
                prepared = scope.prepare(
                    resolve_attention_backend(name),
                    AttentionLayout(
                        query_positions=positions,
                        key_positions=positions,
                        visibility=visible,
                    ),
                    batch_size=1,
                    num_heads=2,
                    is_causal=False,
                    sliding_window=None,
                )
            result = prepared.attend(q, q, q, scale=0.5, dropout_p=0, training=False)
            self.assertEqual(result.output[:, 0, 0].count_nonzero(), 0)
            self.assertEqual(result.output[:, :, 1].count_nonzero(), 0)
            self.assertEqual(result.active_queries.tolist(), [[True, False, True]])
            self.assertEqual(
                result.mask_output(torch.ones(1, 3, 8))[:, 1].count_nonzero(), 0
            )

    def test_preparation(self):
        check_preparation()
