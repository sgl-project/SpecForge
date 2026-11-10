"""Backend parity and preparation lifetime, including checkpoint recomputation."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from specforge.modeling.draft.layers.attention.backends import (
    AttentionPreparation,
    resolve_attention_backend,
)
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from tests.test_modeling.test_draft_gqa import build, make_node


def layout_for(device):
    positions = torch.arange(7, device=device)[None]
    active = torch.tensor([[True, True, True, True, True, True, False]], device=device)
    # Query zero has no visible keys even though marked active: exercise empty rows.
    return AttentionLayout(
        query_positions=positions,
        key_positions=positions,
        visibility=lambda b, h, q, k: (q > 0) & (k % 2 == 0),
        active_queries=active,
    )


def check_backend(backend_name):
    if backend_name == "flex" and not torch.cuda.is_available():
        raise unittest.SkipTest("compiled Flex parity requires CUDA")
    device = "cuda" if backend_name == "flex" else "cpu"
    torch.manual_seed(0)
    layout = layout_for(device)
    values = [torch.randn(1, heads, 7, 16, device=device) for heads in (4, 2, 2)]
    results = []
    for name in ("eager", backend_name):
        q, k, v = [value.clone().requires_grad_() for value in values]
        with AttentionPreparation() as scope:
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
    if backend_name == "flex":
        torch.testing.assert_close(results[1][4], results[0][4], atol=3e-5, rtol=3e-5)


def check_preparation():
    layers = nn.ModuleList(
        [build(make_node(attention_dropout=0), rotary_dim=0) for _ in range(2)]
    )
    hidden = torch.randn(1, 7, 16, requires_grad=True)
    layout = layout_for("cpu")
    backend = layers[0].backend
    with patch.object(backend, "prepare", wraps=backend.prepare) as prepare:
        with AttentionPreparation() as scope:
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
        with AttentionPreparation() as scope:
            fresh = layers[0].prepare(layout, scope, batch_size=1)
        assert fresh is not inputs[0]
        assert prepare.call_count == 3
    with unittest.TestCase().assertRaisesRegex(RuntimeError, "per-forward"):
        layers[0].prepare(layout, scope, batch_size=1)


class TestAttentionBackends(unittest.TestCase):
    def test_sdpa(self):
        check_backend("sdpa")

    def test_flex(self):
        check_backend("flex")

    def test_preparation(self):
        check_preparation()
