"""GQA components preserve projection layouts and reference forward/backward math."""

import unittest
from contextlib import nullcontext
from dataclasses import replace
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5Attention, Qwen3_5Config

from specforge.config.draft.attention import MHASpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec
from specforge.modeling.draft.dflash import Qwen3DFlashAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers import GQAAttention, build_component
from specforge.modeling.draft.layers.attention.backends import AttentionPreparation
from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    dense_attention_mask,
)
from specforge.modeling.draft.layers.rope import apply_rotary, build_rotary
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


def make_node(**changes):
    return replace(
        draft_config().attention_configs["full"],
        **(
            dict(
                num_attention_heads=2,
                num_key_value_heads=1,
                attention_bias=True,
                is_causal=True,
                attention_dropout=0.1,
            )
            | changes
        ),
    )


def build(node, *, input_size=16, rotary_dim=8, offset=0, rope=None, **options):
    return build_component(
        "attention",
        node,
        input_size=input_size,
        hidden_size=16,
        norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=offset),
        rope=(
            rope
            if rope is not None
            else (
                NoRoPESpec()
                if rotary_dim == 0
                else DefaultRoPESpec(
                    interleaved=False,
                    rope_theta=10000.0,
                    partial_rotary_factor=rotary_dim / node.head_dim,
                )
            )
        ),
        backend="eager",
        **options,
    )


def layout_for(hidden, context_length=0):
    b, q = hidden.shape[:2]
    keys = torch.arange(context_length + q, device=hidden.device).expand(b, -1)
    return AttentionLayout(
        query_positions=keys[:, context_length:],
        key_positions=keys,
        visibility=torch.ones(
            b, 1, q, keys.shape[1], device=hidden.device, dtype=torch.bool
        ),
    )


def prepared(module, layout, batch_size):
    with AttentionPreparation() as preparation:
        return module.prepare(layout, preparation, batch_size=batch_size)


class GQATest(unittest.TestCase):
    def _assert_parity(self, device, dtypes, autocast=False):
        for kind in ("qwen3", "gated", "wide", "context", "mha"):
            gated, wide = kind == "gated", kind == "wide"
            node = make_node(
                output_gate=gated,
                qk_norm=not wide,
                attention_bias=not wide,
                sliding_window=2 if kind in ("qwen3", "context") else None,
            )
            if kind == "mha":
                node = MHASpec(**(node.__dict__ | {"num_key_value_heads": 2}))
            input_size, rotary_dim = (32 if wide else 16), (4 if gated else 8)
            config_args = dict(
                hidden_size=16,
                num_attention_heads=2,
                num_key_value_heads=node.num_key_value_heads,
                head_dim=8,
                num_hidden_layers=1,
                layer_types=["full_attention"],
                attention_bias=node.attention_bias,
                attention_dropout=0.1,
                rms_norm_eps=1e-6,
            )
            for dtype in dtypes:
                with self.subTest(
                    device=device, dtype=dtype, kind=kind, autocast=autocast
                ):
                    cfg = Qwen3Config(**config_args)
                    cfg._attn_implementation = "eager"
                    if gated:
                        cfg = Qwen3_5Config(text_config=config_args).text_config
                        cfg._attn_implementation = "eager"
                        reference = Qwen3_5Attention(cfg, 0)
                    elif kind == "context":
                        reference = Qwen3DFlashAttention(cfg, 0, DEFAULT_DFLASH_KERNELS)
                    else:
                        reference = Qwen3Attention(cfg, 0)
                    if wide:
                        reference.q_norm = reference.k_norm = nn.Identity()
                        for name in ("q_proj", "k_proj", "v_proj"):
                            old = getattr(reference, name)
                            setattr(
                                reference,
                                name,
                                nn.Linear(
                                    input_size,
                                    old.out_features,
                                    bias=node.attention_bias,
                                ),
                            )
                    reference.to(device=device, dtype=dtype)
                    actual = build(
                        node,
                        input_size=input_size,
                        rotary_dim=rotary_dim,
                        offset=int(gated),
                        device=device,
                        dtype=dtype,
                    )
                    actual.load_state_dict(reference.state_dict(), strict=True)
                    hidden = torch.randn(
                        2, 3, input_size, device=device, dtype=dtype, requires_grad=True
                    )
                    other = hidden.detach().clone().requires_grad_()
                    context = (
                        torch.randn(
                            2,
                            2,
                            input_size,
                            device=device,
                            dtype=dtype,
                            requires_grad=True,
                        )
                        if kind == "context"
                        else None
                    )
                    other_context = (
                        None
                        if context is None
                        else context.detach().clone().requires_grad_()
                    )
                    layout = layout_for(hidden, 0 if context is None else 2)
                    rope = build_rotary(
                        DefaultRoPESpec(
                            interleaved=False,
                            rope_theta=10000.0,
                            partial_rotary_factor=rotary_dim / 8,
                        ),
                        head_dim=8,
                    )
                    embeddings = rope(hidden, layout.key_positions)
                    mask = dense_attention_mask(
                        layout,
                        is_causal=node.is_causal,
                        sliding_window=node.sliding_window,
                    )
                    bias = torch.zeros_like(mask, dtype=dtype).masked_fill(
                        ~mask, -torch.inf
                    )
                    torch.manual_seed(9)
                    with (
                        torch.autocast(device_type=device, dtype=torch.bfloat16)
                        if autocast
                        else nullcontext()
                    ):
                        output = actual(
                            hidden,
                            attention=prepared(actual, layout, hidden.shape[0]),
                            position_embeddings=embeddings,
                            target_context=context,
                        )
                    torch.manual_seed(9)
                    with (
                        torch.autocast(device_type=device, dtype=torch.bfloat16)
                        if autocast
                        else nullcontext()
                    ):
                        if context is None:
                            expected = reference(
                                other,
                                position_embeddings=embeddings,
                                attention_mask=bias,
                            )[0]
                        else:
                            expected = reference(
                                other, other_context, embeddings, bias
                            )[0]
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    expected.backward(upstream)
                    torch.testing.assert_close(hidden.grad, other.grad, rtol=0, atol=0)
                    if context is not None:
                        torch.testing.assert_close(
                            context.grad, other_context.grad, rtol=0, atol=0
                        )
                    for name, parameter in actual.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(reference.named_parameters())[name].grad,
                            rtol=0,
                            atol=0,
                        )
                    self.assertIs(actual.config, node)

    def test_cpu_reference_outputs_and_gradients(self):
        self._assert_parity("cpu", (torch.float32,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_reference_outputs_and_gradients(self):
        self._assert_parity("cuda", (torch.float32,))

    def test_autocast_keeps_projected_attention_dtype(self):
        module = build(make_node(attention_dropout=0)).eval()
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        layout = layout_for(hidden)
        tables = build_rotary(
            DefaultRoPESpec(
                interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
            ),
            head_dim=8,
        )(hidden, layout.key_positions)
        with torch.autocast("cpu", dtype=torch.bfloat16):
            output = module(
                hidden,
                attention=prepared(module, layout, 2),
                position_embeddings=tables,
            )
        self.assertEqual(output.dtype, torch.bfloat16)
        output.float().sum().backward()
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_qk_norm_order_scope_and_interleaved_rope(self):
        for scope, position, weighted in (
            ("head", "before_rope", True),
            ("head", "after_rope", False),
            ("projection", "before_rope", True),
        ):
            norm = RMSNormSpec(eps=1e-6, weight_offset=0, has_weight=weighted)
            node = make_node(
                attention_dropout=0,
                qk_norm=True,
                qk_norm_scope=scope,
                qk_norm_position=position,
                qk_norm_config=norm,
            )
            rope = DefaultRoPESpec(
                interleaved=True, rope_theta=10000.0, partial_rotary_factor=1.0
            )
            module = build(node, rope=rope).eval()
            hidden = torch.randn(2, 3, 16, requires_grad=True)
            layout = layout_for(hidden)
            tables = build_rotary(rope, head_dim=8)(hidden, layout.key_positions)
            q = module.q_proj(hidden).view(2, 3, 2, 8).transpose(1, 2)
            k = module.k_proj(hidden).view(2, 3, 1, 8).transpose(1, 2)

            def normalize(tensor):
                shape = tensor.shape
                value = tensor.transpose(1, 2)
                if scope == "projection":
                    value = value.flatten(2)
                value = value * torch.rsqrt(
                    value.square().mean(-1, keepdim=True) + norm.eps
                )
                return value.reshape(shape[0], shape[2], shape[1], shape[3]).transpose(
                    1, 2
                )

            if position == "before_rope":
                q, k = normalize(q), normalize(k)
            q, k = (apply_rotary(t, tables, interleaved=True) for t in (q, k))
            if position == "after_rope":
                q, k = normalize(q), normalize(k)
            with patch.object(
                module.backend, "attend", wraps=module.backend.attend
            ) as attend:
                output = module(
                    hidden,
                    attention=prepared(module, layout, 2),
                    position_embeddings=tables,
                )
                actual_q, actual_k, _ = attend.call_args.args
            torch.testing.assert_close(actual_q, q, rtol=0, atol=0)
            torch.testing.assert_close(actual_k, k, rtol=0, atol=0)
            output.sum().backward()
            self.assertTrue(torch.isfinite(hidden.grad).all())
            self.assertEqual(module.q_norm.weight is not None, weighted)

    def test_disabled_rope_and_masked_queries_do_not_leak_output_bias(self):
        node = make_node(rope_parameters=NoRoPESpec(), output_gate=True)
        module = build(node, rotary_dim=0)
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        layout = layout_for(hidden)
        layout.visibility.zero_()
        with torch.no_grad():
            module.o_proj.bias.fill_(7)
        output = module(
            hidden,
            attention=prepared(module, layout, hidden.shape[0]),
            position_embeddings=None,
        )
        torch.testing.assert_close(output, torch.zeros_like(output), rtol=0, atol=0)
        output.sum().backward()
        for parameter in (*module.parameters(), hidden):
            torch.testing.assert_close(
                parameter.grad, torch.zeros_like(parameter), rtol=0, atol=0
            )
        empty = torch.empty(2, 0, 16)
        self.assertEqual(
            module(
                empty,
                attention=prepared(module, layout_for(empty), empty.shape[0]),
                position_embeddings=None,
            ).shape,
            (2, 0, 16),
        )

    def test_meta_materialization_uses_generic_initialization(self):
        owner = DFlashDraftModel(model_config(initializer_range=0))
        with torch.device("meta"):
            module = build(make_node())
        module.to_empty(device="cpu")
        module.apply(owner._initialize_weights)
        for name, parameter in module.named_parameters():
            expected = (
                torch.ones_like(parameter)
                if "_norm.weight" in name
                else torch.zeros_like(parameter)
            )
            torch.testing.assert_close(parameter, expected, rtol=0, atol=0)

    def test_shared_positions_and_external_rope_dtype(self):
        module = build(make_node(), dtype=torch.bfloat16).eval()
        hidden = torch.randn(2, 3, 16, dtype=torch.bfloat16)
        layout = layout_for(hidden)
        shared = replace(
            layout,
            query_positions=layout.query_positions[:1],
            key_positions=layout.key_positions[:1],
        )
        tables = (
            torch.ones(1, 3, 8, dtype=torch.float32),
            torch.zeros(1, 3, 8, dtype=torch.float32),
        )
        expected = module(
            hidden,
            attention=prepared(module, layout, hidden.shape[0]),
            position_embeddings=tables,
        )
        actual = module(
            hidden,
            attention=prepared(module, shared, hidden.shape[0]),
            position_embeddings=tables,
        )
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, "FP32"):
            module(
                hidden,
                attention=prepared(module, shared, hidden.shape[0]),
                position_embeddings=tuple(t.bfloat16() for t in tables),
            )
        mismatched = replace(
            shared, visibility=torch.ones(3, 1, 3, 3, dtype=torch.bool)
        )
        with self.assertRaisesRegex(ValueError, "layout"):
            module(
                hidden,
                attention=prepared(module, mismatched, hidden.shape[0]),
                position_embeddings=tables,
            )

    def test_invalid_construction_and_runtime_inputs_fail_early(self):
        node = make_node()
        with patch("torch.nn.Linear") as allocate:
            for changes in ({"input_size": 0}, {"rotary_dim": 3}, {"rotary_dim": 10}):
                with self.assertRaises(ValueError):
                    build(node, **changes)
            with self.assertRaises(ValueError):
                GQAAttention(
                    node,
                    input_size=16,
                    hidden_size=16,
                    norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
                    rope=DefaultRoPESpec(
                        interleaved=False, rope_theta=10000.0, partial_rotary_factor=1.0
                    ),
                    backend="unknown",
                )
            with self.assertRaises(ValueError):
                build(replace(node, rope_parameters=NoRoPESpec()))
            allocate.assert_not_called()
        module = build(node)
        hidden = torch.randn(2, 3, 16)
        layout = layout_for(hidden)
        with patch.object(module.q_proj, "forward") as project:
            with self.assertRaises(ValueError):
                module(
                    hidden,
                    attention=prepared(module, layout, hidden.shape[0]),
                    position_embeddings=None,
                )
            with self.assertRaises(ValueError):
                module(
                    hidden,
                    attention=prepared(module, layout, hidden.shape[0]),
                    position_embeddings=(torch.ones(2, 4, 8), torch.zeros(2, 4, 8)),
                )
            with self.assertRaises(ValueError):
                module(
                    hidden,
                    attention=prepared(module, layout, hidden.shape[0]),
                    position_embeddings=None,
                    target_context=torch.zeros(2, 1, 32),
                )
            project.assert_not_called()


if __name__ == "__main__":
    unittest.main()
