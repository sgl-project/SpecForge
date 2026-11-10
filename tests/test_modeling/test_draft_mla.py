"""Node-driven MLA matches the existing eager projections and full-QK scaling."""

import unittest
from dataclasses import dataclass, replace
from typing import ClassVar
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from specforge.config.draft.attention import MLASpec
from specforge.config.draft.codec.components import decode_component
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.registry import _SCHEMAS, register_component_schema
from specforge.config.draft.residual import StandardResidualSpec
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec, YaRNSpec
from specforge.modeling.draft.dflash import Qwen3DFlashMLAAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers import DecoderLayer, MLAAttention, build_component
from specforge.modeling.draft.layers.attention.backends import AttentionPreparationScope
from specforge.modeling.draft.layers.attention.masks import dense_attention_mask
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.layers.residual import StandardResidualConnection
from specforge.modeling.draft.layers.rope import build_rotary
from tests.test_modeling.test_draft_gqa import layout_for


def mla_node(**changes):
    return MLASpec(
        **(
            dict(
                num_attention_heads=2,
                q_lora_rank=8,
                kv_lora_rank=6,
                qk_nope_head_dim=4,
                qk_rope_head_dim=4,
                v_head_dim=6,
                attention_bias=True,
                is_causal=True,
                sliding_window=3,
                attention_dropout=0.1,
                rope_parameters=None,
            )
            | changes
        )
    )


def build(node, rope, *, backend="eager", **options):
    return build_component(
        "attention",
        node,
        input_size=16,
        hidden_size=16,
        norm=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0),
        rope=rope,
        backend=backend,
        **options,
    )


def prepare(module, layout, batch_size):
    with AttentionPreparationScope() as scope:
        return module.prepare(layout, scope, batch_size=batch_size)


class MLATest(unittest.TestCase):
    def test_backend_limits_fail_before_allocation(self):
        for backend, message in (
            ("flex_attention", "dropout"),
            ("fa", "head dimensions"),
        ):
            with patch("torch.nn.Linear") as allocate:
                with self.assertRaisesRegex(ValueError, message):
                    build(mla_node(), NoRoPESpec(), backend=backend)
                allocate.assert_not_called()
        module = build(
            mla_node(), NoRoPESpec(), backend="flex_attention", training=False
        )
        with self.assertRaisesRegex(ValueError, "dropout"):
            module.train()

    def _parity(self, device, dtypes, autocast=False):
        for rank, interleaved, rope_kind in (
            (None, False, "default"),
            (8, True, "default"),
            (None, True, "yarn"),
            (8, False, "yarn"),
            (8, True, "none"),
        ):
            node = mla_node(q_lora_rank=rank)
            rope_dict = dict(rope_type="default", rope_theta=10000.0)
            rope = DefaultRoPESpec(
                interleaved=interleaved, rope_theta=10000.0, partial_rotary_factor=1.0
            )
            if rope_kind == "yarn":
                rope_dict.update(rope_type="yarn", factor=4.0, mscale_all_dim=0.707)
                rope = YaRNSpec(
                    interleaved=interleaved,
                    rope_theta=10000.0,
                    partial_rotary_factor=1.0,
                    factor=4.0,
                    original_max_position_embeddings=32,
                    attention_factor=1.2,
                    beta_fast=32.0,
                    beta_slow=1.0,
                    truncate=True,
                    mscale_all_dim=0.707,
                )
            elif rope_kind == "none":
                rope = NoRoPESpec()
            for dtype in dtypes:
                with self.subTest(
                    device=device,
                    dtype=dtype,
                    rank=rank,
                    rope=rope_kind,
                    autocast=autocast,
                ):
                    config = Qwen3Config(
                        hidden_size=16,
                        num_attention_heads=node.num_attention_heads,
                        num_hidden_layers=1,
                        layer_types=["full_attention"],
                        q_lora_rank=rank,
                        kv_lora_rank=node.kv_lora_rank,
                        qk_nope_head_dim=node.qk_nope_head_dim,
                        qk_rope_head_dim=node.qk_rope_head_dim,
                        v_head_dim=node.v_head_dim,
                        rope_interleave=interleaved,
                        attention_bias=node.attention_bias,
                        rms_norm_eps=1e-6,
                        attention_dropout=node.attention_dropout,
                        rope_parameters=rope_dict,
                    )
                    config._attn_implementation = "eager"
                    reference = Qwen3DFlashMLAAttention(
                        config, 0, DEFAULT_DFLASH_KERNELS
                    ).to(device=device, dtype=dtype)
                    actual = build(node, rope, device=device, dtype=dtype)
                    actual.load_state_dict(reference.state_dict(), strict=True)
                    self.assertEqual(actual.scaling, reference.scaling)
                    hidden = torch.randn(
                        2, 4, 16, device=device, dtype=dtype, requires_grad=True
                    )
                    other = hidden.detach().clone().requires_grad_()
                    context = torch.randn(
                        2,
                        3,
                        16,
                        device=device,
                        dtype=torch.bfloat16 if autocast else dtype,
                        requires_grad=True,
                    )
                    other_context = context.detach().clone().requires_grad_()
                    layout = layout_for(hidden, 3)
                    layout.visibility[:, :, -1] = False
                    mask = dense_attention_mask(
                        layout,
                        is_causal=node.is_causal,
                        sliding_window=node.sliding_window,
                    )
                    rotary = build_rotary(rope, head_dim=node.qk_rope_head_dim)
                    embeddings = (
                        None if rotary is None else rotary(hidden, layout.key_positions)
                    )
                    reference_embeddings = embeddings or (
                        # Emulate disabled rotation without promoting autocast
                        # projections: FP32 identity tensors would change the
                        # softmax/dropout dtype compared with no rotation.
                        torch.ones(
                            2,
                            7,
                            4,
                            device=device,
                            dtype=torch.bfloat16 if autocast else dtype,
                        ),
                        torch.zeros(
                            2,
                            7,
                            4,
                            device=device,
                            dtype=torch.bfloat16 if autocast else dtype,
                        ),
                    )
                    torch.manual_seed(79)
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                        expected = reference(
                            other, other_context, reference_embeddings, mask
                        )[0]
                    torch.manual_seed(79)
                    with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                        output = actual(
                            hidden,
                            attention=prepare(actual, layout, hidden.shape[0]),
                            position_embeddings=embeddings,
                            target_context=context,
                        )
                    torch.testing.assert_close(output, expected, rtol=0, atol=0)
                    self.assertEqual(torch.count_nonzero(output[:, -1]).item(), 0)
                    upstream = torch.randn_like(output)
                    output.backward(upstream)
                    expected.backward(upstream)
                    for left, right in ((hidden, other), (context, other_context)):
                        torch.testing.assert_close(
                            left.grad, right.grad, rtol=0, atol=0
                        )
                    for name, parameter in actual.named_parameters():
                        self.assertIsNotNone(parameter.grad, name)
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(reference.named_parameters())[name].grad,
                            rtol=0,
                            atol=0,
                        )
                    self.assertIs(actual.config, node)

    def test_cpu_outputs_and_gradients_match(self):
        self._parity("cpu", (torch.float32,))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_outputs_and_gradients_match(self):
        self._parity("cuda", (torch.float32,))

    def test_registered_rope_controls_mla_score_scale(self):
        @dataclass(frozen=True, kw_only=True)
        class ScaledRoPE(DefaultRoPESpec):
            rope_type: ClassVar[str] = "test_scaled_rope"

            @property
            def mla_qk_scale_multiplier(self):
                return 1.75

        with patch.dict(_SCHEMAS):
            register_component_schema("rope", ScaledRoPE)
            rope = decode_component(
                "rope",
                dict(
                    rope_type=ScaledRoPE.rope_type,
                    rope_theta=10000.0,
                    partial_rotary_factor=1.0,
                    interleaved=True,
                ),
            )
            node = mla_node()
            self.assertEqual(build(node, rope).scaling, node.qk_head_dim**-0.5 * 1.75)
            self.assertEqual(build(node, NoRoPESpec()).scaling, node.qk_head_dim**-0.5)

    def _decoder_backends(self, device):
        node = mla_node(attention_dropout=0)
        rope = DefaultRoPESpec(
            interleaved=True, rope_theta=10000.0, partial_rotary_factor=1.0
        )
        norm = RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=0)
        for autocast in (False, True):
            dtype = torch.float32 if autocast else torch.bfloat16
            eager = DecoderLayer(
                build(node, rope, device=device, dtype=dtype),
                torch.nn.Linear(16, 16, device=device, dtype=dtype),
                attention_connection=StandardResidualConnection(StandardResidualSpec()),
                mlp_connection=StandardResidualConnection(StandardResidualSpec()),
                hidden_size=16,
                norm=norm,
                device=device,
                dtype=dtype,
            )
            sdpa = DecoderLayer(
                build(node, rope, backend="sdpa", device=device, dtype=dtype),
                torch.nn.Linear(16, 16, device=device, dtype=dtype),
                attention_connection=StandardResidualConnection(StandardResidualSpec()),
                mlp_connection=StandardResidualConnection(StandardResidualSpec()),
                hidden_size=16,
                norm=norm,
                device=device,
                dtype=dtype,
            )
            sdpa.load_state_dict(eager.state_dict())
            hidden = torch.randn(1, 4, 16, device=device, dtype=dtype)
            context = torch.randn(1, 3, 16, device=device, dtype=torch.bfloat16)
            layout = layout_for(hidden, 3)
            layout.visibility[:, :, -1] = False
            embeddings = build_rotary(rope, head_dim=node.rotary_input_dim)(
                hidden, layout.key_positions
            )
            self.assertEqual(embeddings[0].dtype, torch.float32)
            results = []
            for layer in (eager, sdpa):
                x, ctx = (
                    hidden.clone().requires_grad_(),
                    context.clone().requires_grad_(),
                )
                entry = LayerInputs(
                    attention=prepare(layer.self_attn, layout, 1),
                    target_context=ctx,
                    position_embeddings=embeddings,
                )
                with torch.autocast(device, dtype=torch.bfloat16, enabled=autocast):
                    output = layer(x, entry)
                output.float().square().mean().backward()
                results.append((output, x.grad, ctx.grad))
                for parameter in layer.parameters():
                    self.assertIsNotNone(parameter.grad)
                    self.assertTrue(torch.isfinite(parameter.grad).all())
            torch.testing.assert_close(results[0], results[1], rtol=0.04, atol=0.01)

    def test_decoder_fp32_rope_bf16_and_autocast_backends(self):
        self._decoder_backends("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_decoder_fp32_rope_bf16_and_autocast_backends(self):
        self._decoder_backends("cuda")

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_flex_mla_outputs_and_gradients(self):
        node = mla_node(
            qk_nope_head_dim=16, qk_rope_head_dim=16, v_head_dim=16, attention_dropout=0
        )
        eager = build(node, NoRoPESpec(), device="cuda")
        flex = build(node, NoRoPESpec(), backend="flex_attention", device="cuda")
        flex.load_state_dict(eager.state_dict())
        hidden = torch.randn(1, 4, 16, device="cuda")
        layout = replace(
            layout_for(hidden),
            active_queries=torch.ones(1, 4, device="cuda", dtype=torch.bool),
        )
        layout.visibility[:, :, -1] = False
        results = []
        for module in (eager, flex):
            x = hidden.clone().requires_grad_()
            output = module(
                x, attention=prepare(module, layout, 1), position_embeddings=None
            )
            output.square().sum().backward()
            self.assertEqual(output[:, -1].count_nonzero().item(), 0)
            results.append((output, x.grad, *(p.grad for p in module.parameters())))
        torch.testing.assert_close(results[0], results[1], rtol=3e-4, atol=3e-5)

    def test_gqa_and_mla_share_preparation(self):
        from tests.test_modeling.test_draft_gqa import build as build_gqa
        from tests.test_modeling.test_draft_gqa import make_node

        gqa = build_gqa(make_node(sliding_window=3))
        mla = build(mla_node(), NoRoPESpec())
        layout = layout_for(torch.randn(1, 4, 16))
        with AttentionPreparationScope() as scope:
            self.assertIs(
                gqa.prepare(layout, scope, batch_size=1),
                mla.prepare(layout, scope, batch_size=1),
            )

    def test_self_attention_partial_rotation_zero_nope_and_bias_rules(self):
        node = mla_node(
            qk_nope_head_dim=0,
            q_lora_rank=None,
            attention_bias=False,
            attention_dropout=0.0,
        )
        rope = DefaultRoPESpec(
            interleaved=True, rope_theta=10000.0, partial_rotary_factor=0.5
        )
        module = build(node, rope)
        self.assertFalse(
            any(name.endswith("bias") for name, _ in module.named_parameters())
        )
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        layout = layout_for(hidden)
        embeddings = build_rotary(rope, head_dim=4)(hidden, layout.key_positions)
        result = module(
            hidden,
            attention=prepare(module, layout, hidden.shape[0]),
            position_embeddings=embeddings,
        )
        # Empty context is equivalent to no context; neither projects query features into context.
        expected = module(
            hidden,
            attention=prepare(module, layout, hidden.shape[0]),
            position_embeddings=embeddings,
            target_context=hidden[:, :0],
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        result.square().sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in module.parameters())
        )
        # Rotation changes only the declared prefix of the MLA rotary subspace.
        from specforge.modeling.draft.layers.attention.backends.dense import (
            eager_attention,
        )

        disabled = build(node, NoRoPESpec())
        disabled.load_state_dict(module.state_dict(), strict=True)
        with patch(
            "specforge.modeling.draft.layers.attention.backends.dense.eager_attention",
            wraps=eager_attention,
        ) as backend:
            module(
                hidden,
                attention=prepare(module, layout, hidden.shape[0]),
                position_embeddings=embeddings,
            )
            rotated_q, rotated_k = backend.call_args.args[:2]
            disabled(
                hidden,
                attention=prepare(disabled, layout, hidden.shape[0]),
                position_embeddings=None,
            )
            plain_q, plain_k = backend.call_args.args[:2]
        for rotated, plain in ((rotated_q, plain_q), (rotated_k, plain_k)):
            torch.testing.assert_close(rotated[..., 2:], plain[..., 2:], rtol=0, atol=0)
            self.assertFalse(torch.equal(rotated[..., :2], plain[..., :2]))

    def test_constructor_checks_precede_allocation_and_inputs_use_shared_contract(self):
        node, rope = mla_node(), NoRoPESpec()
        with patch("torch.nn.Linear") as allocate:
            for changes in (
                {"input_size": 0},
                {"hidden_size": True},
                {"backend": "unknown"},
                {"rope": {}},
                {"norm": {}},
            ):
                with self.assertRaises((ValueError, TypeError)):
                    MLAAttention(
                        node,
                        **(
                            dict(
                                input_size=16,
                                hidden_size=16,
                                norm=RMSNormSpec(
                                    has_weight=True, eps=1e-6, weight_offset=0
                                ),
                                rope=rope,
                                backend="eager",
                            )
                            | changes
                        ),
                    )
            with self.assertRaisesRegex(ValueError, "override"):
                build(
                    replace(
                        node,
                        rope_parameters=DefaultRoPESpec(
                            interleaved=True,
                            rope_theta=10000.0,
                            partial_rotary_factor=1.0,
                        ),
                    ),
                    rope,
                )
            allocate.assert_not_called()
        module = build(node, rope, device="meta")
        self.assertTrue(all(parameter.is_meta for parameter in module.parameters()))
        module = build(node, rope)
        hidden = torch.randn(1, 3, 16)
        with self.assertRaisesRegex(ValueError, "context-then-query"):
            module(
                hidden,
                attention=prepare(module, layout_for(hidden, 2), hidden.shape[0]),
                position_embeddings=None,
            )
        with self.assertRaisesRegex(ValueError, "disabled RoPE"):
            module(
                hidden,
                attention=prepare(module, layout_for(hidden), hidden.shape[0]),
                position_embeddings=(hidden, hidden),
            )


if __name__ == "__main__":
    unittest.main()
