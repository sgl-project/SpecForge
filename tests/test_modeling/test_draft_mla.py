"""Node-driven MLA matches the existing eager projections and full-QK scaling."""

import unittest
from dataclasses import replace
from unittest.mock import patch

import torch
from transformers import Qwen3Config

from specforge.config.draft.attention import MLASpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.rope import DefaultRoPESpec, NoRoPESpec, YaRNSpec
from specforge.modeling.draft.dflash import Qwen3DFlashMLAAttention
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers import MLAAttention, build_component
from specforge.modeling.draft.layers.attention.masks import dense_attention_mask
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
                rope_interleave=True,
                attention_bias=True,
                is_causal=True,
                sliding_window=3,
                attention_dropout=0.1,
                rope_parameters=None,
            )
            | changes
        )
    )


def build(node, rope, **options):
    return build_component(
        "attention",
        node,
        input_size=16,
        hidden_size=16,
        norm=RMSNormSpec(eps=1e-6, weight_offset=0),
        rope=rope,
        backend="eager",
        **options,
    )


class MLATest(unittest.TestCase):
    def _parity(self, device, dtypes, autocast=False):
        for rank, interleaved, rope_kind in (
            (None, False, "default"),
            (8, True, "default"),
            (None, True, "yarn"),
            (8, False, "yarn"),
            (8, True, "none"),
        ):
            node = mla_node(q_lora_rank=rank, rope_interleave=interleaved)
            rope_dict = dict(rope_type="default", rope_theta=10000.0)
            rope = DefaultRoPESpec(rope_theta=10000.0, partial_rotary_factor=1.0)
            if rope_kind == "yarn":
                rope_dict.update(rope_type="yarn", factor=4.0, mscale_all_dim=0.707)
                rope = YaRNSpec(
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
                            layout=layout,
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
        self._parity("cpu", (torch.float32, torch.bfloat16))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_outputs_and_gradients_match(self):
        self._parity("cuda", (torch.float32, torch.float16, torch.bfloat16))

    def test_autocast_with_bf16_context_and_fp32_parameters(self):
        self._parity("cpu", (torch.float32,), autocast=True)
        if torch.cuda.is_available():
            self._parity("cuda", (torch.float32,), autocast=True)

    def test_self_attention_partial_rotation_zero_nope_and_bias_rules(self):
        node = mla_node(
            qk_nope_head_dim=0,
            q_lora_rank=None,
            attention_bias=False,
            attention_dropout=0.0,
        )
        rope = DefaultRoPESpec(rope_theta=10000.0, partial_rotary_factor=0.5)
        module = build(node, rope)
        self.assertFalse(
            any(name.endswith("bias") for name, _ in module.named_parameters())
        )
        hidden = torch.randn(2, 4, 16, requires_grad=True)
        layout = layout_for(hidden)
        embeddings = build_rotary(rope, head_dim=4)(hidden, layout.key_positions)
        result = module(hidden, layout=layout, position_embeddings=embeddings)
        # Empty context is equivalent to no context; neither projects query features into context.
        expected = module(
            hidden,
            layout=layout,
            position_embeddings=embeddings,
            target_context=hidden[:, :0],
        )
        torch.testing.assert_close(result, expected, rtol=0, atol=0)
        result.square().sum().backward()
        self.assertTrue(
            all(parameter.grad is not None for parameter in module.parameters())
        )
        # Rotation changes only the declared prefix of the MLA rotary subspace.
        from specforge.modeling.draft.layers.attention.mla import eager_attention

        disabled = build(node, NoRoPESpec())
        disabled.load_state_dict(module.state_dict(), strict=True)
        with patch(
            "specforge.modeling.draft.layers.attention.mla.eager_attention",
            wraps=eager_attention,
        ) as backend:
            module(hidden, layout=layout, position_embeddings=embeddings)
            rotated_q, rotated_k = backend.call_args.args[:2]
            disabled(hidden, layout=layout, position_embeddings=None)
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
                {"backend": "sdpa"},
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
                                norm=RMSNormSpec(eps=1e-6, weight_offset=0),
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
                            rope_theta=10000.0, partial_rotary_factor=1.0
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
            module(hidden, layout=layout_for(hidden, 2), position_embeddings=None)
        with self.assertRaisesRegex(ValueError, "disabled RoPE"):
            module(
                hidden, layout=layout_for(hidden), position_embeddings=(hidden, hidden)
            )


if __name__ == "__main__":
    unittest.main()
