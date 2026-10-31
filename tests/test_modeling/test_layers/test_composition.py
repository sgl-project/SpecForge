"""Contracts and numerical checks for reusable draft components."""

import copy
import unittest

import torch
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_close
from transformers import LlamaConfig, Qwen3Config

from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers.attention.base import (
    Attention,
    AttentionContext,
    AttentionOutput,
)
from specforge.modeling.draft.layers.attention.gqa import GQAAttention
from specforge.modeling.draft.layers.attention.mla import MLAAttention
from specforge.modeling.draft.layers.config import AttentionConfig
from specforge.modeling.draft.layers.decoder import DecoderLayer
from specforge.modeling.draft.layers.factory import (
    build_attention,
    build_ffn,
    build_residual,
)
from specforge.modeling.draft.layers.ffn.dense import GatedMLP


def config(kind="gqa"):
    cfg = Qwen3Config(
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
        vocab_size=32,
        num_target_layers=4,
        block_size=4,
        layer_types=["full_attention"] * 2,
        attention_dropout=0.0,
        dflash_config={"attention_mode": kind, "target_layer_ids": [1]},
        kv_lora_rank=8,
        q_lora_rank=8,
        qk_nope_head_dim=4,
        qk_rope_head_dim=4,
        v_head_dim=4,
    )
    cfg._attn_implementation = "sdpa"
    return cfg


def context(hidden, prefix=None):
    length = hidden.shape[1] + (0 if prefix is None else prefix.shape[1])
    return AttentionContext(
        position_embeddings=(
            hidden.new_ones(1, length, 4),
            hidden.new_zeros(1, length, 4),
        ),
        attention_mask=torch.ones(1, 1, hidden.shape[1], length, dtype=torch.bool),
        key_value_prefix=prefix,
    )


class ComponentFactoryTest(unittest.TestCase):
    def test_factory_resolves_attention_structure(self):
        for kind, cls in (("gqa", GQAAttention), ("mla", MLAAttention)):
            with self.subTest(kind=kind):
                attention = build_attention(
                    config(kind), 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind)
                )
                self.assertIsInstance(attention, cls)
                hidden = torch.randn(2, 3, 16, requires_grad=True)
                output = attention(hidden, context(hidden))
                self.assertEqual(output.hidden_states.shape, hidden.shape)
                output.hidden_states.square().sum().backward()
                self.assertTrue(torch.isfinite(hidden.grad).all())
                self.assertTrue(all(p.grad is not None for p in attention.parameters()))

    def test_attention_weights_are_opt_in(self):
        cfg = config()
        cfg._attn_implementation = "eager"
        attention = build_attention(cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig())
        hidden = torch.randn(1, 3, 16)
        ctx = context(hidden)
        default = attention(hidden, ctx)
        detailed = attention(hidden, ctx, output_attentions=True)
        self.assertIsNone(default.attention_weights)
        self.assertEqual(detailed.attention_weights.shape, (1, 4, 3, 3))
        assert_close(default.hidden_states, detailed.hidden_states, atol=0, rtol=0)

    def test_invalid_combinations_do_not_fall_back(self):
        for kind in ("qsa", "dsa", "unknown"):
            with self.assertRaisesRegex(ValueError, "Unsupported attention kind"):
                AttentionConfig(kind)
        for window in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "sliding_window"):
                AttentionConfig(sliding_window=window)
        with self.assertRaisesRegex(ValueError, "MHA requires"):
            build_attention(config(), 0, DEFAULT_DFLASH_KERNELS, AttentionConfig("mha"))
        cfg = config()
        cfg.num_key_value_heads = 3
        with self.assertRaisesRegex(ValueError, "divide"):
            build_attention(cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig())

    def test_self_attention_matches_explicit_empty_prefix(self):
        for kind in ("gqa", "mla"):
            attention = build_attention(
                config(kind), 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind)
            )
            hidden = torch.randn(2, 3, 16, requires_grad=True)
            ordinary = attention(hidden, context(hidden)).hidden_states
            explicit = attention(hidden, context(hidden, hidden[:, :0])).hidden_states
            assert_close(ordinary, explicit, atol=0, rtol=0)
            first = torch.autograd.grad(ordinary.sum(), hidden, retain_graph=True)[0]
            second = torch.autograd.grad(explicit.sum(), hidden)[0]
            assert_close(first, second, atol=0, rtol=0)

    def test_gqa_matches_independent_attention_with_prefix(self):
        cfg = config()
        attention = build_attention(cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig())
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        prefix = torch.randn(2, 4, 16, requires_grad=True)
        ctx = context(hidden, prefix)
        actual = attention(hidden, ctx).hidden_states
        keys = torch.cat((prefix, hidden), dim=1)
        query = attention.q_norm(attention.q_proj(hidden).view(2, 3, 4, 4)).transpose(
            1, 2
        )
        key = attention.k_norm(attention.k_proj(keys).view(2, 7, 2, 4)).transpose(1, 2)
        value = attention.v_proj(keys).view(2, 7, 2, 4).transpose(1, 2)
        weights = (
            query @ key.repeat_interleave(2, dim=1).transpose(-1, -2) / 2
        ).softmax(-1)
        expected = attention.o_proj(
            (weights @ value.repeat_interleave(2, dim=1))
            .transpose(1, 2)
            .reshape(2, 3, 16)
        )
        assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        tensors = (hidden, prefix, *attention.parameters())
        first = torch.autograd.grad(actual.square().sum(), tensors, retain_graph=True)
        second = torch.autograd.grad(expected.square().sum(), tensors)
        for a, b in zip(first, second):
            assert_close(a, b, atol=2e-6, rtol=2e-5)


class DecoderCompositionTest(unittest.TestCase):
    def test_decoder_preserves_non_kv_attention_state(self):
        class CounterAttention(Attention[int]):
            def forward(self, hidden_states, context, state=None, **kwargs):
                return AttentionOutput(hidden_states, state + 1)

        layer = DecoderLayer(
            CounterAttention(), nn.Identity(), nn.Identity(), nn.Identity()
        )
        hidden = torch.randn(1, 3, 16)
        output = layer(hidden, context(hidden), state=4)
        self.assertEqual(output.attention.state, 5)
        assert_close(output.hidden_states, 4 * hidden, atol=0, rtol=0)

    def test_output_and_gradients_match_explicit_pre_norm_decoder(self):
        cfg = config()
        kernels = DEFAULT_DFLASH_KERNELS
        layer = DecoderLayer(
            build_attention(cfg, 0, kernels, AttentionConfig()),
            build_ffn(cfg, 0, kernels),
            kernels.make_rms_norm(16, cfg.rms_norm_eps),
            kernels.make_rms_norm(16, cfg.rms_norm_eps),
        )
        reference = copy.deepcopy(layer)
        hidden = torch.randn(2, 3, 16, requires_grad=True)
        reference_hidden = hidden.detach().clone().requires_grad_(True)
        ctx = context(hidden)
        actual = layer(hidden, ctx).hidden_states
        intermediate = (
            reference_hidden
            + reference.self_attn(
                reference.input_layernorm(reference_hidden), ctx
            ).hidden_states
        )
        expected = intermediate + reference.mlp(
            reference.post_attention_layernorm(intermediate)
        )
        assert_close(actual, expected, atol=0, rtol=0)
        actual.square().sum().backward()
        expected.square().sum().backward()
        assert_close(hidden.grad, reference_hidden.grad, atol=0, rtol=0)
        self.assertEqual(
            list(dict(layer.named_parameters())),
            list(dict(reference.named_parameters())),
        )
        for (name, p), (_, q) in zip(
            layer.named_parameters(), reference.named_parameters()
        ):
            assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)
        self.assertIsNot(layer.attention_residual, layer.ffn_residual)

    def test_checkpointing_preserves_model_gradients(self):
        for kind in ("gqa", "mla"):
            with self.subTest(kind=kind):
                original = DFlashDraftModel(config(kind)).train()
                checkpointed = copy.deepcopy(original)
                checkpointed.gradient_checkpointing_enable(
                    gradient_checkpointing_kwargs={"use_reentrant": False}
                )
                noise = torch.randn(1, 4, 16)
                target = torch.randn(1, 3, 16)
                outputs = []
                for model in (original, checkpointed):
                    output = model(
                        noise_embedding=noise,
                        target_hidden=target,
                        position_ids=torch.arange(7)[None],
                        attention_mask=torch.ones(1, 1, 4, 7, dtype=torch.bool),
                    )
                    output.square().sum().backward()
                    outputs.append(output)
                assert_close(*outputs, atol=0, rtol=0)
                for (name, p), (_, q) in zip(
                    original.named_parameters(), checkpointed.named_parameters()
                ):
                    assert_close(p.grad, q.grad, atol=0, rtol=0, msg=name)

    def test_standard_residual_keeps_original_state_for_write(self):
        residual = build_residual()
        hidden = torch.randn(2, 3, 8, requires_grad=True)
        normalized, state = residual.read(hidden, nn.LayerNorm(8))
        self.assertIs(state, hidden)
        assert_close(
            residual.write(normalized, state), hidden + normalized, atol=0, rtol=0
        )


class DenseFFNTest(unittest.TestCase):
    def test_tensor_parallel_ffn_matches_unsharded_formula(self):
        for tp in (1, 2):
            cfg = LlamaConfig(
                hidden_size=8,
                intermediate_size=16,
                num_attention_heads=2,
                num_key_value_heads=2,
                pretraining_tp=tp,
            )
            module = GatedMLP(cfg)
            hidden = torch.randn(2, 3, 8, requires_grad=True)
            actual = module(hidden)
            expected = F.linear(
                F.silu(F.linear(hidden, module.gate_proj.weight))
                * F.linear(hidden, module.up_proj.weight),
                module.down_proj.weight,
            )
            assert_close(actual, expected)
            params = (hidden, *module.parameters())
            actual_grads = torch.autograd.grad(
                actual.square().sum(), params, retain_graph=True
            )
            expected_grads = torch.autograd.grad(expected.square().sum(), params)
            for a, b in zip(actual_grads, expected_grads):
                assert_close(a, b, atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
