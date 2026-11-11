"""Canonical MTP matches transformers' Qwen3.5 decoder layer around its own fusion."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import nn
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
    apply_rotary_pos_emb,
)

from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.config.draft.rope import DefaultRoPESpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import MTPDraftModel
from tests.test_config.test_draft_mtp_spec import mtp_config
from tests.test_modeling.test_canonical_dflash import count_layer_calls
from tests.test_modeling.test_draft_gqa import layout_for


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.tensor([[1, 4, 7, 2], [3, 8, 6, 5]], device=device)
    features = torch.randn(
        2,
        4,
        config.spec_config.target_features.hidden_size,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    layout = layout_for(features)
    return (
        ids,
        features,
        [
            LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
            for _ in config.layer_types
        ],
    )


def reference_model(config):
    """The same parameter names, computed by transformers' Qwen3.5 modules."""
    attention = config.attention_configs["full"]
    hf_config = Qwen3_5TextConfig(
        hidden_size=config.hidden_size,
        intermediate_size=config.mlp_configs["dense"].intermediate_size,
        num_attention_heads=attention.num_attention_heads,
        num_key_value_heads=attention.num_key_value_heads,
        head_dim=attention.head_dim,
        num_hidden_layers=1,
        layer_types=["full_attention"],
        rms_norm_eps=config.norm_config.eps,
        attention_bias=False,
        attention_dropout=0.0,
        vocab_size=config.vocab_size,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": config.rope_parameters.rope_theta,
            "partial_rotary_factor": config.rope_parameters.partial_rotary_factor,
        },
    )
    hf_config._attn_implementation = "eager"
    width = config.hidden_size
    reference = nn.Module()
    reference.embed_tokens = nn.Embedding(config.vocab_size, width)
    reference.fusion = nn.Module()
    reference.fusion.pre_fc_norm_embedding = Qwen3_5RMSNorm(
        width, eps=hf_config.rms_norm_eps
    )
    reference.fusion.pre_fc_norm_hidden = Qwen3_5RMSNorm(
        width, eps=hf_config.rms_norm_eps
    )
    reference.fusion.fc = nn.Linear(2 * width, width, bias=False)
    reference.backbone = nn.Module()
    reference.backbone.layers = nn.ModuleList(
        [Qwen3_5DecoderLayer(hf_config, layer_idx=0)]
    )
    reference.backbone.final_norm = Qwen3_5RMSNorm(width, eps=hf_config.rms_norm_eps)
    reference.lm_head = nn.Linear(width, config.vocab_size, bias=False)
    reference.rotary = Qwen3_5TextRotaryEmbedding(hf_config)
    return reference


def reference_forward(reference, ids, features, positions):
    embeddings = reference.embed_tokens(ids)
    fusion = reference.fusion
    hidden = fusion.fc(
        torch.cat(
            (
                fusion.pre_fc_norm_embedding(embeddings),
                fusion.pre_fc_norm_hidden(features),
            ),
            dim=-1,
        )
    )
    length = ids.shape[1]
    causal = torch.full(
        (length, length), float("-inf"), dtype=hidden.dtype, device=hidden.device
    ).triu(1)
    # Match the canonical/SGLang FP32 rotation; retain HF as the independent oracle.
    rotary = reference.rotary.float()(hidden.float(), positions)
    for layer in reference.backbone.layers:
        hidden = layer(
            hidden, position_embeddings=rotary, attention_mask=causal[None, None]
        )
    hidden = reference.backbone.final_norm(hidden)
    return hidden, reference.lm_head(hidden)


class CanonicalMTPTest(unittest.TestCase):
    def _parity(self, device, dtype):
        torch.manual_seed(0)
        base = mtp_config(target_bindings=(), target_logit_transform=None)
        config = replace(
            base,
            norm_config=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1),
            attention_configs={
                "full": replace(base.attention_configs["full"], output_gate=True)
            },
            rope_parameters=DefaultRoPESpec(
                interleaved=False, rope_theta=10_000_000, partial_rotary_factor=0.5
            ),
        )
        actual = build_draft_model(config, device=device, dtype=dtype)
        with torch.no_grad():
            for parameter in actual.parameters():
                parameter.normal_(std=0.2)
        reference = reference_model(config).to(device=device, dtype=dtype)
        reference.rotary = Qwen3_5TextRotaryEmbedding(reference.rotary.config).to(
            device
        )
        reference.load_state_dict(actual.state_dict(), strict=True)
        ids, features, entries = model_inputs(config, device, dtype)
        other = features.detach().clone().requires_grad_()
        output = actual(ids, features, entries)

        def rotate(query, key, cos, sin, **kwargs):
            q, k = apply_rotary_pos_emb(query.float(), key.float(), cos, sin, **kwargs)
            return q.to(query.dtype), k.to(key.dtype)

        with patch(
            "transformers.models.qwen3_5.modeling_qwen3_5.apply_rotary_pos_emb", rotate
        ):
            hidden, expected = reference_forward(
                reference, ids, other, entries[0].position_ids
            )
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
        upstream = torch.randn_like(expected)
        output.logits.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        references = dict(reference.named_parameters())
        for name, parameter in actual.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad, references[name].grad, rtol=0, atol=0
            )

    def test_cpu_native_computation(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_native_computation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_mixed_width_multilayer_checkpoint_and_tied_weights(self):
        base = mtp_config(
            target_bindings=(),
            target_logit_transform=None,
            tie_word_embeddings=True,
            layer_types=("full_dense",) * 2,
        )
        config = replace(
            base,
            spec_config=replace(
                base.spec_config, target_features=FinalHiddenFeatureSpec(hidden_size=48)
            ),
        )
        actual = build_draft_model(config)
        self.assertIs(actual.embed_tokens.weight, actual.lm_head.weight)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        calls = count_layer_calls(checkpointed)
        ids, features, entries = model_inputs(config)
        other = features.detach().clone().requires_grad_()
        output, expected = actual(ids, features, entries), checkpointed(
            ids, other, entries
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        grad = torch.randn_like(output.logits)
        output.logits.backward(grad)
        expected.logits.backward(grad)
        self.assertEqual(calls, [2] * checkpointed.backbone.num_layers)
        for name, parameter in actual.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(
                parameter.grad,
                dict(checkpointed.named_parameters())[name].grad,
                rtol=0,
                atol=0,
            )
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        self.assertIsNone(actual(ids, features, entries, output_logits=False).logits)
        with self.assertRaisesRegex(ValueError, "target KV"):
            actual(
                ids, features, [replace(e, target_context=features) for e in entries]
            )
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )

    def test_bound_modules_and_hf_round_trip(self):
        config = mtp_config()
        actual = build_draft_model(config)
        self.assertIsInstance(actual, MTPDraftModel)
        self.assertIs(actual.config.draft, config)
        args = model_inputs(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            actual(*args)
        for name, value in weights.items():
            getattr(actual, name).load_target_weight(value)
        output = actual(*args)
        output.logits.sum().backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
        self.assertNotIn("embed_tokens.weight", actual.state_dict())
        self.assertNotIn("lm_head.weight", actual.state_dict())
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = MTPDraftModel.from_pretrained(path)
        self.assertEqual(restored.config.draft, config)
        for name, value in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(value)
        torch.testing.assert_close(
            restored(*args).logits, output.logits, rtol=0, atol=0
        )


if __name__ == "__main__":
    unittest.main()
