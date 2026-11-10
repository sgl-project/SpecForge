"""Canonical MTP matches the existing native computation using own components."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

import torch
from transformers import Qwen3Config

from specforge.config.draft.architecture.features import FinalHiddenFeatureSpec
from specforge.config.draft.norm import RMSNormSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import MTPDraftModel
from specforge.modeling.draft.mtp.qwen3_5 import PartialRotaryEmbedding, Qwen3_5MTPModel
from tests.test_config.test_draft_mtp_spec import mtp_config
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


def legacy_key(name):
    for prefix, replacement in (
        ("fusion.", ""),
        ("backbone.layers.", "layers."),
        ("backbone.final_norm.", "norm."),
    ):
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return name


class CanonicalMTPTest(unittest.TestCase):
    def _parity(self, device, dtype):
        base = mtp_config(target_bindings=(), target_logit_transform=None)
        config = replace(
            base,
            norm_config=RMSNormSpec(has_weight=True, eps=1e-6, weight_offset=1),
            attention_configs={
                "full": replace(base.attention_configs["full"], output_gate=True)
            },
            rope_parameters=replace(base.rope_parameters, partial_rotary_factor=0.5),
        )
        actual = build_draft_model(config, device=device, dtype=dtype)
        legacy = Qwen3Config(
            hidden_size=32,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
            num_hidden_layers=1,
            vocab_size=64,
            rms_norm_eps=1e-6,
            attn_output_gate=True,
            attention_dropout=0,
            partial_rotary_factor=0.5,
            rope_theta=10000,
        )
        legacy._attn_implementation = "eager"
        reference = Qwen3_5MTPModel(legacy).to(device=device, dtype=dtype)
        # Use the same FP32-derived frequencies as canonical RoPE after casts.
        reference.rotary_emb = PartialRotaryEmbedding(legacy, 8).to(device)
        reference.load_state_dict(
            {
                legacy_key(name): value
                for name, value in actual.state_dict().items()
                if name != "embed_tokens.weight"
            },
            strict=True,
        )
        embedding = copy.deepcopy(actual.embed_tokens)
        ids, features, entries = model_inputs(config, device, dtype)
        other = features.detach().clone().requires_grad_()
        output = actual(ids, features, entries)
        hidden = reference(embedding(ids), other, position_ids=entries[0].position_ids)
        expected = reference.lm_head(hidden)
        torch.testing.assert_close(output.hidden_states, hidden, rtol=0, atol=0)
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
        upstream = torch.randn_like(expected)
        output.logits.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        for name, parameter in actual.named_parameters():
            ref = (
                embedding.weight
                if name == "embed_tokens.weight"
                else dict(reference.named_parameters())[legacy_key(name)]
            )
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=0, atol=0)

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
        ids, features, entries = model_inputs(config)
        other = features.detach().clone().requires_grad_()
        output, expected = actual(ids, features, entries), checkpointed(
            ids, other, entries
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        grad = torch.randn_like(output.logits)
        output.logits.backward(grad)
        expected.logits.backward(grad)
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
