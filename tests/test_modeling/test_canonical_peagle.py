"""P-EAGLE learns feature replacement inside the same forward used for inference."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling import build_draft_model
from specforge.modeling.draft.llama3_eagle import LlamaRotaryEmbedding
from specforge.modeling.draft.models import PEagleDraftModel
from specforge.modeling.draft.peagle import PEagleDraftModel as LegacyPEagleDraftModel
from tests.test_modeling.legacy_rope_reference import (
    eager_legacy_reference,
    legacy_fp32_rope,
)
from tests.test_modeling.test_canonical_eagle3 import install_mapping, model_inputs
from tests.test_modeling.test_draft_eagle_backbone import config_for
from tests.test_modeling.test_draft_eagle_fusion import eager_reference


def model_config(*, normalized=False, bound=False, target_width=48):
    base = config_for("peagle", normalized)
    return replace(
        base,
        target_bindings=("embed_tokens",) if bound else (),
        spec_config=replace(
            base.spec_config,
            target_features=replace(
                base.spec_config.target_features, hidden_size=target_width
            ),
        ),
    )


def feature_mask(device="cpu"):
    return torch.tensor(
        [[False, True, False, True], [True, False, True, False]], device=device
    )


def legacy_name(name):
    for prefix, target in (
        ("backbone.layers.", "layers."),
        ("backbone.final_norm.", "norm."),
        ("fusion.", ""),
        ("vocabulary.", ""),
    ):
        if name.startswith(prefix):
            return target + name[len(prefix) :]
    return name


def peagle_eager_reference(*, query, key, value, **kwargs):
    # Legacy applies RoPE inline; cast after rotation, as canonical RoPE does.
    return eager_reference(
        query=query.to(value.dtype), key=key.to(value.dtype), value=value, **kwargs
    )


class CanonicalPEagleTest(unittest.TestCase):
    @eager_legacy_reference()
    @legacy_fp32_rope()
    def _parity(self, device, dtype):
        # Canonical RoPE computes in FP32 before casting back to the model dtype.
        tolerance = (
            2e-2
            if dtype == torch.bfloat16
            else 1e-2 if dtype == torch.float16 else 1e-5
        )
        torch.manual_seed(0)
        for normalized in (False, True):
            with self.subTest(device=device, dtype=dtype, normalized=normalized):
                config = model_config(normalized=normalized)
                actual = build_draft_model(config, device=device, dtype=dtype)
                install_mapping(actual)
                with torch.no_grad():
                    actual.backbone.final_norm.weight.uniform_(0.5, 1.5)
                legacy = LlamaConfig(
                    hidden_size=32,
                    target_hidden_size=48,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    num_hidden_layers=3,
                    vocab_size=64,
                    draft_vocab_size=32,
                    max_position_embeddings=128,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                    pretraining_tp=1,
                    norm_before_residual=normalized,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                )
                reference = LegacyPEagleDraftModel(legacy).to(
                    device=device, dtype=dtype
                )
                reference.rotary_emb = LlamaRotaryEmbedding(
                    8, max_position_embeddings=128, base=10000, device=device
                )
                # Compute the legacy inline rotation in FP32 without changing
                # projection weights or the attention/MLP arithmetic.
                for layer in reference.layers:
                    for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj):
                        projection.register_forward_hook(
                            lambda module, args, output: output.float()
                        )
                reference.load_state_dict(
                    {legacy_name(n): v for n, v in actual.state_dict().items()},
                    strict=True,
                )
                ids, features, entries = model_inputs(config, device, dtype)
                other = features.detach().clone().requires_grad_()
                masked = feature_mask(device)
                output = actual(ids, features, entries, feature_mask=masked)
                visibility = (
                    entries[0].attention_layout.key_positions[:, None, None, :]
                    <= entries[0].attention_layout.query_positions[:, None, :, None]
                )
                # Legacy training performs feature replacement outside the model.
                selected = torch.where(
                    masked[..., None], reference.mask_hidden.to(dtype), other
                )
                with patch(
                    "specforge.modeling.draft.peagle.compile_friendly_flex_attention",
                    peagle_eager_reference,
                ):
                    hidden = reference(
                        selected,
                        reference.embed_tokens(
                            ids.masked_fill(masked, config.spec_config.mask_token_id)
                        ),
                        attention_mask=visibility,
                        position_ids=entries[0].position_ids,
                    )
                logits = reference.lm_head(hidden)
                torch.testing.assert_close(
                    output.hidden_states, hidden, rtol=tolerance, atol=tolerance
                )
                torch.testing.assert_close(
                    output.logits, logits, rtol=tolerance, atol=tolerance
                )
                upstream = torch.randn_like(logits)
                output.logits.backward(upstream)
                logits.backward(upstream)
                torch.testing.assert_close(
                    features.grad, other.grad, rtol=tolerance, atol=tolerance
                )
                self.assertEqual(torch.count_nonzero(features.grad[masked]).item(), 0)
                self.assertGreater(
                    torch.count_nonzero(actual.mask_hidden.grad).item(), 0
                )
                for name, parameter in actual.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(reference.named_parameters())[legacy_name(name)].grad,
                        rtol=tolerance,
                        atol=tolerance,
                    )

    def test_linear_initialization_matches_pytorch_default(self):
        model = build_draft_model(model_config())
        for module in model.modules():
            if isinstance(module, torch.nn.Linear):
                reference = torch.nn.Linear(
                    module.in_features,
                    module.out_features,
                    bias=module.bias is not None,
                )
                torch.manual_seed(17)
                reference.reset_parameters()
                torch.manual_seed(17)
                model._init_weights(module)
                torch.testing.assert_close(
                    module.weight, reference.weight, rtol=0, atol=0
                )
                if module.bias is not None:
                    torch.testing.assert_close(
                        module.bias, reference.bias, rtol=0, atol=0
                    )

    def test_cpu_legacy_composition(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_composition(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_checkpointing_binding_and_hf_restore_keep_learned_mask(self):
        config = model_config(bound=True, target_width=32)
        actual = build_draft_model(config)
        install_mapping(actual)
        embedding = torch.randn(64, 32)
        actual.embed_tokens.load_target_weight(embedding)
        with torch.no_grad():
            actual.mask_hidden.fill_(0.37)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, mask = model_inputs(config), feature_mask()
        output, expected = actual(*args, feature_mask=mask), checkpointed(
            *args, feature_mask=mask
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        output.logits.sum().backward()
        expected.logits.sum().backward()
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = PEagleDraftModel.from_pretrained(path)
        self.assertFalse(restored.embed_tokens.weight.is_meta)
        restored.to("cpu")
        restored.embed_tokens.load_target_weight(embedding)
        torch.testing.assert_close(
            restored.mask_hidden, actual.mask_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored(*args, feature_mask=mask).logits, output.logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.vocabulary.to_target(torch.arange(32)),
            torch.arange(32) * 2,
            rtol=0,
            atol=0,
        )
        self.assertNotIn("embed_tokens.weight", actual.state_dict())

    def test_mask_application_hidden_only_meta_and_tied_embeddings(self):
        config = model_config()
        actual = build_draft_model(config)
        args = model_inputs(config)
        output = actual(*args, output_logits=False)
        unmasked = actual(
            *args,
            feature_mask=torch.zeros_like(args[0], dtype=torch.bool),
            output_logits=False,
        )
        torch.testing.assert_close(
            output.hidden_states, unmasked.hidden_states, rtol=0, atol=0
        )
        self.assertIsNone(output.logits)
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            actual.compute_logits(output.hidden_states)
        for mask in (torch.ones_like(args[0]), feature_mask()[:1]):
            with self.assertRaisesRegex(ValueError, "feature_mask"):
                actual(*args, feature_mask=mask, output_logits=False)
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        self.assertEqual(meta.mask_hidden.shape, config.spec_config.mask_hidden_shape)
        full = build_draft_model(
            replace(config, draft_vocab_size=64, tie_word_embeddings=True)
        )
        self.assertIs(full.embed_tokens.weight, full.lm_head.weight)
        self.assertTrue(full.vocabulary.ready)


if __name__ == "__main__":
    unittest.main()
