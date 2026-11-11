"""DFlash2 composition includes convolution, unary transforms and selector gradients."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.architecture import DFlash2Spec
from specforge.config.draft.conv import GroupedCausalConvSpec
from specforge.config.draft.layer import LayerSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dflash2 import (
    DFlash2DraftModel as LegacyDFlash2DraftModel,
)
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DFlash2DraftModel, DFlashDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_dflash import count_layer_calls, legacy_name
from tests.test_modeling.test_draft_gqa import layout_for


def dflash2_config(*, bound=False):
    base = (
        draft_config()
        if bound
        else draft_config(target_bindings=(), target_logit_transform=None)
    )
    return replace(
        base,
        architecture="DFlash2DraftModel",
        conv_configs={"grouped": GroupedCausalConvSpec(kernel_size=3, group_size=4)},
        layer_configs={
            "full_dense": LayerSpec(
                attention="full", mlp="dense", sublayer_conv="grouped"
            )
        },
        spec_config=DFlash2Spec(
            block=base.spec_config.block,
            target_features=base.spec_config.target_features,
            selector_rank=4,
            selector_top_k=3,
            output_multiplier=1.7,
            final_logit_softcapping=2.5,
        ),
    )


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.arange(16, device=device).reshape(2, 8)
    features = torch.randn(2, 3, 64, device=device, dtype=dtype, requires_grad=True)
    layout = layout_for(torch.empty(2, 8, 32, device=device), context_length=3)
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    candidates = torch.randint(64, (2, 8, 3), device=device)
    predecessors = torch.randint(64, (2, 8), device=device)
    return (ids, features, entries), dict(
        candidate_ids=candidates, predecessor_ids=predecessors
    )


def score(model, output, options):
    return model.candidate_selector.score_candidates(
        **options,
        unary_logits=output.logits.gather(-1, options["candidate_ids"]),
        hidden_states=output.hidden_states,
    )


class CanonicalDFlash2Test(unittest.TestCase):
    @legacy_fp32_rope()
    def _parity(self, device, dtype):
        torch.manual_seed(0)
        config = dflash2_config()
        actual = build_draft_model(config, device=device, dtype=dtype)
        self.assertNotIsInstance(actual, DFlashDraftModel)
        with torch.no_grad():
            actual.candidate_selector.successor_codebook.normal_(std=0.1)
            for layer in actual.backbone.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.kernel_projection.weight.normal_(std=0.03)
        legacy = legacy_dflash_config(
            block_size=8,
            dflash_config={
                "target_layer_ids": [1, 3],
                "mask_token_id": 63,
                "conv_kernel_size": 3,
                "conv_group_size": 4,
                "selector_rank": 4,
                "selector_top_k": 3,
                "output_multiplier": 1.7,
                "final_logit_softcapping": 2.5,
            },
        )
        reference = LegacyDFlash2DraftModel(legacy).to(device=device, dtype=dtype)
        reference.rotary_emb = Qwen3RotaryEmbedding(legacy).to(device)
        reference.load_state_dict(
            {
                legacy_name(name) or name: value
                for name, value in actual.state_dict().items()
                if name not in ("embed_tokens.weight", "lm_head.weight")
            },
            strict=True,
        )
        embedding, head = copy.deepcopy(actual.embed_tokens), copy.deepcopy(
            actual.lm_head
        )
        (ids, features, entries), options = model_inputs(config, device, dtype)
        other = features.detach().clone().requires_grad_()
        output = actual(ids, features, entries)
        actual_scores = score(actual, output, options)
        with patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}):
            hidden = reference(
                entries[0].position_ids,
                noise_embedding=embedding(ids),
                target_hidden=other,
            )
        logits = reference.transform_unary_logits(head(hidden))
        scores = reference.candidate_selector.score_candidates(
            **options,
            unary_logits=logits.gather(-1, options["candidate_ids"]),
            hidden_states=hidden,
        )
        torch.testing.assert_close(
            (output.hidden_states, output.logits, actual_scores),
            (hidden, logits, scores),
            rtol=0,
            atol=0,
        )
        self.assertEqual(output.logits.dtype, torch.float32)
        upstream = torch.randn_like(scores)
        actual_scores.backward(upstream)
        scores.backward(upstream)
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        for name, parameter in actual.named_parameters():
            ref = (
                embedding.weight
                if name == "embed_tokens.weight"
                else (
                    head.weight
                    if name == "lm_head.weight"
                    else dict(reference.named_parameters())[legacy_name(name) or name]
                )
            )
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=0, atol=0)

    def test_cpu_complete_legacy_computation(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_complete_legacy_computation(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_bound_weights_checkpointing_and_hf_restore(self):
        config = dflash2_config(bound=True)
        actual = build_draft_model(config)
        # Restore trained, non-noop factors so accidental reinitialization of
        # either the selector or convolution branch cannot hide behind zeros.
        with torch.no_grad():
            actual.candidate_selector.successor_codebook.normal_(std=0.1)
            for layer in actual.backbone.layers:
                for conv in (layer.attention_conv, layer.mlp_conv):
                    conv.kernel_projection.weight.normal_(std=0.03)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, weight in weights.items():
            getattr(actual, name).load_target_weight(weight)
        other = copy.deepcopy(actual)
        other.gradient_checkpointing_enable()
        calls = count_layer_calls(other)
        args, options = model_inputs(config)
        output = score(actual, actual(*args), options)
        expected = score(other, other(*args), options)
        torch.testing.assert_close(output, expected, rtol=0, atol=0)
        grad = torch.randn_like(output)
        output.backward(grad)
        expected.backward(grad)
        self.assertEqual(calls, [2] * other.backbone.num_layers)
        for name, parameter in actual.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(other.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        with TemporaryDirectory() as path:
            actual.save_pretrained(path)
            restored = DFlash2DraftModel.from_pretrained(path)
        for name, weight in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(weight)
        torch.testing.assert_close(
            score(restored, restored(*args), options), output, rtol=0, atol=0
        )
        self.assertNotIn("lm_head.weight", actual.state_dict())
        self.assertTrue(all(e.target_context is None for e in args[2]))

    def test_initialization_noops_and_meta(self):
        config = dflash2_config()
        actual = build_draft_model(config)
        torch.testing.assert_close(
            actual.candidate_selector.successor_codebook,
            torch.zeros_like(actual.candidate_selector.successor_codebook),
            rtol=0,
            atol=0,
        )
        for layer in actual.backbone.layers:
            for conv in (layer.attention_conv, layer.mlp_conv):
                torch.testing.assert_close(
                    conv.kernel_projection.weight,
                    torch.zeros_like(conv.kernel_projection.weight),
                    rtol=0,
                    atol=0,
                )
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        identity = replace(
            config,
            spec_config=replace(
                config.spec_config, output_multiplier=1, final_logit_softcapping=None
            ),
        )
        identity_model = build_draft_model(identity)
        self.assertTrue(identity_model.unary_logits_transform_is_identity())
        self.assertFalse(actual.unary_logits_transform_is_identity())
        logits = torch.randn(2, 3, 64, dtype=torch.bfloat16)
        torch.testing.assert_close(
            identity_model.transform_unary_logits(logits),
            logits.float(),
            rtol=0,
            atol=0,
        )

    def test_materialized_meta_models_initialize_all_parameters(self):
        for config in (
            draft_config(target_bindings=(), target_logit_transform=None),
            dflash2_config(),
        ):
            with self.subTest(architecture=config.architecture):
                model = build_draft_model(config, device="meta")
                model.to_empty(device="cpu")
                with torch.no_grad():
                    for parameter in model.parameters():
                        parameter.fill_(float("nan"))
                model.init_weights()
                for name, parameter in model.named_parameters():
                    self.assertTrue(parameter.isfinite().all(), name)

    def test_convolution_is_optional(self):
        base = dflash2_config()
        config = replace(
            base,
            conv_configs={},
            layer_configs={"full_dense": LayerSpec(attention="full", mlp="dense")},
        )
        model = build_draft_model(config)
        args, _ = model_inputs(config)
        output = model(*args)
        self.assertTrue(output.logits.isfinite().all())
        self.assertIsNone(model.backbone.layers[0].attention_conv)
        self.assertIsNone(model.backbone.layers[0].mlp_conv)

    def test_hidden_only_output(self):
        actual = build_draft_model(dflash2_config())
        args, _ = model_inputs(actual.config.draft)
        hidden_only = actual(*args, output_logits=False)
        self.assertIsNone(hidden_only.logits)
        self.assertEqual(hidden_only.hidden_states.shape, (2, 8, 32))


if __name__ == "__main__":
    unittest.main()
