"""The canonical DFlash model preserves full forward/backward and bindings."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.logits import MuPLogitTransformSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dflash import DFlashDraftModel as LegacyDFlashDraftModel
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DFlashDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.test_draft_gqa import layout_for


def inputs_for(config, device="cpu", dtype=torch.float32):
    ids = torch.tensor([[1, 3, 4, 7], [2, 5, 6, 8]], device=device)
    features = torch.randn(
        2, 3, config.hidden_size * 2, device=device, dtype=dtype, requires_grad=True
    )
    layout = layout_for(
        torch.empty(2, 4, config.hidden_size, device=device), context_length=3
    )
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    return ids, features, entries


def legacy_name(name):
    for prefix, replacement in (
        ("backbone.layers.", "layers."),
        ("backbone.final_norm.", "norm."),
        ("fusion.", ""),
    ):
        if name.startswith(prefix):
            return replacement + name[len(prefix) :]
    return None


class CanonicalDFlashTest(unittest.TestCase):
    def _legacy_parity(self, device, dtype):
        config = draft_config(target_bindings=(), target_logit_transform=None)
        actual = build_draft_model(config, device=device, dtype=dtype)
        legacy_config = legacy_dflash_config(
            block_size=8,
            dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 63},
        )
        reference = LegacyDFlashDraftModel(legacy_config).to(device=device, dtype=dtype)
        # Canonical RoPE derives FP32 frequencies after module casts. Rebuild the
        # legacy non-persistent buffer so both models use that same frequency
        # rule; casting the legacy buffer to BF16/FP16 would quantize it first.
        reference.rotary_emb = Qwen3RotaryEmbedding(legacy_config, device=device)
        reference.load_state_dict(
            {
                legacy_name(name): value
                for name, value in actual.state_dict().items()
                if legacy_name(name) is not None
            },
            strict=True,
        )
        embeddings, head = copy.deepcopy(actual.embed_tokens), copy.deepcopy(
            actual.lm_head
        )
        ids, features, entries = inputs_for(config, device, dtype)
        other_features = features.detach().clone().requires_grad_()
        output = actual(ids, features, entries)
        # Bidirectional full attention matches the fixture's explicit node.
        expected_hidden = reference(
            entries[0].position_ids,
            noise_embedding=embeddings(ids),
            target_hidden=other_features,
        )
        expected = head(expected_hidden)
        torch.testing.assert_close(
            output.hidden_states, expected_hidden, rtol=0, atol=0
        )
        torch.testing.assert_close(output.logits, expected, rtol=0, atol=0)
        upstream = torch.randn_like(expected)
        output.logits.backward(upstream)
        expected.backward(upstream)
        torch.testing.assert_close(features.grad, other_features.grad, rtol=0, atol=0)
        for name, parameter in actual.named_parameters():
            if legacy_name(name) is not None:
                ref = dict(reference.named_parameters())[legacy_name(name)]
            else:
                ref = (
                    embeddings.weight if name == "embed_tokens.weight" else head.weight
                )
            self.assertIsNotNone(parameter.grad, name)
            torch.testing.assert_close(parameter.grad, ref.grad, rtol=0, atol=0)

    def test_cpu_complete_legacy_forward_and_backward(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._legacy_parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_complete_legacy_forward_and_backward(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._legacy_parity("cuda", dtype)

    def test_bindings_round_trip_and_only_trainable_parameters_receive_gradients(self):
        for bindings in (
            (),
            ("embed_tokens",),
            ("lm_head",),
            ("embed_tokens", "lm_head"),
        ):
            with self.subTest(bindings=bindings):
                config = draft_config(
                    target_bindings=bindings,
                    target_logit_transform=(
                        MuPLogitTransformSpec(multiplier=3)
                        if "lm_head" in bindings
                        else None
                    ),
                )
                model = build_draft_model(config)
                self.assertIsInstance(model, DFlashDraftModel)
                self.assertIs(model.config.draft, config)
                values = {name: torch.randn(64, 32) for name in bindings}
                args = inputs_for(config)
                if bindings:
                    with self.assertRaisesRegex(RuntimeError, "not bound"):
                        model(*args)
                for name, value in values.items():
                    getattr(model, name).load_target_weight(value)
                output = model(*args)
                output.logits.sum().backward()
                for name, parameter in model.named_parameters():
                    self.assertEqual(
                        parameter.grad is not None, parameter.requires_grad, name
                    )
                state = model.state_dict()
                for name in bindings:
                    self.assertNotIn(name + ".weight", state)
                restored = build_draft_model(config)
                restored.load_state_dict(state, strict=True)
                for name, value in values.items():
                    getattr(restored, name).load_target_weight(value)
                torch.testing.assert_close(
                    restored(*args).logits, output.logits, rtol=0, atol=0
                )
                self.assertTrue(all(entry.target_context is None for entry in args[2]))

    def test_checkpointing_tied_weights_and_meta_construction(self):
        config = draft_config(
            target_bindings=(), target_logit_transform=None, tie_word_embeddings=True
        )
        model = build_draft_model(config)
        self.assertIs(model.embed_tokens.weight, model.lm_head.weight)
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing_enable()
        ids, features, entries = inputs_for(config)
        other = features.detach().clone().requires_grad_()
        output, expected = model(ids, features, entries), checkpointed(
            ids, other, entries
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        upstream = torch.randn_like(output.logits)
        output.logits.backward(upstream)
        expected.logits.backward(upstream)
        torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(
                parameter.grad,
                dict(checkpointed.named_parameters())[name].grad,
                rtol=0,
                atol=0,
            )
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )

    def test_hf_round_trip_preserves_configuration_and_requires_rebinding(self):
        config = draft_config()
        model = build_draft_model(config)
        values = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, value in values.items():
            getattr(model, name).load_target_weight(value)
        args = inputs_for(config)
        expected = model(*args).logits
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored = DFlashDraftModel.from_pretrained(path)
        self.assertEqual(restored.config.draft, config)
        with self.assertRaisesRegex(RuntimeError, "not bound"):
            restored(*args)
        for name, value in values.items():
            bound = getattr(restored, name)
            # Omitted target parameters remain meta in HF's low-memory loader.
            # Materialize only these modules, never the restored draft weights.
            if bound.weight.is_meta:
                bound.to_empty(device=restored.fusion.fc.weight.device)
            bound.load_target_weight(value)
        torch.testing.assert_close(restored(*args).logits, expected, rtol=0, atol=0)

    def test_explicit_hidden_only_output_and_rejecting_context_bypass(self):
        config = draft_config(target_bindings=("lm_head",))
        model = build_draft_model(config)
        ids, features, entries = inputs_for(config)
        with patch.object(model.lm_head, "forward") as head:
            output = model(ids, features, entries, output_logits=False)
            self.assertIsNone(output.logits)
            self.assertEqual(output.hidden_states.shape, (2, 4, 32))
            head.assert_not_called()
        entries[0] = replace(entries[0], target_context=torch.randn(2, 3, 32))
        with self.assertRaisesRegex(ValueError, "trainable fusion"):
            model(ids, features, entries)


if __name__ == "__main__":
    unittest.main()
