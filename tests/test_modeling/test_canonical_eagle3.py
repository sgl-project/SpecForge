"""Full EAGLE3 composition preserves feature projection and recurrent outputs."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from transformers import LlamaConfig

from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.llama3_eagle import (
    LlamaForCausalLMEagle3,
    LlamaRotaryEmbedding,
)
from specforge.modeling.draft.models import Eagle3DraftModel
from tests.test_modeling.test_draft_eagle_backbone import config_for
from tests.test_modeling.test_draft_eagle_fusion import eager_reference
from tests.test_modeling.test_draft_gqa import layout_for


def model_config(*, normalize=False, norm_output=False, bound=False, target_width=48):
    base = config_for("eagle3", norm_output)
    return replace(
        base,
        target_bindings=("embed_tokens",) if bound else (),
        spec_config=replace(
            base.spec_config,
            normalize_features=normalize,
            target_features=replace(
                base.spec_config.target_features, hidden_size=target_width
            ),
        ),
    )


def install_mapping(model):
    vocabulary = model.vocabulary
    selected = torch.arange(vocabulary.draft_vocab_size) * 2
    mask = torch.zeros(vocabulary.vocab_size, dtype=torch.bool)
    mask[selected] = True
    vocabulary.load_mapping(t2d=mask, d2t=selected - torch.arange(len(selected)))


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.arange(8, device=device).reshape(2, 4)
    features = torch.randn(
        2,
        4,
        config.spec_config.target_features.input_width,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    layout = layout_for(torch.empty(2, 4, 32, device=device))
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    return ids, features, entries


def sdpa_eager(query, key, value, *, attn_mask, **kwargs):
    return eager_reference(
        query=query, key=key, value=value, block_mask=attn_mask, enable_gqa=True
    )


def legacy_name(name):
    for prefix, target in (
        ("backbone.layers.0.", "midlayer."),
        ("backbone.final_norm.", "norm."),
        ("logit_norm.", "norm."),
        ("fusion.", ""),
        ("vocabulary.", ""),
    ):
        if name.startswith(prefix):
            return target + name[len(prefix) :]
    return name


class CanonicalEagle3Test(unittest.TestCase):
    # Compare eager math on both sides, without compiler fusion differences.
    @patch("torch._dynamo.config.disable", True)
    def _parity(self, device, dtype):
        for normalize, norm_output in (
            (False, False),
            (False, True),
            (True, False),
            (True, True),
        ):
            with self.subTest(
                device=device, dtype=dtype, normalize=normalize, norm_output=norm_output
            ):
                config = model_config(normalize=normalize, norm_output=norm_output)
                actual = build_draft_model(config, device=device, dtype=dtype)
                install_mapping(actual)
                legacy = LlamaConfig(
                    hidden_size=32,
                    target_hidden_size=48,
                    intermediate_size=64,
                    num_attention_heads=4,
                    num_key_value_heads=2,
                    head_dim=8,
                    num_hidden_layers=1,
                    vocab_size=64,
                    draft_vocab_size=32,
                    max_position_embeddings=128,
                    rms_norm_eps=1e-6,
                    hidden_act="silu",
                    pretraining_tp=1,
                    fc_norm=normalize,
                    norm_output=norm_output,
                    rope_parameters={"rope_type": "default", "rope_theta": 10000.0},
                )
                reference = LlamaForCausalLMEagle3(legacy).to(
                    device=device, dtype=dtype
                )
                # Keep legacy frequency buffers in FP32, matching canonical RoPE.
                reference.midlayer.self_attn.rotary_emb = LlamaRotaryEmbedding(
                    8, max_position_embeddings=128, base=10000, device=device
                )
                reference.load_state_dict(
                    {
                        legacy_name(n): v
                        for n, v in actual.state_dict().items()
                        if n != "vocabulary.ready"
                    },
                    strict=True,
                )
                ids, features, entries = model_inputs(config, device, dtype)
                other = features.detach().clone().requires_grad_()
                mask = (
                    entries[0].attention_layout.key_positions[:, None, None, :]
                    <= entries[0].attention_layout.query_positions[:, None, :, None]
                )
                first = actual(ids, features, entries)
                # Match the composed stage's concat/split boundary so residual
                # gradients accumulate before returning to the recurrent input.
                projected = reference.project_hidden_states(other)
                embeds, projected = torch.cat(
                    (reference.embed_tokens(ids), projected), dim=-1
                ).split(32, dim=-1)
                with patch(
                    "torch.nn.functional.scaled_dot_product_attention", sdpa_eager
                ):
                    initial = reference.backbone(
                        embeds,
                        projected,
                        None,
                        mask,
                        entries[0].position_ids,
                        use_cache=False,
                    )
                initial_logits = reference.compute_logits(initial)
                second = actual(
                    ids, None, entries, recurrent_hidden=first.hidden_states
                )
                embeds, projected = torch.cat(
                    (reference.embed_tokens(ids), initial), dim=-1
                ).split(32, dim=-1)
                with patch(
                    "torch.nn.functional.scaled_dot_product_attention", sdpa_eager
                ):
                    recurrent = reference.backbone(
                        embeds,
                        projected,
                        None,
                        mask,
                        entries[0].position_ids,
                        use_cache=False,
                    )
                recurrent_logits = reference.compute_logits(recurrent)
                values = (
                    first.hidden_states,
                    first.logits,
                    second.hidden_states,
                    second.logits,
                )
                expected = (initial, initial_logits, recurrent, recurrent_logits)
                torch.testing.assert_close(values, expected, rtol=0, atol=0)
                upstream = [torch.randn_like(value) for value in values]
                sum(
                    value.mul(grad).sum() for value, grad in zip(values, upstream)
                ).backward()
                sum(
                    value.mul(grad).sum() for value, grad in zip(expected, upstream)
                ).backward()
                torch.testing.assert_close(features.grad, other.grad, rtol=0, atol=0)
                for name, parameter in actual.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad,
                        dict(reference.named_parameters())[legacy_name(name)].grad,
                        rtol=0,
                        atol=0,
                    )

    def test_cpu_legacy_composition(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_composition(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_checkpointing_binding_and_hf_restore(self):
        config = model_config(normalize=True, bound=True, target_width=32)
        actual = build_draft_model(config)
        install_mapping(actual)
        embedding = torch.randn(64, 32)
        actual.embed_tokens.load_target_weight(embedding)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args = model_inputs(config)
        output, expected = actual(*args), checkpointed(*args)
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
            restored = Eagle3DraftModel.from_pretrained(path)
        if restored.embed_tokens.weight.is_meta:
            restored.embed_tokens.to_empty(device="cpu")
        restored.embed_tokens.load_target_weight(embedding)
        torch.testing.assert_close(
            restored(*args).logits, output.logits, rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.vocabulary.to_target(torch.arange(32)),
            torch.arange(32) * 2,
            rtol=0,
            atol=0,
        )
        self.assertNotIn("embed_tokens.weight", actual.state_dict())

    def test_hidden_only_recurrent_contract_and_meta(self):
        config = model_config()
        actual = build_draft_model(config)
        ids, features, entries = model_inputs(config)
        hidden = actual(ids, features, entries, output_logits=False).hidden_states
        with self.assertRaisesRegex(RuntimeError, "mapping"):
            actual.compute_logits(hidden)
        with patch.object(
            actual.fusion,
            "forward",
            side_effect=AssertionError("projection must be skipped"),
        ):
            actual(ids, None, entries, recurrent_hidden=hidden, output_logits=False)
        for target, recurrent in (
            (features, hidden),
            (None, None),
            (None, hidden[..., :-1]),
        ):
            with self.assertRaises(ValueError):
                actual(
                    ids,
                    target,
                    entries,
                    recurrent_hidden=recurrent,
                    output_logits=False,
                )
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        full = build_draft_model(
            replace(config, draft_vocab_size=64, tie_word_embeddings=True)
        )
        self.assertIs(full.embed_tokens.weight, full.lm_head.weight)
        self.assertTrue(full.vocabulary.ready.item())


if __name__ == "__main__":
    unittest.main()
