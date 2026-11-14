"""Full EAGLE3 composition preserves feature projection and recurrent outputs."""

import copy
import unittest
from dataclasses import replace
from itertools import product
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.config.draft.rope import DynamicNTKRoPESpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import Eagle3DraftModel
from tests.test_modeling.test_draft_eagle_backbone import config_for
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


class CanonicalEagle3Test(unittest.TestCase):

    def test_projected_prefix_matches_replay_and_gradients(self):
        config = model_config(normalize=True)
        variants = (
            config,
            replace(
                config,
                rope_parameters=DynamicNTKRoPESpec(
                    interleaved=False,
                    rope_theta=10000.0,
                    partial_rotary_factor=1.0,
                    factor=2.0,
                    original_max_position_embeddings=4,
                ),
            ),
        )
        devices = ("cpu", "cuda") if torch.cuda.is_available() else ("cpu",)
        for config, device in product(variants, devices):
            for checkpointing in (False, True):
                with self.subTest(
                    attention=type(config.attention_configs["full"]).__name__,
                    rope=type(config.rope_parameters).__name__,
                    device=device,
                    checkpointing=checkpointing,
                ):
                    cached = build_draft_model(config, device=device)
                    install_mapping(cached)
                    replay = copy.deepcopy(cached)
                    if checkpointing:
                        cached.gradient_checkpointing_enable()
                    ids, features, entries = model_inputs(config, device)
                    other = features.detach().clone().requires_grad_()
                    first = cached(ids, features, entries)
                    reference = replay(ids, other, entries)
                    positions = torch.arange(8, device=device).expand(2, -1)
                    # Earlier queries cannot see the newly appended depth.
                    visibility = (
                        positions[:, None, None, :] <= positions[:, None, :, None]
                    )
                    layout = replace(
                        entries[0].attention_layout,
                        query_positions=positions,
                        key_positions=positions,
                        visibility=visibility,
                    )
                    full_inputs = [
                        LayerInputs(attention_layout=layout, position_ids=positions)
                    ]
                    next_inputs = [
                        LayerInputs(
                            attention_layout=replace(
                                layout,
                                query_positions=positions[:, 4:],
                                visibility=visibility[:, :, 4:],
                            ),
                            position_ids=positions[:, 4:],
                        )
                    ]
                    second = cached(
                        ids,
                        None,
                        next_inputs,
                        recurrent_hidden=first.hidden_states,
                        kv_prefix=first.kv,
                    )
                    expected = replay(
                        torch.cat((ids, ids), dim=1),
                        None,
                        full_inputs,
                        recurrent_hidden=torch.cat(
                            (replay.fusion(other), reference.hidden_states), dim=1
                        ),
                    )
                    torch.testing.assert_close(
                        second.logits, expected.logits[:, 4:], rtol=1e-5, atol=1e-5
                    )
                    self.assertEqual(second.kv.key.shape[2], 4)
                    self.assertTrue(second.kv.key.requires_grad)
                    (
                        first.logits.square().sum() + second.logits.square().sum()
                    ).backward()
                    (
                        reference.logits.square().sum()
                        + expected.logits[:, 4:].square().sum()
                    ).backward()
                    torch.testing.assert_close(
                        features.grad, other.grad, rtol=1e-4, atol=1e-5
                    )
                    for name, parameter in cached.named_parameters():
                        torch.testing.assert_close(
                            parameter.grad,
                            dict(replay.named_parameters())[name].grad,
                            rtol=1e-4,
                            atol=1e-5,
                            msg=name,
                        )

    def test_autocast_fusion_preserves_feature_dtype(self):
        from tests.test_modeling.test_canonical_peagle import (
            model_config as peagle_config,
        )

        for config in (model_config(), peagle_config()):
            model = build_draft_model(config, backend="sdpa")
            ids, features, entries = model_inputs(config, dtype=torch.bfloat16)
            with (
                torch.autocast("cpu", dtype=torch.bfloat16),
                patch.object(
                    model.backbone, "begin", wraps=model.backbone.begin
                ) as begin,
            ):
                output = model(ids, features, entries, output_logits=False)
            self.assertEqual(begin.call_args.args[0].dtype, torch.bfloat16)
            output.hidden_states.float().sum().backward()
            self.assertIsNotNone(features.grad)

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
        self.assertFalse(restored.embed_tokens.weight.is_meta)
        restored.to("cpu")
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
        self.assertTrue(full.vocabulary.ready)


if __name__ == "__main__":
    unittest.main()
