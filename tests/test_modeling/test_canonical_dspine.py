"""DSpine composition preserves injection order, intermediate gradients and refinement."""

import copy
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from safetensors.torch import load_file, save_file
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.architecture.block import AnchoredBlockSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.dflash_mask import create_dflash_sdpa_mask
from specforge.modeling.draft.dspine import DSpineDraftModel as LegacyDSpineDraftModel
from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DSpineDraftModel
from tests.test_config.test_draft_dspine_spec import dspine_config
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_dflash import legacy_name


def model_config():
    base = dspine_config()
    return replace(
        base,
        spec_config=replace(base.spec_config, block=AnchoredBlockSpec(block_size=4)),
    )


def initialized_model(device="cpu", dtype=torch.float32):
    model = build_draft_model(model_config(), device=device, dtype=dtype)
    head = torch.randn(64, 32, device=device, dtype=dtype)
    model.embed_tokens.load_target_weight(torch.randn_like(head))
    model.lm_head.load_target_weight(head)
    model.initialize_transfer_space(chunk_size=7)
    with torch.no_grad():
        for write in model.injection.writes:
            write.weight.normal_(std=0.1)
    return model


def model_inputs(config, device="cpu", dtype=torch.float32):
    anchors = torch.tensor([[1, 3], [2, 1]], device=device)
    positions = torch.cat(
        (
            torch.arange(5, device=device).expand(2, -1),
            (anchors[..., None] + torch.arange(4, device=device)).flatten(1),
        ),
        dim=1,
    )
    mask = create_dflash_sdpa_mask(
        anchors,
        torch.ones_like(anchors, dtype=torch.bool),
        5,
        4,
        device,
        is_causal=True,
    )
    layout = AttentionLayout(
        query_positions=positions[:, 5:], key_positions=positions, visibility=mask
    )
    entries = [
        LayerInputs(attention_layout=layout, position_ids=positions)
        for _ in config.layer_types
    ]
    ids = torch.randint(64, (2, 8), device=device)
    features = torch.randn(2, 5, 64, device=device, dtype=dtype, requires_grad=True)
    tokens = torch.randint(64, (2, 2, 4), device=device)
    replace_mask = torch.tensor([[True, False], [False, True]], device=device)
    return (ids, features, entries), tokens, replace_mask


class CanonicalDSpineTest(unittest.TestCase):
    @legacy_fp32_rope()
    def _parity(self, device, dtype):
        # Canonical RoPE computes in FP32 before casting back to the model dtype.
        tolerance = (
            2e-2
            if dtype == torch.bfloat16
            else 1e-2 if dtype == torch.float16 else 1e-5
        )
        if device == "cpu":
            tolerance = 0
        torch.manual_seed(0)
        for cutoff in (0, 1, 2):
            with self.subTest(device=device, dtype=dtype, cutoff=cutoff):
                actual = initialized_model(device, dtype)
                config = actual.config.draft
                legacy = legacy_dflash_config(
                    block_size=4,
                    dflash_config={"target_layer_ids": [1, 3], "mask_token_id": 63},
                    dspine_config={
                        "transfer_rank": 8,
                        "message_size": 4,
                        "replacement_layers": cutoff,
                    },
                )
                reference = LegacyDSpineDraftModel(legacy).to(
                    device=device, dtype=dtype
                )
                # Rebuild FP32 inv_freq on the test device to match canonical RoPE;
                # legacy .to(bfloat16) rounds it, changing the parity reference.
                with torch.device(device):
                    reference.rotary_emb = Qwen3RotaryEmbedding(legacy)
                reference.load_state_dict(
                    {legacy_name(n) or n: v for n, v in actual.state_dict().items()},
                    strict=True,
                )
                embedding, head = actual.embed_tokens, actual.lm_head
                (ids, features, entries), tokens, mask = model_inputs(
                    config, device, dtype
                )
                other = features.detach().clone().requires_grad_()
                output = actual(
                    ids,
                    features,
                    entries,
                    block_token_ids=tokens,
                    replace_masks=[mask if i < cutoff else None for i in range(2)],
                )
                expected = reference(
                    entries[0].position_ids,
                    attention_mask=entries[0].attention_layout.visibility,
                    noise_embedding=embedding(ids),
                    target_hidden=other,
                    block_token_ids=tokens,
                    replace_mask=mask,
                )
                expected_logits = head(expected.hidden_states)
                refined = actual.refine(
                    output.pre_injection[..., 1:, :], tokens[..., :-1]
                )
                expected_refined = reference.refine(
                    expected.pre_injection[..., 1:, :], tokens[..., :-1]
                )
                values = (
                    output.hidden_states,
                    output.pre_injection,
                    output.layer_features,
                    output.logits,
                    refined,
                )
                references = (*expected, expected_logits, expected_refined)
                gradients = [torch.randn_like(value) for value in values]
                torch.testing.assert_close(
                    values, references, rtol=tolerance, atol=tolerance
                )
                sum(
                    value.mul(grad).sum() for value, grad in zip(values, gradients)
                ).backward()
                sum(
                    value.mul(grad).sum() for value, grad in zip(references, gradients)
                ).backward()
                torch.testing.assert_close(
                    features.grad, other.grad, rtol=tolerance, atol=tolerance
                )
                for name, parameter in actual.named_parameters():
                    if not parameter.requires_grad:
                        self.assertIsNone(parameter.grad, name)
                        continue
                    ref = dict(reference.named_parameters())[legacy_name(name) or name]
                    self.assertIsNotNone(parameter.grad, name)
                    torch.testing.assert_close(
                        parameter.grad, ref.grad, rtol=tolerance, atol=tolerance
                    )

    def test_cpu_legacy_composition(self):
        for dtype in (torch.float32, torch.bfloat16):
            self._parity("cpu", dtype)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_legacy_composition(self):
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            self._parity("cuda", dtype)

    def test_explicit_late_replacement_and_block_isolation(self):
        model = initialized_model()
        args, tokens, mask = model_inputs(model.config.draft)
        output = model(*args, block_token_ids=tokens, replace_masks=[None, mask])
        plain = model(*args, block_token_ids=tokens)
        self.assertFalse(torch.equal(output.hidden_states, plain.hidden_states))
        changed = tokens.clone()
        changed[:, 0] = (changed[:, 0] + 1) % 64
        other = model(*args, block_token_ids=changed, replace_masks=[None, mask])
        torch.testing.assert_close(
            other.hidden_states[:, 1], output.hidden_states[:, 1], rtol=0, atol=0
        )
        # Causal decoder and adjacent injection cannot send future input changes backward.
        ids = args[0].clone()
        ids[:, 3] = (ids[:, 3] + 1) % 64
        future = model(
            ids, *args[1:], block_token_ids=tokens, replace_masks=[None, mask]
        )
        torch.testing.assert_close(
            future.hidden_states[:, 0, :3],
            output.hidden_states[:, 0, :3],
            rtol=0,
            atol=0,
        )

    def test_checkpointing_and_hf_restore_keep_transfer_state(self):
        model = initialized_model()
        checkpointed = copy.deepcopy(model)
        checkpointed.gradient_checkpointing_enable()
        args, tokens, mask = model_inputs(model.config.draft)
        output = model(*args, block_token_ids=tokens, replace_masks=[mask, None])
        expected = checkpointed(
            *args, block_token_ids=tokens, replace_masks=[mask, None]
        )
        torch.testing.assert_close(output.logits, expected.logits, rtol=0, atol=0)
        (output.logits.sum() + output.layer_features.sum()).backward()
        (expected.logits.sum() + expected.layer_features.sum()).backward()
        for name, parameter in model.named_parameters():
            self.assertEqual(parameter.grad is not None, parameter.requires_grad, name)
            if parameter.requires_grad:
                torch.testing.assert_close(
                    parameter.grad,
                    dict(checkpointed.named_parameters())[name].grad,
                    rtol=0,
                    atol=0,
                )
        self.assertFalse(model.transfer_codes.requires_grad)
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            restored = DSpineDraftModel.from_pretrained(path)
        self.assertTrue(restored.transfer_ready.item())
        for name in model.config.draft.target_bindings:
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(getattr(model, name).weight)
        result = restored(*args, block_token_ids=tokens, replace_masks=[mask, None])
        torch.testing.assert_close(
            restored.state_dict(), model.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(result.logits, output.logits, rtol=0, atol=0)
        self.assertNotIn("lm_head.weight", model.state_dict())

    def test_transfer_space_uses_bound_head(self):
        model = build_draft_model(model_config())
        with self.assertRaisesRegex(RuntimeError, "bound"):
            model.initialize_transfer_space()
        model.lm_head.load_target_weight(torch.randn(64, 32))
        with patch(
            "specforge.modeling.draft.models.dspine.initialize_transfer_space"
        ) as initialize:
            model.initialize_transfer_space(chunk_size=7)
        initialize.assert_called_once_with(
            model.lm_head.bound_weight(),
            model.transfer_codes,
            model.injection.readouts,
            chunk_size=7,
        )
        ids = torch.tensor([[1, 7, 9]])
        weights = model.lm_head.candidate_weights(ids)
        torch.testing.assert_close(weights, model.lm_head.bound_weight()[ids])
        self.assertFalse(weights.requires_grad)

    def test_transfer_space_cannot_overwrite_trained_state(self):
        model = initialized_model()
        before = copy.deepcopy(model.state_dict())
        with self.assertRaisesRegex(RuntimeError, "already initialized"):
            model.initialize_transfer_space()
        torch.testing.assert_close(model.state_dict(), before, rtol=0, atol=0)

    def test_hf_restore_rejects_incomplete_or_unexpected_state(self):
        model = initialized_model()
        with TemporaryDirectory() as path:
            model.save_pretrained(path)
            weights = Path(path) / "model.safetensors"
            original = load_file(weights)
            for key in (
                "transfer_codes",
                "injection.readouts.0.weight",
                "extra.tensor",
            ):
                with self.subTest(key=key):
                    state = dict(original)
                    if key == "extra.tensor":
                        state[key] = torch.zeros(1)
                    else:
                        del state[key]
                    save_file(state, weights)
                    with self.assertRaises(ValueError) as error:
                        DSpineDraftModel.from_pretrained(path)
                    self.assertIn(key, str(error.exception))
            save_file(original, weights)
            restored, info = DSpineDraftModel.from_pretrained(
                path, output_loading_info=True
            )
            self.assertFalse(info["missing_keys"])
            self.assertTrue(restored.transfer_ready.item())

    def test_uninitialized_hidden_only_meta_and_invalid_inputs(self):
        config = model_config()
        model = build_draft_model(config)
        args, tokens, mask = model_inputs(config)
        with self.assertRaisesRegex(RuntimeError, "transfer space"):
            model(*args, block_token_ids=tokens)
        model.embed_tokens.load_target_weight(torch.randn(64, 32))
        model.lm_head.load_target_weight(torch.randn(64, 32))
        model.initialize_transfer_space()
        self.assertTrue(
            all(
                torch.count_nonzero(write.weight) == 0
                for write in model.injection.writes
            )
        )
        self.assertIsNone(
            model(*args, block_token_ids=tokens, output_logits=False).logits
        )
        for kwargs in (
            {"block_token_ids": tokens.float()},
            {"block_token_ids": tokens, "replace_masks": [mask]},
            {"block_token_ids": tokens, "replace_masks": [mask.float(), None]},
        ):
            with self.assertRaises(ValueError):
                model(*args, **kwargs)
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        self.assertTrue(meta.transfer_codes.is_meta)
        self.assertEqual(meta.transfer_ready.dtype, torch.bool)
        # Mixed-precision distributed wrappers may retain fixed buffers in FP32.
        model = initialized_model(dtype=torch.bfloat16)
        model.transfer_codes = model.transfer_codes.float()
        args, tokens, _ = model_inputs(config, dtype=torch.bfloat16)
        output = model(*args, block_token_ids=tokens)
        refined = model.refine(output.pre_injection[..., 1:, :], tokens[..., :-1])
        (output.logits.float().sum() + refined.float().sum()).backward()
        self.assertEqual(model.transfer_codes.dtype, torch.float32)
        self.assertTrue(torch.isfinite(refined).all())


if __name__ == "__main__":
    unittest.main()
