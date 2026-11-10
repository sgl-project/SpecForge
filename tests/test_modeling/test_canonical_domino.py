"""Domino composition preserves shifted alignment and independent block state."""

import copy
import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from specforge.config.draft.architecture.domino import DominoBlockSpec
from specforge.modeling import build_draft_model
from specforge.modeling.draft.domino import DominoDraftModel as LegacyDominoDraftModel
from specforge.modeling.draft.layers.inputs import LayerInputs
from specforge.modeling.draft.models import DominoDraftModel
from tests.test_config.draft_fixtures import draft_config
from tests.test_config.test_draft_domino_spec import domino_spec
from tests.test_modeling.legacy_fixtures import legacy_dflash_config
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_dflash import legacy_name
from tests.test_modeling.test_draft_gqa import layout_for


def model_config(*, shifted=False, prefix=1, bound=False):
    return draft_config(
        architecture="DominoDraftModel",
        spec_config=domino_spec(
            block=DominoBlockSpec(block_size=4, shift_label=shifted),
            pure_draft_prefix_len=prefix,
        ),
        **({} if bound else {"target_bindings": (), "target_logit_transform": None}),
    )


def model_inputs(config, device="cpu", dtype=torch.float32):
    ids = torch.arange(16, device=device).reshape(2, 8)
    features = torch.randn(2, 3, 64, device=device, dtype=dtype, requires_grad=True)
    layout = layout_for(torch.empty(2, 8, 32, device=device), context_length=3)
    entries = [
        LayerInputs(attention_layout=layout, position_ids=layout.key_positions)
        for _ in config.layer_types
    ]
    return (ids, features, entries), torch.randint(64, (2, 8), device=device)


def reference_name(name):
    return legacy_name(name) or name.removeprefix("gru_head.")


class CanonicalDominoTest(unittest.TestCase):
    @legacy_fp32_rope()
    def _parity(self, device, dtype):
        # Canonical RoPE computes in FP32 before casting back to the model dtype.
        tolerance = (
            2e-2
            if dtype == torch.bfloat16
            else 1e-2 if dtype == torch.float16 else 1e-5
        )
        torch.manual_seed(0)
        for shifted, prefix in ((False, 0), (False, 2), (True, 0), (True, 3)):
            with self.subTest(
                device=device, dtype=dtype, shifted=shifted, prefix=prefix
            ):
                config = model_config(shifted=shifted, prefix=prefix)
                actual = build_draft_model(config, device=device, dtype=dtype)
                legacy = legacy_dflash_config(
                    block_size=4,
                    dflash_config={
                        "target_layer_ids": [1, 3],
                        "mask_token_id": 63,
                        "emb_dim": 8,
                        "gru_hidden_dim": 16,
                        "pure_draft_prefix_len": prefix,
                        "shift_label": shifted,
                    },
                )
                reference = LegacyDominoDraftModel(legacy).to(
                    device=device, dtype=dtype
                )
                # Canonical RoPE derives frequencies in FP32 even for low-precision weights.
                reference.rotary_emb = Qwen3RotaryEmbedding(legacy).to(device)
                reference.load_state_dict(
                    {
                        reference_name(n): v
                        for n, v in actual.state_dict().items()
                        if n not in ("embed_tokens.weight", "lm_head.weight")
                    },
                    strict=True,
                )
                embedding, head = copy.deepcopy(actual.embed_tokens), copy.deepcopy(
                    actual.lm_head
                )
                (ids, features, entries), previous = model_inputs(config, device, dtype)
                other = features.detach().clone().requires_grad_()
                output = actual(ids, features, entries, prev_token_ids=previous)
                hidden = reference(
                    entries[0].position_ids,
                    noise_embedding=embedding(ids),
                    target_hidden=other,
                )
                states = hidden.reshape(2, 2, 4, 32)
                previous_embeddings = embedding(previous).reshape_as(states)
                correction = reference.compute_correction_logits(
                    hidden_states=states, prev_token_embeddings=previous_embeddings
                )
                base = head(hidden).reshape(2, 2, 4, 64)
                start = reference.suffix_start
                logits = torch.cat(
                    (base[..., :start, :], base[..., start:, :] + correction), dim=-2
                ).reshape(2, 8, 64)
                torch.testing.assert_close(
                    (output.hidden_states, output.logits, output.correction_logits),
                    (hidden, logits, correction),
                    rtol=tolerance,
                    atol=tolerance,
                )
                torch.testing.assert_close(
                    output.logits.reshape_as(base)[..., :start, :],
                    base[..., :start, :],
                    rtol=tolerance,
                    atol=tolerance,
                )
                grad = torch.randn_like(logits)
                output.logits.mul(grad).sum().backward()
                logits.mul(grad).sum().backward()
                torch.testing.assert_close(
                    features.grad, other.grad, rtol=tolerance, atol=tolerance
                )
                for name, parameter in actual.named_parameters():
                    ref = (
                        embedding.weight
                        if name == "embed_tokens.weight"
                        else (
                            head.weight
                            if name == "lm_head.weight"
                            else dict(reference.named_parameters())[
                                reference_name(name)
                            ]
                        )
                    )
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

    def test_gru_resets_at_block_boundaries(self):
        for shifted in (False, True):
            config = model_config(shifted=shifted)
            actual = build_draft_model(config)
            args, previous = model_inputs(config)
            output = actual(*args, prev_token_ids=previous)
            changed = previous.clone()
            changed[:, :4] = (changed[:, :4] + 1) % 64
            other = actual(*args, prev_token_ids=changed)
            torch.testing.assert_close(
                other.logits[:, 4:], output.logits[:, 4:], rtol=0, atol=0
            )
            torch.testing.assert_close(
                other.correction_logits[:, 1],
                output.correction_logits[:, 1],
                rtol=0,
                atol=0,
            )
            self.assertFalse(
                torch.equal(
                    other.correction_logits[:, 0], output.correction_logits[:, 0]
                )
            )

    def test_binding_checkpoint_and_hf_restore(self):
        config = model_config(bound=True)
        actual = build_draft_model(config)
        weights = {name: torch.randn(64, 32) for name in config.target_bindings}
        for name, value in weights.items():
            getattr(actual, name).load_target_weight(value)
        checkpointed = copy.deepcopy(actual)
        checkpointed.gradient_checkpointing_enable()
        args, previous = model_inputs(config)
        output = actual(*args, prev_token_ids=previous)
        expected = checkpointed(*args, prev_token_ids=previous)
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
            restored = DominoDraftModel.from_pretrained(path)
        for name, value in weights.items():
            module = getattr(restored, name)
            if module.weight.is_meta:
                module.to_empty(device="cpu")
            module.load_target_weight(value)
        result = restored(*args, prev_token_ids=previous)
        torch.testing.assert_close(result.logits, output.logits, rtol=0, atol=0)
        torch.testing.assert_close(
            result.correction_logits, output.correction_logits, rtol=0, atol=0
        )
        self.assertNotIn("lm_head.weight", actual.state_dict())

    def test_hidden_only_meta_tying_and_input_contract(self):
        config = model_config()
        actual = build_draft_model(config)
        args, previous = model_inputs(config)
        hidden = actual(*args, output_logits=False)
        self.assertIsNone(hidden.logits)
        self.assertIsNone(hidden.correction_logits)
        with self.assertRaisesRegex(ValueError, "requires prev_token_ids"):
            actual(*args)
        with self.assertRaisesRegex(ValueError, "whole draft blocks"):
            actual(args[0][:, :-1], args[1], args[2], prev_token_ids=previous)
        with self.assertRaisesRegex(ValueError, "integer IDs"):
            actual(*args, prev_token_ids=previous.float())
        meta = build_draft_model(config, device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in meta.parameters())
        )
        tied = build_draft_model(replace(config, tie_word_embeddings=True))
        self.assertIs(tied.embed_tokens.weight, tied.lm_head.weight)


if __name__ == "__main__":
    unittest.main()
