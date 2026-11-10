"""Aligned Domino heads retain GRU numerics, gradients and HF loading."""

import unittest
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.config.draft.architecture.domino import GRUHeadSpec
from specforge.modeling.draft.heads import DominoGRUHead
from specforge.modeling.draft.pretrained import DraftPreTrainedModel
from tests.test_modeling.test_domino_model import _bare_domino
from tests.test_modeling.test_draft_pretrained import model_config


class DFlashDraftModel(DraftPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.head = DominoGRUHead(
            GRUHeadSpec(hidden_size=3, projection_size=2), hidden_size=4, vocab_size=7
        )
        self.post_init()


class DominoGRUTest(unittest.TestCase):
    def test_aligned_head_matches_legacy_interface_for_both_label_conventions(self):
        cases = [("cpu", torch.float32), ("cpu", torch.bfloat16)]
        if torch.cuda.is_available():
            cases.extend(
                ("cuda", dtype)
                for dtype in (torch.float32, torch.float16, torch.bfloat16)
            )
        for device, dtype in cases:
            for shifted in (False, True):
                for prefix in (0, 1, 2):
                    with self.subTest(
                        device=device, dtype=dtype, shifted=shifted, prefix=prefix
                    ):
                        legacy = _bare_domino().to(device=device, dtype=dtype)
                        legacy.shift_label, legacy.pure_draft_prefix_len = (
                            shifted,
                            prefix,
                        )
                        head = DominoGRUHead(
                            GRUHeadSpec(hidden_size=3, projection_size=2),
                            hidden_size=4,
                            vocab_size=7,
                            device=device,
                            dtype=dtype,
                        )
                        head.load_state_dict(legacy.state_dict(), strict=True)
                        hidden = torch.randn(
                            2, 2, 4, 4, device=device, dtype=dtype, requires_grad=True
                        )
                        embeddings = torch.randn_like(hidden, requires_grad=True)
                        other_hidden = hidden.detach().clone().requires_grad_()
                        other_embeddings = embeddings.detach().clone().requires_grad_()
                        # Independent alignment reference from the original Domino path.
                        prefix_tokens = embeddings if shifted else embeddings[:, :, :-1]
                        states = legacy.prefix_gru(
                            prefix_tokens.reshape(-1, prefix_tokens.shape[-2], 4)
                        )[0]
                        states = states.reshape(2, 2, prefix_tokens.shape[-2], 3)
                        start = (
                            legacy.suffix_start if shifted else legacy.suffix_start - 1
                        )
                        expected = legacy.embed_proj(
                            torch.cat(
                                (
                                    hidden[:, :, legacy.suffix_start :],
                                    states[:, :, start:],
                                ),
                                -1,
                            )
                        )
                        torch.testing.assert_close(
                            legacy.compute_correction_logits(
                                prev_token_embeddings=embeddings, hidden_states=hidden
                            ),
                            expected,
                            rtol=0,
                            atol=0,
                        )
                        actual = head(
                            other_hidden[:, :, legacy.suffix_start :],
                            (
                                other_embeddings
                                if shifted
                                else other_embeddings[:, :, :-1]
                            ),
                        )
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                        upstream = torch.randn_like(actual)
                        actual.backward(upstream)
                        expected.backward(upstream)
                        for a, b in (
                            (other_hidden, hidden),
                            (other_embeddings, embeddings),
                        ):
                            torch.testing.assert_close(a.grad, b.grad, rtol=0, atol=0)
                        for name, parameter in head.named_parameters():
                            torch.testing.assert_close(
                                parameter.grad,
                                dict(legacy.named_parameters())[name].grad,
                                rtol=0,
                                atol=0,
                            )

    def test_prefix_causality_and_empty_suffix(self):
        head = DominoGRUHead(
            GRUHeadSpec(hidden_size=3, projection_size=2), hidden_size=4, vocab_size=7
        )
        hidden, prefix = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
        expected = head(hidden, prefix)
        changed = prefix.clone()
        changed[:, -1] += 10
        torch.testing.assert_close(
            head(hidden, changed)[:, :-1], expected[:, :-1], rtol=0, atol=0
        )
        self.assertEqual(head(hidden[:, :0], prefix).shape, (2, 0, 7))
        with self.assertRaises(ValueError):
            head(hidden, prefix[:, :2])

    def test_meta_and_loaded_gru_parameters(self):
        head = DominoGRUHead(
            GRUHeadSpec(hidden_size=3, projection_size=2),
            hidden_size=4,
            vocab_size=7,
            device="meta",
        )
        head.to_empty(device="cpu")
        head.prefix_gru.reset_draft_parameters()
        for parameter in head.prefix_gru.parameters():
            self.assertTrue(torch.isfinite(parameter).all())
            self.assertLessEqual(parameter.abs().max().item(), 3**-0.5)
            with torch.no_grad():
                parameter.fill_(7)
            parameter._is_hf_initialized = True
        head.prefix_gru.reset_draft_parameters()
        for parameter in head.prefix_gru.parameters():
            torch.testing.assert_close(parameter, torch.full_like(parameter, 7))

    def test_generic_hf_initialization_and_roundtrip(self):
        model = DFlashDraftModel(model_config())
        hidden, prefix = torch.randn(2, 3, 4), torch.randn(2, 5, 4)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.normal_(std=0.1)
        expected = model.head(hidden, prefix)
        model.post_init()
        torch.testing.assert_close(model.head(hidden, prefix), expected, rtol=0, atol=0)
        with TemporaryDirectory() as directory:
            model.save_pretrained(directory)
            restored = DFlashDraftModel.from_pretrained(directory)
        torch.testing.assert_close(
            restored.state_dict(), model.state_dict(), rtol=0, atol=0
        )
        torch.testing.assert_close(
            restored.head(hidden, prefix), expected, rtol=0, atol=0
        )

    def test_invalid_constructor_inputs_fail_before_allocation(self):
        with patch("torch.nn.Linear") as allocate:
            with self.assertRaises(TypeError):
                DominoGRUHead({}, hidden_size=4, vocab_size=7)
            for width, vocab in ((0, 7), (4, True)):
                with self.assertRaises(ValueError):
                    DominoGRUHead(
                        GRUHeadSpec(hidden_size=3, projection_size=2),
                        hidden_size=width,
                        vocab_size=vocab,
                    )
            allocate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
