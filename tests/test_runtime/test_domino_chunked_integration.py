# coding=utf-8
"""CPU integration checks for OnlineDominoModel's chunked-logit path."""

import copy
import importlib.util
import sys
import types
import unittest
from pathlib import Path

import torch
import torch.nn as nn


def _load_online_domino_model():
    """Load the focused module without importing SpecForge's GPU-heavy package."""
    repo = Path(__file__).resolve().parents[2]
    names = (
        "specforge",
        "specforge.core",
        "specforge.core.loss",
        "specforge.modeling",
        "specforge.modeling.draft",
        "specforge.modeling.draft.dflash",
    )
    saved = {name: sys.modules.get(name) for name in names}
    try:
        for name in (
            "specforge",
            "specforge.core",
            "specforge.modeling",
            "specforge.modeling.draft",
        ):
            module = types.ModuleType(name)
            module.__path__ = []
            sys.modules[name] = module

        draft_module = types.ModuleType("specforge.modeling.draft.dflash")
        draft_module.DFlashDraftModel = nn.Module
        sys.modules[draft_module.__name__] = draft_module

        loss_spec = importlib.util.spec_from_file_location(
            "specforge.core.loss", repo / "specforge" / "core" / "loss.py"
        )
        loss_module = importlib.util.module_from_spec(loss_spec)
        sys.modules[loss_spec.name] = loss_module
        loss_spec.loader.exec_module(loss_module)

        dflash_spec = importlib.util.spec_from_file_location(
            "_domino_test_dflash", repo / "specforge" / "core" / "dflash.py"
        )
        dflash_module = importlib.util.module_from_spec(dflash_spec)
        dflash_spec.loader.exec_module(dflash_module)
        return dflash_module.OnlineDominoModel
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


OnlineDominoModel = _load_online_domino_model()


class _TinyDominoDraft(nn.Module):
    """Only the Domino head seam needed to exercise the online wrapper."""

    def __init__(self, hidden_size, prefix_size, vocab_size, shift_label):
        super().__init__()
        self.shift_label = shift_label
        self.pure_draft_prefix_len = 0
        self.backbone = nn.Linear(hidden_size, hidden_size, bias=False)
        self.prefix_gru = nn.GRU(hidden_size, prefix_size, batch_first=True, bias=False)
        self.embed_proj = nn.Sequential(
            nn.Linear(hidden_size + prefix_size, hidden_size, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_size, vocab_size, bias=False),
        )

    @property
    def suffix_start(self):
        return 0 if self.shift_label else 1

    def compute_prefix_states(self, prev_token_embeddings):
        bsz, n_blocks, block_size, hidden_size = prev_token_embeddings.shape
        if self.shift_label:
            gru_inputs = prev_token_embeddings.reshape(
                bsz * n_blocks, block_size, hidden_size
            )
            return self.prefix_gru(gru_inputs)[0].reshape(bsz, n_blocks, block_size, -1)

        gru_inputs = prev_token_embeddings[:, :, :-1, :].reshape(
            bsz * n_blocks, block_size - 1, hidden_size
        )
        return self.prefix_gru(gru_inputs)[0].reshape(bsz, n_blocks, block_size - 1, -1)

    def apply_logits_head(
        self,
        base_logits,
        *,
        hidden_states,
        prev_token_embeddings,
        prev_token_ids=None,
    ):
        del prev_token_ids
        prefix_states = self.compute_prefix_states(prev_token_embeddings)
        suffix_hidden = hidden_states[:, :, self.suffix_start :, :]
        correction = self.embed_proj(torch.cat([suffix_hidden, prefix_states], dim=-1))
        return torch.cat(
            [
                base_logits[:, :, : self.suffix_start, :],
                base_logits[:, :, self.suffix_start :, :] + correction,
            ],
            dim=2,
        )


class _DeterministicOnlineDomino(OnlineDominoModel):
    """Replace attention/anchor sampling while retaining the production loss path."""

    def _forward_draft_blocks(self, input_ids, hidden_states, loss_mask):
        del input_ids, loss_mask
        anchors = torch.tensor([[0, 4], [1, 3]], device=hidden_states.device)
        keep = torch.tensor([[True, True], [True, False]], device=hidden_states.device)
        output_hidden = self.draft_model.backbone(hidden_states[:, :8, :])
        return anchors, keep, output_hidden


def _build_model(shift_label):
    hidden_size = 6
    prefix_size = 4
    vocab_size = 17
    draft = _TinyDominoDraft(hidden_size, prefix_size, vocab_size, shift_label).double()
    lm_head = nn.Linear(hidden_size, vocab_size, bias=False, dtype=torch.double)
    embed_tokens = nn.Embedding(vocab_size, hidden_size, dtype=torch.double)
    lm_head.requires_grad_(False)
    embed_tokens.requires_grad_(False)
    return _DeterministicOnlineDomino(
        draft_model=draft,
        target_lm_head=lm_head,
        target_embed_tokens=embed_tokens,
        mask_token_id=0,
        block_size=4,
        attention_backend="sdpa",
        num_anchors=2,
        loss_decay_gamma=2.5,
        shift_label=shift_label,
    )


class TestOnlineDominoChunkedIntegration(unittest.TestCase):
    def _run(self, model, input_ids, hidden_states, loss_mask, chunk_size):
        loss, accuracy, metrics = model(
            input_ids=input_ids,
            hidden_states=hidden_states,
            loss_mask=loss_mask,
            lambda_base=0.35,
            logit_chunk_size=chunk_size,
            compute_metrics=True,
        )
        loss.backward()
        grads = {
            name: parameter.grad.detach().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        return loss.detach(), accuracy.detach(), metrics, grads

    def test_full_and_chunked_forward_match_for_both_label_modes(self):
        input_ids = torch.tensor(
            [
                [1, 2, 3, 4, 5, 6, 7, 8, 9],
                [9, 8, 7, 6, 5, 4, 3, 2, 1],
            ]
        )
        torch.manual_seed(2026)
        hidden_states = torch.randn(2, 9, 6, dtype=torch.double)
        loss_mask = torch.ones(2, 9, dtype=torch.double)
        loss_mask[0, 6] = 0

        for shift_label, expected_denom in ((False, 8.0), (True, 11.0)):
            with self.subTest(shift_label=shift_label):
                torch.manual_seed(1234)
                full_model = _build_model(shift_label)
                chunked_model = copy.deepcopy(full_model)

                full = self._run(
                    full_model, input_ids, hidden_states, loss_mask, chunk_size=0
                )
                chunked = self._run(
                    chunked_model, input_ids, hidden_states, loss_mask, chunk_size=3
                )

                torch.testing.assert_close(chunked[0], full[0], rtol=1e-10, atol=1e-10)
                torch.testing.assert_close(chunked[1], full[1], rtol=0, atol=0)
                self.assertEqual(set(chunked[2]), set(full[2]))
                for name in full[2]:
                    torch.testing.assert_close(
                        chunked[2][name], full[2][name], rtol=1e-10, atol=1e-10
                    )
                self.assertEqual(float(chunked[2]["accuracy_denom"]), expected_denom)

                self.assertEqual(set(chunked[3]), set(full[3]))
                for name in full[3]:
                    torch.testing.assert_close(
                        chunked[3][name], full[3][name], rtol=1e-9, atol=1e-10
                    )


if __name__ == "__main__":
    unittest.main()
