"""Selector initialization works under the generic HF base and meta loading."""

import unittest

import torch
from torch import nn

from specforge.modeling.draft.dflash2 import CandidateSelector as LegacySelector
from specforge.modeling.draft.heads.candidate_selector import CandidateSelector
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


def selector(**options):
    return CandidateSelector(
        hidden_size=8,
        vocab_size=16,
        state_rank=4,
        top_k=3,
        initializer_range=0.2,
        **options,
    )


class SelectorInitializationTest(unittest.TestCase):
    def test_standalone_initialization_and_legacy_alias_are_unchanged(self):
        self.assertIs(LegacySelector, CandidateSelector)
        torch.manual_seed(71)
        module = selector()
        # Preserve the old constructor's random draw order: child Linear first,
        # then predecessor factor; the successor factor is exactly zero.
        torch.manual_seed(71)
        projection = nn.Linear(8, 4, bias=False)
        predecessor = torch.empty(16, 4).normal_(std=0.2)
        torch.testing.assert_close(
            module.hidden_projection.weight, projection.weight, rtol=0, atol=0
        )
        torch.testing.assert_close(
            module.predecessor_codebook, predecessor, rtol=0, atol=0
        )
        torch.testing.assert_close(
            module.successor_codebook, torch.zeros_like(predecessor), rtol=0, atol=0
        )

    def test_meta_materialization_and_generic_initialization_preserve_unary_noop(self):
        module = selector(device="meta", dtype=torch.bfloat16)
        self.assertTrue(
            all(p.is_meta and p.dtype == torch.bfloat16 for p in module.parameters())
        )
        module.to_empty(device="cpu")
        owner = DFlashDraftModel(model_config(initializer_range=0.2))
        module.apply(owner._initialize_weights)
        logits = torch.randn(2, 3, 3)
        scores = module.score_candidates(
            candidate_ids=torch.randint(16, (2, 3, 3)),
            unary_logits=logits,
            hidden_states=torch.randn(2, 3, 8, dtype=torch.bfloat16),
            predecessor_ids=torch.randint(16, (2, 3)),
        )
        torch.testing.assert_close(scores, logits, rtol=0, atol=0)
        before = {name: value.clone() for name, value in module.state_dict().items()}
        module.apply(owner._initialize_weights)
        torch.testing.assert_close(module.state_dict(), before, rtol=0, atol=0)

    def test_partial_restore_never_overwrites_loaded_factors(self):
        module = selector()
        with torch.no_grad():
            module.predecessor_codebook.fill_(3)
            module.successor_codebook.fill_(7)
        module.predecessor_codebook._is_hf_initialized = True
        module.reset_draft_parameters()
        torch.testing.assert_close(
            module.predecessor_codebook,
            torch.full_like(module.predecessor_codebook, 3),
            rtol=0,
            atol=0,
        )
        torch.testing.assert_close(
            module.successor_codebook,
            torch.zeros_like(module.successor_codebook),
            rtol=0,
            atol=0,
        )
        module.successor_codebook._is_hf_initialized = True
        with torch.no_grad():
            module.successor_codebook.fill_(5)
        module.reset_draft_parameters()
        torch.testing.assert_close(
            module.successor_codebook,
            torch.full_like(module.successor_codebook, 5),
            rtol=0,
            atol=0,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_explicit_cuda_placement_and_dtype(self):
        module = selector(device="cuda", dtype=torch.float16)
        self.assertTrue(
            all(
                p.device.type == "cuda" and p.dtype == torch.float16
                for p in module.parameters()
            )
        )


if __name__ == "__main__":
    unittest.main()
