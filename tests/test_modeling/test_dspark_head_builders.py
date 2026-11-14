"""Node-driven DSpark heads retain legacy initialization and explicit placement."""

import unittest

import torch

from specforge.config.draft.architecture.dspark import (
    ConfidenceHeadSpec,
    MarkovHeadSpec,
)
from specforge.modeling.draft.heads.confidence import AcceptRatePredictor
from specforge.modeling.draft.heads.markov import build_markov_head
from tests.test_modeling.test_draft_pretrained import DFlashDraftModel, model_config


class DSparkHeadBuilderTest(unittest.TestCase):
    def test_meta_materialization_and_generic_initialization(self):
        owner = DFlashDraftModel(model_config(initializer_range=0))
        for kind in ("vanilla", "gated", "rnn"):
            node = MarkovHeadSpec(type=kind, rank=4)
            head = build_markov_head(
                node, vocab_size=16, hidden_size=8, device="meta", dtype=torch.bfloat16
            )
            confidence = AcceptRatePredictor(
                ConfidenceHeadSpec(with_markov=True).input_width(8, node),
                device="meta",
                dtype=torch.bfloat16,
            )
            for module in (head, confidence):
                self.assertTrue(
                    all(
                        p.is_meta and p.dtype == torch.bfloat16
                        for p in module.parameters()
                    )
                )
                module.to_empty(device="cpu")
                module.apply(owner._initialize_weights)
                for parameter in module.parameters():
                    torch.testing.assert_close(
                        parameter, torch.zeros_like(parameter), rtol=0, atol=0
                    )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_cuda_placement(self):
        for kind in ("vanilla", "gated", "rnn"):
            head = build_markov_head(
                MarkovHeadSpec(type=kind, rank=4),
                vocab_size=16,
                hidden_size=8,
                device="cuda",
                dtype=torch.float16,
            )
            self.assertTrue(
                all(
                    p.device.type == "cuda" and p.dtype == torch.float16
                    for p in head.parameters()
                )
            )

    def test_none_and_invalid_nodes(self):
        self.assertIsNone(build_markov_head(None, vocab_size=16, hidden_size=8))
        with self.assertRaises(TypeError):
            build_markov_head({"type": "rnn", "rank": 4}, vocab_size=16, hidden_size=8)


if __name__ == "__main__":
    unittest.main()
