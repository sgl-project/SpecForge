"""Future sparse integration contracts, without claiming QSA/DSA execution."""

import unittest
from dataclasses import dataclass, replace

import torch
from torch import nn
from torch.testing import assert_close

from specforge.modeling.draft.dflash_kernels import DEFAULT_DFLASH_KERNELS
from specforge.modeling.draft.layers.attention import (
    Attention,
    AttentionAuxiliaryLoss,
    AttentionOutput,
    IndexedSelection,
    SlidingWindowSelection,
    SparseSelection,
)
from specforge.modeling.draft.layers.attention.backends.sparse import (
    SparseAttentionBackend,
    SparseAttentionResult,
)
from specforge.modeling.draft.layers.config import AttentionConfig
from specforge.modeling.draft.layers.decoder import DecoderLayer
from specforge.modeling.draft.layers.factory import build_attention
from tests.test_modeling.test_layers.test_composition import config, context


class SparseContractTest(unittest.TestCase):
    def test_plan_retains_block_token_and_window_sources_without_expansion(self):
        blocks = IndexedSelection(
            torch.tensor([[[[2, -99], [-99, -99]]]]),
            torch.tensor([[[1, 0]]]),
            block_size=4,
        )
        tokens = IndexedSelection(
            torch.tensor([[[[1], [2]]]]),
            torch.ones(1, 1, 2, dtype=torch.int32),
            source="compressed",
        )
        plan = SparseSelection((blocks, tokens), (SlidingWindowSelection(8),))
        self.assertIs(plan.indexed[0].indices, blocks.indices)
        self.assertEqual(plan.indexed[0].block_size, 4)
        self.assertEqual(plan.indexed[1].source, "compressed")
        self.assertEqual(plan.windows[0].source, "main")
        self.assertEqual(SparseSelection().indexed, ())
        self.assertEqual(SparseSelection().windows, ())

    def test_selection_rejects_ambiguous_metadata(self):
        indices = torch.zeros(1, 1, 2, 3, dtype=torch.long)
        lengths = torch.zeros(1, 1, 2, dtype=torch.long)
        with self.assertRaisesRegex(ValueError, "lengths"):
            IndexedSelection(indices, lengths[..., None])
        with self.assertRaisesRegex(TypeError, "integer"):
            IndexedSelection(indices.float(), lengths)
        for size in (0, -1, True, 1.5):
            with self.assertRaisesRegex(ValueError, "block_size"):
                IndexedSelection(indices, lengths, block_size=size)
            with self.assertRaisesRegex(ValueError, "window size"):
                SlidingWindowSelection(size)
        with self.assertRaisesRegex(ValueError, "source"):
            IndexedSelection(indices, lengths, source="")

    def test_decoder_preserves_custom_state_and_indexer_loss_gradients(self):
        # Test adapter with a custom payload: no dense Q/K/V or HF Cache needed.
        @dataclass
        class LatentInputs:
            latent: torch.Tensor

        @dataclass
        class SelectionState:
            selection: SparseSelection

        class Adapter:
            def __call__(self, inputs, selection, context, options):
                self.selection = selection
                values = inputs.latent.unsqueeze(2)
                valid = torch.ones(values.shape[:-1], dtype=torch.bool)
                return SparseAttentionResult(values, valid)

        adapter: SparseAttentionBackend[LatentInputs] = Adapter()

        class TestAttention(Attention[SelectionState]):
            def __init__(self):
                super().__init__()
                self.indexer_weight = nn.Parameter(torch.tensor(2.0))

            def forward(self, hidden_states, context, state=None, **kwargs):
                result = adapter(
                    LatentInputs(hidden_states),
                    state.selection,
                    context,
                    AttentionConfig("qsa"),
                )
                loss = (hidden_states.mean() * self.indexer_weight).square()
                return AttentionOutput(
                    result.values.flatten(2),
                    state,
                    auxiliary_losses=(AttentionAuxiliaryLoss("indexer", loss),),
                )

        attention = TestAttention()
        layer = DecoderLayer(attention, nn.Identity(), nn.Identity(), nn.Identity())
        hidden = torch.ones(1, 2, 4, requires_grad=True)
        state = SelectionState(SparseSelection(windows=(SlidingWindowSelection(4),)))
        output = layer(hidden, context(hidden), state)
        self.assertIs(output.attention.state, state)
        self.assertIs(adapter.selection, state.selection)
        objective = output.attention.auxiliary_losses[0]
        self.assertEqual(objective.name, "indexer")
        self.assertIsNone(attention.indexer_weight.grad)
        objective.value.backward()
        assert_close(attention.indexer_weight.grad, torch.tensor(4.0))
        self.assertTrue(torch.isfinite(hidden.grad).all())

    def test_swa_matches_explicit_visibility_for_gqa_and_mla(self):
        # SWA already executes through dense backends, independently of sparse
        # selection. Check causal/noncausal windows and non-contiguous positions.
        for kind in ("gqa", "mla"):
            for backend in ("eager", "sdpa"):
                for causal in (False, True):
                    with self.subTest(kind=kind, backend=backend, causal=causal):
                        cfg = config(kind)
                        cfg._attn_implementation = backend
                        windowed = build_attention(
                            cfg,
                            0,
                            DEFAULT_DFLASH_KERNELS,
                            AttentionConfig(kind, is_causal=causal, sliding_window=3),
                        )
                        explicit = build_attention(
                            cfg, 0, DEFAULT_DFLASH_KERNELS, AttentionConfig(kind)
                        )
                        explicit.load_state_dict(windowed.state_dict())
                        hidden = torch.randn(1, 3, 16, requires_grad=True)
                        positions = torch.tensor([[0, 2, 7]])
                        distance = positions[:, :, None] - positions[:, None, :]
                        mask = distance.abs() < 3
                        if causal:
                            mask &= distance >= 0
                        ctx = replace(
                            context(hidden),
                            query_positions=positions,
                            key_positions=positions,
                        )
                        actual = windowed(hidden, ctx).hidden_states
                        expected = explicit(
                            hidden, replace(ctx, attention_mask=mask[:, None])
                        ).hidden_states
                        assert_close(actual, expected)
                        actual_grad = torch.autograd.grad(actual.sum(), hidden)[0]
                        expected_grad = torch.autograd.grad(expected.sum(), hidden)[0]
                        assert_close(actual_grad, expected_grad)


if __name__ == "__main__":
    unittest.main()
