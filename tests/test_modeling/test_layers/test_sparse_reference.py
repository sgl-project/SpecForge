import unittest

import torch
from torch.nn import functional as F
from torch.testing import assert_close

from specforge.modeling.draft.layers.attention.backends.reference import (
    ExplicitSparseInputs,
    SparseKVSource,
    reference_sparse_attention,
)
from specforge.modeling.draft.layers.attention.base import AttentionContext
from specforge.modeling.draft.layers.attention.selection import (
    IndexedSelection,
    SlidingWindowSelection,
    SparseSelection,
)
from specforge.modeling.draft.layers.config import AttentionConfig


class SparseReferenceTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.q = torch.randn(1, 2, 3, 4, dtype=torch.float64, requires_grad=True)
        self.k = torch.randn(1, 1, 5, 4, dtype=torch.float64, requires_grad=True)
        self.v = torch.randn(1, 1, 5, 3, dtype=torch.float64, requires_grad=True)
        self.ctx = AttentionContext((torch.empty(0), torch.empty(0)))

    def inputs(self):
        return ExplicitSparseInputs(
            self.q,
            (SparseKVSource("main", self.k, self.v, torch.arange(5)[None]),),
            torch.tensor([[2, 3, 4]]),
            0.5,
        )

    def test_overlap_partial_blocks_and_causal_window_match_sdpa(self):
        # Row 0 sees block 0 plus window {1,2}; row 1 block 0 plus {2,3};
        # row 2 partial block 2 plus {3,4}. Each key counts exactly once.
        selection = SparseSelection(
            (
                IndexedSelection(
                    torch.tensor([[[[0, 99], [0, -99], [2, -99]]]]),
                    torch.ones(1, 1, 3, dtype=torch.long),
                    block_size=2,
                ),
            ),
            (SlidingWindowSelection(2),),
        )
        actual = reference_sparse_attention(
            self.inputs(), selection, self.ctx, AttentionConfig(is_causal=True)
        )
        mask = torch.tensor(
            [[[[1, 1, 1, 0, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 1]]]], dtype=torch.bool
        )
        expected = F.scaled_dot_product_attention(
            self.q,
            self.k.repeat_interleave(2, 1),
            self.v.repeat_interleave(2, 1),
            attn_mask=mask,
            scale=0.5,
        ).transpose(1, 2)
        assert_close(actual.values, expected)
        a = torch.autograd.grad(
            actual.values.square().sum(), (self.q, self.k, self.v), retain_graph=True
        )
        b = torch.autograd.grad(expected.square().sum(), (self.q, self.k, self.v))
        for x, y in zip(a, b):
            assert_close(x, y)

    def test_multiple_sources_share_normalization(self):
        second = SparseKVSource(
            "compressed",
            self.k[..., :1, :],
            self.v[..., :1, :] * 2,
            torch.tensor([[0]]),
        )
        inputs = ExplicitSparseInputs(
            self.q, (*self.inputs().sources, second), self.inputs().query_positions, 0.5
        )
        selection = SparseSelection(
            (
                IndexedSelection(
                    torch.zeros(1, 1, 3, 1, dtype=torch.long),
                    torch.ones(1, 1, 3, dtype=torch.long),
                    source="compressed",
                ),
            ),
            (SlidingWindowSelection(1),),
        )
        actual = reference_sparse_attention(
            inputs, selection, self.ctx, AttentionConfig()
        )
        rows = []
        for i in range(3):
            k = torch.cat((self.k[:, :, i + 2 : i + 3], second.key), 2).expand(
                1, 2, 2, 4
            )
            v = torch.cat((self.v[:, :, i + 2 : i + 3], second.value), 2).expand(
                1, 2, 2, 3
            )
            rows.append(
                F.scaled_dot_product_attention(self.q[:, :, i : i + 1], k, v, scale=0.5)
            )
        assert_close(actual.values, torch.cat(rows, 2).transpose(1, 2))

    def test_empty_selection_zero_output_and_gradients(self):
        result = reference_sparse_attention(
            self.inputs(), SparseSelection(), self.ctx, AttentionConfig()
        )
        assert_close(result.values, torch.zeros_like(result.values))
        self.assertFalse(result.valid_queries.any())
        for grad in torch.autograd.grad(result.values.sum(), (self.q, self.k, self.v)):
            assert_close(grad, torch.zeros_like(grad))

    def test_bad_valid_indices_and_unknown_sources_fail(self):
        for indices, source in ((-1, "main"), (5, "main"), (0, "missing")):
            plan = SparseSelection(
                (
                    IndexedSelection(
                        torch.full((1, 1, 3, 1), indices),
                        torch.ones(1, 1, 3, dtype=torch.long),
                        source,
                    ),
                )
            )
            with self.assertRaises(ValueError):
                reference_sparse_attention(
                    self.inputs(), plan, self.ctx, AttentionConfig()
                )

    def test_additive_bias_and_global_mask_override_selection(self):
        bias = torch.zeros(1, 1, 3, 5, dtype=torch.float64)
        bias[..., 0, :] = float("-inf")
        bias[..., 1, 1] = 0.7
        ctx = AttentionContext(self.ctx.position_embeddings, attention_mask=bias)
        plan = SparseSelection(windows=(SlidingWindowSelection(100),))
        actual = reference_sparse_attention(self.inputs(), plan, ctx, AttentionConfig())
        expected = F.scaled_dot_product_attention(
            self.q,
            self.k.expand(1, 2, 5, 4),
            self.v.expand(1, 2, 5, 3),
            attn_mask=bias,
            scale=0.5,
        ).transpose(1, 2)
        assert_close(actual.values, expected)
        self.assertFalse(actual.valid_queries[:, 0].any())


if __name__ == "__main__":
    unittest.main()
