"""The batched (row-shared, host-planned) context scan must match the row-wise scan exactly."""

from __future__ import annotations

import importlib.util
import unittest
from unittest import mock

import torch
from torch.testing import assert_close
from transformers import Qwen3Config

import specforge.modeling.draft.kda as kda_module
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.kda import (
    build_scan_layout,
    fla_kda,
    reference_kda,
    scan_kda_context_states,
    scan_kda_context_states_rowwise,
)

CUDA_AND_FLA = torch.cuda.is_available() and importlib.util.find_spec("fla") is not None
HIDDEN, HEADS, HEAD_DIM, BLOCK, CONV = 24, 4, 6, 2, 3


def _config() -> Qwen3Config:
    config = Qwen3Config(
        hidden_size=HIDDEN,
        intermediate_size=48,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=3,
        head_dim=HEAD_DIM,
        max_position_embeddings=128,
        vocab_size=64,
        tie_word_embeddings=False,
        attention_bias=False,
        attention_dropout=0.0,
    )
    config._attn_implementation = "eager"
    config.architectures = ["DFlashDraftModel"]
    config.num_target_layers = 8
    config.block_size = BLOCK
    config.draft_vocab_size = 64
    config.layer_types = ["full_attention"] * config.num_hidden_layers
    config.use_sliding_window = False
    config.dflash_config = {"attention_modes": ["kda", "gqa", "kda"], "mask_token_id": 0}
    config.linear_attn_config = {
        "head_dim": HEAD_DIM,
        "num_heads": HEADS,
        "short_conv_kernel_size": CONV,
        "use_full_rank_gate": False,
        "gate_lower_bound": -5.0,
        "backend": "reference",
        "context_state": "scan",
    }
    return config


def _inputs(batch_size: int, context_len: int, heads: int, dim: int, **kw):
    k = torch.randn(batch_size, context_len, heads, dim, requires_grad=True, **kw)
    v = torch.randn(batch_size, context_len, heads, dim, requires_grad=True, **kw)
    raw_gate = (torch.randn(batch_size, context_len, heads, dim, **kw) * 0.1).requires_grad_(True)
    beta = torch.randn(batch_size, context_len, heads, requires_grad=True, **kw)
    A_log = torch.rand(heads, **kw).add_(0.5).log_().requires_grad_(True)
    dt_bias = (torch.randn(heads * dim, **kw) * 0.1).requires_grad_(True)
    return k, v, raw_gate, beta, A_log, dt_bias


ANCHORS = torch.tensor(
    [
        [0, 7, 7, 31, 3, 31, 37, 12, 19, 2],  # duplicates, empty context, full context
        [5, 5, 5, 5, 5, 5, 5, 5, 5, 5],  # one unique anchor
        [36, 1, 2, 3, 4, 6, 8, 10, 13, 17],  # many segments, unsorted
    ]
)


class TestBatchedScanMatchesRowwise(unittest.TestCase):
    def _compare(self, group_size, anchors=ANCHORS, context_len=37, seed=3):
        torch.manual_seed(seed)
        inputs = _inputs(anchors.shape[0], context_len, 2, 4)
        k, v, raw_gate, beta, A_log, dt_bias = inputs
        weight = torch.randn(anchors.shape[0], anchors.shape[1], 2, 4, 4)
        results = []
        for fn in (scan_kda_context_states_rowwise, scan_kda_context_states):
            states = fn(
                reference_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors,
                group_size=group_size,
            )
            grads = (
                torch.autograd.grad((states * weight).sum(), inputs)
                if states.requires_grad  # all-zero anchors yield a constant zero state
                else None
            )
            results.append((states, grads))
        (expected, expected_grads), (actual, actual_grads) = results
        assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        self.assertEqual(actual_grads is None, expected_grads is None)
        for got, want in zip(actual_grads or (), expected_grads or ()):
            assert_close(got, want, rtol=1e-5, atol=1e-6)

    def test_default_group_size(self):
        self._compare(None)

    def test_small_and_large_groups(self):
        for group_size in (1, 2, 3, 5, 10, 64):
            with self.subTest(group_size=group_size):
                self._compare(group_size)

    def test_single_row_and_all_zero_anchors(self):
        self._compare(None, anchors=torch.tensor([[0, 0, 0]]), context_len=9)
        self._compare(2, anchors=torch.tensor([[9, 4, 0, 9]]), context_len=9)

    def test_layout_is_reusable_and_shape_checked(self):
        torch.manual_seed(5)
        k, v, raw_gate, beta, A_log, dt_bias = _inputs(3, 37, 2, 4)
        layout = build_scan_layout(ANCHORS, 37, k.device)
        expected = scan_kda_context_states_rowwise(
            reference_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, ANCHORS
        )
        for _ in range(2):
            actual = scan_kda_context_states(
                reference_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, ANCHORS,
                layout=layout,
            )
            assert_close(actual, expected, rtol=1e-6, atol=1e-6)
        with self.assertRaises(ValueError):
            scan_kda_context_states(
                reference_kda, k[:2], v[:2], raw_gate[:2], beta[:2], A_log, dt_bias,
                -5.0, ANCHORS[:2], layout=layout,
            )

    def test_launch_count_is_two_level(self):
        anchors = torch.arange(1, 65).unsqueeze(0) * 3  # 64 segments in one row
        layout = build_scan_layout(anchors, 200, torch.device("cpu"))
        self.assertEqual(layout.group, 8)
        self.assertEqual(len(layout.level1), 8)
        self.assertLessEqual(len(layout.level2), 7)
        two_rows = build_scan_layout(torch.cat([anchors, anchors - 1]), 200, torch.device("cpu"))
        # Batching rows adds no launches.
        self.assertEqual(len(two_rows.level1), len(layout.level1))
        self.assertEqual(len(two_rows.level2), len(layout.level2))


class TestModelUsesSharedLayout(unittest.TestCase):
    def _model_inputs(self, batch_size, num_blocks, context_len):
        draft_len = num_blocks * BLOCK
        return {
            "position_ids": torch.arange(context_len + draft_len).expand(batch_size, -1),
            "noise_embedding": torch.randn(batch_size, draft_len, HIDDEN),
            "target_hidden": torch.randn(batch_size, context_len, 3 * HIDDEN),
            "attention_mask": torch.ones(
                batch_size, 1, draft_len, context_len + draft_len, dtype=torch.bool
            ),
            "anchor_positions": torch.tensor([[0, 3, 3, 9, 6], [9, 1, 4, 4, 2]]),
        }

    def test_forward_matches_rowwise_scan_and_builds_one_layout(self):
        torch.manual_seed(11)
        model = DFlashDraftModel(_config()).eval()
        inputs = self._model_inputs(batch_size=2, num_blocks=5, context_len=9)

        def rowwise(*args, layout=None, **kwargs):
            return scan_kda_context_states_rowwise(*args, **kwargs)

        with mock.patch.object(kda_module, "scan_kda_context_states", rowwise):
            expected = model(**inputs)
        with mock.patch.object(
            kda_module, "build_scan_layout", wraps=kda_module.build_scan_layout
        ) as build:
            actual = model(**inputs)
        assert_close(actual, expected, rtol=1e-5, atol=1e-6)
        self.assertEqual(build.call_count, 1)  # once per forward, not once per layer

    def test_generation_path_builds_no_layout(self):
        model = DFlashDraftModel(_config()).eval()
        self.assertIsNone(model._build_scan_layout(torch.zeros(1, 4, HIDDEN), None, None))


@unittest.skipUnless(CUDA_AND_FLA, "requires CUDA and fla")
class TestBatchedScanWithFLA(unittest.TestCase):
    def test_fla_batched_matches_rowwise_and_reference(self):
        torch.manual_seed(16)
        device = torch.device("cuda")
        heads, dim, context_len = 2, 128, 200
        k, v, raw_gate, beta, A_log, dt_bias = _inputs(
            2, context_len, heads, dim, device=device, dtype=torch.bfloat16
        )
        A_log = A_log.detach().float().requires_grad_(True)
        dt_bias = dt_bias.detach().float().requires_grad_(True)
        anchors = torch.tensor([[0, 37, 37, 130, 200, 64], [5, 199, 12, 12, 100, 3]], device=device)
        rowwise = scan_kda_context_states_rowwise(
            fla_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors
        )
        batched = scan_kda_context_states(
            fla_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors
        )
        reference = scan_kda_context_states_rowwise(
            reference_kda, k, v, raw_gate, beta, A_log, dt_bias, -5.0, anchors
        )
        assert_close(batched, rowwise, rtol=1e-4, atol=1e-4)  # same kernel, same launches
        assert_close(batched, reference, rtol=4e-2, atol=4e-2)
        weight = torch.randn_like(batched)
        grads_rowwise = torch.autograd.grad((rowwise * weight).sum(), (k, v, raw_gate, beta))
        grads_batched = torch.autograd.grad((batched * weight).sum(), (k, v, raw_gate, beta))
        for got, want in zip(grads_batched, grads_rowwise):
            assert_close(got.float(), want.float(), rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    unittest.main()
