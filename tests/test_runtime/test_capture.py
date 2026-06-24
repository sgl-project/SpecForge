# coding=utf-8
"""CaptureConfig assertions, incl. the M4 gate test_capture_layer_mismatch_fails (CPU)."""

import unittest

import torch

from specforge.runtime.inference.capture import (
    CaptureConfig,
    CaptureMismatchError,
    verify_capture,
)

H = 8
DRAFT_V = 16
TARGET_V = 64


def _capture(layer_ids=(10, 15, 20), target_repr="hidden_state", **kw):
    return CaptureConfig.from_strategy(
        required_features={"input_ids", "loss_mask", "hidden_state", "target"},
        aux_hidden_state_layer_ids=layer_ids,
        target_repr=target_repr,
        target_hidden_size=H,
        target_vocab_size=TARGET_V,
        draft_vocab_size=DRAFT_V,
        **kw,
    )


def _good_tensors(target_dim=H):
    return {
        "input_ids": torch.zeros(1, 4, dtype=torch.long),
        "loss_mask": torch.ones(1, 4, dtype=torch.long),
        "hidden_state": torch.randn(1, 4, 3 * H),  # 3 aux layers * H
        "target": torch.randn(1, 4, target_dim),
    }


class TestCapture(unittest.TestCase):
    def test_ok(self):
        verify_capture(
            _good_tensors(),
            _capture(),
            sample_id="s0",
            recorded_aux_layer_ids=(10, 15, 20),
        )

    def test_capture_layer_mismatch_fails(self):
        """Requested aux layers [10,15,20] vs recorded [10,15,21] -> loud failure."""
        with self.assertRaises(CaptureMismatchError) as ctx:
            verify_capture(
                _good_tensors(),
                _capture(layer_ids=(10, 15, 20)),
                sample_id="s0",
                recorded_aux_layer_ids=(10, 15, 21),
            )
        self.assertIn("aux-layer id mismatch", str(ctx.exception))

    def test_missing_feature_fails(self):
        t = _good_tensors()
        del t["target"]
        with self.assertRaises(CaptureMismatchError):
            verify_capture(t, _capture(), sample_id="s0")

    def test_aux_width_mismatch_fails(self):
        t = _good_tensors()
        t["hidden_state"] = torch.randn(1, 4, 2 * H)  # only 2 layers' worth
        with self.assertRaises(CaptureMismatchError) as ctx:
            verify_capture(t, _capture(), sample_id="s0")
        self.assertIn("aux width", str(ctx.exception))

    def test_target_dim_mismatch_pruned_logits(self):
        cap = _capture(target_repr="pruned_logits", vocab_map_version="v1")
        # pruned_logits expects draft_vocab_size on the last dim
        ok = _good_tensors(target_dim=DRAFT_V)
        verify_capture(ok, cap, sample_id="s0")
        bad = _good_tensors(target_dim=TARGET_V)  # full vocab, wrong for pruned
        with self.assertRaises(CaptureMismatchError):
            verify_capture(bad, cap, sample_id="s0")

    def test_pruned_logits_requires_vocab_map_version(self):
        cap = _capture(target_repr="pruned_logits")  # no vocab_map_version
        with self.assertRaises(CaptureMismatchError):
            verify_capture(_good_tensors(target_dim=DRAFT_V), cap, sample_id="s0")

    def test_logits_expects_full_vocab(self):
        cap = _capture(target_repr="logits")
        verify_capture(_good_tensors(target_dim=TARGET_V), cap, sample_id="s0")
        with self.assertRaises(CaptureMismatchError):
            verify_capture(_good_tensors(target_dim=DRAFT_V), cap, sample_id="s0")


if __name__ == "__main__":
    unittest.main(verbosity=2)
