import unittest

import torch

from specforge.core.edsd import (
    OnlineEdsdModel,
    _edsd_apply_curriculum_mask,
    _edsd_compute_target_p_padded,
)


class TestComputeStepN(unittest.TestCase):
    """Tests for OnlineEdsdModel.compute_step_n (EDSD paper Eq. 6)."""

    def test_epoch0_returns_1(self):
        """At epoch 0 the TTT length should be 1 (minimum)."""
        self.assertEqual(OnlineEdsdModel.compute_step_n(0, 10, s_max=7), 1)

    def test_last_epoch_returns_smax(self):
        """At the last epoch the TTT length should equal s_max."""
        self.assertEqual(OnlineEdsdModel.compute_step_n(9, 10, s_max=7), 7)

    def test_single_epoch_returns_smax(self):
        """When total_epochs <= 1, always return s_max."""
        self.assertEqual(OnlineEdsdModel.compute_step_n(0, 1, s_max=7), 7)
        self.assertEqual(OnlineEdsdModel.compute_step_n(0, 0, s_max=5), 5)

    def test_monotonic_increase(self):
        """TTT length should be non-decreasing across epochs."""
        s_max = 7
        total = 10
        lengths = [OnlineEdsdModel.compute_step_n(e, total, s_max) for e in range(total)]
        for i in range(len(lengths) - 1):
            self.assertGreaterEqual(lengths[i + 1], lengths[i])

    def test_two_epochs(self):
        """With 2 epochs: epoch 0 -> 1, epoch 1 -> s_max."""
        self.assertEqual(OnlineEdsdModel.compute_step_n(0, 2, s_max=7), 1)
        self.assertEqual(OnlineEdsdModel.compute_step_n(1, 2, s_max=7), 7)

    def test_smax_1(self):
        """When s_max=1, all epochs should return 1."""
        for epoch in range(5):
            self.assertEqual(OnlineEdsdModel.compute_step_n(epoch, 5, s_max=1), 1)


class TestComputeActualLength(unittest.TestCase):
    """Tests for OnlineEdsdModel._compute_actual_length."""

    def _make_model(self, length=7, step_n_schedule=None):
        from specforge.modeling.draft.edsd import EdsdDraftModel
        from transformers import LlamaConfig

        config = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            draft_vocab_size=64,
            target_layer_ids=[1, 2],
        )
        draft = EdsdDraftModel(config)
        return OnlineEdsdModel(
            draft_model=draft,
            length=length,
            step_n_schedule=step_n_schedule,
        )

    def test_schedule_overrides_eq6(self):
        """step_n_schedule takes priority over Eq. 6."""
        model = self._make_model(length=7, step_n_schedule=[1, 3, 5, 7])
        self.assertEqual(model._compute_actual_length(0, 10), 1)
        self.assertEqual(model._compute_actual_length(1, 10), 3)
        self.assertEqual(model._compute_actual_length(2, 10), 5)
        self.assertEqual(model._compute_actual_length(3, 10), 7)

    def test_schedule_clamps_to_last(self):
        """Out-of-range epochs use the last schedule value."""
        model = self._make_model(length=7, step_n_schedule=[1, 3, 5])
        self.assertEqual(model._compute_actual_length(5, 10), 5)
        self.assertEqual(model._compute_actual_length(100, 10), 5)

    def test_no_schedule_uses_eq6(self):
        """Without step_n_schedule, delegates to compute_step_n."""
        model = self._make_model(length=7, step_n_schedule=None)
        self.assertEqual(model._compute_actual_length(0, 10), 1)
        self.assertEqual(model._compute_actual_length(9, 10), 7)


class TestCurriculumMask(unittest.TestCase):
    """Tests for _edsd_apply_curriculum_mask."""

    def _make_inputs(self, batch=2, seq=8, draft_vocab=16, num_valid=6):
        """Create simple target_p and position_mask tensors."""
        target_p = torch.zeros(batch, seq, draft_vocab)
        # Uniform distribution on first num_valid positions
        target_p[:, :, :num_valid] = 1.0 / num_valid
        position_mask = torch.ones(batch, seq, 1)
        return target_p, position_mask

    def test_no_drop_at_high_epoch(self):
        """When epoch_idx >= total_epochs, drop_ratio is 0, mask unchanged."""
        target_p, position_mask = self._make_inputs()
        result = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=10, drop_ratio_scale=0.02, total_epochs=10)
        self.assertTrue(torch.equal(result, position_mask))

    def test_no_drop_with_zero_scale(self):
        """When drop_ratio_scale=0, no positions are dropped."""
        target_p, position_mask = self._make_inputs()
        result = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=0, drop_ratio_scale=0.0, total_epochs=10)
        self.assertTrue(torch.equal(result, position_mask))

    def test_drops_some_at_early_epoch(self):
        """At epoch 0 with non-zero scale, some valid positions should be dropped."""
        target_p, position_mask = self._make_inputs(num_valid=10)
        result = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=0, drop_ratio_scale=0.04, total_epochs=10)
        # drop_ratio = (10 - 0) * 0.04 = 0.4, capped at 0.4
        # Some positions should have been zeroed out
        num_original = position_mask.sum().item()
        num_after = result.sum().item()
        self.assertLess(num_after, num_original)

    def test_mask_capped_at_0p4(self):
        """Maximum drop ratio is 0.4."""
        target_p, position_mask = self._make_inputs(num_valid=100)
        # epoch_idx=0, drop_ratio_scale=1.0 -> (10-0)*1.0 = 10, capped at 0.4
        result = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=0, drop_ratio_scale=1.0, total_epochs=10)
        num_original = position_mask.sum().item()
        num_after = result.sum().item()
        # At most 40% dropped, so at least 60% remain
        self.assertGreaterEqual(num_after / num_original, 0.59)

    def test_preserves_zero_positions(self):
        """Positions that were already 0 in position_mask stay 0."""
        target_p, position_mask = self._make_inputs()
        position_mask[:, 5:, :] = 0  # zero out last 3 positions
        result = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=0, drop_ratio_scale=0.04, total_epochs=10)
        self.assertTrue(torch.all(result[:, 5:, :] == 0))

    def test_progressive_unmasking(self):
        """Higher epochs should retain more positions than lower epochs."""
        target_p, position_mask = self._make_inputs(batch=1, seq=20, num_valid=20)
        # Make entropy vary so curriculum has something to drop
        target_p[0, :, :10] = 0.08
        target_p[0, :, 10:] = 0.02
        result_0 = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=0, drop_ratio_scale=0.04, total_epochs=10)
        result_5 = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=5, drop_ratio_scale=0.04, total_epochs=10)
        result_10 = _edsd_apply_curriculum_mask(target_p, position_mask, epoch_idx=10, drop_ratio_scale=0.04, total_epochs=10)
        self.assertLessEqual(result_0.sum().item(), result_5.sum().item())
        self.assertLessEqual(result_5.sum().item(), result_10.sum().item())


class TestEdsdComputeTargetPPadded(unittest.TestCase):
    """Tests for _edsd_compute_target_p_padded."""

    def _make_inputs(self, batch=2, seq=4, vocab=32, draft_vocab=8, length=3):
        target = torch.randn(batch, seq, vocab)
        t2d = torch.zeros(vocab, dtype=torch.bool)
        # Select draft_vocab tokens evenly spaced
        indices = torch.linspace(0, vocab - 1, draft_vocab).long()
        t2d[indices] = True
        loss_mask = torch.ones(batch, seq, 1)
        return target, t2d, loss_mask, length

    def test_output_shapes(self):
        """Check output tensor shapes match expectations."""
        target, t2d, loss_mask, length = self._make_inputs(draft_vocab=8, length=3)
        target_p, target_p_on_draft, target_ids, position_mask = _edsd_compute_target_p_padded(
            target, t2d, loss_mask, length, epoch_idx=0, compute_on_draft=True
        )
        batch, seq = 2, 4
        self.assertEqual(target_p.shape, (batch, seq + length, 8))
        self.assertEqual(target_p_on_draft.shape, (batch, seq + length, 8))
        self.assertEqual(target_ids.shape, (batch, seq + length))
        self.assertEqual(position_mask.shape, (batch, seq, 1))

    def test_padding_values(self):
        """Padded positions should have uniform target_p and zero target_p_on_draft."""
        target, t2d, loss_mask, length = self._make_inputs(draft_vocab=8, length=3)
        target_p, target_p_on_draft, target_ids, _ = _edsd_compute_target_p_padded(
            target, t2d, loss_mask, length, epoch_idx=0, compute_on_draft=True
        )
        # Padded region: last `length` positions
        padded_target_p = target_p[:, -length:, :]
        self.assertTrue(torch.allclose(padded_target_p, torch.tensor(1.0 / 8)))
        padded_on_draft = target_p_on_draft[:, -length:, :]
        self.assertTrue(torch.allclose(padded_on_draft, torch.zeros_like(padded_on_draft)))

    def test_compute_on_draft_false(self):
        """When compute_on_draft=False, target_p_on_draft should be None."""
        target, t2d, loss_mask, length = self._make_inputs()
        _, target_p_on_draft, _, _ = _edsd_compute_target_p_padded(
            target, t2d, loss_mask, length, epoch_idx=0, compute_on_draft=False
        )
        self.assertIsNone(target_p_on_draft)

    def test_curriculum_mask_applied(self):
        """Curriculum masking should reduce position_mask sum at early epochs."""
        target, t2d, loss_mask, length = self._make_inputs(batch=1, seq=20, vocab=64, draft_vocab=16, length=3)
        # epoch_idx=0 with non-zero scale -> some positions dropped
        _, _, _, pm_early = _edsd_compute_target_p_padded(
            target, t2d, loss_mask, length, epoch_idx=0, drop_ratio_scale=0.04, total_epochs=10
        )
        # epoch_idx=10 -> no dropping
        _, _, _, pm_late = _edsd_compute_target_p_padded(
            target, t2d, loss_mask, length, epoch_idx=10, drop_ratio_scale=0.04, total_epochs=10
        )
        # Original (non-padded) region only
        orig_len = target.shape[1]
        self.assertLessEqual(pm_early[:, :orig_len].sum().item(), pm_late[:, :orig_len].sum().item())


if __name__ == "__main__":
    unittest.main()