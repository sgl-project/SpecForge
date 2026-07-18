import unittest
from unittest.mock import patch

import torch

from tests.test_utils.test_dflash_losses import (
    _dflash_module,
    _FixedDraft,
    _make_model,
    _sample_tensors,
    _targets_and_mask,
)


class _TrainableDraft(_FixedDraft):
    def __init__(self, hidden_size: int, max_draft_length: int = 32):
        super().__init__(hidden_size)
        values = torch.linspace(
            0.05,
            0.5,
            max_draft_length * hidden_size,
            dtype=torch.double,
        )
        self.row_values = torch.nn.Parameter(
            values.reshape(max_draft_length, hidden_size)
        )

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        del position_ids, target_hidden, attention_mask
        batch_size, draft_length = noise_embedding.shape[:2]
        return self.row_values[:draft_length].unsqueeze(0).expand(batch_size, -1, -1)


class TestCompactFrozenCrossEntropyRows(unittest.TestCase):
    def test_compaction_preserves_weighted_loss_and_only_sends_active_rows(self):
        logits, input_ids, loss_mask, hidden_states, anchors, keep_mask = (
            _sample_tensors()
        )
        _targets, binary_mask = _targets_and_mask(
            input_ids, loss_mask, anchors, keep_mask, logits.shape[2]
        )
        calls = []

        def fake_fused_ce(hidden, weight, target, bias):
            del weight, bias
            calls.append((hidden.shape[0], target.detach().clone()))
            values = hidden.to(torch.double).square().mean(dim=-1)
            values = values + target.to(torch.double) * 0.125 + 0.25
            return values, torch.ones_like(values)

        for loss_type in ("dflash", "dpace"):
            with self.subTest(loss_type=loss_type):
                calls.clear()
                common = dict(
                    linear_cross_entropy_backend="liger",
                    loss_decay_gamma=7.0,
                    loss_type=loss_type,
                    dpace_alpha=0.5,
                )
                compact_draft = _TrainableDraft(hidden_size=4)
                full_draft = _TrainableDraft(hidden_size=4)
                full_draft.load_state_dict(compact_draft.state_dict())
                with (
                    patch.object(_dflash_module, "validate_liger_installation"),
                    patch.object(
                        _dflash_module,
                        "frozen_linear_cross_entropy",
                        fake_fused_ce,
                    ),
                ):
                    compact = _make_model(
                        logits,
                        anchors,
                        keep_mask,
                        draft_model=compact_draft,
                        compact_zero_weight_ce_rows=True,
                        **common,
                    )
                    compact_loss, _, _ = compact(
                        input_ids=input_ids,
                        hidden_states=hidden_states,
                        loss_mask=loss_mask,
                    )
                    compact_loss.backward()
                    full = _make_model(
                        logits,
                        anchors,
                        keep_mask,
                        draft_model=full_draft,
                        **common,
                    )
                    full_loss, _, _ = full(
                        input_ids=input_ids,
                        hidden_states=hidden_states,
                        loss_mask=loss_mask,
                    )
                    full_loss.backward()

                self.assertEqual(len(calls), 2)
                self.assertEqual(calls[0][0], int((binary_mask > 0).sum().item()))
                self.assertEqual(calls[1][0], binary_mask.numel())
                torch.testing.assert_close(compact_loss, full_loss, rtol=0, atol=1e-12)
                torch.testing.assert_close(
                    compact_draft.row_values.grad,
                    full_draft.row_values.grad,
                    rtol=0,
                    atol=1e-12,
                )

    def test_compaction_requires_liger_backend(self):
        logits, input_ids, loss_mask, _hidden, anchors, keep_mask = _sample_tensors()
        del input_ids, loss_mask
        with self.assertRaisesRegex(ValueError, "requires.*liger"):
            _make_model(
                logits,
                anchors,
                keep_mask,
                compact_zero_weight_ce_rows=True,
                linear_cross_entropy_backend="torch",
            )


if __name__ == "__main__":
    unittest.main()
