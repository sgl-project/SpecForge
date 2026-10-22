"""DSpark DPACE formula, detached-gradient and chunking regressions."""
import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn
import test_dflash_losses as fixtures


class TestDSparkDPACE(unittest.TestCase):
    def setUp(self):
        (self.logits, self.ids, self.mask, self.hidden,
         self.anchors, self.keep) = fixtures._sample_tensors()

    def make_model(self, **kwargs):
        return fixtures._make_dspark_model(
            self.logits, self.anchors, self.keep, **kwargs
        )

    def run_model(self, model):
        return model(
            input_ids=self.ids, hidden_states=self.hidden, loss_mask=self.mask,
            target_last_hidden_states=self.hidden,
        )

    def test_ce_formula_detached_gradient_and_masking(self):
        targets, mask = fixtures._dspark_targets_and_mask(
            self.ids, self.mask, self.anchors, self.keep, self.logits.shape[2]
        )
        for mode in ('dpace', 'dpace-cumulative-confidence-only',
                     'dpace-continuation-value-only'):
            with self.subTest(mode=mode):
                logits = self.logits.clone().requires_grad_()
                model = self.make_model(
                    lm_head=fixtures._FixedHead(logits.flatten(1, 2)),
                    loss_type=mode, dpace_alpha=0.5, loss_decay_gamma=4.0,
                    dspark_ce_loss_alpha=1.0, dspark_l1_loss_alpha=0.0,
                    dspark_confidence_head_alpha=0.0,
                )
                actual, _, metrics = self.run_model(model)
                ce = fixtures._neg_log_q(logits, targets)
                # Independent per-sequence / per-anchor loop oracle.
                sequence_losses = []
                for b in range(mask.shape[0]):
                    anchor_losses = []
                    for a in range(mask.shape[1]):
                        valid = mask[b, a]
                        if not valid.any():
                            continue
                        prob = ce[b, a, valid].detach().neg().exp()
                        prefix = (0.5 + 0.5 * prob).cumprod(0)
                        credits = torch.stack([prefix[k:].sum() for k in range(len(prefix))])
                        if mode.endswith('cumulative-confidence-only'):
                            credits = prefix
                        elif mode.endswith('continuation-value-only'):
                            credits = credits / prefix
                        anchor_losses.append((ce[b, a, valid] * credits).sum())
                    if anchor_losses:
                        sequence_losses.append(torch.stack(anchor_losses).mean())
                expected = torch.stack(sequence_losses).mean()
                torch.testing.assert_close(actual, expected)
                actual_grad = torch.autograd.grad(actual, logits, retain_graph=True)[0]
                expected_grad = torch.autograd.grad(expected, logits)[0]
                torch.testing.assert_close(actual_grad, expected_grad)
                self.assertEqual(metrics['ratio_metrics']['ce_loss'][1].item(), 2)

    def test_auxiliary_losses_and_metrics_are_unchanged(self):
        head = nn.Linear(4, self.logits.shape[-1], bias=False).double()
        models = [self.make_model(
            draft_model=fixtures._LearnableDSparkDraft(4).double(),
            lm_head=copy.deepcopy(head), loss_type=mode, loss_decay_gamma=4.0,
        ) for mode in ('dflash', 'dpace')]
        old_loss, old_acc, old = self.run_model(models[0])
        new_loss, new_acc, new = self.run_model(models[1])
        torch.testing.assert_close(old_acc, new_acc)
        for name, pair in old['ratio_metrics'].items():
            if name != 'ce_loss':
                for before, after in zip(pair, new['ratio_metrics'][name]):
                    torch.testing.assert_close(before, after)
        old_ce = old['ratio_metrics']['ce_loss']
        new_ce = new['ratio_metrics']['ce_loss']
        torch.testing.assert_close(
            new_loss - old_loss, 0.1 * (new_ce[0]/new_ce[1] - old_ce[0]/old_ce[1]),
        )

    def test_chunked_loss_and_gradient_match_full(self):
        head = nn.Linear(4, self.logits.shape[-1], bias=False).double()
        models = [self.make_model(
            draft_model=fixtures._LearnableDSparkDraft(4).double(),
            lm_head=copy.deepcopy(head), loss_type='dpace',
            objective_chunk_blocks=chunk, loss_decay_gamma=4.0,
        ) for chunk in (0, 1)]
        results = [self.run_model(model) for model in models]
        torch.testing.assert_close(results[0][0], results[1][0])
        for loss, _, _ in results:
            loss.backward()
        torch.testing.assert_close(models[0].draft_model.signal.grad,
                                   models[1].draft_model.signal.grad)

    def test_distributed_denominators_reduce_independently(self):
        model = self.make_model(
            loss_type='dpace', dspark_ce_loss_alpha=1.0,
            dspark_l1_loss_alpha=0.0, dspark_confidence_head_alpha=0.0,
        )
        single_loss, _, _ = self.run_model(model)
        def reduce_den(den, op):
            self.assertEqual(tuple(den.shape), (2,))
            self.assertEqual(den[0].item(), 2)
            # A second rank with one valid sequence and a different token count.
            den.add_(den.new_tensor([1, 7]))
        with patch('torch.distributed.is_initialized', return_value=True), \
             patch('torch.distributed.get_world_size', return_value=2), \
             patch('torch.distributed.all_reduce', side_effect=reduce_den):
            distributed_loss, _, _ = self.run_model(model)
        torch.testing.assert_close(distributed_loss, single_loss * (4 / 3))


if __name__ == '__main__':
    unittest.main()
