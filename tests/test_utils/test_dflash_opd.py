import unittest

import torch

from specforge.algorithms.common.dflash_opd import (
    first_rejection_mask, global_block_loss, opd_block_terms, sparse_overlap,
    probabilities_on_candidates,
)
from specforge.modeling.draft.dflash2 import CandidateSelector


class DraftOPDTests(unittest.TestCase):
    def test_sparse_tv_matches_dense_with_outside_mass_and_gradients(self):
        torch.manual_seed(17)
        scores = torch.randn(4, 7, 3, requires_grad=True)
        target = torch.rand(4, 7, 11)
        target /= target.sum(-1, keepdim=True)
        ids = torch.tensor([2, 5, 8]).expand(4, 7, -1)
        p = target.gather(-1, ids)
        overlap, _ = sparse_overlap(scores, p)
        q_dense = torch.zeros_like(target).scatter(-1, ids, scores.softmax(-1))
        expected = 1 - 0.5 * (q_dense - target).abs().sum(-1)
        torch.testing.assert_close(overlap, expected)
        sparse_grad = torch.autograd.grad(overlap.sum(), scores, retain_graph=True)[0]
        dense_grad = torch.autograd.grad(expected.sum(), scores)[0]
        torch.testing.assert_close(sparse_grad, dense_grad)

    def test_first_rejection_all_accepted_and_request_censoring(self):
        accepted = torch.tensor([0, 1, 6, 7, 7, 0])
        exposed = torch.tensor([7, 7, 7, 7, 3, 0])
        masks = first_rejection_mask(accepted, exposed)
        self.assertEqual(masks.sum(-1).tolist(), [1, 2, 7, 7, 3, 0])
        with self.assertRaises(ValueError):
            first_rejection_mask(torch.tensor([8]), torch.tensor([7]))

    def test_analytic_block_loss_and_early_gradient(self):
        # q=.5/.5; p(C)=.8, with a strict over/under direction at each slot.
        scores = torch.zeros(1, 3, 2, requires_grad=True)
        p = torch.tensor([[[.2, .6], [.2, .6], [.2, .6]]])
        terms = opd_block_terms(scores, p, torch.ones(1, 3, dtype=torch.bool))
        self.assertAlmostEqual(terms.loss_sum.item(), 1 - (.7 + .49 + .343) / 3, places=6)
        grad = torch.autograd.grad(terms.loss_sum, scores)[0][0, :, 0]
        torch.testing.assert_close(grad, torch.tensor([.25 * (1 + .7 + .49) / 3,
                                                       .25 * (.7 + .49) / 3,
                                                       .25 * .49 / 3]))

    def test_empty_padding_does_not_change_loss_or_gradient(self):
        scores = torch.tensor([[[.4, -.2], [float('nan'), float('nan')]],
                               [[float('nan'), float('nan')], [float('nan'), float('nan')]]],
                              requires_grad=True)
        p = torch.tensor([[[.2, .6], [float('nan'), float('nan')]],
                          [[float('nan'), float('nan')], [float('nan'), float('nan')]]])
        mask = torch.tensor([[True, False], [False, False]])
        terms = opd_block_terms(scores, p, mask)
        expected = 1 - (.2 + scores[0, 0].softmax(0)[1])
        torch.testing.assert_close(terms.loss_sum, expected)
        grad = torch.autograd.grad(terms.loss_sum, scores)[0]
        self.assertTrue(torch.isfinite(grad).all())
        self.assertEqual(grad[~mask].abs().sum().item(), 0)

    def test_teacher_is_detached_and_zero_overlap_is_finite(self):
        scores = torch.randn(2, 7, 3, requires_grad=True)
        p = torch.zeros_like(scores, requires_grad=True)
        terms = opd_block_terms(scores, p, torch.ones(2, 7, dtype=torch.bool))
        terms.loss_sum.backward()
        self.assertEqual(terms.loss_sum.item(), 2)
        self.assertTrue(torch.isfinite(scores.grad).all())
        self.assertIsNone(p.grad)

    def test_global_normalization_with_unequal_rank_and_microbatch_counts(self):
        torch.manual_seed(23)
        scores = torch.randn(5, 7, 3, requires_grad=True)
        p = torch.softmax(torch.randn(5, 7, 5), -1)[..., :3]
        masks = first_rejection_mask(torch.tensor([0, 1, 6, 0, 3]), torch.tensor([7]*5))
        full = opd_block_terms(scores, p, masks).loss_sum / 5
        reference = torch.autograd.grad(full, scores, retain_graph=True)[0]
        # Two ranks, with 1 versus4 valid blocks split across microbatches.
        losses = []
        for start, end in [(0, 1), (1, 3), (3, 5)]:
            t = opd_block_terms(scores[start:end], p[start:end], masks[start:end])
            losses.append(global_block_loss(t, global_block_count=5, gradient_average_world_size=2))
        distributed = sum(losses) / 2
        observed = torch.autograd.grad(distributed, scores)[0]
        torch.testing.assert_close(reference, observed)

    def test_selector_backbone_and_predecessor_gradients(self):
        torch.manual_seed(33)
        selector = CandidateSelector(hidden_size=4, vocab_size=13, state_rank=3, top_k=3,
                                     initializer_range=.2)
        torch.nn.init.normal_(selector.successor_codebook, std=.2)
        hidden = torch.randn(2, 3, 4, requires_grad=True)
        unary = torch.randn(2, 3, 3, requires_grad=True)
        ids = torch.tensor([1, 2, 3]).expand(2, 3, 3)
        predecessors = torch.tensor([[9, 1, 2], [8, 3, 1]])
        scores = selector.score_candidates(candidate_ids=ids, unary_logits=unary,
                                          hidden_states=hidden, predecessor_ids=predecessors)
        p = torch.softmax(torch.randn(2, 3, 5), -1)[..., :3]
        terms = opd_block_terms(scores, p, torch.ones(2, 3, dtype=torch.bool))
        terms.loss_sum.backward()
        self.assertGreater(hidden.grad.abs().sum().item(), 0)
        self.assertGreater(unary.grad.abs().sum().item(), 0)
        for param in selector.parameters():
            self.assertGreater(param.grad.abs().sum().item(), 0)
        changed = selector.score_candidates(candidate_ids=ids, unary_logits=unary,
                                             hidden_states=hidden, predecessor_ids=predecessors.flip(0))
        self.assertFalse(torch.allclose(scores, changed))

    def test_invalid_mass_nonfinite_and_nonprefix_masks_fail(self):
        scores = torch.zeros(1, 3, 2)
        p = torch.full_like(scores, .6)
        with self.assertRaises(ValueError):
            sparse_overlap(scores, p)
        with self.assertRaises(ValueError):
            opd_block_terms(scores, p * .5, torch.tensor([[True, False, True]]))
        with self.assertRaises(ValueError):
            sparse_overlap(scores * float('nan'), p * .5)

    def test_sparse_support_join_preserves_original_mass_and_detaches_teacher(self):
        candidates = torch.tensor([[[1, 4, 7]]])
        target_ids = torch.tensor([[[7, 8, 1]]])
        target_probs = torch.tensor([[[.2, .5, .3]]], requires_grad=True)
        result = probabilities_on_candidates(candidates, target_ids, target_probs)
        torch.testing.assert_close(result, torch.tensor([[[.3, 0, .2]]]))
        self.assertFalse(result.requires_grad)
        for probabilities in (torch.tensor([[[float('nan'), .5, .5]]]),
                              torch.tensor([[[-.1, .5, .6]]])):
            with self.assertRaisesRegex(ValueError, 'Invalid sparse teacher'):
                probabilities_on_candidates(candidates, target_ids, probabilities)


if __name__ == '__main__':
    unittest.main()
