# coding=utf-8
"""MoE balancing: the router's load-balance (aux) loss."""

import unittest

import torch
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.router import (
    TopKRouter,
    count_tokens_per_expert,
    load_balance_loss,
)

E, K, B, S, H = 8, 2, 2, 6, 16


def _moe(coef=0.5):
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        router_aux_loss_coef=coef,
    )
    return build_moe(config)


def _real_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _loss(scores, indices, mask=None):
    return load_balance_loss(scores, count_tokens_per_expert(indices, mask, E), mask)


class LoadBalanceLossTest(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(B * S, E)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        for mask in (None, _real_token_mask()):
            with self.subTest(masked=mask is not None):
                want = load_balancing_loss_func((logits,), E, K, mask)
                flat = None if mask is None else mask.reshape(-1)
                torch.testing.assert_close(_loss(scores, indices, flat), want)

    def test_sigmoid_scores_are_normalized_per_token(self):
        torch.manual_seed(0)
        scores = torch.randn(B * S, E).sigmoid()
        indices = scores.topk(K, dim=-1).indices
        probs = scores / scores.sum(dim=-1, keepdim=True)
        counts = count_tokens_per_expert(indices, None, E)
        want = E * (counts / (B * S) * probs.mean(dim=0)).sum()
        torch.testing.assert_close(_loss(scores, indices), want)

    def test_no_kept_slots_gives_a_zero_on_the_graph(self):
        logits = torch.randn(4, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        loss = _loss(scores, indices, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_balanced_routing_gives_top_k(self):
        scores = torch.full((E, E), 1 / E)
        indices = torch.stack([torch.arange(E), torch.arange(E).roll(1)], dim=1)
        torch.testing.assert_close(_loss(scores, indices), torch.tensor(float(K)))


class RouterAuxLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(B * S, H)
        self.mask = _real_token_mask().reshape(-1)

    def test_training_keeps_the_masked_loss_on_the_router(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        scores, indices, _ = router(self.x, self.mask)
        torch.testing.assert_close(router.aux_loss, _loss(scores, indices, self.mask))
        # The loss reaches the gate weight through the scores.
        (grad,) = torch.autograd.grad(router.aux_loss, router.weight)
        self.assertTrue(torch.any(grad != 0))

    def test_without_a_mask_every_token_counts(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        scores, indices, _ = router(self.x)
        torch.testing.assert_close(router.aux_loss, _loss(scores, indices))

    def test_eval_keeps_no_loss(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        router.eval()(self.x, self.mask)
        self.assertIsNone(router.aux_loss)

    def test_coefficient_zero_still_computes_the_balance_metric(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        self.assertEqual(router.aux_loss_coef, 0.0)
        router(self.x)
        self.assertIsNotNone(router.aux_loss)

    def test_family_reads_router_aux_loss_coef(self):
        layer = _moe(coef=0.25)
        self.assertEqual(layer.gate.aux_loss_coef, 0.25)
        layer(torch.randn(B, S, H))
        self.assertIsNotNone(layer.gate.aux_loss)


if __name__ == "__main__":
    unittest.main(verbosity=2)
