# coding=utf-8
"""MoE router: Qwen3.5-MoE parity, norm_topk_prob and the DeepSeek-V4 seams."""

import unittest

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import Qwen3_5MoeTopKRouter

from specforge.modeling.draft.moe.router import TopKRouter

E, H, T, K = 8, 64, 16, 2


def _sqrtsoftplus(logits):
    return F.softplus(logits).sqrt()


def _grads(scores, weights, inputs):
    return torch.autograd.grad(scores.square().sum() + weights.square().sum(), inputs)


class TopKRouterTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(T, H)

    def test_matches_transformers_qwen3_5_moe_with_gradients(self):
        # transformers' Qwen3.5-MoE router always renormalizes the top-k probs.
        router = TopKRouter(E, H, K, norm_topk_prob=True, initializer_range=0.5)
        config = Qwen3_5MoeTextConfig(
            num_experts=E, num_experts_per_tok=K, hidden_size=H
        )
        reference = Qwen3_5MoeTopKRouter(config)
        reference.load_state_dict(router.state_dict())  # strict: same names

        x = self.x.clone().requires_grad_()
        scores, indices, weights = router(x)
        x_ref = self.x.clone().requires_grad_()
        logits, ref_weights, ref_indices = reference(x_ref)
        ref_scores = logits.float().softmax(dim=-1)

        self.assertTrue(torch.equal(indices, ref_indices))
        torch.testing.assert_close(weights, ref_weights)
        torch.testing.assert_close(scores, ref_scores)
        got = _grads(scores, weights, (x, router.weight))
        want = _grads(ref_scores, ref_weights, (x_ref, reference.weight))
        for g, w in zip(got, want):
            torch.testing.assert_close(g, w)

    def test_norm_topk_prob_false_keeps_the_raw_top_k_scores(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        scores, indices, weights = router(self.x)
        top = scores.topk(K, dim=-1)
        self.assertTrue(torch.equal(indices, top.indices))
        torch.testing.assert_close(weights, top.values)

    def test_seams_compose_a_deepseek_v4_style_router(self):
        router = TopKRouter(
            E,
            H,
            K,
            norm_topk_prob=True,
            initializer_range=0.5,
            score_fn=_sqrtsoftplus,
            selection_bias=True,
            route_scale=2.5,
        )
        with torch.no_grad():
            router.selection_bias.copy_(torch.randn(E))
        scores, indices, weights = router(self.x)

        # Biased top-k picks, unbiased renormalized weights, then the route scale.
        want_scores = _sqrtsoftplus(self.x @ router.weight.T)
        want_indices = (want_scores + router.selection_bias).topk(K, dim=-1).indices
        want_weights = want_scores.gather(-1, want_indices)
        want_weights = want_weights / want_weights.sum(-1, keepdim=True) * 2.5
        torch.testing.assert_close(scores, want_scores)
        self.assertTrue(torch.equal(indices, want_indices))
        torch.testing.assert_close(weights, want_weights)
        self.assertIn("selection_bias", router.state_dict())

    def test_bf16_inputs_return_fp32_scores_and_weights(self):
        router = TopKRouter(E, H, K, norm_topk_prob=True, initializer_range=0.5)
        scores, indices, weights = router.bfloat16()(self.x.bfloat16())
        self.assertEqual(
            (scores.dtype, indices.dtype, weights.dtype),
            (torch.float32, torch.int64, torch.float32),
        )

    def test_reset_parameters_reinitializes_with_std(self):
        router = TopKRouter(64, 1024, K, norm_topk_prob=True, initializer_range=0.02)
        with torch.no_grad():
            router.weight.fill_(1e30)
        router.reset_parameters(0.02)
        self.assertAlmostEqual(router.weight.std().item(), 0.02, delta=0.002)

    def test_top_k_outside_one_to_num_experts_raises(self):
        for top_k in (0, E + 1):
            with self.subTest(top_k=top_k), self.assertRaises(ValueError):
                TopKRouter(E, H, top_k, norm_topk_prob=True, initializer_range=0.02)


if __name__ == "__main__":
    unittest.main(verbosity=2)
