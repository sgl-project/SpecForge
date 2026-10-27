# coding=utf-8
"""The router's balancing contract: a stateless forward and the expert bias."""

import unittest

import torch

from specforge.modeling.draft.moe.router import Router, TopKRouter

E, K, T, H = 8, 2, 12, 16


def _router(expert_bias=True, norm_topk_prob=True):
    return TopKRouter(
        E,
        H,
        K,
        norm_topk_prob=norm_topk_prob,
        initializer_range=0.5,
        expert_bias=expert_bias,
    )


class RouterContractTest(unittest.TestCase):
    def test_the_bias_steers_the_choice_and_is_the_only_state(self):
        torch.manual_seed(0)
        router, x = _router(), torch.randn(T, H)
        self.assertIsInstance(router, Router)
        with self.subTest(phase="the only state is the bias"):
            self.assertEqual(
                [name for name, _ in router.named_buffers()], ["expert_bias"]
            )
        with self.subTest(phase="the bias steers the choice, not the weights"):
            router.expert_bias.copy_(torch.linspace(-1.0, 1.0, E))
            scores, indices, weights = router(x)
            biased = (scores + router.expert_bias).topk(K, dim=-1).indices
            self.assertTrue(torch.equal(indices, biased))
            picked = scores.gather(-1, indices)
            torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True))
        with self.subTest(phase="one bias step from pooled counts"):
            counts = torch.tensor([9.0, 1, 1, 1, 1, 1, 1, 1])
            router.expert_bias.zero_()
            router.update_expert_bias(counts)
            delta = 0.001 * torch.sign(counts.mean() - counts)
            torch.testing.assert_close(router.expert_bias, delta - delta.mean())
        with self.subTest(phase="a frozen bias and a missing bias both stay put"):
            frozen = _router()
            frozen.bias_update_rate = 0.0
            frozen.update_expert_bias(counts)
            self.assertTrue(torch.all(frozen.expert_bias == 0))
            off = _router(expert_bias=False)
            self.assertIsNone(off.expert_bias)
            self.assertEqual(list(off.state_dict()), ["weight"])
            off.update_expert_bias(counts)  # no bias: a no-op

    def test_the_bias_keeps_fp32_values_when_cast_to_bf16(self):
        router = _router(norm_topk_prob=False)
        bias = torch.linspace(1.001, 1.002, E)  # bf16 would round these to 1.0
        router.expert_bias.copy_(bias)
        router = router.bfloat16()
        self.assertEqual(router.expert_bias.dtype, torch.float32)
        torch.testing.assert_close(router.expert_bias, bias)
        self.assertEqual(router.weight.dtype, torch.bfloat16)
        self.assertIn("expert_bias", router.state_dict())


if __name__ == "__main__":
    unittest.main(verbosity=2)
