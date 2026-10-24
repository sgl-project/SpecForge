# coding=utf-8
"""MoE balancing math and router behaviour: the loss, the step balance, the bias."""

import unittest
from unittest import mock

import torch
import torch.distributed as dist
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.modeling.draft.moe.balance import (
    count_tokens_per_expert,
    dispatch_balance,
    load_balance_loss,
    step_moe_balancing,
)
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.router import TopKRouter

E, K, B, S, H = 8, 2, 2, 6, 16


def _moe(coef=0.5, selection_bias=False, **dflash_config):
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        dflash_config={
            "moe_router_aux_loss_coef": coef,
            "moe_router_selection_bias": selection_bias,
            **dflash_config,
        },
    )
    return build_moe(config)


def _router(top_k=1):
    return TopKRouter(
        E,
        H,
        top_k,
        norm_topk_prob=False,
        initializer_range=0.5,
        selection_bias=True,
        bias_update_rate=0.001,
    )


def _batch_wise_router(top_k=K, selection_bias=False):
    return TopKRouter(
        E,
        H,
        top_k,
        norm_topk_prob=False,
        initializer_range=0.5,
        selection_bias=selection_bias,
        aux_loss_type="batch_wise",
    )


def _zero_mean_sign_step(counts, rate=0.001):
    delta = rate * torch.sign(counts.mean() - counts)
    return delta - delta.mean()


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
                # transformers sums the K slots per token: K times our per-token form.
                torch.testing.assert_close(K * _loss(scores, indices, flat), want)

    def test_sigmoid_scores_are_normalized_per_token(self):
        torch.manual_seed(0)
        scores = torch.randn(B * S, E).sigmoid()
        indices = scores.topk(K, dim=-1).indices
        probs = scores / scores.sum(dim=-1, keepdim=True)
        counts = count_tokens_per_expert(indices, None, E)
        want = E * (counts / (K * B * S) * probs.mean(dim=0)).sum()
        torch.testing.assert_close(_loss(scores, indices), want)

    def test_no_kept_slots_gives_a_zero_on_the_graph(self):
        logits = torch.randn(4, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        loss = _loss(scores, indices, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_balanced_routing_gives_one(self):
        scores = torch.full((E, E), 1 / E)
        indices = torch.stack([torch.arange(E), torch.arange(E).roll(1)], dim=1)
        torch.testing.assert_close(_loss(scores, indices), torch.tensor(1.0))


class DispatchBalanceTest(unittest.TestCase):
    def test_uniform_scores_one_and_collapse_scores_experts_per_slot(self):
        uniform = torch.full((E,), 3.0)
        torch.testing.assert_close(dispatch_balance(uniform), torch.tensor(1.0))
        one = torch.tensor([24.0, 0, 0, 0, 0, 0, 0, 0])  # top-1 stuck on one: E
        torch.testing.assert_close(dispatch_balance(one), torch.tensor(8.0))
        two = torch.tensor([12.0, 12.0, 0, 0, 0, 0, 0, 0])  # top-2 stuck on two: E/K
        torch.testing.assert_close(dispatch_balance(two), torch.tensor(4.0))

    def test_equals_the_loss_when_the_router_agrees_with_its_dispatch(self):
        counts = torch.tensor([5.0, 3, 2, 1, 1, 0, 0, 0])  # 12 tokens, top-1
        probs = (counts / counts.sum()).expand(12, E)
        torch.testing.assert_close(
            dispatch_balance(counts), load_balance_loss(probs, counts)
        )

    def test_an_empty_step_scores_zero(self):
        self.assertEqual(dispatch_balance(torch.zeros(E)).item(), 0.0)


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

    def test_switching_to_eval_clears_a_stale_training_loss(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        router(self.x)
        self.assertIsNotNone(router.aux_loss)
        router.eval()(self.x)
        self.assertIsNone(router.aux_loss)

    def test_coefficient_zero_still_computes_the_balance_metric(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        self.assertEqual(router.aux_loss_coef, 0.0)
        router(self.x)
        self.assertIsNotNone(router.aux_loss)

    def test_family_reads_the_aux_loss_coefficient(self):
        layer = _moe(coef=0.25)
        self.assertEqual(layer.gate.aux_loss_coef, 0.25)
        layer(torch.randn(B, S, H))
        self.assertIsNotNone(layer.gate.aux_loss)


class SelectionBiasTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_forward_accumulates_counts_and_leaves_the_bias_alone(self):
        router = _router()
        _, first, _ = router(torch.randn(B * S, H))
        _, second, _ = router(torch.randn(B * S, H))
        self.assertTrue(torch.all(router.selection_bias == 0))
        counts = count_tokens_per_expert(torch.cat([first, second]), None, E)
        torch.testing.assert_close(router.tokens_per_expert, counts)

    def test_update_moves_the_bias_once_with_a_zero_mean_delta(self):
        router = _router()
        _, indices, _ = router(torch.randn(B * S, H))
        counts = count_tokens_per_expert(indices, None, E)
        step_moe_balancing(router)
        torch.testing.assert_close(router.selection_bias, _zero_mean_sign_step(counts))
        self.assertAlmostEqual(router.selection_bias.mean().item(), 0.0, places=7)
        self.assertTrue(torch.all(router.tokens_per_expert == 0))

    def test_update_pools_the_counts_of_every_rank(self):
        router = _router()
        _, indices, _ = router(torch.randn(B * S, H))
        counts = count_tokens_per_expert(indices, None, E)
        other = torch.tensor([9.0, 0, 0, 0, 0, 0, 0, 3.0])  # another rank's counts
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                step_moe_balancing(router)
        want = _zero_mean_sign_step(counts + other)
        torch.testing.assert_close(router.selection_bias, want)

    def test_an_empty_step_leaves_the_bias_alone(self):
        router = _router(top_k=K)
        router(torch.randn(B * S, H), torch.zeros(B * S, dtype=torch.bool))
        self.assertTrue(torch.all(router.tokens_per_expert == 0))
        step_moe_balancing(router)
        self.assertTrue(torch.all(router.selection_bias == 0))

    def test_without_a_selection_bias_the_counts_still_score_the_step(self):
        router = TopKRouter(E, H, K, norm_topk_prob=True, initializer_range=0.5)
        _, indices, _ = router(torch.randn(B * S, H))
        counts = count_tokens_per_expert(indices, None, E)
        metrics = step_moe_balancing(router)
        self.assertIsNone(router.selection_bias)
        torch.testing.assert_close(
            metrics["moe/step_balance"], dispatch_balance(counts)
        )
        self.assertTrue(torch.all(router.tokens_per_expert == 0))
        self.assertNotIn("tokens_per_expert", router.state_dict())

    def test_buffers_keep_fp32_values_when_cast_to_bf16(self):
        router = _router()
        bias = torch.linspace(1.001, 1.002, E)  # bf16 would round these to 1.0
        router.selection_bias.copy_(bias)
        router.tokens_per_expert.fill_(1001.0)  # bf16 would round this to 1000
        router = router.bfloat16()
        self.assertEqual(router.selection_bias.dtype, torch.float32)
        self.assertEqual(router.tokens_per_expert.dtype, torch.float32)
        torch.testing.assert_close(router.selection_bias, bias)
        torch.testing.assert_close(router.tokens_per_expert, torch.full((E,), 1001.0))
        self.assertIn("selection_bias", router.state_dict())
        self.assertNotIn("tokens_per_expert", router.state_dict())

    def test_family_reads_the_bias_keys(self):
        self.assertIsNone(_moe().gate.selection_bias)
        gate = _moe(selection_bias=True).gate
        self.assertIsNotNone(gate.selection_bias)
        self.assertEqual(gate.bias_update_rate, 0.001)


class BatchWiseLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x = torch.randn(B * S, H)
        self.mask = _real_token_mask().reshape(-1)

    def test_the_first_micro_batch_matches_the_micro_batch_wise_loss(self):
        router = _batch_wise_router()
        scores, indices, _ = router(self.x, self.mask)
        torch.testing.assert_close(router.aux_loss, _loss(scores, indices, self.mask))

    def test_later_micro_batches_balance_the_step_so_far(self):
        router = _batch_wise_router()
        _, first, _ = router(self.x, self.mask)
        scores, second, _ = router(torch.randn(B * S, H))
        counts = count_tokens_per_expert(second, None, E)
        pooled = count_tokens_per_expert(first, self.mask, E) + counts
        torch.testing.assert_close(router.aux_loss, load_balance_loss(scores, pooled))
        # The loss reaches the gate weight through this micro-batch's scores.
        (grad,) = torch.autograd.grad(router.aux_loss, router.weight)
        self.assertTrue(torch.any(grad != 0))

    def test_the_load_pools_the_counts_of_every_rank(self):
        router = _batch_wise_router()
        other = torch.tensor([9.0, 0, 0, 0, 0, 0, 0, 3.0])  # another rank's counts
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                scores, indices, _ = router(self.x)
        counts = count_tokens_per_expert(indices, None, E)
        load = counts + other
        torch.testing.assert_close(router.aux_loss, load_balance_loss(scores, load))
        # The buffer stays rank-local: the bias update pools it once per step.
        torch.testing.assert_close(router.tokens_per_expert, counts)

    def test_the_step_boundary_starts_a_new_window_without_a_bias(self):
        router = _batch_wise_router()
        self.assertIsNone(router.selection_bias)
        router(self.x)
        self.assertTrue(torch.any(router.tokens_per_expert != 0))
        step_moe_balancing(router)
        self.assertTrue(torch.all(router.tokens_per_expert == 0))

    def test_with_a_bias_the_same_counts_serve_both(self):
        router = _batch_wise_router(selection_bias=True)
        _, indices, _ = router(self.x)
        counts = count_tokens_per_expert(indices, None, E)
        step_moe_balancing(router)
        torch.testing.assert_close(router.selection_bias, _zero_mean_sign_step(counts))
        self.assertTrue(torch.all(router.tokens_per_expert == 0))

    def test_family_reads_the_aux_loss_type(self):
        self.assertEqual(_moe().gate.aux_loss_type, "microbatch_wise")
        gate = _moe(moe_router_aux_loss_type="batch_wise").gate
        self.assertEqual(gate.aux_loss_type, "batch_wise")


if __name__ == "__main__":
    unittest.main(verbosity=2)
