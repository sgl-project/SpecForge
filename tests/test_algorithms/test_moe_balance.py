"""MoE load balancing: counts per expert over a step and the expert-bias update."""

import unittest

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.moe_balance import (
    MoEBalancer,
    count_tokens_per_expert,
    dispatch_balance,
)
from specforge.algorithms.objectives.registry import resolve_objective
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.mlp import MoEFFN
from tests.test_algorithms.test_objectives import training_batch
from tests.test_modeling.test_canonical_dspark import model_config as dense_config
from tests.test_modeling.test_canonical_moe import moe_model_config


class MoEBalanceTest(unittest.TestCase):
    def test_counts_and_dispatch_balance(self):
        expert_ids = torch.tensor([[0, 1], [0, 2], [3, 3], [1, 2]])
        keep = torch.tensor([True, True, False, True])
        counts = count_tokens_per_expert(expert_ids, keep, 4)
        self.assertEqual(counts.tolist(), [2, 2, 2, 0])
        self.assertEqual(counts.dtype, torch.long)
        uniform, collapsed = torch.tensor([3, 3, 3, 3]), torch.tensor([8, 0, 0, 0])
        torch.testing.assert_close(dispatch_balance(uniform), torch.tensor(1.0))
        torch.testing.assert_close(dispatch_balance(collapsed), torch.tensor(4.0))
        self.assertEqual(dispatch_balance(torch.zeros(4, dtype=torch.long)).item(), 0)

    def test_option_is_validated_and_needs_a_biased_moe_draft(self):
        options = {"num_anchors": 2, "bias_update_rate": 0.001}
        resolved = resolve_objective("DSparkDraftModel", options=options)
        self.assertEqual(resolved.objective.bias_update_rate, 0.001)
        for architecture in (
            "DFlashDraftModel",
            "DFlash2DraftModel",
            "DominoDraftModel",
        ):
            with self.subTest(architecture=architecture):
                resolve_objective(architecture, options={"bias_update_rate": 0.5})
        with self.assertRaises(ValueError):
            resolve_objective("DSparkDraftModel", options={"bias_update_rate": -1.0})
        with self.assertRaisesRegex(ValueError, "MoE layers"):
            build_objective_model(build_draft_model(dense_config()), options=options)
        with self.assertRaisesRegex(ValueError, "expert_bias"):
            build_objective_model(
                build_draft_model(moe_model_config(expert_bias=False)), options=options
            )
        runtime = build_objective_model(
            build_draft_model(moe_model_config()), options={"num_anchors": 2}
        )
        self.assertIsNone(runtime.balancer)

    def test_a_step_pools_the_window_and_moves_each_bias_zero_mean(self):
        config = moe_model_config()
        model = build_draft_model(config)
        runtime = build_objective_model(
            model,
            options={"num_anchors": 2, "bias_update_rate": 0.01},
            target_head=nn.Linear(
                32, config.vocab_size, bias=False
            ),  # DSpark's teacher
        )
        balancer = runtime.balancer
        self.assertIsInstance(balancer, MoEBalancer)
        self.assertEqual(balancer.step(), {})  # nothing observed yet
        (layer,) = [m for m in model.modules() if isinstance(m, MoEFFN)]
        before = layer.gate.expert_bias.clone()
        torch.manual_seed(0)
        for _ in range(2):  # two micro-batches of one optimizer step
            runtime(training_batch(config), step=0, total_steps=2).loss.backward()
        with torch.no_grad():  # a forward without gradients counts nothing
            runtime(training_batch(config), step=0, total_steps=2)
        (counts,) = balancer.counts
        slots = config.spec_config.protocol.num_model_slots
        # Both anchors of each micro-batch are kept: every slot routes to top-2 experts.
        self.assertEqual(counts.sum().item(), 2 * 2 * slots * 2)
        metrics = balancer.step()
        self.assertEqual(set(metrics), {"moe/step_balance"})
        self.assertEqual(balancer.counts, [None])
        load = counts.float()
        expected = 0.01 * torch.sign(load.mean() - load)
        expected = expected - expected.mean()
        torch.testing.assert_close(layer.gate.expert_bias - before, expected)
        torch.testing.assert_close(
            metrics["moe/step_balance"], dispatch_balance(counts)
        )
        self.assertEqual(balancer.step(), {})


if __name__ == "__main__":
    unittest.main()
