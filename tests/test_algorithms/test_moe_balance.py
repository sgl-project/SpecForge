"""MoE load balancing: counts per expert over a step and the expert-bias update."""

import unittest
from dataclasses import replace

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
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_modeling.test_canonical_dspark import model_config as dense_config
from tests.test_modeling.test_canonical_dspark import moe_model_config
from tests.test_modeling.test_draft_moe_ffn import expert_spec, moe_spec, router_spec

# The objectives whose models carry expert routing out of the forward today.
BALANCED = ("DFlashDraftModel", "DSparkDraftModel")


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
        # The registry takes the balancing options for objectives that provide routing.
        self.assertFalse(hasattr(resolved.objective, "bias_update_rate"))
        self.assertEqual(
            resolved.statistics_settings,
            {"bias_update_rate": 0.001, "aux_loss_coef": 0.0},
        )
        resolve_objective("DFlashDraftModel", options={"bias_update_rate": 0.5})
        for architecture in ("DFlash2DraftModel", "DominoDraftModel"):
            # Their models do not carry expert routing out yet, so no option either.
            with self.subTest(architecture=architecture), self.assertRaises(TypeError):
                resolve_objective(architecture, options={"bias_update_rate": 0.5})
        with self.assertRaises(ValueError):
            resolve_objective("DSparkDraftModel", options={"bias_update_rate": -1.0})
        with self.assertRaises(ValueError):
            resolve_objective("DFlashDraftModel", options={"aux_loss_coef": -0.1})
        with self.assertRaisesRegex(ValueError, "MoE layers"):
            build_objective_model(build_draft_model(dense_config()), options=options)
        with self.assertRaisesRegex(ValueError, "expert_bias"):
            build_objective_model(
                build_draft_model(moe_model_config(expert_bias=False)), options=options
            )
        runtime = build_objective_model(
            build_draft_model(moe_model_config()), options={"num_anchors": 2}
        )
        self.assertIsNone(runtime.statistics)
        # The auxiliary loss alone needs no expert bias.
        runtime = build_objective_model(
            build_draft_model(moe_model_config(expert_bias=False)),
            options={"num_anchors": 2, "aux_loss_coef": 0.01},
        )
        self.assertEqual(runtime.statistics.rate, 0.0)
        self.assertEqual(runtime.statistics.settings["aux_loss_coef"], 0.01)

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
        balancer = runtime.statistics
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
        moved = (
            layer.gate.expert_bias - before
        )  # the sign step is test_router_contract's
        self.assertTrue(torch.any(moved != 0))
        torch.testing.assert_close(moved.sum(), torch.zeros(()))
        torch.testing.assert_close(
            metrics["moe/step_balance"], dispatch_balance(counts)
        )
        self.assertEqual(balancer.step(), {})

    def test_dflash_and_dspark_count_and_move_the_bias_end_to_end(self):
        """The two objectives that provide expert routing count kept slots and move the bias."""
        node = moe_spec(experts=expert_spec(16), router=router_spec(expert_bias=True))
        configs = [c for c in configurations() if c.architecture in BALANCED]
        self.assertEqual([c.architecture for c in configs], list(BALANCED))
        for config in configs:
            with self.subTest(architecture=config.architecture):
                moe_config = replace(
                    config,
                    mlp_configs={"moe": node},
                    layer_configs={
                        name: replace(layer, mlp="moe")
                        for name, layer in config.layer_configs.items()
                    },
                )
                torch.manual_seed(11)
                model = build_draft_model(moe_config)
                for module in model.modules():
                    if isinstance(module, MoEFFN):
                        module.experts_kernel = "loop"
                head = nn.Linear(32, config.vocab_size, bias=False)
                batch = training_batch(config)
                plain = build_objective_model(
                    model, options={"num_anchors": 2}, target_head=head
                )
                runtime = build_objective_model(
                    model,
                    options={"num_anchors": 2, "bias_update_rate": 0.01},
                    target_head=head,
                )
                self.assertIsNone(plain.statistics)
                self.assertIn("expert_routing", runtime.registration.provides)
                balancer = runtime.statistics
                self.assertEqual(len(balancer.layers), len(config.layer_types))
                torch.manual_seed(1)
                runtime(batch, step=1, total_steps=10).loss.backward()
                for counts in balancer.counts:
                    self.assertGreater(counts.sum().item(), 0)
                before = [layer.gate.expert_bias.clone() for layer in balancer.layers]
                self.assertEqual(set(balancer.step()), {"moe/step_balance"})
                for layer, old in zip(balancer.layers, before):
                    torch.testing.assert_close(
                        (layer.gate.expert_bias - old).sum(), torch.zeros(())
                    )

    def test_layers_with_different_expert_counts_keep_their_own_statistics(self):
        """An 8-expert and a 4-expert layer each move their bias on their own load."""
        config = moe_model_config()
        wide = replace(config.mlp_configs["moe"], num_experts=8)
        config = replace(
            config,
            mlp_configs={"moe8": wide, "moe4": config.mlp_configs["moe"]},
            layer_configs={
                "wide": replace(config.layer_configs["full_moe"], mlp="moe8"),
                "narrow": replace(config.layer_configs["full_moe"], mlp="moe4"),
            },
            layer_types=("wide", "narrow"),
        )
        model = build_draft_model(config)
        layers = [m for m in model.modules() if isinstance(m, MoEFFN)]
        self.assertEqual([layer.config.num_experts for layer in layers], [8, 4])
        runtime = build_objective_model(
            model,
            options={"num_anchors": 2, "bias_update_rate": 0.01},
            target_head=nn.Linear(32, config.vocab_size, bias=False),
        )
        torch.manual_seed(0)
        runtime(training_batch(config), step=0, total_steps=2).loss.backward()
        counts = [c.clone() for c in runtime.statistics.counts]
        self.assertEqual([tuple(c.shape) for c in counts], [(8,), (4,)])
        before = [layer.gate.expert_bias.clone() for layer in layers]
        metrics = runtime.statistics.step()
        for layer, old, load in zip(layers, before, counts):
            load = load.float()
            delta = 0.01 * torch.sign(load.mean() - load)
            torch.testing.assert_close(
                layer.gate.expert_bias - old, delta - delta.mean()
            )
        torch.testing.assert_close(
            metrics["moe/step_balance"],
            torch.stack([dispatch_balance(c) for c in counts]).mean(),
        )


if __name__ == "__main__":
    unittest.main()
