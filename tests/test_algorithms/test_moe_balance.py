"""MoE load balancing: counts per expert over a step and the expert-bias update."""

import unittest
from dataclasses import replace
from types import SimpleNamespace

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.base import ObjectiveResult
from specforge.algorithms.objectives.moe_balance import (
    AUX_LOSS_TYPES,
    MoEBalancer,
    balance_from_terms,
    count_tokens_per_expert,
    dispatch_balance,
)
from specforge.algorithms.objectives.registry import resolve_objective
from specforge.modeling import build_draft_model
from specforge.modeling.draft.layers.inputs import ExpertRouting
from specforge.modeling.draft.layers.mlp import MoEFFN
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_algorithms.test_objectives import configurations, training_batch
from tests.test_modeling.test_canonical_dspark import model_config as dense_config
from tests.test_modeling.test_canonical_moe import moe_model_config
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
        self.assertEqual(resolved.objective.bias_update_rate, 0.001)
        resolve_objective("DFlashDraftModel", options={"bias_update_rate": 0.5})
        for architecture in ("DFlash2DraftModel", "DominoDraftModel"):
            # Their models do not carry expert routing out yet, so no option either.
            with self.subTest(architecture=architecture), self.assertRaises(TypeError):
                resolve_objective(architecture, options={"bias_update_rate": 0.5})
        with self.assertRaises(ValueError):
            resolve_objective("DSparkDraftModel", options={"bias_update_rate": -1.0})
        with self.assertRaisesRegex(ValueError, "aux_loss_type"):
            resolve_objective(
                "DSparkDraftModel", options={"aux_loss_type": "sequence_wise"}
            )
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
        self.assertIsNone(runtime.balancer)
        # The auxiliary loss alone needs no expert bias.
        runtime = build_objective_model(
            build_draft_model(moe_model_config(expert_bias=False)),
            options={"num_anchors": 2, "aux_loss_coef": 0.01},
        )
        self.assertEqual(runtime.balancer.rate, 0.0)
        self.assertEqual(runtime.balancer.aux_loss_type, "batch_wise")

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
        moved = (
            layer.gate.expert_bias - before
        )  # the sign step is test_router_contract's
        self.assertTrue(torch.any(moved != 0))
        torch.testing.assert_close(moved.sum(), torch.zeros(()))
        torch.testing.assert_close(
            metrics["moe/step_balance"], dispatch_balance(counts)
        )
        self.assertEqual(balancer.step(), {})

    def test_dflash_and_dspark_balance_and_fold_the_loss_end_to_end(self):
        """The two objectives that provide expert routing move the bias and fold the loss."""
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
                    options={
                        "num_anchors": 2,
                        "bias_update_rate": 0.01,
                        "aux_loss_coef": 0.5,
                    },
                    target_head=head,
                )
                self.assertIsNone(plain.balancer)
                self.assertIn("expert_routing", runtime.registration.provides)
                balancer = runtime.balancer
                self.assertEqual(len(balancer.layers), len(config.layer_types))
                torch.manual_seed(1)
                base = plain(batch, step=1, total_steps=10)
                torch.manual_seed(1)
                result = runtime(batch, step=1, total_steps=10)
                with self.assertRaisesRegex(ValueError, "unfinished"):
                    result.loss
                self.assertEqual(
                    tuple(result.balance.counts.shape), (len(config.layer_types), 4)
                )
                finished = balancer.finish(result, pooled=True)
                aux, one = finished.ratios["moe/aux_loss"]
                self.assertEqual(one.item(), 1.0)
                self.assertGreater(aux.item(), 0.0)
                self.assertNotIn("moe/aux_loss", base.ratios)
                torch.testing.assert_close(finished.loss, base.loss + 0.5 * aux)
                finished.loss.backward()
                for counts in balancer.counts:
                    self.assertGreater(counts.sum().item(), 0)
                self.assertEqual(set(balancer.step()), {"moe/step_balance"})

    def test_balancing_reads_the_registration_not_the_model(self):
        """An objective that does not declare expert routing cannot ask for balancing."""
        model = build_draft_model(moe_model_config())
        registration = resolve_objective("DSparkDraftModel").registration
        self.assertIn("expert_routing", registration.provides)
        silent = replace(
            registration, provides=registration.provides - {"expert_routing"}
        )
        objective = SimpleNamespace(bias_update_rate=0.01)
        self.assertIsInstance(
            MoEBalancer.for_objective(model, objective, registration), MoEBalancer
        )
        with self.assertRaisesRegex(ValueError, "does not provide expert routing"):
            MoEBalancer.for_objective(model, objective, silent)
        objective.bias_update_rate = 0.0
        self.assertIsNone(MoEBalancer.for_objective(model, objective, silent))

    def test_the_loss_matches_transformers(self):
        from transformers.models.qwen3_moe.modeling_qwen3_moe import (
            load_balancing_loss_func,
        )

        model = build_draft_model(moe_model_config(expert_bias=False))
        balancer = MoEBalancer(model, rate=0.0, aux_loss_coef=1.0)
        torch.manual_seed(0)
        B, S, E, K = 2, 6, 4, 2
        logits = torch.randn(B * S, E, requires_grad=True)
        scores = logits.softmax(-1)
        expert_ids = scores.topk(K, dim=-1).indices
        output = SimpleNamespace(expert_routing=(ExpertRouting(expert_ids, scores),))
        mask = torch.tensor([[1, 1, 1, 0, 0, 0], [1, 1, 1, 1, 1, 0]])
        for attention_mask in (None, mask):
            with self.subTest(masked=attention_mask is not None):
                want = load_balancing_loss_func((logits,), E, K, attention_mask)
                keep = (
                    torch.ones(B * S, dtype=torch.bool)
                    if attention_mask is None
                    else attention_mask.reshape(-1).bool()
                )
                term = balancer.observe(output, keep=keep)
                self.assertTrue(
                    torch.equal(
                        term.counts[0], count_tokens_per_expert(expert_ids, keep, E)
                    )
                )
                # Transformers sums the K slots of each token: K times the per-token form.
                torch.testing.assert_close(
                    K * balance_from_terms(term.mean_probs, term.counts), want
                )
        uniform = torch.full((1, E), 1.0 / E)
        torch.testing.assert_close(
            balance_from_terms(uniform, torch.full((1, E), 2)), torch.tensor(1.0)
        )
        none = torch.zeros(B * S, dtype=torch.bool)
        term = balancer.observe(output, keep=none)
        loss = balance_from_terms(term.mean_probs, term.counts)
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_the_term_is_finished_with_its_own_or_this_steps_pooled_load(self):
        model = build_draft_model(moe_model_config(expert_bias=False))
        (layer,) = [m for m in model.modules() if isinstance(m, MoEFFN)]
        E = layer.config.num_experts
        torch.manual_seed(0)
        keep = torch.tensor([True] * 8 + [False] * 2)
        outputs = []
        for _ in range(2):
            scores = torch.randn(10, E, requires_grad=True).softmax(-1)
            ids = scores.topk(2, dim=-1).indices
            outputs.append(
                SimpleNamespace(expert_routing=(ExpertRouting(ids, scores),))
            )
        counts = [
            count_tokens_per_expert(o.expert_routing[0].expert_ids, keep, E)
            for o in outputs
        ]
        for aux_loss_type in AUX_LOSS_TYPES:
            with self.subTest(aux_loss_type=aux_loss_type):
                balancer = MoEBalancer(
                    model, rate=0.0, aux_loss_coef=0.5, aux_loss_type=aux_loss_type
                )
                terms, finished = [], []
                for output, own in zip(outputs, counts):
                    term = balancer.observe(output, keep=keep)
                    self.assertEqual(tuple(term.mean_probs.shape), (1, E))
                    self.assertTrue(term.mean_probs.requires_grad)
                    self.assertTrue(torch.equal(term.counts[0], own))
                    result = ObjectiveResult(
                        torch.tensor(6.0), torch.tensor(2.0), {}, term
                    )
                    with self.assertRaisesRegex(ValueError, "unfinished"):
                        result.loss
                    terms.append(term)
                    finished.append(balancer.finish(result, pooled=True))
                first, second = finished
                self.assertIsNone(second.balance)
                # A step's first forward balances against its own load either way.
                aux = first.ratios["moe/aux_loss"][0]
                torch.testing.assert_close(
                    aux,
                    balance_from_terms(terms[0].mean_probs, counts[0][None]).detach(),
                )
                torch.testing.assert_close(first.loss, 3.0 + 0.5 * aux)
                load = (
                    counts[0] + counts[1]
                    if aux_loss_type == "batch_wise"
                    else counts[1]
                )
                torch.testing.assert_close(
                    second.ratios["moe/aux_loss"][0],
                    balance_from_terms(terms[1].mean_probs, load[None]).detach(),
                )
                if aux_loss_type == "batch_wise":
                    self.assertTrue(
                        torch.equal(balancer.pooled[0], counts[0] + counts[1])
                    )
                else:
                    self.assertIsNone(balancer.pooled)
                # Without pooling (evaluation) every forward balances against its own load.
                again = balancer.finish(
                    ObjectiveResult(
                        torch.tensor(6.0),
                        torch.tensor(2.0),
                        {},
                        balancer.observe(outputs[1], keep=keep),
                    ),
                    pooled=False,
                )
                torch.testing.assert_close(
                    again.ratios["moe/aux_loss"][0],
                    balance_from_terms(terms[1].mean_probs, counts[1][None]).detach(),
                )
                # The optimizer step resets the pooled load; rate 0 moves no bias.
                self.assertEqual(set(balancer.step()), {"moe/step_balance"})
                self.assertIsNone(balancer.pooled)
        off = MoEBalancer(model, rate=0.0, aux_loss_coef=0.0)
        self.assertIsNone(off.observe(outputs[0], keep=keep))
        plain = ObjectiveResult(torch.tensor(6.0), torch.tensor(2.0))
        self.assertIs(off.finish(plain, pooled=True), plain)

    def test_the_strategy_finishes_the_term_and_pools_only_while_training(self):
        config = moe_model_config(expert_bias=False)
        model = build_draft_model(config)
        runtime = build_objective_model(
            model,
            options={"num_anchors": 2, "aux_loss_coef": 0.5},
            target_head=nn.Linear(32, config.vocab_size, bias=False),
        )
        strategy = CanonicalTrainStrategy(
            runtime, name="dspark", objective=runtime.objective, required_features=()
        )
        batch = TrainBatch(
            sample_ids=["a"], strategy="dspark", tensors=training_batch(config)
        )
        out = strategy.forward_loss(batch)
        aux = out.ratio_metrics["moe/aux_loss"][0]
        self.assertGreater(aux.item(), 0.0)
        self.assertIsNotNone(runtime.balancer.pooled)  # training: this step's load
        numerator, denominator = out.loss_terms
        torch.testing.assert_close(out.loss, numerator / denominator)
        runtime.balancer.step()
        runtime.eval()
        with torch.no_grad():
            out = strategy.forward_loss(batch)
        self.assertIn("moe/aux_loss", out.ratio_metrics)
        self.assertIsNone(runtime.balancer.pooled)  # evaluation: its own load


if __name__ == "__main__":
    unittest.main()
