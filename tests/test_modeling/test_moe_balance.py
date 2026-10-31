# coding=utf-8
"""The balancing: the router's expert bias, the layer's counts and loss, the step hook."""

import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.dflash2 import DFlash2DraftModel
from specforge.modeling.draft.layers.ffn.moe.balance import (
    MoEBalancing,
    dispatch_balance,
    moe_layers,
)
from specforge.modeling.draft.layers.ffn.moe.experts import RoutedExperts
from specforge.modeling.draft.layers.ffn.moe.layer import (
    MoELayer,
    count_tokens_per_expert,
    load_balance_loss,
)
from specforge.modeling.draft.layers.ffn.moe.router import Router, TopKRouter
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

E, K, T, H = 8, 2, 12, 16
B, S, V = 2, 6, 32


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
        with self.subTest(phase="the bias keeps fp32 values through a bf16 cast"):
            bias = torch.linspace(1.001, 1.002, E)  # bf16 would round these to 1.0
            router.expert_bias.copy_(bias)
            cast = router.bfloat16()
            self.assertEqual(cast.expert_bias.dtype, torch.float32)
            torch.testing.assert_close(cast.expert_bias, bias)
            self.assertEqual(cast.weight.dtype, torch.bfloat16)
            self.assertIn("expert_bias", cast.state_dict())


def _layer(expert_bias=False, bias_update_rate=0.001, **aux):
    gate = TopKRouter(
        E,
        H,
        K,
        norm_topk_prob=True,
        initializer_range=0.5,
        expert_bias=expert_bias,
        bias_update_rate=bias_update_rate,
    )
    experts = RoutedExperts(E, H, 8, initializer_range=0.5)
    return MoELayer(gate=gate, experts=experts, **aux)


def _real_moe_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _zero_mean_sign_step(counts, rate=0.001):
    delta = rate * torch.sign(counts.mean() - counts)
    return delta - delta.mean()


class LayerCountsTest(unittest.TestCase):
    def test_the_layer_counts_kept_slots_in_training(self):
        torch.manual_seed(0)
        layer, x = _layer(), torch.randn(B, S, H)
        mask = torch.ones(B, S, dtype=torch.bool)
        mask[1, 4:] = False  # the second sample ends in padding
        _, indices, _ = layer.gate(x.reshape(-1, H))
        with self.subTest(phase="masked micro-batch"):
            out = layer(x, mask)
            want = count_tokens_per_expert(indices, mask.reshape(-1), E)
            self.assertEqual(layer.tokens_per_expert.dtype, torch.int64)
            self.assertTrue(torch.equal(layer.tokens_per_expert, want))
            self.assertEqual(want.sum().item(), mask.sum().item() * K)
            # The mask only feeds the counts; the layer's output does not depend on it.
            torch.testing.assert_close(layer(x), out)
        with self.subTest(phase="the unmasked call accumulated"):
            want = want + count_tokens_per_expert(indices, None, E)
            self.assertTrue(torch.equal(layer.tokens_per_expert, want))
        with self.subTest(phase="eval does not count"):
            layer.eval()(x, mask)
            self.assertTrue(torch.equal(layer.tokens_per_expert, want))
        with self.subTest(phase="transient and cast-proof"):
            self.assertNotIn("tokens_per_expert", layer.state_dict())
            cast = layer.bfloat16()
            self.assertEqual(cast.tokens_per_expert.dtype, torch.int64)
            self.assertTrue(torch.equal(cast.tokens_per_expert, want))


class LoadBalanceLossTest(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(B * S, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        for mask in (None, _real_moe_token_mask()):
            with self.subTest(masked=mask is not None):
                # transformers sums the K slots per token: K times our per-token form.
                want = load_balancing_loss_func((logits,), E, K, mask)
                flat = None if mask is None else mask.reshape(-1)
                counts = count_tokens_per_expert(indices, flat, E)
                torch.testing.assert_close(
                    K * load_balance_loss(scores, counts, flat), want
                )
        with self.subTest(masked="every token"):
            none = torch.zeros(B * S, dtype=torch.bool)
            loss = load_balance_loss(
                scores, count_tokens_per_expert(indices, none, E), none
            )
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.all(logits.grad == 0))


class LayerAuxLossTest(unittest.TestCase):
    """The layer's per-forward loss: its load, its gradient, and when it is skipped."""

    def test_the_loss_follows_the_coefficient_and_the_load_type(self):
        torch.manual_seed(0)
        x, x2, mask = torch.randn(B, S, H), torch.randn(B, S, H), _real_moe_token_mask()
        flat = mask.reshape(-1)
        with self.subTest(phase="coefficient 0 skips the loss and its collectives"):
            layer = _layer(aux_loss_coef=0.0)
            with mock.patch.object(dist, "is_initialized", return_value=True):
                with mock.patch.object(dist, "all_reduce") as all_reduce:
                    layer(x, mask)
            self.assertIsNone(layer.aux_loss)
            all_reduce.assert_not_called()
        with self.subTest(phase="micro-batch-wise: this micro-batch's counts"):
            layer = _layer(aux_loss_coef=0.5, aux_loss_type="microbatch_wise")
            layer(x, mask)
            scores, indices, _ = layer.gate(x.reshape(-1, H))
            want = load_balance_loss(
                scores, count_tokens_per_expert(indices, flat, E), flat
            )
            torch.testing.assert_close(layer.aux_loss, want)
            (grad,) = torch.autograd.grad(layer.aux_loss, layer.gate.weight)
            self.assertTrue(torch.any(grad != 0))
        with self.subTest(phase="batch-wise: the step's counts so far, pooled"):
            layer = _layer(aux_loss_coef=0.5)  # the default type
            layer(x, mask)
            first = layer.tokens_per_expert.clone()
            other = torch.tensor([9, 0, 0, 0, 0, 0, 0, 3])  # another rank's counts
            with mock.patch.object(dist, "is_initialized", return_value=True):
                with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                    layer(x2)
            scores, indices, _ = layer.gate(x2.reshape(-1, H))
            pooled = first + count_tokens_per_expert(indices, None, E) + other
            torch.testing.assert_close(
                layer.aux_loss, load_balance_loss(scores, pooled)
            )
            self.assertTrue(torch.all(layer.tokens_per_expert == pooled - other))
        with self.subTest(phase="eval and bad types"):
            layer.eval()(x, mask)
            self.assertIsNone(layer.aux_loss)
            with self.assertRaises(ValueError):
                _layer(aux_loss_type="x")


class StepHookTest(unittest.TestCase):
    """Two micro-batches and one optimizer step boundary through MoEBalancing."""

    def test_the_step_pools_the_counts_and_moves_the_biases(self):
        torch.manual_seed(0)
        model = nn.ModuleList(
            [
                _layer(expert_bias=True),
                _layer(expert_bias=True, bias_update_rate=0.0),  # frozen
                _layer(expert_bias=False),
            ]
        )
        hook = MoEBalancing(model)
        self.assertEqual(hook.layers, list(model))
        x = torch.randn(B, S, H)
        for layer in model:
            layer(x)
            layer(x)  # the same routing again: counts double
        totals = [layer.tokens_per_expert.clone() for layer in model]
        other = torch.tensor([9, 0, 0, 0, 0, 0, 0, 3])  # another rank's counts
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                metrics = hook.on_step()
        with self.subTest(phase="the bias moves from the pooled counts"):
            want = _zero_mean_sign_step((totals[0] + other).float())
            torch.testing.assert_close(model[0].gate.expert_bias, want)
            self.assertTrue(torch.all(model[1].gate.expert_bias == 0))  # rate 0
            self.assertIsNone(model[2].gate.expert_bias)
            for layer in model:
                self.assertTrue(torch.all(layer.tokens_per_expert == 0))
        with self.subTest(phase="the step balance scores the pooled load"):
            want = torch.stack([dispatch_balance(t + other) for t in totals]).mean()
            torch.testing.assert_close(metrics["moe/step_balance"], want)
            uniform, collapsed = torch.full((E,), 3), torch.tensor([24] + [0] * 7)
            torch.testing.assert_close(dispatch_balance(uniform), torch.tensor(1.0))
            torch.testing.assert_close(dispatch_balance(collapsed), torch.tensor(8.0))
        with self.subTest(phase="dense models have no layers"):
            dense = MoEBalancing(nn.Linear(2, 2))
            self.assertEqual(dense.layers, [])
            self.assertEqual(dense.on_step(), {})


def _tiny_moe_draft(cls=DFlashDraftModel, **dflash_config):
    """A one-layer DFlash-family draft whose FFN is a 4-expert top-2 MoE, on CPU."""
    config = Qwen3Config(
        architectures=[cls.__name__],
        hidden_size=H,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        num_hidden_layers=1,
        num_target_layers=4,
        head_dim=4,
        max_position_embeddings=64,
        vocab_size=V,
        layer_types=["full_attention"],
        block_size=2,
        dflash_config={"target_layer_ids": [1], **dflash_config},
        moe_preset="qwen3_5_moe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
    )
    config._attn_implementation = "eager"
    return cls(config)


def _online(draft, **overrides):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=nn.Linear(H, V, bias=False),
        target_embed_tokens=nn.Embedding(V, H),
        mask_token_id=V - 1,
        block_size=2,
        attention_backend="eager",
        num_anchors=2,
    )
    return OnlineDFlashModel(**{**kwargs, **overrides})


class MaskPlumbingTest(unittest.TestCase):
    """The draft blocks hand the kept-block mask down to every MoE layer."""

    def _run_blocks(self, model, keep, anchors):
        model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
        torch.manual_seed(0)
        input_ids = torch.randint(1, V, (2, 6))
        model._forward_draft_blocks(input_ids, torch.randn(2, 6, H), torch.ones(2, 6))

    def test_the_mask_reaches_the_layers_as_an_argument(self):
        keep = torch.tensor([[True, True], [True, False]])
        anchors = torch.tensor([[1, 3], [2, 0]])
        for cls in (DFlashDraftModel, DFlash2DraftModel):
            with self.subTest(draft=cls.__name__):
                extra = {}
                if cls is DFlash2DraftModel:
                    extra = dict(
                        conv_group_size=4,
                        conv_kernel_size=2,
                        selector_rank=4,
                        selector_top_k=3,
                    )
                draft = _tiny_moe_draft(cls, **extra)
                (layer,) = moe_layers(draft)
                seen = []  # the decoder layer passes the mask as the second argument
                layer.register_forward_pre_hook(lambda _, args: seen.append(args[1]))
                self._run_blocks(_online(draft), keep, anchors)
                # The kept-block mask, expanded to draft tokens (block_size 2).
                self.assertTrue(torch.equal(seen[0], keep.repeat_interleave(2, dim=1)))
                # Only the kept blocks' tokens were counted: block_size 2, top-2.
                self.assertEqual(
                    layer.tokens_per_expert.sum().item(), keep.sum().item() * 2 * 2
                )


class _Backend(TrainingBackend):
    name = "test"

    def __init__(self, module=None):
        self.module = module

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()

    def scale_gradients(self, factor):
        for parameter in self.module.parameters():
            if parameter.grad is not None:
                parameter.grad.mul_(factor)

    def step(self):
        return torch.tensor(1.0)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class _MoEStrategy(DraftTrainStrategy):
    """A squared-output loss, with loss_terms (token sum over token count) on request."""

    name = "moe-test"
    required_features = {"hidden_states"}

    def __init__(self, model, terms=False):
        self.model = model
        self.terms = terms

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        out = self.model(batch.tensors["hidden_states"])
        numerator, denominator = out.square().sum(), torch.tensor(float(out.numel()))
        loss_terms = (numerator, denominator) if self.terms else None
        return StepOutput(
            loss=numerator / denominator, metrics={}, loss_terms=loss_terms
        )


def _batch(x):
    tensors = {"hidden_states": x}
    return TrainBatch(sample_ids=["s0"], strategy="moe-test", tensors=tensors)


class FoldTest(unittest.TestCase):
    """before_backward adds coef x loss per layer to the objective and logs the raw mean."""

    def test_the_hook_folds_the_layers_losses_once(self):
        torch.manual_seed(0)
        model = nn.ModuleList(
            [
                _layer(aux_loss_coef=0.5, aux_loss_type="microbatch_wise"),
                _layer(aux_loss_coef=0.25),
                _layer(aux_loss_coef=0.0),  # no loss, no term
            ]
        )
        hook, x, mask = (
            MoEBalancing(model),
            torch.randn(B, S, H),
            _real_moe_token_mask(),
        )
        for layer in model:
            layer(x, mask)
        losses = [model[0].aux_loss, model[1].aux_loss]
        weighted, raw = 0.5 * losses[0] + 0.25 * losses[1], (losses[0] + losses[1]) / 2
        out = StepOutput(
            loss=torch.tensor(2.0),
            metrics={},
            loss_terms=(torch.tensor(6.0), torch.tensor(3.0)),
        )
        with self.subTest(phase="with loss_terms: the numerator carries the fold"):
            folded = hook.before_backward(out)
            torch.testing.assert_close(folded.loss, 2.0 + weighted)
            numerator, denominator = folded.loss_terms
            torch.testing.assert_close(numerator, 6.0 + weighted * 3.0)
            self.assertEqual(denominator.item(), 3.0)
            metric, weight = folded.ratio_metrics["moe/aux_loss"]
            torch.testing.assert_close(metric, raw.detach())
            self.assertEqual(weight.item(), 1.0)
            self.assertFalse(metric.requires_grad)
            self.assertTrue(all(layer.aux_loss is None for layer in model))
        with self.subTest(phase="consumed: a second call changes nothing"):
            self.assertIs(hook.before_backward(out), out)
        with self.subTest(phase="without loss_terms: the loss carries the fold"):
            for layer in model:
                layer(x, mask)
            plain = StepOutput(loss=torch.tensor(2.0), metrics={})
            folded = hook.before_backward(plain)
            torch.testing.assert_close(folded.loss, 2.0 + weighted)
            self.assertIsNone(folded.loss_terms)


class TrainerStepHookTest(unittest.TestCase):
    """TrainerCore runs the registered hook: the fold before backward, the step work after."""

    def test_the_trainer_runs_the_registered_hook(self):
        torch.manual_seed(0)
        x = torch.randn(B, S, H)
        with self.subTest(phase="the fold reaches the gradient like an objective fold"):
            layer = _layer(aux_loss_coef=0.5, aux_loss_type="microbatch_wise")
            strategy = _MoEStrategy(layer, terms=True)
            core = TrainerCore(strategy, _Backend(layer), accumulation_steps=1)
            core.register_step_hook(MoEBalancing(layer))
            result = core.train_step(_batch(x))
            got = layer.gate.weight.grad.clone()
            # The reference: #1003's objective fold, numerator += coef * aux * denominator.
            layer.zero_grad()
            out = layer(x)
            numerator, denominator = out.square().sum(), float(out.numel())
            ((numerator + 0.5 * layer.aux_loss * denominator) / denominator).backward()
            torch.testing.assert_close(got, layer.gate.weight.grad)
            self.assertAlmostEqual(
                float(result.metrics["moe/aux_loss"]), layer.aux_loss.item(), places=5
            )
        with self.subTest(phase="the step work runs at the accumulation boundary"):
            layer = _layer(expert_bias=True)  # coefficient 0: no loss, no aux metric
            core = TrainerCore(_MoEStrategy(layer), _Backend(), accumulation_steps=2)
            core.register_step_hook(MoEBalancing(layer))
            first = core.train_step(_batch(x))
            self.assertNotIn("moe/step_balance", first.metrics)
            self.assertNotIn("moe/aux_loss", first.metrics)
            counts = layer.tokens_per_expert.clone()
            second = core.train_step(_batch(x))  # the same routing again: counts double
            self.assertAlmostEqual(
                float(second.metrics["moe/step_balance"]),
                dispatch_balance(2 * counts).item(),
                places=5,
            )
            self.assertTrue(torch.all(layer.tokens_per_expert == 0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
