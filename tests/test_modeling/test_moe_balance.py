# coding=utf-8
"""balance.py and its wiring: loss, hooks, objective folds and the step boundary."""

import copy
import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.algorithms.common.dflash_family_model import (
    OnlineDFlashModel,
    OnlineDominoModel,
    OnlineDSparkModel,
)
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.domino import DominoDraftModel
from specforge.modeling.draft.dspark import DSparkDraftModel
from specforge.modeling.draft.moe.balance import (
    count_tokens_per_expert,
    dispatch_balance,
    load_balance_loss,
    set_moe_token_mask,
    step_moe_balancing,
    take_moe_aux_loss,
)
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.router import TopKRouter
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

E, K, B, S, H, V = 8, 2, 2, 6, 16, 32


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


def _zero_mean_sign_step(counts, rate=0.001):
    delta = rate * torch.sign(counts.mean() - counts)
    return delta - delta.mean()


def _real_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _loss(scores, indices, mask=None):
    return load_balance_loss(scores, count_tokens_per_expert(indices, mask, E), mask)


def _scores_and_indices(gate, x):
    """The router's scores and picks for ``x`` without touching its training state."""
    scores, indices, _ = gate.eval()(x.reshape(-1, H))
    gate.train()
    return scores, indices


def _tiny_moe_draft(coef=0.5, cls=DFlashDraftModel, **dflash_config):
    """A one-layer DFlash-family draft whose FFN is a 4-expert MoE, on CPU."""
    config = Qwen3Config(
        architectures=["DFlashDraftModel"],
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
        dflash_config={
            "target_layer_ids": [1],
            "moe_router_aux_loss_coef": coef,
            **dflash_config,
        },
        moe_preset="qwen3_5_moe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
    )
    config._attn_implementation = "eager"
    return cls(config)


class _MaskRecordingDraft(nn.Module):
    """A stand-in draft: one MoE layer over the noise embedding."""

    def __init__(self, moe):
        super().__init__()
        self.moe = moe
        self.sliding_window = None
        self.seen_mask = None

    def forward(self, position_ids, noise_embedding, target_hidden, attention_mask):
        self.seen_mask = self.moe.gate.token_mask
        return self.moe(noise_embedding)


def _online(draft, cls=OnlineDFlashModel, **overrides):
    kwargs = dict(
        draft_model=draft,
        target_lm_head=nn.Linear(H, V, bias=False),
        target_embed_tokens=nn.Embedding(V, H),
        mask_token_id=V - 1,
        block_size=2,
        attention_backend="eager",
        num_anchors=2,
    )
    return cls(**{**kwargs, **overrides})


class LoadBalanceLossTest(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(B * S, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        for mask in (None, _real_token_mask()):
            with self.subTest(masked=mask is not None):
                # transformers sums the K slots per token: K times our per-token form.
                want = load_balancing_loss_func((logits,), E, K, mask)
                flat = None if mask is None else mask.reshape(-1)
                torch.testing.assert_close(K * _loss(scores, indices, flat), want)
        with self.subTest(masked="every token"):
            loss = _loss(scores, indices, torch.zeros(B * S, dtype=torch.bool))
            self.assertEqual(loss.item(), 0.0)
            loss.backward()
            self.assertTrue(torch.all(logits.grad == 0))


class FP32BuffersTest(unittest.TestCase):
    def test_buffers_keep_fp32_values_when_cast_to_bf16(self):
        router = TopKRouter(
            E, H, 1, norm_topk_prob=False, initializer_range=0.5, selection_bias=True
        )
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


class BalancingHooksTest(unittest.TestCase):
    """Two micro-batches and one optimizer step through the hooks in balance.py."""

    def test_micro_batches_feed_the_loss_and_the_step_moves_the_bias(self):
        torch.manual_seed(0)
        layers = nn.ModuleList(
            [
                _moe(
                    coef=0.5,
                    selection_bias=True,
                    moe_router_aux_loss_type="microbatch_wise",
                ),
                _moe(coef=0.25, selection_bias=True),  # the default: batch-wise
            ]
        )
        gates = [layer.gate for layer in layers]
        x, mask = torch.randn(B, S, H), _real_token_mask()

        with self.subTest(phase="first micro-batch, masked"):
            set_moe_token_mask(layers, mask)
            for layer in layers:
                self.assertIs(layer.gate.token_mask, mask)
                layer(x)
            losses = [gate.aux_loss for gate in gates]
            first = [gate.tokens_per_expert.clone() for gate in gates]
            aux, raw = take_moe_aux_loss(layers)
            torch.testing.assert_close(aux, 0.5 * losses[0] + 0.25 * losses[1])
            torch.testing.assert_close(raw, (losses[0] + losses[1]) / 2)
            for layer in layers:
                self.assertIsNone(layer.gate.aux_loss)
                self.assertIsNone(layer.gate.token_mask)
            # On the step's first forward both loss types read this micro-batch alone.
            for gate, loss in zip(gates, losses):
                scores, indices = _scores_and_indices(gate, x)
                torch.testing.assert_close(
                    loss, _loss(scores, indices, mask.reshape(-1))
                )
                (grad,) = torch.autograd.grad(loss, gate.weight)
                self.assertTrue(torch.any(grad != 0))

        with self.subTest(phase="second micro-batch, batch-wise load"):
            x2 = torch.randn(B, S, H)
            for layer in layers:
                layer(x2)
            got = gates[1].aux_loss
            scores, indices = _scores_and_indices(gates[1], x2)
            pooled = first[1] + count_tokens_per_expert(indices, None, E)
            torch.testing.assert_close(got, load_balance_loss(scores, pooled))
            take_moe_aux_loss(layers)

        with self.subTest(phase="optimizer step"):
            totals = [gate.tokens_per_expert.clone() for gate in gates]
            other = torch.tensor([9.0, 0, 0, 0, 0, 0, 0, 3.0])  # another rank's counts
            with mock.patch.object(dist, "is_initialized", return_value=True):
                with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other)):
                    metrics = step_moe_balancing(layers)
            for gate, total in zip(gates, totals):
                want = _zero_mean_sign_step(total + other)
                torch.testing.assert_close(gate.selection_bias, want)
                self.assertTrue(torch.all(gate.tokens_per_expert == 0))
            want = torch.stack([dispatch_balance(t + other) for t in totals]).mean()
            torch.testing.assert_close(metrics["moe/step_balance"], want)
            uniform, collapsed = torch.full((E,), 3.0), torch.tensor([24.0] + [0.0] * 7)
            torch.testing.assert_close(dispatch_balance(uniform), torch.tensor(1.0))
            torch.testing.assert_close(dispatch_balance(collapsed), torch.tensor(8.0))

        with self.subTest(phase="the bias steers the choice, not the weights"):
            scores, indices, weights = gates[0](x.reshape(-1, H))
            biased = (scores + gates[0].selection_bias).topk(K, dim=-1).indices
            self.assertTrue(torch.equal(indices, biased))
            picked = scores.gather(-1, indices)
            torch.testing.assert_close(weights, picked / picked.sum(-1, keepdim=True))

        with self.subTest(phase="eval and dense models"):
            set_moe_token_mask(layers, mask)
            layers[0].eval()(x)
            self.assertIsNone(gates[0].aux_loss)
            self.assertEqual(step_moe_balancing(nn.Linear(2, 2)), {})


class ObjectiveTest(unittest.TestCase):
    """Each objective adds coef x the raw loss; the draft blocks hand over the mask."""

    def _inputs(self, **extra):
        torch.manual_seed(0)
        return dict(
            input_ids=torch.randint(0, V - 1, (1, 8)),
            hidden_states=torch.randn(1, 8, H),
            loss_mask=torch.ones(1, 8),
            **extra,
        )

    def _check_fold(self, with_aux, inputs, has_loss_terms):
        plain = copy.deepcopy(with_aux)
        plain.draft_model.layers[0].mlp.gate.aux_loss_coef = 0.0
        torch.manual_seed(1)
        loss, _, metrics = with_aux(**inputs)
        torch.manual_seed(1)
        plain_loss, _, plain_metrics = plain(**inputs)
        raw, _ = metrics["ratio_metrics"]["moe/aux_loss"]
        # The metric is the unweighted loss, reported at any coefficient.
        plain_raw = plain_metrics["ratio_metrics"]["moe/aux_loss"][0]
        torch.testing.assert_close(plain_raw, raw)
        torch.testing.assert_close(loss - plain_loss, 0.5 * raw)
        if has_loss_terms:
            numerator, denominator = metrics["loss_terms"]
            plain_numerator, _ = plain_metrics["loss_terms"]
            torch.testing.assert_close(
                numerator - plain_numerator, 0.5 * raw * denominator
            )
        else:
            self.assertNotIn("loss_terms", metrics)
        # The per-forward state is consumed.
        self.assertIsNone(with_aux.draft_model.layers[0].mlp.gate.aux_loss)
        self.assertIsNone(with_aux.draft_model.layers[0].mlp.gate.token_mask)

    def test_each_objective_folds_the_loss_and_hands_the_block_mask(self):
        with self.subTest(family="dflash"):
            self._check_fold(_online(_tiny_moe_draft()), self._inputs(), True)
        with self.subTest(family="domino"):
            draft = _tiny_moe_draft(cls=DominoDraftModel, emb_dim=8, gru_hidden_dim=8)
            model = _online(draft, cls=OnlineDominoModel)
            self._check_fold(model, self._inputs(lambda_base=0.3), True)
        with self.subTest(family="dspark"):
            draft = _tiny_moe_draft(cls=DSparkDraftModel, markov_rank=0)
            model = _online(draft, cls=OnlineDSparkModel)
            inputs = self._inputs(target_last_hidden_states=torch.randn(1, 8, H))
            self._check_fold(model, inputs, False)
        with self.subTest(family="dense"):
            config = copy.deepcopy(_tiny_moe_draft().config)
            config.moe_preset = None
            _, _, metrics = _online(DFlashDraftModel(config))(**self._inputs())
            self.assertNotIn("moe/aux_loss", metrics["ratio_metrics"])
        with self.subTest(phase="the draft blocks hand the layers the kept-block mask"):
            block_size = 3
            draft = _MaskRecordingDraft(_moe())
            model = _online(draft, block_size=block_size)
            keep = torch.tensor([[True, True], [True, False]])
            anchors = torch.tensor([[1, 3], [2, 0]])
            model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
            input_ids = torch.randint(1, V, (2, 6))
            model._forward_draft_blocks(
                input_ids, torch.randn(2, 6, H), torch.ones(2, 6)
            )
            want = keep.repeat_interleave(block_size, dim=1)
            torch.testing.assert_close(draft.seen_mask, want)


class _Backend(TrainingBackend):
    name = "test"

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        loss.backward()

    def step(self):
        return torch.tensor(1.0)

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


class _Wrapped(nn.Module):
    """Stands in for DDP and FSDP, which keep the model under ``.module``."""

    def __init__(self, module):
        super().__init__()
        self.module = module


class _MoEStrategy(DraftTrainStrategy):
    name = "moe-test"
    required_features = {"hidden_states"}

    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        draft = getattr(self.model, "module", self.model).draft_model
        out = draft.moe(batch.tensors["hidden_states"])
        return StepOutput(loss=out.square().mean(), metrics={})


def _batch(x):
    tensors = {"hidden_states": x}
    return TrainBatch(sample_ids=["s0"], strategy="moe-test", tensors=tensors)


class StepHookTest(unittest.TestCase):
    def test_trainer_runs_the_hook_at_the_optimizer_step_boundary(self):
        torch.manual_seed(0)
        draft, x = _MaskRecordingDraft(_moe(selection_bias=True)), torch.randn(B, S, H)
        # DDP and FSDP keep the model under .module; the hook is found through it.
        strategy = _MoEStrategy(_Wrapped(_online(draft)))
        core = TrainerCore(strategy, _Backend(), accumulation_steps=2)
        first = core.train_step(_batch(x))
        counts = draft.moe.gate.tokens_per_expert.clone()
        self.assertTrue(torch.all(draft.moe.gate.selection_bias == 0))
        self.assertNotIn("moe/step_balance", first.metrics)
        second = core.train_step(_batch(x))  # the same routing again: counts double
        want = _zero_mean_sign_step(2 * counts)
        torch.testing.assert_close(draft.moe.gate.selection_bias, want)
        self.assertTrue(torch.all(draft.moe.gate.tokens_per_expert == 0))
        # The step's pooled balance is reported with the boundary step's metrics.
        self.assertAlmostEqual(
            second.metrics["moe/step_balance"],
            dispatch_balance(2 * counts).item(),
            places=5,
        )
        self.assertIsNone(_MoEStrategy(nn.Linear(2, 2)).on_optimizer_step())


if __name__ == "__main__":
    unittest.main(verbosity=2)
