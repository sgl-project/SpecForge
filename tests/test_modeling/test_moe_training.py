# coding=utf-8
"""MoE training hooks: the kept-token mask, the objective folds, and the step hook."""

import copy
import unittest

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
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
    take_moe_aux_loss,
)
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
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
        self.seen_mask = self.moe.token_mask
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


class TokenMaskTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x, self.mask = torch.randn(B, S, H), _real_token_mask()

    def test_set_moe_token_mask_reaches_every_layer_and_masks_the_loss(self):
        layers = nn.ModuleList([_moe(), _moe()])
        set_moe_token_mask(layers, self.mask)
        for layer in layers:
            self.assertIs(layer.token_mask, self.mask)
            layer(self.x)
            got = layer.gate.aux_loss
            scores, indices, _ = layer.gate.eval()(self.x.reshape(-1, H))
            want = _loss(scores, indices, self.mask.reshape(-1))
            torch.testing.assert_close(got, want)

    def test_eval_ignores_the_mask(self):
        layer = _moe().eval()
        set_moe_token_mask(layer, self.mask)
        layer(self.x)
        self.assertIsNone(layer.gate.aux_loss)

    def test_draft_blocks_hand_moe_layers_the_kept_block_mask(self):
        block_size = 3
        draft = _MaskRecordingDraft(_moe())
        model = _online(draft, block_size=block_size)
        keep = torch.tensor([[True, True], [True, False]])
        anchors = torch.tensor([[1, 3], [2, 0]])
        model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
        input_ids = torch.randint(1, V, (2, 6))
        model._forward_draft_blocks(input_ids, torch.randn(2, 6, H), torch.ones(2, 6))
        want = keep.repeat_interleave(block_size, dim=1)
        torch.testing.assert_close(draft.seen_mask, want)

    def test_moe_drafts_refuse_activation_checkpointing(self):
        with self.assertRaisesRegex(ValueError, "activation checkpointing"):
            _tiny_moe_draft().gradient_checkpointing_enable()


class TakeAuxLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x, self.mask = torch.randn(B, S, H), _real_token_mask()

    def test_take_returns_the_layer_sum_and_mean_and_clears_layer_state(self):
        layers = nn.ModuleList([_moe(coef=0.5), _moe(coef=0.25)])
        set_moe_token_mask(layers, self.mask)
        for layer in layers:
            layer(self.x)
        losses = [layer.gate.aux_loss for layer in layers]
        aux, raw = take_moe_aux_loss(layers)
        torch.testing.assert_close(aux, 0.5 * losses[0] + 0.25 * losses[1])
        torch.testing.assert_close(raw, (losses[0] + losses[1]) / 2)
        for layer in layers:
            self.assertIsNone(layer.gate.aux_loss)
            self.assertIsNone(layer.token_mask)

    def test_without_moe_layers_there_is_nothing_to_take(self):
        self.assertIsNone(take_moe_aux_loss(nn.Linear(2, 2)))


class ObjectiveTest(unittest.TestCase):
    """Each objective adds 0.5 x the raw loss versus the same model at coefficient 0."""

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
        raw, one = metrics["ratio_metrics"]["moe/aux_loss"]
        self.assertEqual(one.item(), 1.0)
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
        self.assertIsNone(with_aux.draft_model.layers[0].mlp.token_mask)

    def test_dflash_objective(self):
        model = _online(_tiny_moe_draft())
        self._check_fold(model, self._inputs(), has_loss_terms=True)

    def test_domino_objective(self):
        draft = _tiny_moe_draft(cls=DominoDraftModel, emb_dim=8, gru_hidden_dim=8)
        model = _online(draft, cls=OnlineDominoModel)
        self._check_fold(model, self._inputs(lambda_base=0.3), has_loss_terms=True)

    def test_dspark_objective(self):
        draft = _tiny_moe_draft(cls=DSparkDraftModel, markov_rank=0)
        model = _online(draft, cls=OnlineDSparkModel)
        inputs = self._inputs(target_last_hidden_states=torch.randn(1, 8, H))
        self._check_fold(model, inputs, has_loss_terms=False)

    def test_a_dense_draft_reports_no_moe_metric(self):
        torch.manual_seed(0)
        config = copy.deepcopy(_tiny_moe_draft().config)
        config.moe_preset = None
        model = _online(DFlashDraftModel(config))
        _, _, metrics = model(
            input_ids=torch.randint(0, V - 1, (1, 8)),
            hidden_states=torch.randn(1, 8, H),
            loss_mask=torch.ones(1, 8),
        )
        self.assertNotIn("moe/aux_loss", metrics["ratio_metrics"])


class _RecordingBackend(TrainingBackend):
    name = "recording"

    def __init__(self):
        self.losses = []

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        self.losses.append(loss.detach())
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
    def setUp(self):
        torch.manual_seed(0)

    def test_strategy_hook_moves_the_bias_through_a_wrapper(self):
        draft = _MaskRecordingDraft(_moe(selection_bias=True))
        strategy = _MoEStrategy(_Wrapped(_online(draft)))
        _, indices, _ = draft.moe.gate(torch.randn(B * S, H))
        counts = count_tokens_per_expert(indices, None, E)
        metrics = strategy.on_optimizer_step()
        torch.testing.assert_close(
            draft.moe.gate.selection_bias, _zero_mean_sign_step(counts)
        )
        torch.testing.assert_close(
            metrics["moe/step_balance"], dispatch_balance(counts)
        )

    def test_a_model_without_a_hook_is_a_no_op(self):
        self.assertIsNone(_MoEStrategy(nn.Linear(2, 2)).on_optimizer_step())

    def test_the_bias_moves_at_the_optimizer_step_boundary_only(self):
        draft, x = _MaskRecordingDraft(_moe(selection_bias=True)), torch.randn(B, S, H)
        core = TrainerCore(
            _MoEStrategy(_online(draft)), _RecordingBackend(), accumulation_steps=2
        )
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
