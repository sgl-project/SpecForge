# coding=utf-8
"""MoE util: Switch aux loss, router centering, fp32 buffers, and the training wiring."""

import unittest
from unittest import mock

import torch
import torch.distributed as dist
from torch import nn
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.util import (
    RouterCentering,
    include_moe_aux_loss,
    set_moe_token_mask,
    switch_aux_loss,
)
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.controller import TrainerCore
from specforge.training.strategies.base import DraftTrainStrategy, StepOutput

E, K, B, S, H = 8, 2, 2, 6, 16


def _moe(coef=0.5, centering=True):
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        router_aux_loss_coef=coef,
        dflash_config={"moe_router_centering": centering},
    )
    return build_moe(config)


def _real_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _sample_means(x, mask):
    return torch.stack([x[b][mask[b]].mean(dim=0) for b in range(len(x))])


class SwitchAuxLossTest(unittest.TestCase):
    def test_matches_transformers_load_balancing_loss(self):
        torch.manual_seed(0)
        logits = torch.randn(B * S, E)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        for mask in (None, _real_token_mask()):
            with self.subTest(masked=mask is not None):
                want = load_balancing_loss_func((logits,), E, K, mask)
                flat = None if mask is None else mask.reshape(-1)
                torch.testing.assert_close(switch_aux_loss(scores, indices, flat), want)

    def test_no_kept_slots_gives_a_zero_on_the_graph(self):
        logits = torch.randn(4, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        aux = switch_aux_loss(scores, indices, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(aux.item(), 0.0)
        aux.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_balanced_routing_gives_top_k(self):
        scores = torch.full((E, E), 1 / E)
        indices = torch.stack([torch.arange(E), torch.arange(E).roll(1)], dim=1)
        aux = switch_aux_loss(scores, indices)
        torch.testing.assert_close(aux, torch.tensor(float(K)))


class RouterCenteringTest(unittest.TestCase):
    def test_training_subtracts_each_samples_real_token_mean(self):
        torch.manual_seed(0)
        centering = RouterCentering(H, momentum=0.9)
        x, mask = torch.randn(B, S, H), _real_token_mask()
        torch.testing.assert_close(
            centering(x, mask), x - _sample_means(x, mask)[:, None]
        )
        # One momentum step from zero toward the batch's real-token mean.
        torch.testing.assert_close(centering.running_mean, 0.1 * x[mask].mean(dim=0))

    def test_running_mean_pools_the_kept_tokens_of_every_rank(self):
        torch.manual_seed(0)
        centering = RouterCentering(H, momentum=0.9)
        x, mask = torch.randn(B, S, H), _real_token_mask()
        other = torch.randn(B, S, H)  # another rank's batch, without padding
        other_stats = torch.cat([other.sum(dim=(0, 1)), torch.tensor([float(B * S)])])
        with mock.patch.object(dist, "is_initialized", return_value=True):
            with mock.patch.object(dist, "all_reduce", lambda t: t.add_(other_stats)):
                out = centering(x, mask)
        pooled = (x[mask].sum(dim=0) + other.sum(dim=(0, 1))) / (mask.sum() + B * S)
        torch.testing.assert_close(centering.running_mean, 0.1 * pooled)
        # Centering itself stays per sample and rank-local.
        torch.testing.assert_close(out, x - _sample_means(x, mask)[:, None])

    def test_eval_subtracts_the_running_mean_without_updating_it(self):
        centering = RouterCentering(H, momentum=0.9).eval()
        centering.running_mean.fill_(0.5)
        x = torch.randn(B, S, H)
        torch.testing.assert_close(centering(x), x - 0.5)
        torch.testing.assert_close(centering.running_mean, torch.full((H,), 0.5))

    def test_running_mean_keeps_fp32_values_when_the_layer_is_cast_to_bf16(self):
        layer = _moe()
        mean = torch.linspace(1.001, 1.002, H)  # bf16 would round these to 1.0
        layer.centering.running_mean.copy_(mean)
        layer = layer.bfloat16()
        self.assertEqual(layer.centering.running_mean.dtype, torch.float32)
        torch.testing.assert_close(layer.centering.running_mean, mean)
        self.assertEqual(layer.gate.weight.dtype, torch.bfloat16)
        out = layer.centering(torch.randn(B, S, H, dtype=torch.bfloat16))
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(layer.centering.running_mean.dtype, torch.float32)


class MoEAuxLossTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)
        self.x, self.mask = torch.randn(B, S, H), _real_token_mask()

    def test_training_routes_centered_inputs_and_keeps_a_masked_aux_loss(self):
        layer = _moe()
        set_moe_token_mask(layer, self.mask)
        layer(self.x)
        centered = self.x - _sample_means(self.x, self.mask)[:, None]
        scores, indices, _ = layer.gate(centered.reshape(-1, H))
        want = switch_aux_loss(scores, indices, self.mask.reshape(-1))
        torch.testing.assert_close(layer.aux_loss, want)

    def test_eval_keeps_no_aux_loss(self):
        layer = _moe().eval()
        set_moe_token_mask(layer, self.mask)
        layer(self.x)
        self.assertIsNone(layer.aux_loss)

    def test_include_adds_the_scaled_layer_mean_and_clears_layer_state(self):
        layers = nn.ModuleList([_moe(), _moe()])
        set_moe_token_mask(layers, self.mask)
        for layer in layers:
            layer(self.x)
        raw = torch.stack([layer.aux_loss for layer in layers]).mean().detach()
        step = StepOutput(loss=torch.tensor(1.0), metrics={})
        out = include_moe_aux_loss(layers, step)
        torch.testing.assert_close(out.loss.detach(), 1.0 + 0.5 * raw)
        torch.testing.assert_close(out.ratio_metrics["moe/aux_loss"][0], raw)
        for layer in layers:
            self.assertIsNone(layer.aux_loss)
            self.assertIsNone(layer.token_mask)

    def test_include_scales_the_aux_loss_into_loss_terms(self):
        layer = _moe()
        layer(self.x)
        aux = (0.5 * layer.aux_loss).detach()
        terms = (torch.tensor(6.0), torch.tensor(3.0))
        step = StepOutput(loss=torch.tensor(2.0), metrics={}, loss_terms=terms)
        out = include_moe_aux_loss(layer, step)
        torch.testing.assert_close(out.loss_terms[0].detach(), 6.0 + 3.0 * aux)

    def test_without_moe_layers_the_step_output_is_unchanged(self):
        step = StepOutput(loss=torch.tensor(1.0), metrics={})
        self.assertIs(include_moe_aux_loss(nn.Linear(2, 2), step), step)


class _MoEStrategy(DraftTrainStrategy):
    name = "moe-test"
    required_features = {"hidden_states"}

    def __init__(self, model):
        self.model = model

    def trainable_module(self):
        return self.model

    def forward_loss(self, batch, ctx=None):
        out = self.model(batch.tensors["hidden_states"])
        return StepOutput(loss=out.square().mean(), metrics={})


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


class TrainingWiringTest(unittest.TestCase):
    def test_train_step_adds_the_moe_aux_loss(self):
        torch.manual_seed(0)
        layer, x = _moe(), torch.randn(B, S, H)
        backend = _RecordingBackend()
        core = TrainerCore(_MoEStrategy(layer), backend)
        tensors = {"hidden_states": x}
        core.train_step(
            TrainBatch(sample_ids=["s0"], strategy="moe-test", tensors=tensors)
        )
        # The fake backend never updates weights, so a rerun reproduces the step.
        task = layer(x).square().mean()
        torch.testing.assert_close(
            backend.losses[0], (task + 0.5 * layer.aux_loss).detach()
        )

    def test_draft_blocks_hand_moe_layers_the_kept_block_mask(self):
        block_size, vocab = 3, 11
        draft = _MaskRecordingDraft(_moe())
        model = OnlineDFlashModel(
            draft_model=draft,
            target_lm_head=nn.Linear(H, vocab),
            target_embed_tokens=nn.Embedding(vocab, H),
            mask_token_id=0,
            block_size=block_size,
            attention_backend="sdpa",
            num_anchors=2,
        )
        keep = torch.tensor([[True, True], [True, False]])
        anchors = torch.tensor([[1, 3], [2, 0]])
        model._sample_anchor_positions = lambda *args, **kwargs: (anchors, keep)
        input_ids = torch.randint(1, vocab, (2, 6))
        model._forward_draft_blocks(input_ids, torch.randn(2, 6, H), torch.ones(2, 6))
        want = keep.repeat_interleave(block_size, dim=1)
        torch.testing.assert_close(draft.seen_mask, want)


if __name__ == "__main__":
    unittest.main(verbosity=2)
