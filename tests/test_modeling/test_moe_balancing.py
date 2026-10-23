# coding=utf-8
"""MoE balancing: the router's load-balance (aux) loss and the kept-token mask."""

import unittest

import torch
from torch import nn
from transformers import Qwen3Config
from transformers.models.qwen3_5_moe.configuration_qwen3_5_moe import (
    Qwen3_5MoeTextConfig,
)
from transformers.models.qwen3_5_moe.modeling_qwen3_5_moe import (
    load_balancing_loss_func,
)

from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
from specforge.modeling.draft.dflash import DFlashDraftModel
from specforge.modeling.draft.moe.families.qwen3_5_moe import build_moe
from specforge.modeling.draft.moe.layer import set_moe_token_mask
from specforge.modeling.draft.moe.router import (
    TopKRouter,
    count_tokens_per_expert,
    load_balance_loss,
)

E, K, B, S, H, V = 8, 2, 2, 6, 16, 32


def _moe(coef=0.5):
    config = Qwen3_5MoeTextConfig(
        num_experts=E,
        num_experts_per_tok=K,
        hidden_size=H,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=8,
        router_aux_loss_coef=coef,
    )
    return build_moe(config)


def _real_token_mask():
    mask = torch.ones(B, S, dtype=torch.bool)
    mask[1, 4:] = False  # the second sample ends in padding
    return mask


def _loss(scores, indices, mask=None):
    return load_balance_loss(scores, count_tokens_per_expert(indices, mask, E), mask)


def _tiny_moe_draft(coef=0.5):
    """A one-layer DFlash draft whose FFN is a 4-expert MoE, on CPU."""
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
        dflash_config={"target_layer_ids": [1]},
        moe_preset="qwen3_5_moe",
        num_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=8,
        shared_expert_intermediate_size=0,
        norm_topk_prob=False,
        router_aux_loss_coef=coef,
    )
    config._attn_implementation = "eager"
    return DFlashDraftModel(config)


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
                torch.testing.assert_close(_loss(scores, indices, flat), want)

    def test_sigmoid_scores_are_normalized_per_token(self):
        torch.manual_seed(0)
        scores = torch.randn(B * S, E).sigmoid()
        indices = scores.topk(K, dim=-1).indices
        probs = scores / scores.sum(dim=-1, keepdim=True)
        counts = count_tokens_per_expert(indices, None, E)
        want = E * (counts / (B * S) * probs.mean(dim=0)).sum()
        torch.testing.assert_close(_loss(scores, indices), want)

    def test_no_kept_slots_gives_a_zero_on_the_graph(self):
        logits = torch.randn(4, E, requires_grad=True)
        scores = logits.softmax(dim=-1)
        indices = scores.topk(K, dim=-1).indices
        loss = _loss(scores, indices, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(loss.item(), 0.0)
        loss.backward()
        self.assertTrue(torch.all(logits.grad == 0))

    def test_balanced_routing_gives_top_k(self):
        scores = torch.full((E, E), 1 / E)
        indices = torch.stack([torch.arange(E), torch.arange(E).roll(1)], dim=1)
        torch.testing.assert_close(_loss(scores, indices), torch.tensor(float(K)))


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

    def test_coefficient_zero_still_computes_the_balance_metric(self):
        router = TopKRouter(E, H, K, norm_topk_prob=False, initializer_range=0.5)
        self.assertEqual(router.aux_loss_coef, 0.0)
        router(self.x)
        self.assertIsNotNone(router.aux_loss)

    def test_family_reads_router_aux_loss_coef(self):
        layer = _moe(coef=0.25)
        self.assertEqual(layer.gate.aux_loss_coef, 0.25)
        layer(torch.randn(B, S, H))
        self.assertIsNotNone(layer.gate.aux_loss)


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


if __name__ == "__main__":
    unittest.main(verbosity=2)
