"""Replay alignment tests with the real small DFlash2 backbone and selector."""

import copy
import unittest

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.common.dflash_opd import OPDDFlash2Model
from specforge.modeling.draft.dflash2 import DFlash2DraftModel


class ReplayModelTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        config = Qwen3Config(
            hidden_size=16,
            intermediate_size=32,
            num_attention_heads=4,
            num_key_value_heads=2,
            num_hidden_layers=1,
            num_target_layers=4,
            head_dim=4,
            max_position_embeddings=64,
            vocab_size=32,
            layer_types=["full_attention"],
            dflash_config=dict(
                block_size=8,
                conv_group_size=4,
                conv_kernel_size=2,
                mask_token_id=31,
                selector_rank=4,
                selector_top_k=3,
                target_layer_ids=[1],
            ),
        )
        draft = DFlash2DraftModel(config)
        draft.config._attn_implementation = "eager"
        # A trained selector has a nonzero successor codebook.
        with torch.no_grad():
            draft.candidate_selector.successor_codebook.normal_(std=0.1)
        self.model = OPDDFlash2Model(
            draft,
            nn.Linear(16, 32, bias=False).requires_grad_(False),
            nn.Embedding(32, 16).requires_grad_(False),
            31,
            block_size=8,
            attention_backend="eager",
            num_anchors=2,
            loss_type="dpace",
            lk_loss_type="lambda",
        )
        self.ids = torch.arange(12)[None]
        self.taps = torch.randn(1, 12, 16)
        self.mask = torch.tensor([[0] * 4 + [1] * 8])
        self.anchors = torch.tensor([4])

    def replay(self):
        with torch.no_grad():
            _, _, h = self.model._forward_draft_blocks(
                self.ids,
                self.taps,
                self.mask,
                anchor_positions=self.anchors[None],
                block_keep_mask=torch.tensor([[True]]),
            )
            h = h.reshape(1, 8, 16)[:, 1:]
            unary, ids = self.model.draft_model.transform_unary_logits(
                self.model.lm_head(h)
            ).topk(3)
            proposed = ids[..., 0]
            previous = torch.cat(
                [self.ids[0, self.anchors, None], proposed[:, :-1]], -1
            )
            scores = self.model.draft_model.candidate_selector.score_candidates(
                candidate_ids=ids,
                unary_logits=unary,
                hidden_states=h,
                predecessor_ids=previous,
            )
        p = torch.full((1, 7, 32), 0.3 / 31)
        p.scatter_(-1, ids[..., :1], 0.7)
        return dict(
            anchor_positions=self.anchors,
            proposed_ids=proposed,
            candidate_ids=ids,
            q=scores.softmax(-1),
            accepted_lengths=torch.tensor([1]),
            exposed_lengths=torch.tensor([7]),
            target_ids=torch.arange(32).expand(1, 7, 32),
            target_probs=p,
        )

    def test_full_backbone_and_selector_receive_finite_gradients(self):
        loss, metrics = self.model(
            input_ids=self.ids,
            hidden_states=self.taps,
            loss_mask=self.mask,
            replay=self.replay(),
        )
        loss.backward()
        self.assertEqual(metrics["opd_terms"].block_count, 1)
        self.assertEqual(metrics["opd_terms"].position_count, 2)
        for name, p in self.model.draft_model.named_parameters():
            self.assertIsNotNone(p.grad, name)
            self.assertTrue(torch.isfinite(p.grad).all(), name)
        selector = self.model.draft_model.candidate_selector
        self.assertGreater(selector.hidden_projection.weight.grad.abs().sum(), 0)
        self.assertGreater(
            self.model.draft_model.layers[0].self_attn.q_proj.weight.grad.abs().sum(), 0
        )

    def test_future_target_context_cannot_change_earlier_block(self):
        replay = self.replay()
        before = self.model(
            input_ids=self.ids,
            hidden_states=self.taps,
            loss_mask=self.mask,
            replay=replay,
            auxiliary_coefficient=0,
        )[0]
        poisoned = self.taps.clone()
        poisoned[:, 4:] = torch.randn_like(poisoned[:, 4:]) * 1000
        after = self.model(
            input_ids=self.ids,
            hidden_states=poisoned,
            loss_mask=self.mask,
            replay=replay,
            auxiliary_coefficient=0,
        )[0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_candidate_and_policy_probability_drift_rejected(self):
        replay = self.replay()
        changed = copy.deepcopy(replay)
        changed["candidate_ids"][0, 0, 0] = (changed["candidate_ids"][0, 0, 0] + 1) % 32
        with self.assertRaisesRegex(ValueError, "candidate identity"):
            self.model(
                input_ids=self.ids,
                hidden_states=self.taps,
                loss_mask=self.mask,
                replay=changed,
            )
        changed = copy.deepcopy(replay)
        changed["q"][0, 0] = torch.tensor([1.0, 0.0, 0.0])
        with self.assertRaisesRegex(ValueError, "probability drift"):
            self.model(
                input_ids=self.ids,
                hidden_states=self.taps,
                loss_mask=self.mask,
                replay=changed,
            )

    def test_final_uncaptured_bonus_is_never_used_as_context(self):
        replay = self.replay()
        before = self.model(
            input_ids=self.ids,
            hidden_states=self.taps,
            loss_mask=self.mask,
            replay=replay,
            auxiliary_coefficient=0,
        )[0]
        after = self.model(
            input_ids=self.ids,
            hidden_states=self.taps[:, :-1],
            loss_mask=self.mask,
            replay=replay,
            auxiliary_coefficient=0,
        )[0]
        torch.testing.assert_close(before, after, rtol=0, atol=0)

    def test_nonfinite_recorded_probability_cannot_bypass_parity_check(self):
        replay = self.replay()
        replay["q"][0, 0, 0] = float("nan")
        with self.assertRaisesRegex(
            ValueError, "Invalid recorded replay probabilities"
        ):
            self.model(
                input_ids=self.ids,
                hidden_states=self.taps,
                loss_mask=self.mask,
                replay=replay,
            )

    def test_disabled_auxiliary_does_not_require_supervised_anchor_sampling(self):
        loss, metrics = self.model(
            input_ids=self.ids,
            hidden_states=self.taps,
            loss_mask=torch.zeros_like(self.mask),
            replay=self.replay(),
            auxiliary_coefficient=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["auxiliary_loss"], 0)

    def test_replay_uses_configured_block_width(self):
        config = copy.deepcopy(self.model.draft_model.config)
        config.dflash_config["block_size"] = 4
        draft = DFlash2DraftModel(config)
        self.model = OPDDFlash2Model(
            draft,
            self.model.lm_head,
            self.model.embed_tokens,
            31,
            block_size=4,
            attention_backend="eager",
            num_anchors=2,
        )
        with torch.no_grad():
            _, _, h = self.model._forward_draft_blocks(
                self.ids,
                self.taps,
                self.mask,
                anchor_positions=self.anchors[None],
                block_keep_mask=torch.tensor([[True]]),
            )
            h = h.reshape(1, 4, 16)[:, 1:]
            unary, ids = self.model.draft_model.transform_unary_logits(
                self.model.lm_head(h)
            ).topk(3)
            proposed = ids[..., 0]
            previous = torch.cat(
                [self.ids[0, self.anchors, None], proposed[:, :-1]], -1
            )
            scores = self.model.draft_model.candidate_selector.score_candidates(
                candidate_ids=ids,
                unary_logits=unary,
                hidden_states=h,
                predecessor_ids=previous,
            )
        replay = dict(
            anchor_positions=self.anchors,
            proposed_ids=proposed,
            candidate_ids=ids,
            q=scores.softmax(-1),
            accepted_lengths=torch.tensor([1]),
            exposed_lengths=torch.tensor([3]),
            target_ids=torch.arange(32).expand(1, 3, 32),
            target_probs=torch.full((1, 3, 32), 1 / 32),
        )
        loss, metrics = self.model(
            input_ids=self.ids,
            hidden_states=self.taps,
            loss_mask=self.mask,
            replay=replay,
            auxiliary_coefficient=0,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(metrics["opd_terms"].position_count, 2)


if __name__ == "__main__":
    unittest.main()
