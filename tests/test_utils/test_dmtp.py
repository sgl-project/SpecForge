"""DMTP token alignment, visibility, frozen teacher, and training gradients."""

import unittest
from unittest.mock import patch

import torch
from torch import nn
from transformers import Qwen3Config

from specforge.algorithms.dmtp.model import OnlineDMTPModel
from specforge.inference.mtp_capture import FrozenMTPCapture
from specforge.modeling.draft.dmtp import DMTPDraftModel
from specforge.modeling.draft.mtp.qwen3_5 import Qwen3_5MTPDraftModel


def config():
    c = Qwen3Config(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=8,
        layer_types=["full_attention"],
        num_target_layers=4,
        dflash_config={"target_layer_ids": [1], "mask_token_id": 0},
        block_size=3,
        attn_output_gate=True,
        partial_rotary_factor=0.5,
    )
    c._attn_implementation = "sdpa"
    return c


class DMTPTest(unittest.TestCase):
    def model(self):
        draft = DMTPDraftModel(config())
        return OnlineDMTPModel(
            draft,
            nn.Linear(16, 32, bias=False).requires_grad_(False),
            nn.Embedding(32, 16).requires_grad_(False),
            mask_token_id=0,
            block_size=3,
            attention_backend="sdpa",
            num_anchors=2,
            fused_logits_loss=False,
            teacher_metrics=False,
        )

    def test_teacher_alignment_prefix_equivalence_and_no_future_leak(self):
        torch.manual_seed(9)
        teacher = FrozenMTPCapture(Qwen3_5MTPDraftModel(config()))
        ids = torch.tensor([1, 2, 3, 4, 5, 6])
        target = torch.randn(6, 16, requires_grad=True)
        full = teacher(ids, target)
        self.assertFalse(full.requires_grad)
        self.assertEqual(full[0].abs().sum().item(), 0)
        for end in range(2, 7):
            prefix = teacher(ids[:end], target[:end])
            torch.testing.assert_close(full[end - 1], prefix[-1], atol=1e-6, rtol=1e-5)
        changed = target.detach().clone()
        changed[3:] += 100
        torch.testing.assert_close(teacher(ids, changed)[:4], full[:4])

    def test_mtp_backbone_matches_independent_hf_qwen_layer(self):
        from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        from transformers.models.qwen3_5.modeling_qwen3_5 import (
            Qwen3_5DecoderLayer,
            Qwen3_5TextRotaryEmbedding,
        )

        torch.manual_seed(17)
        cfg = config()
        model = Qwen3_5MTPDraftModel(cfg).eval()
        native = Qwen3_5TextConfig(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=8,
            layer_types=["full_attention"],
            attention_dropout=0.0,
            rope_parameters={
                "rope_type": "default",
                "rope_theta": 10000.0,
                "partial_rotary_factor": 0.5,
                "mrope_section": [1, 1, 0],
            },
        )
        native._attn_implementation = "eager"
        reference = Qwen3_5DecoderLayer(native, layer_idx=0).eval()
        reference.load_state_dict(model.mtp.layers[0].state_dict(), strict=True)
        rotary = Qwen3_5TextRotaryEmbedding(native)
        ids, hidden = torch.tensor([[1, 2, 3, 4]]), torch.randn(1, 4, 16)
        position = torch.arange(4)[None]
        fused = model.mtp.fc(
            torch.cat(
                [
                    model.mtp.pre_fc_norm_embedding(model.embed_tokens(ids)),
                    model.mtp.pre_fc_norm_hidden(hidden),
                ],
                dim=-1,
            )
        )
        rope = rotary(fused, position.unsqueeze(0).expand(3, -1, -1))
        mask = torch.full((4, 4), float("-inf")).triu(1)[None, None]
        expected = model.mtp.norm(
            reference(fused, position_embeddings=rope, attention_mask=mask)
        )
        actual = model.forward_hidden(ids, hidden, position_ids=position)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)

    def test_context_excludes_a_and_t1_but_keeps_absolute_positions(self):
        model = self.model()
        anchors = torch.tensor([[4]])  # A is 3, T1 is 4.
        mask = model._attention_mask(anchors, torch.tensor([[True]]), 8)
        self.assertTrue(mask[0, 0, :, :3].all())
        self.assertFalse(mask[0, 0, :, 3:8].any())
        self.assertEqual(model._create_position_ids(anchors).tolist(), [[4, 5, 6]])

    def test_fusion_reads_previous_row_and_only_replaces_t1(self):
        model = self.model()
        ids = torch.arange(8)[None]
        mtp = torch.randn(1, 8, 16)
        anchors, keep = torch.tensor([[3, 5]]), torch.tensor([[True, True]])
        with patch.object(
            model.draft_model, "fuse_mtp", wraps=model.draft_model.fuse_mtp
        ) as fuse:
            out = model._create_noise_embed(ids, anchors, keep, mtp_hidden=mtp).reshape(
                1, 2, 3, 16
            )
        torch.testing.assert_close(fuse.call_args.args[1], mtp[:, [2, 4]])
        expected = model.embed_tokens(torch.zeros(1, 2, 2, dtype=torch.long))
        torch.testing.assert_close(out[:, :, 1:], expected)

    def test_backward_updates_fusion_and_backbone_only(self):
        model = self.model()
        ids = torch.arange(8)[None]
        target = torch.randn(1, 8, 16, requires_grad=True)
        mtp = torch.randn(1, 8, 16, requires_grad=True)
        loss, _, _ = model(ids, target, torch.ones_like(ids), mtp_hidden=mtp)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(model.draft_model.mtp_fusion.weight.grad.abs().sum(), 0)
        self.assertIsNone(mtp.grad)
        self.assertIsNone(target.grad)
        self.assertIsNone(model.embed_tokens.weight.grad)
        self.assertIsNone(model.lm_head.weight.grad)
        self.assertTrue(
            any(p.grad is not None for p in model.draft_model.layers.parameters())
        )

    def test_checkpoint_round_trip_preserves_fusion(self):
        first, second = self.model(), self.model()
        second.draft_model.load_state_dict(first.draft_model.state_dict())
        x, h = torch.randn(2, 16), torch.randn(2, 16)
        torch.testing.assert_close(
            first.draft_model.fuse_mtp(x, h), second.draft_model.fuse_mtp(x, h)
        )

    def test_standard_dflash_generation_cannot_silently_skip_mtp(self):
        with self.assertRaisesRegex(NotImplementedError, "native MTP"):
            self.model().draft_model.spec_generate()

    def test_offline_normalization_validates_version_and_lengths(self):
        from specforge.algorithms.dmtp.data import normalize

        raw = {
            "input_ids": torch.arange(6),
            "loss_mask": torch.ones(6),
            "hidden_states": torch.randn(6, 16),
            "mtp_hidden": torch.randn(6, 16),
            "mtp_hidden_version": torch.tensor(1),
        }
        self.assertEqual(normalize(raw, 6)["mtp_hidden"].shape, (1, 6, 16))
        with self.assertRaisesRegex(ValueError, "version"):
            normalize({**raw, "mtp_hidden_version": torch.tensor(2)}, 6)
        with self.assertRaisesRegex(ValueError, "lengths"):
            normalize({**raw, "mtp_hidden": raw["mtp_hidden"][:-1]}, 6)


if __name__ == "__main__":
    unittest.main()
