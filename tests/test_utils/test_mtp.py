# coding=utf-8
"""CPU unit tests for the MTP (Qwen3.5 native-head fine-tune) algorithm.

Covers the pieces that do not need a GPU or a real target checkpoint:
  - OnlineMTPModel forward/shift/loss/accuracy plumbing
  - MTPTrainStrategy batch -> StepOutput adaptation
  - native ``mtp.*`` weight initialization from a target checkpoint
  - draft-architecture and built-in algorithm registration
"""

from __future__ import annotations

import tempfile
import unittest
from types import SimpleNamespace

import torch
from transformers.models.qwen3.modeling_qwen3 import Qwen3Config

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.algorithms.mtp.providers import _init_from_native_mtp
from specforge.core.mtp import OnlineMTPModel
from specforge.modeling.draft import available_drafts, resolve_draft
from specforge.modeling.draft.mtp import Qwen3_5MTPDraftModel
from specforge.training.strategies.base import MTPTrainStrategy


def _tiny_config(**overrides) -> Qwen3Config:
    payload = dict(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=512,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        rms_norm_eps=1e-6,
        attn_output_gate=True,
        partial_rotary_factor=0.25,
        mtp_config={"share_lm_head": True},
        tie_word_embeddings=False,
    )
    payload.update(overrides)
    config = Qwen3Config(**payload)
    config._attn_implementation = "eager"
    return config


def _tiny_batch(config: Qwen3Config, seq_len: int = 16, batch: int = 2):
    input_ids = torch.randint(0, config.vocab_size, (batch, seq_len))
    hidden_states = torch.randn(batch, seq_len, config.hidden_size)
    loss_mask = torch.ones(batch, seq_len)
    return input_ids, hidden_states, loss_mask


class OnlineMTPModelTest(unittest.TestCase):
    def test_forward_returns_finite_loss_and_accuracy_lists(self):
        config = _tiny_config()
        model = OnlineMTPModel(Qwen3_5MTPDraftModel(config))
        input_ids, hidden_states, loss_mask = _tiny_batch(config)

        loss, corrects, denoms = model(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
        )

        self.assertEqual(0, loss.dim())
        self.assertTrue(torch.isfinite(loss))
        self.assertEqual(1, len(corrects))
        self.assertEqual(1, len(denoms))
        # next-token shift drops one position
        self.assertEqual((2, 15), corrects[0].shape)
        self.assertEqual((2, 15), denoms[0].shape)

    def test_loss_backward_populates_mtp_grads(self):
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        model = OnlineMTPModel(draft)
        input_ids, hidden_states, loss_mask = _tiny_batch(config)

        loss, _, _ = model(
            input_ids=input_ids, hidden_states=hidden_states, loss_mask=loss_mask
        )
        loss.backward()

        self.assertIsNotNone(draft.mtp.fc.weight.grad)
        self.assertTrue(torch.isfinite(draft.mtp.fc.weight.grad).all())

    def test_shift_for_next_token_matches_serving_alignment(self):
        model = OnlineMTPModel(Qwen3_5MTPDraftModel(_tiny_config()))
        logits = torch.zeros(1, 5, 7)
        input_ids = torch.tensor([[10, 11, 12, 13, 14]])
        loss_mask = torch.ones(1, 5)

        shift_logits, shift_labels, shift_mask = model._shift_for_next_token(
            logits, input_ids, loss_mask
        )

        self.assertEqual((1, 4, 7), shift_logits.shape)
        # labels are x_2..x_T padded with one ignore index
        self.assertEqual([12, 13, 14, -100], shift_labels[0].tolist())
        # the padded position is masked out
        self.assertEqual([1, 1, 1, 0], shift_mask[0].tolist())


class MTPTrainStrategyTest(unittest.TestCase):
    def test_forward_loss_adapts_model_outputs(self):
        loss = torch.tensor(1.5, requires_grad=True)
        corrects = [torch.tensor([[1.0, 0.0, 1.0]])]
        denoms = [torch.tensor([[1.0, 1.0, 1.0]])]

        class _Stub(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.proj = torch.nn.Linear(1, 1)

            def forward(self, input_ids, hidden_states, loss_mask):
                return loss, corrects, denoms

        strategy = MTPTrainStrategy(_Stub())
        batch = SimpleNamespace(
            tensors={
                "input_ids": torch.zeros(1, 3, dtype=torch.long),
                "loss_mask": torch.ones(1, 3),
                "target_last_hidden_states": torch.zeros(1, 3, 4),
            }
        )

        out = strategy.forward_loss(batch)

        self.assertIs(out.loss, loss)
        self.assertAlmostEqual(2.0 / 3.0, out.metrics["accuracy"].item())
        self.assertEqual(3.0, out.metrics["accuracy_denom"].item())
        num, den = out.ratio_metrics["accuracy"]
        self.assertEqual(2.0, num.item())
        self.assertEqual(3.0, den.item())

    def test_forward_loss_rejects_missing_features(self):
        strategy = MTPTrainStrategy(torch.nn.Linear(1, 1))
        batch = SimpleNamespace(tensors={"input_ids": torch.zeros(1, 3)})
        with self.assertRaisesRegex(ValueError, "missing required features"):
            strategy.forward_loss(batch)

    def test_checkpoint_state_filter_strips_draft_prefix(self):
        strategy = MTPTrainStrategy(torch.nn.Linear(1, 1))
        state = {
            "draft_model.mtp.fc.weight": torch.zeros(2, 2),
            "draft_model.embed_tokens.weight": torch.zeros(4, 2),
            "other.weight": torch.zeros(1),
        }
        filtered = strategy.checkpoint_state_filter(state)
        self.assertEqual({"mtp.fc.weight", "embed_tokens.weight"}, set(filtered.keys()))


class NativeMTPInitTest(unittest.TestCase):
    def test_loads_native_mtp_weights_from_target_checkpoint(self):
        from safetensors.torch import save_file

        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        replacement = torch.ones_like(draft.mtp.fc.weight)

        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            save_file({"mtp.fc.weight": replacement}, f"{tmpdir}/model.safetensors")
            cfg = SimpleNamespace(model=SimpleNamespace(target_model_path=tmpdir))
            _init_from_native_mtp(cfg, draft)

        self.assertTrue(torch.equal(draft.mtp.fc.weight, replacement))

    def test_missing_native_weights_warns_and_keeps_random_init(self):
        config = _tiny_config()
        draft = Qwen3_5MTPDraftModel(config)
        before = draft.mtp.fc.weight.detach().clone()

        with tempfile.TemporaryDirectory(prefix="mtp-native-init-") as tmpdir:
            cfg = SimpleNamespace(model=SimpleNamespace(target_model_path=tmpdir))
            _init_from_native_mtp(cfg, draft)  # must not raise

        self.assertTrue(torch.equal(draft.mtp.fc.weight, before))


class MTPRegistrationTest(unittest.TestCase):
    def test_draft_architecture_is_registered(self):
        self.assertIn("Qwen3_5MTPDraftModel", available_drafts())
        self.assertIs(resolve_draft("Qwen3_5MTPDraftModel"), Qwen3_5MTPDraftModel)

    def test_builtin_registry_resolves_mtp(self):
        registration = builtin_algorithm_registry().resolve("mtp")
        self.assertEqual("mtp", registration.spec.name)
        self.assertEqual(
            "Qwen3_5MTPDraftModel",
            registration.providers.model.draft_config.architecture,
        )

    def test_offline_layout_persists_only_final_hidden(self):
        providers = builtin_algorithm_registry().resolve("mtp").providers
        layout = providers.offline_for("text").capture_layout
        self.assertEqual(
            ("input_ids", "loss_mask", "target_last_hidden_states"),
            layout.output_names,
        )
        self.assertIsNone(layout.aux_feature)

    def test_streaming_layout_exposes_final_hidden(self):
        providers = builtin_algorithm_registry().resolve("mtp").providers
        layout = providers.server_streaming_for("text").layout
        self.assertEqual("target_last_hidden_states", layout.last_hidden_feature)


if __name__ == "__main__":
    unittest.main()
