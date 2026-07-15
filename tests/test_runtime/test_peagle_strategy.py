# coding=utf-8
"""CPU contract tests for the P-EAGLE DataFlow strategy."""

from __future__ import annotations

import unittest
from types import SimpleNamespace
from typing import get_args

import torch
import torch.nn as nn

from specforge.algorithms.builtin import builtin_algorithm_registry
from specforge.runtime.contracts import DraftStrategyName, TrainBatch
from specforge.training.strategies.base import PEagleTrainStrategy

REGISTRY = builtin_algorithm_registry()
EAGLE3 = REGISTRY.resolve("eagle3")
PEAGLE = REGISTRY.resolve("peagle")


class _FakeDraftModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embed_tokens = nn.Embedding(16, 4)
        self.mask_hidden = nn.Parameter(torch.ones(1, 4))
        self.scale = nn.Parameter(torch.tensor(2.0))


class _FakePEagleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.draft_model = _FakeDraftModel()
        self.forward_kwargs = None

    def forward(self, **kwargs):
        self.forward_kwargs = kwargs
        loss = self.draft_model.scale * kwargs["target"].float().mean()
        metrics = {
            "loss_sum": loss.detach(),
            "loss_total": torch.tensor(1.0),
            "position_0_acc_sum": torch.tensor(2.0),
            "position_0_acc_total": torch.tensor(3.0),
            "full_acc_sum": torch.tensor(3.0),
            "full_acc_total": torch.tensor(4.0),
        }
        return loss, metrics


class _FakeTargetHead(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.projection = nn.Linear(2, 5, bias=False)
        self.preprocess_calls = 0

    def preprocess(self, input_ids, target, loss_mask):
        self.preprocess_calls += 1
        shifted_ids = torch.cat(
            [input_ids[:, 1:], torch.zeros_like(input_ids[:, :1])], dim=1
        )
        shifted_target = torch.cat(
            [target[:, 1:], torch.zeros_like(target[:, :1])], dim=1
        )
        return shifted_ids, shifted_target, loss_mask.unsqueeze(-1)

    def forward(self, hidden_states):
        return self.projection(hidden_states)


def _batch(*, target_repr: str = "logits", include_lengths: bool = False) -> TrainBatch:
    tensors = {
        "input_ids": torch.tensor([[1, 2, 3, 0]]),
        "attention_mask": torch.tensor([[1, 1, 1, 0]]),
        "loss_mask": torch.tensor([[1, 1, 1, 0]]),
        "hidden_state": torch.randn(1, 4, 6),
        "target": torch.randn(1, 4, 5 if target_repr == "logits" else 2),
    }
    if include_lengths:
        tensors["lengths"] = torch.tensor([2])
    return TrainBatch(
        sample_ids=["sample-0"],
        strategy="peagle",
        tensors=tensors,
        metadata={"target_repr": target_repr},
    )


class TestPEagleStrategy(unittest.TestCase):
    def test_strategy_name_is_in_runtime_contract_and_builtin_registration(self):
        self.assertIn("peagle", get_args(DraftStrategyName))
        contract = PEAGLE.spec.feature_contract("streaming", "text")
        self.assertEqual(
            contract.required_tensors,
            frozenset(PEagleTrainStrategy.required_features),
        )
        self.assertTrue(PEAGLE.spec.supports_online)
        self.assertTrue(PEAGLE.providers.step.uses_external_target_head)
        self.assertEqual((), PEAGLE.providers.offline)

    def test_provider_reuses_eagle3_online_capture(self):
        peagle = PEAGLE.providers.server_streaming_for("text")
        eagle3 = EAGLE3.providers.server_streaming_for("text")
        self.assertEqual(peagle.capture_method, eagle3.capture_method)
        self.assertEqual(peagle.layout, eagle3.layout)
        self.assertIs(peagle.build_collator, eagle3.build_collator)

    def test_online_logits_map_hidden_state_and_derive_length(self):
        model = _FakePEagleModel()
        head = _FakeTargetHead()
        strategy = PEagleTrainStrategy(model, target_head=head)
        batch = _batch(target_repr="logits")

        output = strategy.forward_loss(batch)

        self.assertEqual(head.preprocess_calls, 0)
        self.assertEqual(output.loss.ndim, 0)
        self.assertIs(
            model.forward_kwargs["hidden_states"], batch.tensors["hidden_state"]
        )
        self.assertEqual(model.forward_kwargs["lengths"].tolist(), [3])
        self.assertAlmostEqual(float(output.metrics["accuracy"]), 0.75)
        self.assertFalse(output.metrics["loss_sum"].requires_grad)

    def test_offline_target_hidden_state_is_shifted_and_projected(self):
        model = _FakePEagleModel()
        head = _FakeTargetHead()
        strategy = PEagleTrainStrategy(model, target_head=head)
        batch = _batch(target_repr="hidden_state", include_lengths=True)
        expected_target = head(
            torch.cat(
                [
                    batch.tensors["target"][:, 1:],
                    torch.zeros_like(batch.tensors["target"][:, :1]),
                ],
                dim=1,
            )
        )

        strategy.forward_loss(batch)

        self.assertEqual(head.preprocess_calls, 1)
        self.assertEqual(model.forward_kwargs["input_ids"].tolist(), [[2, 3, 0, 0]])
        torch.testing.assert_close(model.forward_kwargs["target"], expected_target)
        self.assertEqual(model.forward_kwargs["loss_mask"].shape, (1, 4, 1))
        self.assertEqual(model.forward_kwargs["lengths"].tolist(), [2])

    def test_hidden_state_target_requires_target_head(self):
        strategy = PEagleTrainStrategy(_FakePEagleModel())
        with self.assertRaisesRegex(ValueError, "requires a target_head"):
            strategy.forward_loss(_batch(target_repr="hidden_state"))

    def test_checkpoint_keeps_embedding_mask_hidden_and_only_draft_keys(self):
        model = _FakePEagleModel()
        strategy = PEagleTrainStrategy(model)
        state = {
            "draft_model.embed_tokens.weight": torch.randn(16, 4),
            "draft_model.mask_hidden": torch.randn(1, 4),
            "draft_model.scale": torch.randn(()),
            "target_head.weight": torch.randn(5, 2),
        }

        filtered = strategy.checkpoint_state_filter(state)

        self.assertEqual(
            set(filtered),
            {"embed_tokens.weight", "mask_hidden", "scale"},
        )
        self.assertTrue(model.draft_model.embed_tokens.weight.requires_grad)

    def test_resume_contract_records_resolved_model_and_objective_semantics(self):
        cfg = SimpleNamespace(
            training=SimpleNamespace(
                strategy="peagle",
                attention_backend="flex_attention",
            )
        )
        draft_model = SimpleNamespace(
            layers=[object(), object()],
            norm_before_residual=True,
        )
        model = SimpleNamespace(
            num_depths=5,
            down_sample_ratio=0.7,
            down_sample_ratio_min=0.3,
            mask_token_id=0,
        )

        runtime = PEAGLE.providers.step.bind_runtime(cfg, draft_model, model)

        self.assertEqual(
            dict(runtime.resume_contract),
            {
                "peagle_num_draft_layers": 2,
                "peagle_norm_before_residual": True,
                "peagle_num_depths": 5,
                "peagle_down_sample_ratio": 0.7,
                "peagle_down_sample_ratio_min": 0.3,
                "peagle_mask_token_id": 0,
                "peagle_attention_backend": "flex_attention",
                "specforge_step_options": (),
            },
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
