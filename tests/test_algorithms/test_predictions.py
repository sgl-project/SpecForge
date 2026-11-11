"""Deferred projection preserves complete recurrence units and loss requirements."""

import unittest
from unittest.mock import patch

import torch
from torch import nn

from specforge.algorithms.objectives import build_objective_model, register_loss
from specforge.algorithms.objectives.execution import evaluate_predictions
from specforge.algorithms.objectives.losses import _LOSSES, TokenLoss
from specforge.modeling import build_draft_model
from tests.test_algorithms.test_objectives import configurations, training_batch


class PredictionsTest(unittest.TestCase):
    def test_projection_chunks_match_full_outputs_and_gradients(self):
        for config in configurations():
            with self.subTest(architecture=config.architecture):
                torch.manual_seed(7)
                model = build_draft_model(config)
                if hasattr(model, "vocabulary"):
                    model.vocabulary.load_mapping(
                        t2d=torch.arange(config.vocab_size) < config.draft_vocab_size,
                        d2t=torch.zeros(config.draft_vocab_size, dtype=torch.long),
                    )
                for name in config.target_bindings:
                    model.get_submodule(name).load_target_weight(
                        torch.randn(config.vocab_size, config.hidden_size)
                    )
                if hasattr(model, "initialize_transfer_space"):
                    model.initialize_transfer_space()
                options = {"num_anchors": 2} if config.spec_config.protocol else {}
                runtime = build_objective_model(model, options=options)
                batch = {k: v.detach() for k, v in training_batch(config).items()}
                with patch.object(
                    model.lm_head, "forward", wraps=model.lm_head.forward
                ) as head:
                    prepared = runtime.objective(
                        runtime,
                        batch,
                        step=1,
                        total_steps=10,
                        alignment_ce=torch.tensor(0.5),
                    )
                    prediction = next(iter(prepared.predictions))
                    self.assertEqual(head.call_count, 0)
                self.assertGreater(prediction.hidden.shape[0], 1)
                with patch.object(
                    model.backbone,
                    "run_next_layer",
                    side_effect=AssertionError("projection reran the backbone"),
                ):
                    full = prediction.project()
                    chunks = [
                        prediction.select(part).project()
                        for part in (slice(0, 1), slice(1, None))
                    ]
                for key, value in full.items():
                    if value is not None:
                        torch.testing.assert_close(
                            torch.cat([chunk[key] for chunk in chunks]), value
                        )
                parameters = [p for p in model.parameters() if p.requires_grad]
                full_grad = torch.autograd.grad(
                    sum(v.square().sum() for v in full.values() if v is not None),
                    parameters,
                    allow_unused=True,
                    retain_graph=True,
                )
                chunk_grad = torch.autograd.grad(
                    sum(
                        v.square().sum()
                        for chunk in chunks
                        for v in chunk.values()
                        if v is not None
                    ),
                    parameters,
                    allow_unused=True,
                )
                for expected, actual in zip(full_grad, chunk_grad):
                    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-4)

    def test_teacher_projection_is_lazy(self):
        config = configurations()[0]
        model = build_draft_model(config)
        runtime = build_objective_model(
            model,
            options={"num_anchors": 2},
            target_head=nn.Linear(32, config.vocab_size),
        )
        batch = training_batch(config)
        with patch.object(
            runtime.target_head, "forward", wraps=runtime.target_head.forward
        ) as teacher:
            prediction = next(iter(runtime.objective(runtime, batch).predictions))
            self.assertEqual(teacher.call_count, 0)
            evaluate_predictions(prediction, TokenLoss(kind="ce"), runtime.weighting)
            self.assertEqual(teacher.call_count, 0)
            evaluate_predictions(prediction, TokenLoss(kind="kl"), runtime.weighting)
            self.assertEqual(teacher.call_count, 1)

    def test_capability_mismatch_fails_before_freezing(self):
        with patch.dict(_LOSSES):

            @register_loss("unsupported", requires=("logits", "acceptance_chain"))
            def unsupported(logits, labels, teacher):
                raise AssertionError("incompatible loss was executed")

            model = build_draft_model(configurations()[0])
            with self.assertRaisesRegex(ValueError, "acceptance_chain"):
                build_objective_model(
                    model, loss={"kind": "unsupported"}, frozen=("backbone",)
                )
            self.assertTrue(all(p.requires_grad for p in model.backbone.parameters()))
            runtime = build_objective_model(model)
            prediction = next(
                iter(
                    runtime.objective(
                        runtime, training_batch(model.config.draft)
                    ).predictions
                )
            )
            with self.assertRaisesRegex(ValueError, "teacher_logits"):
                evaluate_predictions(
                    prediction, TokenLoss(kind="kl"), runtime.weighting
                )
