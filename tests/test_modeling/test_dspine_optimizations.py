import unittest
from dataclasses import replace
from itertools import product
from unittest.mock import patch

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.dspine_fixtures import training_inputs, training_model
from tests.test_runtime._fixtures import build_strategy


class DSpineRefinementLossTest(unittest.TestCase):
    def setUp(self):
        self.model = training_model()
        # Identity projection makes candidate logits exact in both projection paths.
        with torch.no_grad():
            self.model.draft_model.lm_head.weight.zero_()
            self.model.draft_model.lm_head.weight[:16].copy_(torch.eye(16))
            self.model.target_head.weight.copy_(self.model.draft_model.lm_head.weight)

    def refinement_loss(self, refined, teacher, labels, weights):
        hidden = torch.zeros_like(refined)
        hidden[..., :3] = hidden.new_tensor([3, 2, 1])
        with patch.object(self.model.draft_model, "refine", return_value=refined):
            from specforge.algorithms.losses.dspine import dspine_chunk_terms

            return dspine_chunk_terms(
                self.model.draft_model,
                self.model.objective,
                self.model.target_head,
                hidden,
                refined,
                refined.new_ones(1, 3, 3, 8),
                labels,
                torch.zeros_like(labels),
                teacher,
                weights,
                weights > 0,
            )[2]

    def test_identical_candidate_logits_have_zero_l1_and_gradient(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for device, dtype in product(devices, (torch.float32, torch.bfloat16)):
            with self.subTest(device=device, dtype=dtype):
                self.model.to(device=device, dtype=dtype)
                self.model.objective = replace(
                    self.model.objective, ce_weight=0, candidate_mass_threshold=0
                )
                torch.manual_seed(42)
                teacher = torch.randn(1, 3, 16, device=device, dtype=dtype)
                teacher.requires_grad_()
                refined = teacher.detach().clone().requires_grad_()
                loss = self.refinement_loss(
                    refined,
                    teacher,
                    torch.zeros(1, 3, device=device, dtype=torch.long),
                    torch.ones(1, 3, device=device),
                )
                loss.backward()
                torch.testing.assert_close(loss, torch.zeros_like(loss), atol=0, rtol=0)
                torch.testing.assert_close(
                    refined.grad, torch.zeros_like(refined), atol=0, rtol=0
                )
                self.assertIsNone(teacher.grad)

    def test_missing_label_ce_and_masked_positions_have_zero_gradient(self):
        self.model.objective = replace(
            self.model.objective, ce_weight=1, candidate_mass_threshold=0.9
        )
        refined = torch.zeros(1, 3, 16)
        refined[..., :3] = torch.tensor([1.5, -0.5, 0.75])
        refined.requires_grad_()
        loss = self.refinement_loss(
            refined,
            torch.zeros_like(refined),
            torch.tensor([[2, 31, 1]]),
            torch.tensor([[0.5, 2, 0]]),
        )
        expected = -0.5 * refined[0, 0, :3].log_softmax(-1)[2]
        torch.testing.assert_close(loss, expected)
        actual_grad = torch.autograd.grad(loss, refined)[0]
        torch.testing.assert_close(
            actual_grad, torch.autograd.grad(expected, refined)[0]
        )
        torch.testing.assert_close(
            actual_grad[:, 1:], torch.zeros_like(actual_grad[:, 1:]), atol=0, rtol=0
        )

    def test_l1_uses_full_vocab_mass_with_strict_threshold_even_without_label(self):
        # Uniform teacher assigns exactly 3/32 mass to the three candidates.
        mass = 3 / 32
        for threshold in (mass - 0.01, mass, mass + 0.01):
            with self.subTest(threshold=threshold):
                self.model.objective = replace(
                    self.model.objective,
                    ce_weight=0,
                    candidate_mass_threshold=threshold,
                )
                refined = torch.zeros(1, 3, 16)
                refined[..., :3] = torch.tensor([1.5, -0.5, 0.75])
                refined.requires_grad_()
                weights = torch.tensor([[0.5, 2, 0]])
                loss = self.refinement_loss(
                    refined,
                    torch.zeros_like(refined),
                    torch.full((1, 3), 31, dtype=torch.long),
                    weights,
                )
                expected = (
                    (refined[..., :3].softmax(-1) - 1 / 3).abs().sum(-1) * weights
                ).sum() * (mass > threshold)
                torch.testing.assert_close(loss, expected)
                actual_grad = torch.autograd.grad(loss, refined)[0]
                torch.testing.assert_close(
                    actual_grad, torch.autograd.grad(expected, refined)[0]
                )
                if threshold >= mass:
                    torch.testing.assert_close(
                        actual_grad, torch.zeros_like(actual_grad), atol=0, rtol=0
                    )


class DSpineOptimizationsTest(unittest.TestCase):

    def test_ce_prepass_matches_full_without_teacher_or_refinement(self):
        model = training_model(chunk_size=1)
        inputs = training_inputs()
        torch.manual_seed(39)
        with torch.no_grad():
            expected = model(inputs, step=100, total_steps=600)
        torch.manual_seed(39)
        with (
            torch.no_grad(),
            patch.object(
                model.draft_model, "refine", side_effect=AssertionError("refinement")
            ),
            patch.object(
                model.target_head, "forward", side_effect=AssertionError("teacher")
            ),
        ):
            actual = model(inputs, step=100, total_steps=600, ce_only=True)
        for key in ("accuracy", "dspine/backbone_ce"):
            torch.testing.assert_close(actual.ratios[key], expected.ratios[key])

    def test_alignment_ce_reduced_once_per_window(self):
        for accumulation_steps in [2, 4]:
            with self.subTest(accumulation_steps=accumulation_steps):
                strategy = build_strategy(training_model(chunk_size=1))
                batch = TrainBatch(["0", "1"], "dspine", training_inputs())
                ctx = StepContext(
                    global_step=100,
                    total_steps=600,
                    accumulation_steps=accumulation_steps,
                )
                with (
                    patch(
                        "specforge.training.strategies.canonical.dist.is_initialized",
                        return_value=True,
                    ),
                    patch(
                        "specforge.training.strategies.canonical.dist.all_reduce"
                    ) as reduce,
                ):
                    for window in range(2):
                        prepared = [
                            strategy.forward_loss(batch, ctx)
                            for _ in range(accumulation_steps)
                        ]
                        self.assertEqual(reduce.call_count, window)
                        ce_terms = [
                            output.ratio_metrics["dspine/backbone_ce"]
                            for output in prepared
                        ]
                        ratios = {"dspine/backbone_ce": tuple(map(sum, zip(*ce_terms)))}
                        for output in prepared:
                            output.replay_loss(ratios)
                        self.assertEqual(reduce.call_count, window + 1)


if __name__ == "__main__":
    unittest.main()
