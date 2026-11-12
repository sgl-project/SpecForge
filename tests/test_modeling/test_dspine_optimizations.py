import unittest
from dataclasses import replace
from itertools import product
from unittest.mock import patch

import torch
from torch.utils.checkpoint import checkpoint as activation_checkpoint

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.core import fused_loss
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import training_inputs, training_model


class DSpineRefinementLossTest(unittest.TestCase):
    def setUp(self):
        self.model = training_model()
        # Identity projection makes candidate logits exact in both projection paths.
        with torch.no_grad():
            self.model.lm_head.weight.zero_()
            self.model.lm_head.weight[:16].copy_(torch.eye(16))

    def refinement_loss(self, refined, teacher, labels, weights):
        hidden = torch.zeros_like(refined)
        hidden[..., :3] = hidden.new_tensor([3, 2, 1])
        with patch.object(self.model.draft_model, "refine", return_value=refined):
            return self.model._objective_terms(
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
                self.model.dspine_config = replace(
                    self.model.dspine_config, ce_weight=0, candidate_mass_threshold=0
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
        self.model.dspine_config = replace(
            self.model.dspine_config, ce_weight=1, candidate_mass_threshold=0.9
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
                self.model.dspine_config = replace(
                    self.model.dspine_config,
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

    def test_shared_losses_match_reference_values_and_hidden_gradients(self):
        cases = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            cases += [("cuda", torch.float32), ("cuda", torch.bfloat16)]
        for (device, dtype), fused in product(cases, (False, True)):
            with self.subTest(device=device, dtype=dtype, fused=fused):
                model = training_model().to(device=device, dtype=dtype)
                model.fused_logits_loss = fused
                hidden = torch.randn(
                    2, 3, 16, device=device, dtype=dtype, requires_grad=True
                )
                labels = torch.randint(0, 32, (2, 3), device=device)
                supervised = torch.tensor(
                    [[True, False, True], [False, True, True]], device=device
                )
                weights = torch.rand(2, 3, device=device) * supervised
                for identical_teacher, teacher_hidden in (
                    (False, None),
                    (False, torch.randn_like(hidden)),
                    (True, hidden.detach()),
                ):
                    logits = model.lm_head(hidden).float()
                    log_p = logits.log_softmax(-1)
                    expected = -log_p.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
                    if teacher_hidden is None:
                        loss, correct = model._ce_terms(
                            hidden, labels, weights, supervised
                        )
                    else:
                        teacher = model.lm_head(teacher_hidden).float().softmax(-1)
                        if not identical_teacher:
                            expected = expected + (log_p.exp() - teacher).abs().sum(-1)
                        terms = model._objective_terms(
                            hidden,
                            hidden,
                            torch.randn(2, 3, 3, 8, device=device, dtype=dtype),
                            labels,
                            labels,
                            teacher_hidden,
                            weights,
                            supervised,
                        )
                        loss, correct = terms[0] + terms[1], terms[4]
                        if identical_teacher:
                            # Equal distributions use the zero subgradient of abs.
                            torch.testing.assert_close(
                                terms[1], torch.zeros_like(terms[1]), atol=0, rtol=0
                            )
                    expected = (expected * weights).sum()
                    torch.testing.assert_close(loss, expected)
                    torch.testing.assert_close(
                        correct,
                        ((logits.argmax(-1) == labels) * supervised).sum().float(),
                    )
                    torch.testing.assert_close(
                        torch.autograd.grad(loss, hidden)[0],
                        torch.autograd.grad(expected, hidden)[0],
                    )

    def test_ce_prepass_matches_full_without_teacher_or_refinement(self):
        devices = ["cpu", "cuda"] if torch.cuda.is_available() else ["cpu"]
        for step, device, fused in product((100, 300), devices, (False, True)):
            with self.subTest(step=step, device=device, fused=fused):
                model = training_model(chunk_size=1).to(device)
                model.fused_logits_loss = fused
                inputs = {
                    name: value.to(device) for name, value in training_inputs().items()
                }
                torch.manual_seed(39)
                with torch.no_grad():
                    (_, accuracy, metrics) = model(
                        **inputs, global_step=step, total_steps=600
                    )
                torch.manual_seed(39)
                with (
                    torch.no_grad(),
                    patch.object(
                        model,
                        "_objective_terms",
                        side_effect=AssertionError("full objective"),
                    ),
                    patch.object(
                        model.draft_model,
                        "refine",
                        side_effect=AssertionError("refinement"),
                    ),
                    patch(
                        "specforge.algorithms.dspine.model.dist.all_reduce",
                        side_effect=AssertionError("collective"),
                    ),
                ):
                    (_, actual_accuracy, actual) = model(
                        **inputs, global_step=step, total_steps=600, ce_only=True
                    )
                torch.testing.assert_close(actual_accuracy, accuracy)
                torch.testing.assert_close(
                    actual["ratio_metrics"]["acc"], metrics["ratio_metrics"]["acc"]
                )
                correct, count = actual["ratio_metrics"]["acc"]
                torch.testing.assert_close(count, actual["accuracy_denom"])
                torch.testing.assert_close(correct / count.clamp_min(1), accuracy)
                for got, want in zip(
                    actual["ratio_metrics"]["dspine/backbone_ce"],
                    metrics["ratio_metrics"]["dspine/backbone_ce"],
                ):
                    torch.testing.assert_close(got, want)

    @unittest.skipUnless(torch.cuda.is_available(), "Fused loss requires CUDA")
    def test_fusion_and_checkpointing_are_independent(self):
        model = training_model().cuda()
        inputs = {name: value.cuda() for name, value in training_inputs().items()}

        def forward_backward():
            model.zero_grad(set_to_none=True)
            torch.manual_seed(39)
            loss, _, _ = model(**inputs, global_step=100, total_steps=600)
            loss.backward()
            return loss.detach(), {
                name: p.grad.clone()
                for name, p in model.named_parameters()
                if p.requires_grad and p.grad is not None
            }

        model.fused_logits_loss = model.objective_checkpointing = False
        reference = forward_backward()
        for fused, checkpoint, chunk in product((False, True), (False, True), (0, 1)):
            with self.subTest(fused=fused, checkpoint=checkpoint, chunk=chunk):
                model.fused_logits_loss = fused
                model.objective_checkpointing = checkpoint
                model.objective_chunk_blocks = chunk
                with (
                    patch(
                        "torch.utils.checkpoint.checkpoint", wraps=activation_checkpoint
                    ) as checkpoint_call,
                    patch(
                        "specforge.core.fused_loss.fused_ce_l1_loss",
                        wraps=fused_loss.fused_ce_l1_loss,
                    ) as fused_call,
                ):
                    actual = forward_backward()
                self.assertEqual(fused_call.called, fused)
                self.assertEqual(checkpoint_call.called, checkpoint)
                torch.testing.assert_close(actual, reference, rtol=1e-4, atol=1e-5)

    def test_ce_prepass_rejects_autograd(self):
        with self.assertRaisesRegex(ValueError, "gradients to be disabled"):
            training_model()(**training_inputs(), ce_only=True)

    def test_alignment_ce_reduced_once_per_window(self):
        for accumulation_steps in [2, 4]:
            with self.subTest(accumulation_steps=accumulation_steps):
                strategy = DSpineTrainStrategy(training_model(chunk_size=1))
                batch = TrainBatch(["0", "1"], "dspine", training_inputs())
                ctx = StepContext(
                    global_step=100,
                    total_steps=600,
                    accumulation_steps=accumulation_steps,
                )
                with (
                    patch(
                        "specforge.algorithms.dspine.strategy.dist.is_initialized",
                        return_value=True,
                    ),
                    patch(
                        "specforge.algorithms.dspine.strategy.dist.all_reduce"
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
