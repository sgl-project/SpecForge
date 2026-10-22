import unittest
from unittest.mock import patch

import torch

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import training_inputs, training_model


class DSpineOptimizationsTest(unittest.TestCase):

    def test_shared_losses_match_reference_values_and_hidden_gradients(self):
        cases = [("cpu", torch.float32)]
        if torch.cuda.is_available():
            cases += [("cuda", torch.float32), ("cuda", torch.bfloat16)]
        for device, dtype in cases:
            with self.subTest(device=device, dtype=dtype):
                model = training_model().to(device=device, dtype=dtype)
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
        for step in [100, 300]:
            with self.subTest(step=step):
                model = training_model(chunk_size=1)
                inputs = training_inputs()
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
                for got, want in zip(
                    actual["ratio_metrics"]["dspine/backbone_ce"],
                    metrics["ratio_metrics"]["dspine/backbone_ce"],
                ):
                    torch.testing.assert_close(got, want)

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
