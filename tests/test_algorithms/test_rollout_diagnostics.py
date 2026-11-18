"""Rollout diagnostics are free, chunk invariant and absent on non-logging steps."""

import unittest
import warnings
from dataclasses import replace
from itertools import islice
from unittest.mock import patch

import torch

from specforge.algorithms.objectives.losses import PositionWeighting
from specforge.runtime.contracts import TrainBatch
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from specforge.training.strategies.base import StepContext
from tests.test_algorithms.test_block_diagnostics import rng_states
from tests.test_algorithms.test_metric_goldens import (
    ROLLOUT_CASES,
    ROLLOUT_DEPTHS,
    STEP,
    TOTAL_STEPS,
    NullBackend,
    RolloutCase,
    evaluation_batches,
    prepared_rollout,
    rollout_strategy,
)


def run(runtime, tensors, *, detailed=True):
    # P-EAGLE samples its positions: every run draws the same ones.
    torch.manual_seed(48)
    return runtime(
        tensors, step=STEP, total_steps=TOTAL_STEPS, collect_detailed_metrics=detailed
    )


def ratio(result, name):
    numerator, denominator = result.ratios[name]
    return (numerator / denominator).item()


def step_metrics(strategy, batches, *, accumulation):
    """What each optimizer step reports over windows of ``accumulation`` batches."""
    runner = DataParallelRunner(strategy, NullBackend())
    ctx = StepContext(
        global_step=STEP, total_steps=TOTAL_STEPS, accumulation_steps=accumulation
    )
    stream = (
        TrainBatch(sample_ids=["a"], strategy=strategy.name, tensors=tensors)
        for tensors in batches
    )
    torch.manual_seed(11)
    return [
        run_window(runner, islice(stream, accumulation), ctx).materialize()
        for _ in range(len(batches) // accumulation)
    ]


class RolloutDiagnosticsTest(unittest.TestCase):
    def test_diagnostics_preserve_loss_gradients_and_rng(self):
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                runtime, tensors = prepared_rollout(case)
                device = next(runtime.parameters()).device
                results, gradients, generators = [], [], []
                for detailed in (False, True):
                    runtime.zero_grad(set_to_none=True)
                    result = run(runtime, tensors, detailed=detailed)
                    generators.append(rng_states(device))
                    result.loss.backward()
                    results.append(result)
                    gradients.append(
                        {
                            name: p.grad.clone()
                            for name, p in runtime.named_parameters()
                            if p.grad is not None
                        }
                    )
                # Identical, not close: the report runs after the loss graph
                # is built and may neither touch it nor draw random numbers.
                exact = dict(rtol=0, atol=0)
                torch.testing.assert_close(results[0].loss, results[1].loss, **exact)
                torch.testing.assert_close(gradients[0], gradients[1], **exact)
                torch.testing.assert_close(generators[0], generators[1], **exact)
                detailed = set(results[1].ratios) - set(results[0].ratios)
                self.assertTrue(detailed)
                self.assertTrue(
                    all(
                        not value.requires_grad
                        for pair in results[1].ratios.values()
                        for value in pair
                    )
                )
                # what both steps report is the same
                torch.testing.assert_close(
                    {name: results[1].ratios[name] for name in results[0].ratios},
                    results[0].ratios,
                    **exact,
                )

    def test_diagnostics_do_not_depend_on_how_the_loss_executes(self):
        for case in ROLLOUT_CASES:
            runtime, tensors = prepared_rollout(case)
            reference = run(runtime, tensors).ratios
            execution = runtime.execution
            for options in (
                dict(chunk_size=5, checkpoint=False),
                dict(trim=True),
                # the teacher projected in vocabulary tiles, never whole
                dict(teacher_chunk_size=24),
            ):
                with self.subTest(case=case.name, options=options):
                    runtime.execution = replace(execution, **options)
                    with warnings.catch_warnings():
                        # LK-lambda reads the whole set and declines to chunk
                        warnings.simplefilter("ignore", RuntimeWarning)
                        actual = run(runtime, tensors).ratios
                    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    def test_non_logging_step_does_not_project_a_second_time(self):
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                runtime, tensors = prepared_rollout(case)
                model = runtime.draft_model
                sets = ROLLOUT_DEPTHS if case.kind == "eagle3" else 1
                for detailed, projections in ((False, sets), (True, 2 * sets)):
                    with (
                        patch.object(model, "project", wraps=model.project) as project,
                        patch.object(model, "forward", wraps=model.forward) as forward,
                    ):
                        run(runtime, tensors, detailed=detailed)
                    self.assertEqual(project.call_count, projections)
                    # the diagnostics never run the draft again
                    self.assertEqual(forward.call_count, sets)

    def test_depth_losses_do_not_carry_the_depth_weight(self):
        runtime, tensors = prepared_rollout(RolloutCase("eagle3", "eagle3"))
        reported = {}
        for weighting in (
            PositionWeighting(kind="geometric", factor=0.8),
            PositionWeighting(kind="geometric", factor=0.3),
            PositionWeighting(kind="exponential", gamma=2.0),
            PositionWeighting(kind="uniform"),
        ):
            runtime.weighting = weighting
            result = run(runtime, tensors, detailed=False)
            losses = [ratio(result, f"ploss_{depth}") for depth in range(3)]
            weights = weighting(torch.ones(3), torch.arange(3))
            reported[weighting] = losses
            # the weighted depths add up to the objective, under its own name
            self.assertAlmostEqual(
                ratio(result, "kl_loss"), result.loss.item(), places=5
            )
            self.assertAlmostEqual(
                (torch.tensor(losses) * weights).sum().item(),
                result.loss.item(),
                places=5,
            )
            for depth in range(3):
                self.assertEqual(
                    result.ratios[f"kl_loss_{depth}"], result.ratios[f"ploss_{depth}"]
                )
        first, *others = reported.values()
        for losses in others:
            for actual, expected in zip(losses, first):
                self.assertAlmostEqual(actual, expected, places=5)
        # a depth the weighting zeroes has no loss to report
        runtime.weighting = PositionWeighting(kind="geometric", factor=0.0)
        result = run(runtime, tensors, detailed=False)
        self.assertAlmostEqual(ratio(result, "ploss_0"), first[0], places=5)
        for depth in (1, 2):
            numerator, denominator = result.ratios[f"ploss_{depth}"]
            self.assertEqual((numerator.item(), denominator.item()), (0, 0))

    def test_micro_batches_combine_as_the_wrappers_combined_ranks(self):
        """The trainer sums a window as it sums ranks, so a window shows the rule.

        The wrappers weighted a rank's depth means by its loss token count,
        rows times rollout length, and averaged its P-EAGLE counts.
        """
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                runtime, tensors = prepared_rollout(case)
                strategy = rollout_strategy(case, runtime, tensors)
                batches = evaluation_batches(tensors)
                separate = step_metrics(strategy, batches, accumulation=1)
                (combined,) = step_metrics(strategy, batches, accumulation=2)
                if case.kind == "eagle3":
                    weights = [batch["input_ids"].numel() for batch in batches]
                    names = [
                        name
                        for name in combined
                        if name not in ("loss", "acc") and not name.startswith("acc_")
                    ]
                    # depth losses under two names, acceptance under two, totals
                    self.assertEqual(len(names), 4 * ROLLOUT_DEPTHS + 3)
                else:
                    weights = [1, 1]
                    names = [name for name in combined if name.startswith("position_")]
                    self.assertEqual(len(names), 2 * ROLLOUT_DEPTHS)
                for name in names:
                    expected = sum(
                        weight * step[name] for weight, step in zip(weights, separate)
                    ) / sum(weights)
                    self.assertAlmostEqual(
                        combined[name], expected, delta=1e-6 + 1e-5 * expected, msg=name
                    )

    def test_chain_weighting_reports_no_depth_losses(self):
        """D-PACE weights whole chains: its loss is not a sum of depth means."""
        runtime, tensors = prepared_rollout(RolloutCase("eagle3", "eagle3"))
        runtime.weighting = PositionWeighting(kind="dpace", alpha=0.5)
        result = run(runtime, tensors)
        self.assertTrue(torch.isfinite(result.loss))
        self.assertFalse(
            [name for name in result.ratios if name.startswith(("ploss", "kl_loss"))]
        )
        self.assertIn("acceptance_rate_2", result.ratios)
        self.assertIn("acc_2", result.ratios)

    def test_only_the_objective_that_reports_them_measures_in_evaluation(self):
        from specforge.algorithms.objectives import objective_registrations

        measured = {
            name
            for name, entry in objective_registrations().items()
            if entry.training.evaluation_diagnostics
        }
        self.assertEqual(measured, {"eagle3"})


if __name__ == "__main__":
    unittest.main()
