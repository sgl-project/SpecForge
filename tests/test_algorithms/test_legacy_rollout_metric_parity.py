"""Compare what the canonical and legacy rollout routes report: EAGLE3.

Both routes run the same fitted weights and batch through their trainer
strategy, ``TrainerCore`` and the trainer's evaluation pass, so the compared
dictionaries are what the tracker receives. The canonical side is built exactly
as in ``test_metric_goldens``, whose recorded values therefore carry this
comparison past the legacy wrappers.

``python -m tests.test_algorithms.test_legacy_rollout_metric_parity --print``
prints both routes' keys and values for a logging step, a non-logging step and
an evaluation pass.
"""

import importlib.util
import sys
import unittest
from functools import cache
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import torch

from specforge.training.strategies.base import Eagle3TrainStrategy
from tests.test_algorithms.test_metric_goldens import (
    ROLLOUT_CASES,
    ROLLOUT_DEPTHS,
    evaluated_metrics,
    evaluation_batches,
    flattened,
    logged_metrics,
    prepared_rollout,
    rollout_strategy,
)
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope
from tests.test_modeling.test_canonical_eagle3 import legacy_model

# Keys only the legacy route logs, by objective, each with the reason the
# canonical route leaves it out. Any other difference between the key sets fails.
LEGACY_ONLY = {}


# Evaluation metrics whose combination across batches differs, by objective.
# The canonical route had these before the depth metrics were restored. One
# batch, or batches of one size and supervision, agree with the wrappers.
BATCH_WEIGHTING = {
    "eagle3": {
        "eval/avg_loss": "the mean of the batch losses; the wrapper weighted a "
        "batch by its loss token count",
    },
}


def logging_only(kind):
    """Keys of a logging step that cost a second projection of head and teacher."""
    depths = range(ROLLOUT_DEPTHS)
    names = ("acceptance_rate", "expected_acceptance")
    return {*names, *(f"{name}_{depth}" for name in names for depth in depths)}


def soft_cross_entropy(logits, target, mask, acceptance_target=None):
    """The wrappers' GPU-only loss kernel, as the formula it implements."""
    loss = (
        -(logits.float().log_softmax(-1) * target).sum(-1) * mask.squeeze(-1)
    ).mean()
    if acceptance_target is None:
        return loss
    return loss, torch.minimum(logits.float().softmax(-1), acceptance_target).sum(-1)


@cache
def legacy_module(kind):
    """A private copy of the wrapper's module that runs without its GPU kernels."""
    path = (
        Path(__file__).resolve().parents[2] / "specforge/algorithms" / kind / "model.py"
    )
    spec = importlib.util.spec_from_file_location(f"reference_{kind}_metrics", path)
    module = importlib.util.module_from_spec(spec)
    kernel = SimpleNamespace(LogSoftmaxLoss=SimpleNamespace(apply=soft_cross_entropy))
    with patch.dict("sys.modules", {"specforge.losses.soft_cross_entropy": kernel}):
        spec.loader.exec_module(module)
    return module


def legacy_strategy(case, runtime):
    """The wrapper and its strategy on the weights the canonical draft reached."""
    wrapper = legacy_module("eagle3").OnlineEagle3Model(
        legacy_model(runtime.draft_model),
        length=ROLLOUT_DEPTHS,
        lk_loss_type=case.token.removeprefix("lk_") if case.token else None,
    )
    return Eagle3TrainStrategy(wrapper)


def legacy_tensors(runtime, tensors):
    """The same batch in the wrappers' contract: shifted, with teacher logits."""

    def shift(value):
        return torch.cat((value[:, 1:], torch.zeros_like(value[:, :1])), 1)

    with torch.no_grad():
        logits = runtime.target_head(tensors["target_last_hidden_states"])
    return dict(
        input_ids=shift(tensors["input_ids"]),
        attention_mask=tensors["attention_mask"],
        loss_mask=shift(tensors["loss_mask"])[..., None],
        hidden_state=tensors["hidden_states"],
        target=shift(logits),
    )


def both_routes(case, step):
    """Canonical and legacy metrics of one case for a kind of step.

    "evaluation" runs two uneven batches and "evaluation of one batch" the
    first of them.
    """
    with legacy_fp32_rope():
        runtime, tensors = prepared_rollout(case)
        batches = evaluation_batches(tensors) if step == "evaluation" else [tensors]
        routes = (
            (rollout_strategy(case, runtime, tensors), batches),
            (
                legacy_strategy(case, runtime),
                [legacy_tensors(runtime, batch) for batch in batches],
            ),
        )
        reported = []
        for strategy, batches in routes:
            torch.manual_seed(11)
            if step.startswith("evaluation"):
                reported.append(flattened(evaluated_metrics(strategy, batches)))
            else:
                reported.append(
                    logged_metrics(strategy, batches[0], detailed=step == "logging")
                )
    return reported


class LegacyRolloutMetricParityTest(unittest.TestCase):
    def assert_values(self, canonical, legacy, keys):
        for key in sorted(keys):
            # Both routes run the same fp32 weights through different modules,
            # so sums are reordered (1e-7 relative here). A wrong formula moves
            # a value by one position in a few dozen.
            self.assertAlmostEqual(
                canonical[key],
                legacy[key],
                delta=2e-6 + 2e-5 * abs(legacy[key]),
                msg=key,
            )

    def test_logging_step_reports_the_legacy_keys_and_values(self):
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                canonical, legacy = both_routes(case, "logging")
                legacy_only = LEGACY_ONLY.get(case.kind, {})
                self.assertEqual(
                    sorted(set(legacy) - set(canonical)), sorted(legacy_only)
                )
                self.assertEqual(sorted(set(canonical) - set(legacy)), [])
                self.assert_values(canonical, legacy, set(legacy) - set(legacy_only))

    def test_non_logging_step_reports_what_the_loss_computed(self):
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                canonical, legacy = both_routes(case, "non-logging")
                expected = (
                    set(legacy)
                    - set(LEGACY_ONLY.get(case.kind, {}))
                    - logging_only(case.kind)
                )
                self.assertEqual(sorted(canonical), sorted(expected))
                self.assert_values(canonical, legacy, expected)

    def test_evaluation_reports_the_legacy_keys_and_values(self):
        for case in ROLLOUT_CASES:
            with self.subTest(case=case.name):
                canonical, legacy = both_routes(case, "evaluation")
                self.assertEqual(sorted(canonical), sorted(legacy))
                weighted = BATCH_WEIGHTING[case.kind]
                self.assert_values(canonical, legacy, set(legacy) - set(weighted))
                # the listed keys do differ here, and only across batches
                for key in weighted:
                    self.assertNotAlmostEqual(
                        canonical[key], legacy[key], delta=1e-4, msg=key
                    )
                canonical, legacy = both_routes(case, "evaluation of one batch")
                self.assertEqual(sorted(canonical), sorted(legacy))
                self.assert_values(canonical, legacy, legacy)


def print_comparison():
    for case in ROLLOUT_CASES:
        for step in (
            "logging step",
            "non-logging step",
            "evaluation",
            "evaluation of one batch",
        ):
            canonical, legacy = both_routes(case, step.removesuffix(" step"))
            print(
                f"\n{case.name}, {step}: legacy {len(legacy)} keys, "
                f"canonical {len(canonical)} keys"
            )
            print("  legacy only:", sorted(set(legacy) - set(canonical)))
            print("  canonical only:", sorted(set(canonical) - set(legacy)))
            for key in sorted(set(legacy) | set(canonical)):
                values = [
                    f"{route[key]:.8f}" if key in route else "-"
                    for route in (canonical, legacy)
                ]
                print(
                    f"  {key:30s} canonical {values[0]:>12s}  legacy {values[1]:>12s}"
                )


if __name__ == "__main__":
    if "--print" in sys.argv:
        print_comparison()
    else:
        unittest.main()
