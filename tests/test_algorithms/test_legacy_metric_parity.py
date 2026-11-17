"""Compare what a logging step reports on the canonical and legacy block routes.

Both routes run the same weights and batch through their trainer strategy and
``TrainerCore``, so the compared dictionaries are what the tracker receives.
The canonical side is built exactly as in ``test_metric_goldens``, whose
recorded values therefore carry this comparison past the legacy wrappers.
"""

import unittest
from unittest.mock import patch

import torch

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.training.strategies.base import (
    DFlashTrainStrategy,
    DominoTrainStrategy,
    DSparkTrainStrategy,
)
from tests.test_algorithms.test_metric_goldens import (
    CASES,
    canonical_metrics,
    logged_metrics,
    loss_settings,
    prepared,
)
from tests.test_algorithms.test_objective_legacy_parity import block_pair
from tests.test_modeling.legacy_rope_reference import legacy_fp32_rope

LEGACY_STRATEGIES = {
    "dflash": DFlashTrainStrategy,
    "dflash2": DFlashTrainStrategy,
    "domino": DominoTrainStrategy,
    "dspark": DSparkTrainStrategy,
    "dspine": DSpineTrainStrategy,
}

# Keys only the legacy route logs, by case, each with the reason the canonical
# route leaves it out. Any other difference between the key sets fails.
LEGACY_ONLY = {
    "dspark without a confidence head": {
        "confidence_loss": "a constant zero: there is no head to supervise",
        "confidence_abs_error": "a constant zero: there is no head to measure",
    },
}


def configure_legacy(case, old):
    """The wrapper's spelling of the settings ``test_metric_goldens.configure`` makes."""
    old.num_anchors = 5
    if case.kind in ("dflash", "dflash2"):
        old.teacher_metrics = case.teacher_metrics
    if case.kind == "dflash":
        old.metric_top_k = 4
    if case.kind == "dflash2":
        old.selector_warmup_ratio = 0.2
        old.selector_ramp_ratio = 0.5
    if case.dpace:
        old.loss_type, old.dpace_alpha = "dpace", 0.5
    else:
        old.loss_decay_gamma = 4.0
    if case.lk_lambda:
        old.lk_loss_type = "lambda"
    if not case.selector:
        old.selector_loss_alpha, old._selector_objective_enabled = 0.0, False
    if not case.confidence:
        old.draft_model.confidence_head = None


def both_routes(case):
    """Logged metrics of the canonical and the legacy route for one case."""
    torch.manual_seed(7)
    new, old, names = block_pair(
        case.kind, shifted=case.shifted, loss=loss_settings(case)
    )
    configure_legacy(case, old)
    tensors = prepared(case, new)
    # The legacy wrapper continues from the weights the canonical draft reached.
    legacy = dict(old.named_parameters())
    with torch.no_grad():
        for name, parameter in new.draft_model.named_parameters():
            if names.get(name, "").startswith("draft_model."):
                legacy[names[name]].copy_(parameter)
    canonical = canonical_metrics(case, new, tensors)
    torch.manual_seed(11)
    with patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}):
        return canonical, logged_metrics(LEGACY_STRATEGIES[case.kind](old), tensors)


class LegacyMetricParityTest(unittest.TestCase):
    @legacy_fp32_rope()
    def test_logging_step_reports_the_legacy_keys_and_values(self):
        for case in CASES:
            with self.subTest(case=case.name):
                canonical, legacy = both_routes(case)
                legacy_only = LEGACY_ONLY.get(case.name, {})
                self.assertEqual(
                    sorted(set(legacy) - set(canonical)), sorted(legacy_only)
                )
                self.assertEqual(sorted(set(canonical) - set(legacy)), [])
                for key in sorted(set(legacy) - set(legacy_only)):
                    # Both routes run the same fp32 weights through different
                    # modules, so sums are reordered (1e-7 relative here), and
                    # the Domino wrapper adds 1e-6 to its denominators. A wrong
                    # formula moves a value by one token in a few dozen.
                    self.assertAlmostEqual(
                        canonical[key],
                        legacy[key],
                        delta=2e-6 + 2e-5 * abs(legacy[key]),
                        msg=key,
                    )


if __name__ == "__main__":
    unittest.main()
