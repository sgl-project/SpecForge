"""Recorded diagnostics of the block and rollout objectives on a logging step.

``tests/fixtures/metric_goldens.json`` holds every metric a logging step hands
to the tracker, for each case below. The values were recorded while the legacy
training wrappers still existed and agreed with them key by key, so they keep
that reference now that the wrappers are gone. A changed value means a metric
changed: regenerate with
``SPECFORGE_UPDATE_GOLDENS=1`` and say in the commit what moved and why.
"""

import json
import os
import unittest
from dataclasses import dataclass, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.algorithms.objectives import build_objective_model
from specforge.algorithms.objectives.losses import PositionWeighting
from specforge.modeling import build_draft_model
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import TrainingBackend
from specforge.training.controller import TrainerController
from specforge.training.frame import run_window
from specforge.training.runner import DataParallelRunner
from specforge.training.strategies.base import StepContext
from specforge.training.strategies.canonical import CanonicalTrainStrategy
from tests.test_algorithms.test_objective_goldens import block_runtime
from tests.test_algorithms.test_objectives import (
    eagle_config,
    peagle_config,
    training_batch,
)
from tests.test_modeling.test_canonical_eagle3 import install_mapping

GOLDENS = Path(__file__).resolve().parents[1] / "fixtures/metric_goldens.json"
# The target tables stay at their initial values while the draft is fitted.
TARGET_TABLES = ("embed_tokens.weight", "lm_head.weight")

# Step 3 of 11 is inside every schedule: Domino's base weight is 0.4, DSpine
# still replaces blocks, and the DFlash2 selector below is 40% into its ramp.
STEP, TOTAL_STEPS = 3, 11


@dataclass(frozen=True)
class Case:
    name: str
    kind: str
    shifted: bool = False
    teacher_metrics: bool = True
    dpace: bool = False
    lk_lambda: bool = False
    selector: bool = True
    confidence: bool = True
    # Relative tolerance against the recorded values on another platform.
    tolerance: float = 1e-4


CASES = (
    Case("dflash", "dflash"),
    Case("dflash dpace", "dflash", dpace=True),
    Case("dflash2", "dflash2"),
    Case(
        "dflash2 lk_lambda dpace",
        "dflash2",
        teacher_metrics=False,
        dpace=True,
        lk_lambda=True,
    ),
    Case("dflash2 without a selector objective", "dflash2", selector=False),
    Case("domino", "domino"),
    Case("domino shifted", "domino", shifted=True),
    Case("dspark", "dspark"),
    Case("dspark without a confidence head", "dspark", confidence=False),
    # DSpine's metrics move a few hundred times the rounding noise of its
    # inputs (3e-4 between macOS arm64 and Linux x86), whatever the optimizer.
    Case("dspine", "dspine", tolerance=2e-3),
)


@dataclass(frozen=True)
class RolloutCase:
    name: str
    kind: str
    # The token loss: None is the objective's soft cross entropy.
    token: str | None = None
    tolerance: float = 1e-4


ROLLOUT_CASES = (
    RolloutCase("eagle3", "eagle3"),
    # An LK loss changes what a depth's loss is and the name it is logged under.
    RolloutCase("eagle3 lk_lambda", "eagle3", token="lk_lambda"),
    RolloutCase("peagle", "peagle"),
)
ROLLOUT_DEPTHS = 3


class NullBackend(TrainingBackend):
    """Runs the trainer's metric reduction without gradients or an optimizer."""

    name = "null"

    def prepare_model(self, model):
        return model

    def backward(self, loss, *, is_boundary=True):
        pass

    def scale_gradients(self, factor):
        pass

    def step(self):
        return None

    def state_dict(self):
        return {}

    def load_state_dict(self, state):
        pass


def logged_metrics(strategy, tensors, *, detailed=True):
    """The metrics one optimizer step hands to the tracker."""
    batch = TrainBatch(sample_ids=["a", "b"], strategy=strategy.name, tensors=tensors)
    ctx = StepContext(
        global_step=STEP, total_steps=TOTAL_STEPS, collect_detailed_metrics=detailed
    )
    runner = DataParallelRunner(strategy, NullBackend())
    return run_window(runner, [batch], ctx).materialize()


def evaluated_metrics(strategy, batches):
    """The metrics the trainer's evaluation pass reports for these batches."""
    data = [
        TrainBatch(sample_ids=["a", "b"], strategy=strategy.name, tensors=tensors)
        for tensors in batches
    ]
    backend = NullBackend()
    with TemporaryDirectory() as output_dir:
        controller = TrainerController(
            DataParallelRunner(strategy, backend),
            strategy=strategy,
            backend=backend,
            run_id="metrics",
            output_dir=output_dir,
            total_steps=TOTAL_STEPS,
            start_step=STEP,
        )
        return controller.evaluate(data)


def metric_batch(runtime):
    """Two uneven rows with a masked gap and a teacher that is often right."""
    head = runtime.target_head.weight.detach().float()
    device, length = head.device, 20
    feature_width = training_batch(runtime.draft_model.config.draft)[
        "hidden_states"
    ].shape[-1]
    ids = torch.randint(0, 12, (2, length), device=device)
    loss_mask = torch.ones(2, length, device=device)
    loss_mask[0, :2] = 0
    loss_mask[0, 9:11] = 0
    loss_mask[1, 14:] = 0
    attention_mask = torch.ones(2, length, dtype=torch.long, device=device)
    attention_mask[1, 16:] = 0
    # A state along a token's head row puts that token's logit at 4. The
    # teacher predicts the next token, and another one at every third position.
    taught = ids.roll(-1, 1)
    taught[:, ::3] += 1
    rows = head[taught]
    return dict(
        input_ids=ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        hidden_states=torch.randn(2, length, feature_width, device=device),
        target_last_hidden_states=4 * rows / rows.square().sum(-1, keepdim=True),
    )


def configure(case, runtime):
    """Non-default settings, so a metric that ignores one of them shows."""
    runtime.objective.num_anchors = 5
    if case.kind in ("dflash", "dflash2"):
        runtime.objective.teacher_metrics = case.teacher_metrics
    if case.kind == "dflash":
        # the default 16 candidates always cover this batch's twelve tokens
        runtime.objective.metric_top_k = 4
    if case.kind == "dflash2":
        runtime.objective.selector_warmup_ratio = 0.2
        runtime.objective.selector_ramp_ratio = 0.5
    # uniform weights would hide a metric that forgot its loss weights
    runtime.weighting = (
        PositionWeighting(kind="dpace", alpha=0.5)
        if case.dpace
        else PositionWeighting(kind="exponential", gamma=4.0)
    )
    if not case.confidence:
        runtime.draft_model.confidence_head = None


def loss_settings(case):
    loss = {"terms": {}}
    if case.lk_lambda:
        loss["terms"]["token"] = {"kind": "lk_lambda"}
    if not case.selector:
        loss["terms"]["selector"] = {"coefficient": 0.0}
    return loss


def fit(runtime, tensors, *, steps=18, frozen=TARGET_TABLES):
    """Train the draft briefly so the diagnostics are neither all zero nor all one."""
    trained = [
        parameter
        for name, parameter in runtime.draft_model.named_parameters()
        if parameter.requires_grad and name not in frozen
    ]
    # Adam's default epsilon turns a gradient of rounding noise into a full
    # step, so the fitted weights would differ between platforms; this one
    # leaves such gradients where they are.
    optimizer = torch.optim.Adam(trained, lr=1e-2, eps=1e-2)
    for index in range(steps):
        optimizer.zero_grad()
        torch.manual_seed(100 + index)
        runtime(
            tensors,
            step=STEP,
            total_steps=TOTAL_STEPS,
            collect_detailed_metrics=False,
        ).loss.backward()
        optimizer.step()
    runtime.zero_grad()


def prepared(case, runtime):
    """Configure and fit a freshly built runtime; returns its batch."""
    configure(case, runtime)
    # The batch must not depend on how many random numbers construction drew.
    torch.manual_seed(13)
    tensors = metric_batch(runtime)
    with patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}):
        fit(runtime, tensors)
    return tensors


def canonical_metrics(case, runtime, tensors, *, detailed=True):
    """What a step of the canonical strategy reports for a prepared runtime."""
    if not case.lk_lambda:
        # Three of ten blocks per chunk, so merging chunks is covered too.
        # LK-lambda reads the whole prediction set and stays unsplit.
        runtime.execution = replace(runtime.execution, chunk_size=3)
    strategy = CanonicalTrainStrategy(
        runtime, name=case.kind, objective=case.kind, required_features=tuple(tensors)
    )
    torch.manual_seed(11)
    with patch.dict("os.environ", {"SPECFORGE_DFLASH2_FUSED_CONV": "0"}):
        return logged_metrics(strategy, tensors, detailed=detailed)


def block_metrics(case, *, detailed=True):
    torch.manual_seed(7)
    runtime = block_runtime(case.kind, shifted=case.shifted, loss=loss_settings(case))
    return canonical_metrics(case, runtime, prepared(case, runtime), detailed=detailed)


def rollout_runtime(case):
    """The canonical model and objective of one rollout case, before fitting."""
    config = eagle_config() if case.kind == "eagle3" else peagle_config()
    model = build_draft_model(config)
    install_mapping(model)
    loss = None
    if case.token is not None:
        token = {"kind": case.token, "options": {"target": "teacher"}}
        loss = {"terms": {"token": token}}
    return build_objective_model(
        model,
        options={
            "ttt_length" if case.kind == "eagle3" else "num_depths": ROLLOUT_DEPTHS
        },
        target_head=torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False),
        loss=loss,
    )


def rollout_batch(case, runtime):
    """Uneven supervision under a teacher that leaves the draft vocabulary.

    The draft vocabulary holds the even tokens and every input token is even.
    The teacher names the next token, and at every third position the odd
    token after it, which the draft can never propose.
    """
    head = runtime.target_head.weight.detach().float()
    # P-EAGLE samples its positions from a single row.
    rows, length = (2, 16) if case.kind == "eagle3" else (1, 20)
    width = training_batch(runtime.draft_model.config.draft)["hidden_states"].shape[-1]
    ids = 2 * torch.randint(0, 6, (rows, length))
    loss_mask = torch.ones(rows, length)
    loss_mask[0, :2] = 0
    loss_mask[0, 9:11] = 0
    attention_mask = torch.ones(rows, length, dtype=torch.long)
    if rows > 1:
        loss_mask[1, 12:] = 0
        attention_mask[1, 14:] = 0
    taught = ids.roll(-1, 1)
    taught[:, ::3] += 1
    states = head[taught]
    return dict(
        input_ids=ids,
        loss_mask=loss_mask,
        attention_mask=attention_mask,
        hidden_states=torch.randn(rows, length, width),
        target_last_hidden_states=4 * states / states.square().sum(-1, keepdim=True),
    )


def prepared_rollout(case):
    """A fitted rollout runtime and its batch."""
    torch.manual_seed(7)
    runtime = rollout_runtime(case)
    torch.manual_seed(13)
    tensors = rollout_batch(case, runtime)
    # The rollout drafts own their output head: fit it with the rest.
    fit(runtime, tensors, frozen=())
    return runtime, tensors


def evaluation_batches(tensors):
    """The batch and a shorter one with another share of supervised positions.

    Evaluation weights a batch's depth means by its loss token count, which
    two batches of one size and supervision would not show.
    """
    return [tensors, {name: value[:1, :9] for name, value in tensors.items()}]


def rollout_strategy(case, runtime, tensors):
    return CanonicalTrainStrategy(
        runtime, name=case.kind, objective=case.kind, required_features=tuple(tensors)
    )


def rollout_metrics(case, *, detailed=True, evaluation=False):
    """What the canonical strategy reports for a rollout case."""
    runtime, tensors = prepared_rollout(case)
    strategy = rollout_strategy(case, runtime, tensors)
    torch.manual_seed(11)
    if evaluation:
        return evaluated_metrics(strategy, evaluation_batches(tensors))
    return logged_metrics(strategy, tensors, detailed=detailed)


def flattened(metrics):
    """Metrics with a list spread over one key per element."""
    return {
        name: value
        for key, values in metrics.items()
        for name, value in (
            [(f"{key}[{index}]", item) for index, item in enumerate(values)]
            if isinstance(values, list)
            else [(key, values)]
        )
    }


class MetricGoldenTest(unittest.TestCase):
    def test_logging_step_reports_the_recorded_keys_and_values(self):
        recorded = json.loads(GOLDENS.read_text()) if GOLDENS.exists() else {}
        update = os.environ.get("SPECFORGE_UPDATE_GOLDENS")
        measured = [(case.name, case, block_metrics) for case in CASES]
        for case in ROLLOUT_CASES:
            measured.append((case.name, case, rollout_metrics))
            if case.kind == "eagle3":
                # its evaluation reports the depth losses and acceptance too
                measured.append(
                    (
                        f"{case.name} evaluation",
                        case,
                        lambda case: flattened(rollout_metrics(case, evaluation=True)),
                    )
                )
        for name, case, measure in measured:
            with self.subTest(case=name):
                actual = measure(case)
                if update:
                    recorded[name] = actual
                    continue
                expected = recorded[name]
                self.assertEqual(sorted(actual), sorted(expected))
                for key, value in expected.items():
                    # Loose enough for another BLAS after eighteen optimizer
                    # steps; a wrong formula moves a value by a token in dozens.
                    self.assertAlmostEqual(
                        actual[key],
                        value,
                        delta=1e-5 + case.tolerance * abs(value),
                        msg=key,
                    )
        if update:
            GOLDENS.write_text(json.dumps(recorded, indent=1, sort_keys=True) + "\n")

    def test_non_logging_step_reports_only_what_the_loss_already_computed(self):
        for kind, expected in (
            ("dflash", ["acc", "ce_loss", "loss"]),
            ("dflash2", ["acc", "ce_loss", "loss"]),
            ("domino", ["acc", "loss"]),
            ("dspark", ["acc", "loss"]),
        ):
            with self.subTest(kind=kind):
                self.assertEqual(
                    sorted(block_metrics(Case(kind, kind), detailed=False)), expected
                )
        depths = range(ROLLOUT_DEPTHS)
        for case in ROLLOUT_CASES:
            # A depth's accuracy and loss come with the loss; its acceptance
            # and the P-EAGLE counts need a second projection.
            expected = ["acc", "loss"]
            if case.kind == "eagle3":
                name = "lk_loss" if case.token else "kl_loss"
                expected += [name] + [
                    f"{family}_{depth}"
                    for family in ("acc", "ploss", name)
                    for depth in depths
                ]
            with self.subTest(case=case.name):
                self.assertEqual(
                    sorted(rollout_metrics(case, detailed=False)), sorted(expected)
                )


if __name__ == "__main__":
    unittest.main()
