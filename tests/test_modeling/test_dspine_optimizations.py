import copy
from unittest.mock import patch

import pytest
import torch

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.runtime.contracts import TrainBatch
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import training_inputs, training_model


@pytest.mark.parametrize("chunk_size", [0, 1, 3])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_flattened_projection_preserves_full_objective_and_gradients(chunk_size, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    model = training_model(chunk_size=chunk_size)
    if device == "cuda":
        model = model.to(device=device, dtype=torch.bfloat16)
    reference = copy.deepcopy(model)
    reference._project_logits = reference.lm_head.forward
    inputs = {name: value.to(device) for name, value in training_inputs().items()}
    if device == "cuda":
        for name in ("hidden_states", "target_last_hidden_states"):
            inputs[name] = inputs[name].bfloat16()
    torch.manual_seed(39)
    expected = reference(**inputs, global_step=100, total_steps=600)[0]
    expected.backward()
    torch.manual_seed(39)
    actual = model(**inputs, global_step=100, total_steps=600)[0]
    actual.backward()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=2e-3)
    for got, want in zip(model.parameters(), reference.parameters()):
        if want.grad is not None:
            if device == "cpu":
                torch.testing.assert_close(got.grad, want.grad, atol=2e-5, rtol=2e-4)
            else:
                difference = (got.grad.float() - want.grad.float()).norm()
                assert difference <= 0.02 * want.grad.float().norm() + 1e-6


@pytest.mark.parametrize("step", [100, 300])
def test_ce_prepass_matches_full_without_teacher_or_refinement(step):
    model = training_model(chunk_size=1)
    inputs = training_inputs()
    torch.manual_seed(39)
    with torch.no_grad():
        _, accuracy, metrics = model(**inputs, global_step=step, total_steps=600)
    torch.manual_seed(39)
    with (
        torch.no_grad(),
        patch.object(
            model, "_objective_terms", side_effect=AssertionError("full objective")
        ),
        patch.object(
            model.draft_model, "refine", side_effect=AssertionError("refinement")
        ),
        patch(
            "specforge.algorithms.dspine.model.dist.all_reduce",
            side_effect=AssertionError("collective"),
        ),
    ):
        _, actual_accuracy, actual = model(
            **inputs, global_step=step, total_steps=600, ce_only=True
        )
    torch.testing.assert_close(actual_accuracy, accuracy)
    for got, want in zip(
        actual["ratio_metrics"]["dspine/backbone_ce"],
        metrics["ratio_metrics"]["dspine/backbone_ce"],
    ):
        torch.testing.assert_close(got, want)


def test_ce_prepass_rejects_autograd():
    with pytest.raises(ValueError, match="gradients to be disabled"):
        training_model()(**training_inputs(), ce_only=True)


@pytest.mark.parametrize("accumulation_steps", [2, 4])
def test_alignment_ce_reduced_once_per_window(accumulation_steps):
    strategy = DSpineTrainStrategy(training_model(chunk_size=1))
    batch = TrainBatch(["0", "1"], "dspine", training_inputs())
    ctx = StepContext(
        global_step=100, total_steps=600, accumulation_steps=accumulation_steps
    )
    with (
        patch(
            "specforge.algorithms.dspine.strategy.dist.is_initialized", return_value=True
        ),
        patch("specforge.algorithms.dspine.strategy.dist.all_reduce") as reduce,
    ):
        for window in range(2):
            prepared = [
                strategy.forward_loss(batch, ctx) for _ in range(accumulation_steps)
            ]
            assert reduce.call_count == window
            ratios = {
                "dspine/backbone_ce": tuple(
                    sum(
                        output.ratio_metrics["dspine/backbone_ce"][index]
                        for output in prepared
                    )
                    for index in range(2)
                )
            }
            for output in prepared:
                output.replay_loss(ratios)
            assert reduce.call_count == window + 1
