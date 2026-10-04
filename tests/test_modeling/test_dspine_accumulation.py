"""DSpine accumulation preserves the combined-batch objective and sampling."""

import copy

import pytest
import torch

from specforge.algorithms.dspine.strategy import DSpineTrainStrategy
from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.controller import TrainerController, TrainerCore
from specforge.training.strategies.base import StepContext
from tests.test_modeling.test_dspine import training_inputs, training_model
from tests.test_modeling.test_dspine_distributed import fixed_anchors, uneven_batch


class RecordingOptimizer:
    def __init__(self, model):
        self.parameters = list(model.draft_model.parameters())
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.01)
        self.gradients = []

    def step(self):
        self.gradients.append(
            [parameter.grad.detach().clone() for parameter in self.parameters]
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return torch.tensor(0.0)


def training_core(model, accumulation_steps):
    backend = FSDPTrainingBackend(ParallelConfig())
    backend.prepare_model(model, wrap=False)
    optimizer = RecordingOptimizer(model)
    backend.set_optimizer(optimizer)
    return (
        TrainerCore(
            DSpineTrainStrategy(model), backend, accumulation_steps=accumulation_steps
        ),
        optimizer,
    )


@pytest.mark.parametrize("accumulation_steps", [2, 4])
def test_optimizer_windows_match_combined_batches_with_uneven_masks(accumulation_steps):
    full = training_model(chunk_size=1)
    full._sample_anchor_positions = fixed_anchors
    split = copy.deepcopy(full)
    full_core, full_optimizer = training_core(full, 1)
    split_core, split_optimizer = training_core(split, accumulation_steps)
    for window in range(2):
        batches = [uneven_batch(), uneven_batch()]
        inputs = {
            name: torch.cat([batch[name] for batch in batches]) for name in batches[0]
        }
        ctx = StepContext(global_step=300 + window, total_steps=600)
        expected = full_core.train_step(
            TrainBatch(["0", "1", "2", "3"], "dspine", inputs), ctx
        )
        width = 4 // accumulation_steps
        for microstep in range(accumulation_steps):
            start = microstep * width
            actual = split_core.train_step(
                TrainBatch(
                    [str(start)],
                    "dspine",
                    {
                        name: value[start : start + width]
                        for name, value in inputs.items()
                    },
                ),
                ctx,
            )
            boundary = microstep == accumulation_steps - 1
            assert actual.optimizer_stepped == boundary
            assert (
                split_core.accumulation_remainder
                == (microstep + 1) % accumulation_steps
            )
            assert len(split_optimizer.gradients) == window + int(boundary)
        assert actual.loss == pytest.approx(expected.loss, rel=2e-6)
        for got, want in zip(
            split_optimizer.gradients[-1], full_optimizer.gradients[-1]
        ):
            torch.testing.assert_close(got, want, atol=2e-5, rtol=2e-4)
        for got, want in zip(split.parameters(), full.parameters()):
            torch.testing.assert_close(got, want, atol=2e-6, rtol=2e-5)


def test_window_replay_preserves_anchor_curriculum_draws_and_rng():
    model = training_model(chunk_size=1)
    reference = copy.deepcopy(model)
    inputs = training_inputs()
    batches = [
        TrainBatch(
            [str(index)],
            "dspine",
            {name: tensor[index : index + 1] for name, tensor in inputs.items()},
        )
        for index in range(2)
    ]
    ctx = StepContext(global_step=100, total_steps=600)
    torch.manual_seed(321)
    with torch.no_grad():
        for batch in batches:
            reference(**batch.tensors, global_step=100, total_steps=600)
    expected_rng = torch.get_rng_state()

    forwards = []

    def record_inputs(_module, _args, kwargs):
        forwards.append(
            (
                torch.is_grad_enabled(),
                kwargs["position_ids"].clone(),
                kwargs["replace_mask"].clone(),
            )
        )

    model.draft_model.register_forward_pre_hook(record_inputs, with_kwargs=True)
    core, _ = training_core(model, 2)
    torch.manual_seed(321)
    for batch in batches:
        core.train_step(batch, ctx)
    torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
    assert [entry[0] for entry in forwards] == [False, False, True, True]
    for prepass, replay in zip(forwards[:2], forwards[2:]):
        for got, want in zip(replay[1:], prepass[1:]):
            torch.testing.assert_close(got, want, atol=0, rtol=0)


def test_evaluation_does_not_request_window_replay():
    model = training_model(chunk_size=1)
    with torch.no_grad():
        output = DSpineTrainStrategy(model).forward_loss(
            TrainBatch(["0", "1"], "dspine", training_inputs()),
            StepContext(global_step=300, total_steps=600, accumulation_steps=2),
        )
    assert output.replay_loss is None
    assert torch.isfinite(output.loss)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("attention_backend", ["sdpa", "flex_attention"])
def test_cuda_bf16_window_replay_has_finite_updates(attention_backend):
    model = training_model(chunk_size=1, head_dim=16).to(
        device="cuda", dtype=torch.bfloat16
    )
    model.attention_backend = attention_backend
    model.draft_model.config._attn_implementation = attention_backend
    model.draft_model.transfer_codes = model.draft_model.transfer_codes.float()
    core, optimizer = training_core(model, 2)
    inputs = training_inputs()
    for name in ("hidden_states", "target_last_hidden_states"):
        inputs[name] = inputs[name].to(torch.bfloat16)
    for step in (100, 300):
        for index in range(2):
            result = core.train_step(
                TrainBatch(
                    [str(index)],
                    "dspine",
                    {
                        name: tensor[index : index + 1]
                        for name, tensor in inputs.items()
                    },
                ),
                StepContext(global_step=step, total_steps=600),
            )
            assert result.optimizer_stepped == (index == 1)
        assert torch.isfinite(torch.tensor(result.loss))
        assert all(
            torch.isfinite(gradient).all() for gradient in optimizer.gradients[-1]
        )
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert len(optimizer.gradients) == 2
    assert model.draft_model.transfer_codes.dtype == torch.float32
    assert model.lm_head.weight.grad is None
    assert model.embed_tokens.weight.grad is None


def test_incomplete_replay_window_is_rejected_without_an_optimizer_step(tmp_path):
    model = training_model(chunk_size=1)
    core, optimizer = training_core(model, 2)
    controller = TrainerController(
        core, run_id="dspine-incomplete", output_dir=tmp_path
    )
    with pytest.raises(RuntimeError, match="incomplete gradient accumulation"):
        controller.fit([TrainBatch(["0", "1"], "dspine", training_inputs())])
    assert controller.global_step == 0
    assert core.accumulation_remainder == 1
    assert not optimizer.gradients
