"""DSpine accumulation preserves the combined-batch objective and sampling."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch

from specforge.runtime.contracts import TrainBatch
from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
from specforge.training.contracts import IncompleteWindow, StepContext
from specforge.training.controller import TrainerController
from tests.test_modeling.dspine_fixtures import training_inputs, training_model
from tests.test_modeling.test_dspine_distributed import (
    fixed_anchors,
    per_microbatch_objective,
    uneven_batch,
)
from tests.test_runtime._fixtures import build_runner, train_window


class RecordingOptimizer:
    def __init__(self, model):
        self.parameters = [p for p in model.draft_model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.SGD(self.parameters, lr=0.01)
        self.gradients = []

    def step(self):
        self.gradients.append(
            [parameter.grad.detach().clone() for parameter in self.parameters]
        )
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)
        return torch.tensor(0.0)


def training_runner(model):
    backend = FSDPTrainingBackend(ParallelConfig())
    backend.prepare_model(model, wrap=False)
    optimizer = RecordingOptimizer(model)
    backend.set_optimizer(optimizer)
    return build_runner(model, backend), optimizer


class DSpineAccumulationTest(unittest.TestCase):

    def test_optimizer_windows_match_combined_batches_with_uneven_masks(self):
        for accumulation_steps in [2, 4]:
            with self.subTest(accumulation_steps=accumulation_steps):
                full = training_model(chunk_size=1)
                self.enterContext(
                    patch(
                        "specforge.algorithms.objectives.inputs.sample_anchors",
                        fixed_anchors,
                    )
                )
                split = copy.deepcopy(full)
                (_, full_optimizer) = training_runner(full)
                (split_runner, split_optimizer) = training_runner(split)
                for window in range(2):
                    batches = [uneven_batch(), uneven_batch()]
                    inputs = {
                        name: torch.cat([batch[name] for batch in batches])
                        for name in batches[0]
                    }
                    ctx = StepContext(global_step=300 + window, total_steps=600)
                    width = 4 // accumulation_steps
                    expected = per_microbatch_objective(full, inputs, width, ctx)
                    expected.backward()
                    full_optimizer.step()
                    actual = train_window(
                        split_runner,
                        [
                            TrainBatch(
                                [str(start)],
                                "dspine",
                                {
                                    name: value[start : start + width]
                                    for (name, value) in inputs.items()
                                },
                            )
                            for start in range(0, 4, width)
                        ],
                        ctx,
                    )
                    # One optimizer step per window, however it is split.
                    self.assertEqual(len(split_optimizer.gradients), window + 1)
                    self.assertLessEqual(
                        abs(actual.loss - expected.item()),
                        max(1e-12, 2e-06 * abs(expected.item())),
                    )
                    for got, want in zip(
                        split_optimizer.gradients[-1], full_optimizer.gradients[-1]
                    ):
                        torch.testing.assert_close(got, want, atol=2e-05, rtol=0.0002)
                    for got, want in zip(split.parameters(), full.parameters()):
                        torch.testing.assert_close(got, want, atol=2e-06, rtol=2e-05)

    def test_window_replay_preserves_anchor_curriculum_draws_and_rng(self):
        model = training_model(chunk_size=1)
        reference = copy.deepcopy(model)
        inputs = training_inputs()
        batches = [
            TrainBatch(
                [str(index)],
                "dspine",
                {name: tensor[index : index + 1] for (name, tensor) in inputs.items()},
            )
            for index in range(2)
        ]
        ctx = StepContext(global_step=100, total_steps=600)
        torch.manual_seed(321)
        with torch.no_grad():
            for batch in batches:
                reference(batch.tensors, step=100, total_steps=600)
        expected_rng = torch.get_rng_state()
        forwards = []

        def record_inputs(_module, _args, kwargs):
            forwards.append(
                (
                    torch.is_grad_enabled(),
                    _args[2][0].position_ids.clone(),
                    kwargs["replace_masks"][0].clone(),
                )
            )

        model.draft_model.register_forward_pre_hook(record_inputs, with_kwargs=True)
        (runner, _) = training_runner(model)
        torch.manual_seed(321)
        train_window(runner, batches, ctx)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        self.assertEqual([entry[0] for entry in forwards], [False, False, True, True])
        for prepass, replay in zip(forwards[:2], forwards[2:]):
            for got, want in zip(replay[1:], prepass[1:]):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_evaluation_does_not_replay_the_window(self):
        model = training_model(chunk_size=1)
        (runner, _) = training_runner(model)
        forwards = []
        model.register_forward_pre_hook(
            lambda _module, _args, kwargs: forwards.append(
                (
                    torch.is_grad_enabled(),
                    kwargs.get("prepass", False),
                    kwargs.get("alignment_ce"),
                )
            ),
            with_kwargs=True,
        )
        with TemporaryDirectory() as directory:
            controller = TrainerController(
                runner,
                strategy_name=runner.spec.name,
                accumulation_steps=2,
                run_id="dspine-evaluation",
                output_dir=directory,
                total_steps=600,
                start_step=300,
            )
            metrics = controller.evaluate(
                [TrainBatch(["0", "1"], "dspine", training_inputs())]
            )
        # One forward without gradients, on the batch's own statistic: neither
        # a prepass nor the replay of one.
        self.assertEqual(forwards, [(False, False, None)])
        self.assertTrue(torch.isfinite(torch.tensor(metrics["eval/avg_loss"])))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_window_replay_has_finite_updates(self):
        for attention_backend in ["sdpa", "flex_attention"]:
            with self.subTest(attention_backend=attention_backend):
                model = training_model(
                    chunk_size=1,
                    head_dim=16,
                    backend=attention_backend,
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                model.attention_backend = attention_backend
                model.draft_model.transfer_codes = (
                    model.draft_model.transfer_codes.float()
                )
                (runner, optimizer) = training_runner(model)
                inputs = training_inputs()
                for name in ("hidden_states", "target_last_hidden_states"):
                    inputs[name] = inputs[name].to(torch.bfloat16)
                batches = [
                    TrainBatch(
                        [str(index)],
                        "dspine",
                        {
                            name: tensor[index : index + 1]
                            for (name, tensor) in inputs.items()
                        },
                    )
                    for index in range(2)
                ]
                for window, step in enumerate((100, 300)):
                    metrics = train_window(
                        runner, batches, StepContext(global_step=step, total_steps=600)
                    )
                    self.assertEqual(len(optimizer.gradients), window + 1)
                    self.assertTrue(torch.isfinite(torch.tensor(metrics.loss)))
                    self.assertTrue(
                        all(
                            (
                                torch.isfinite(gradient).all()
                                for gradient in optimizer.gradients[-1]
                            )
                        )
                    )
                    self.assertTrue(
                        all(
                            (
                                torch.isfinite(parameter).all()
                                for parameter in model.parameters()
                            )
                        )
                    )
                self.assertEqual(len(optimizer.gradients), 2)
                self.assertEqual(model.draft_model.transfer_codes.dtype, torch.float32)
                self.assertIs(model.target_head.weight.grad, None)

    def test_incomplete_replay_window_is_rejected_without_an_optimizer_step(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model = training_model(chunk_size=1, replacement_probability=0)
            (runner, optimizer) = training_runner(model)
            controller = TrainerController(
                runner,
                strategy_name=runner.spec.name,
                accumulation_steps=2,
                run_id="dspine-incomplete",
                output_dir=tmp_path,
            )
            with self.assertRaisesRegex(
                IncompleteWindow,
                "incomplete gradient accumulation: received 1 of 2 micro-batches",
            ):
                controller.fit([TrainBatch(["0", "1"], "dspine", training_inputs())])
            self.assertEqual(controller.global_step, 0)
            self.assertFalse(optimizer.gradients)


if __name__ == "__main__":
    unittest.main()
