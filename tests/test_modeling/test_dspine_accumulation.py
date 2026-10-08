"""DSpine accumulation preserves the combined-batch objective and sampling."""

import copy
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

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


class DSpineAccumulationTest(unittest.TestCase):

    def test_optimizer_windows_match_combined_batches_with_uneven_masks(self):
        for accumulation_steps in [2, 4]:
            with self.subTest(accumulation_steps=accumulation_steps):
                full = training_model(chunk_size=1)
                full._sample_anchor_positions = fixed_anchors
                split = copy.deepcopy(full)
                (full_core, full_optimizer) = training_core(full, 1)
                (split_core, split_optimizer) = training_core(split, accumulation_steps)
                for window in range(2):
                    batches = [uneven_batch(), uneven_batch()]
                    inputs = {
                        name: torch.cat([batch[name] for batch in batches])
                        for name in batches[0]
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
                                    for (name, value) in inputs.items()
                                },
                            ),
                            ctx,
                        )
                        boundary = microstep == accumulation_steps - 1
                        self.assertEqual(actual.optimizer_stepped, boundary)
                        self.assertEqual(
                            split_core.accumulation_remainder,
                            (microstep + 1) % accumulation_steps,
                        )
                        self.assertEqual(
                            len(split_optimizer.gradients), window + int(boundary)
                        )
                    self.assertLessEqual(
                        abs(actual.loss - expected.loss),
                        max(1e-12, 2e-06 * abs(expected.loss)),
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
        (core, _) = training_core(model, 2)
        torch.manual_seed(321)
        for batch in batches:
            core.train_step(batch, ctx)
        torch.testing.assert_close(torch.get_rng_state(), expected_rng, atol=0, rtol=0)
        self.assertEqual([entry[0] for entry in forwards], [False, False, True, True])
        for prepass, replay in zip(forwards[:2], forwards[2:]):
            for got, want in zip(replay[1:], prepass[1:]):
                torch.testing.assert_close(got, want, atol=0, rtol=0)

    def test_evaluation_does_not_request_window_replay(self):
        model = training_model(chunk_size=1)
        with torch.no_grad():
            output = DSpineTrainStrategy(model).forward_loss(
                TrainBatch(["0", "1"], "dspine", training_inputs()),
                StepContext(global_step=300, total_steps=600, accumulation_steps=2),
            )
        self.assertIs(output.replay_loss, None)
        self.assertTrue(torch.isfinite(output.loss))

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_cuda_bf16_window_replay_has_finite_updates(self):
        for attention_backend in ["sdpa", "flex_attention"]:
            with self.subTest(attention_backend=attention_backend):
                model = training_model(chunk_size=1, head_dim=16).to(
                    device="cuda", dtype=torch.bfloat16
                )
                model.attention_backend = attention_backend
                model.draft_model.config._attn_implementation = attention_backend
                model.draft_model.transfer_codes = (
                    model.draft_model.transfer_codes.float()
                )
                (core, optimizer) = training_core(model, 2)
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
                                    for (name, tensor) in inputs.items()
                                },
                            ),
                            StepContext(global_step=step, total_steps=600),
                        )
                        self.assertEqual(result.optimizer_stepped, index == 1)
                    self.assertTrue(torch.isfinite(torch.tensor(result.loss)))
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
                self.assertIs(model.lm_head.weight.grad, None)
                self.assertIs(model.embed_tokens.weight.grad, None)

    def test_incomplete_replay_window_is_rejected_without_an_optimizer_step(self):
        with TemporaryDirectory() as directory:
            tmp_path = Path(directory)
            model = training_model(chunk_size=1)
            (core, optimizer) = training_core(model, 2)
            controller = TrainerController(
                core, run_id="dspine-incomplete", output_dir=tmp_path
            )
            with self.assertRaisesRegex(
                RuntimeError, "incomplete gradient accumulation"
            ):
                controller.fit([TrainBatch(["0", "1"], "dspine", training_inputs())])
            self.assertEqual(controller.global_step, 0)
            self.assertEqual(core.accumulation_remainder, 1)
            self.assertFalse(optimizer.gradients)


if __name__ == "__main__":
    unittest.main()
