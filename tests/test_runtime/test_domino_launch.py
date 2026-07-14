# coding=utf-8
"""Domino strategy schedule and canonical online launcher.

Domino is the third algorithm on the composable launch. Beyond a StrategySpec it
needs exactly one shared-contract extension: its loss blends a base loss with a
step-decayed weight, so DominoTrainStrategy reads the StepContext threaded through
forward_loss. These tests prove the lambda schedule logic on CPU and that Domino
trains end to end through the typed online runtime on GPU.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


class TestDominoLambdaSchedule(unittest.TestCase):
    """CPU: the StepContext-driven lambda_base schedule (no model needed)."""

    def test_lambda_base_decays_over_total_steps(self):
        from specforge.training.strategies.base import DominoTrainStrategy, StepContext

        # _lambda_base reads only ctx + lambda_start/decay_ratio, not the model.
        s = DominoTrainStrategy(None, lambda_start=1.0, decay_ratio=0.5)
        self.assertAlmostEqual(
            s._lambda_base(StepContext(global_step=0, total_steps=10)), 1.0
        )
        # decay_steps = total_steps*decay_ratio = 5; step 5+ -> 0
        self.assertEqual(
            s._lambda_base(StepContext(global_step=5, total_steps=10)), 0.0
        )
        self.assertEqual(
            s._lambda_base(StepContext(global_step=10, total_steps=10)), 0.0
        )
        # no schedule info -> pure final loss
        self.assertEqual(s._lambda_base(None), 0.0)
        self.assertEqual(
            s._lambda_base(StepContext(global_step=3, total_steps=None)), 0.0
        )


@unittest.skipUnless(CUDA, "Domino online launcher path requires CUDA")
class TestDominoOnlineLaunch(unittest.TestCase):
    def test_online_rollout_is_interleaved_with_fsdp_train(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29572")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_online_runtime
        from specforge.inference.target_engine import get_target_engine
        from specforge.optimizer import BF16Optimizer

        HIDDEN, V, SEQ, ACC, MAX_OPT_STEPS, N = 64, fx.V, 16, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="domino_online_")

        domino_model, width, target_dir, layers = fx.build_domino(
            workdir,
            hidden=HIDDEN,
            vocab=V,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        self.assertEqual(width, HIDDEN)

        # domino captures the same hidden_states as DFlash -> reuse the DFlash target
        target = get_target_engine(
            target_dir,
            strategy="domino",
            backend="hf",
            torch_dtype=torch.bfloat16,
            device="cuda",
        )
        target.set_capture_layers(layers)

        g = torch.Generator().manual_seed(7)
        prompts = [
            {
                "payload": {
                    "input_ids": torch.randint(0, V, (SEQ,), generator=g).tolist(),
                    "loss_mask": [1] * SEQ,
                }
            }
            for _ in range(N)
        ]

        def optimizer_factory(draft_module):
            return BF16Optimizer(
                draft_module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=10,
            )

        trainer, loader, workers, controller, run_interleaved = build_online_runtime(
            strategy="domino",
            target_model=target,
            prompts=prompts,
            draft_model=domino_model,
            optimizer_factory=optimizer_factory,
            run_id="domino-online",
            output_dir=os.path.join(workdir, "out"),
            target_hidden_size=HIDDEN,
            batch_size=1,
            accumulation_steps=ACC,
            max_steps=MAX_OPT_STEPS,
        )

        self.assertEqual(controller.sample_queue.depth(), 0)

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)

        step = run_interleaved()
        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(loader.queue.produced_count, ACC * MAX_OPT_STEPS)
        self.assertLessEqual(loader.queue.peak_resident_samples, 1)
        self.assertTrue(
            all(torch.isfinite(p).all() for p in module.parameters()),
            "draft params became non-finite — loss was NaN/inf?",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
