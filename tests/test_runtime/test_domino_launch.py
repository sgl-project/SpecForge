# coding=utf-8
"""Domino step schedule and offline launcher.

Domino's immutable algorithm registration injects its concrete step provider.
Its loss blends a base loss with a step-decayed weight, so
DominoTrainStrategy reads the StepContext threaded through forward_loss. These
tests prove the lambda schedule logic on CPU and that Domino trains end to end
from precomputed features on GPU.
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


@unittest.skipUnless(CUDA, "Domino offline launcher path requires CUDA")
class TestDominoOfflineLaunch(unittest.TestCase):
    def test_domino_trains_from_precomputed_dflash_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29571")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        hidden, sequence_length = 64, 32
        workdir = tempfile.mkdtemp(prefix="domino_offline_")
        feature_dir = fx.write_offline_files_dflash(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            hidden=hidden,
        )
        model, width, _target_dir, _layers = fx.build_domino(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        self.assertEqual(width, hidden)

        def optimizer_factory(module):
            return BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=2,
            )

        trainer = build_offline_runtime(
            strategy="domino",
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=None,
            optimizer_factory=optimizer_factory,
            run_id="domino-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
            total_steps=2,
            strategy_kwargs={"lambda_start": 1.0, "decay_ratio": 0.5},
        )

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
