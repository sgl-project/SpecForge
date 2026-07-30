# coding=utf-8
"""DSpark offline capability through the canonical trainer runtime."""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
ALGORITHM = builtin_algorithm_registry().resolve("dspark")


@unittest.skipUnless(CUDA, "DSpark offline launcher requires CUDA")
class TestDSparkOfflineLaunch(unittest.TestCase):
    def test_dspark_trains_from_precomputed_target_features(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29580")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.training.strategies.base import DSparkTrainStrategy

        hidden, sequence_length = 64, 12
        workdir = tempfile.mkdtemp(prefix="dspark_offline_")
        model, captured_width = fx.build_dspark(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=2,
            attention_backend="sdpa",
        )
        feature_dir = fx.write_offline_files_dspark(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            captured_width=captured_width,
            target_hidden=hidden,
        )

        trainer = build_offline_runtime(
            algorithm=ALGORITHM,
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=None,
            optimizer_factory=lambda module: BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=2,
            ),
            run_id="dspark-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
            total_steps=2,
        )

        strategy = trainer.core.strategy
        self.assertIsInstance(strategy, DSparkTrainStrategy)
        module = strategy.trainable_module()
        self.assertIsInstance(module, FSDP)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
