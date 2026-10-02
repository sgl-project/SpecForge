# coding=utf-8
"""DFlash offline capability through the canonical strategy-neutral trainer."""

import os
import tempfile
import unittest

import torch

from specforge.algorithms.builtin import builtin_algorithm_registry

CUDA = torch.cuda.is_available()
ALGORITHM = builtin_algorithm_registry().resolve("dflash")


@unittest.skipUnless(CUDA, "DFlash offline launcher requires CUDA")
class TestDFlashOfflineLaunch(unittest.TestCase):
    def test_dflash_trains_from_precomputed_features(self):
        self._check_backend("fsdp")

    def test_fsdp2_trains_through_the_same_launcher(self):
        self._check_backend("fsdp2")

    def test_dflash2_nested_blocks_train_with_fsdp2(self):
        self._check_backend("fsdp2", dflash2=True)

    def _check_backend(self, training_backend, *, dflash2=False):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29567")

        from torch.distributed.fsdp import FSDPModule
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        expected_wrapper = FSDPModule if training_backend == "fsdp2" else FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        hidden, sequence_length = 64, 32
        workdir = tempfile.mkdtemp(prefix="dflash_offline_")
        feature_dir = fx.write_offline_files_dflash(
            os.path.join(workdir, "features"),
            n=4,
            seq=sequence_length,
            hidden=hidden,
        )
        model, width, _target_dir, _layers = fx.build_dflash(
            workdir,
            hidden=hidden,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        self.assertEqual(width, hidden)
        if dflash2:
            from specforge.modeling.draft.dflash2 import DFlash2DraftModel

            config = model.draft_model.config
            config.dflash_config.update(
                conv_kernel_size=3,
                conv_group_size=16,
                selector_rank=8,
                selector_top_k=4,
            )
            model.draft_model = DFlash2DraftModel(config).cuda().bfloat16()
            model.selector_loss_alpha = 0.1

        def optimizer_factory(module):
            return BF16Optimizer(
                module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=2,
            )

        trainer = build_offline_runtime(
            algorithm=ALGORITHM,
            hidden_states_path=feature_dir,
            draft_model=model,
            target_head=None,
            optimizer_factory=optimizer_factory,
            training_backend=training_backend,
            run_id="dflash-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=sequence_length,
            batch_size=1,
            num_epochs=1,
            max_steps=2,
        )

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, expected_wrapper)
        self.assertEqual(trainer.fit(), 2)
        self.assertTrue(all(torch.isfinite(p).all() for p in module.parameters()))


if __name__ == "__main__":
    unittest.main(verbosity=2)
