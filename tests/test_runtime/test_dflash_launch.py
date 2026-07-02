# coding=utf-8
"""Launcher path (DFlash, offline): build_offline_runtime(strategy="dflash") end to end.

Proves the strategy-parameterized launch layer trains a DIFFERENT model family
(DFlash: block-parallel, scalar loss, no target distribution) through the SAME
runtime spine (FeatureDataLoader -> _assemble_trainer -> FSDP -> DFlashTrainStrategy)
with zero new builder — only a StrategySpec entry. Synthetic offline features
(no DFlash dumper exists yet). GPU-only; run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "DFlash launcher FSDP path requires CUDA")
class TestDFlashOfflineLaunch(unittest.TestCase):
    def test_offline_dflash_trains_through_fsdp(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29567")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.launch import build_offline_runtime
        from specforge.optimizer import BF16Optimizer

        HIDDEN, SEQ, N, MAX_OPT_STEPS = 64, 32, 4, 2
        workdir = tempfile.mkdtemp(prefix="dflash_launch_")
        feat_dir = fx.write_offline_files_dflash(
            os.path.join(workdir, "features"), n=N, seq=SEQ, hidden=HIDDEN
        )
        dflash_model, width, _target_dir, _layers = fx.build_dflash(
            workdir,
            hidden=HIDDEN,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        # single draft layer -> one capture layer -> width == hidden
        self.assertEqual(width, HIDDEN)

        def optimizer_factory(draft_module):
            return BF16Optimizer(
                draft_module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=10,
            )

        logs = []
        trainer, loader = build_offline_runtime(
            strategy="dflash",
            hidden_states_path=feat_dir,
            eagle3_model=dflash_model,  # legacy param name = "the composite draft model"
            target_head=None,  # DFlash owns its own (frozen) head
            optimizer_factory=optimizer_factory,
            run_id="dflash-offline",
            output_dir=os.path.join(workdir, "out"),
            max_len=SEQ,
            batch_size=1,
            num_epochs=3,
            max_steps=MAX_OPT_STEPS,
            log_interval=1,
            logger=lambda metrics, step: logs.append((step, metrics)),
        )

        # the strategy holds the FSDP-wrapped DFlash model
        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)

        step = trainer.fit(loader)

        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertTrue(logs, "trainer logged no metrics")
        # loss must have been finite: NaN/inf loss would corrupt the params via AdamW
        self.assertTrue(
            all(torch.isfinite(p).all() for p in module.parameters()),
            "draft params became non-finite — loss was NaN/inf?",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
