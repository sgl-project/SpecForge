# coding=utf-8
"""Launcher path: FSDP is in the forward path + global_step is optimizer steps.

Exercises build_offline_eagle3_runtime end to end (single rank) with
accumulation_steps=2, which the equivalence tests (wrap=False) do not cover:
- the strategy must run forward through the FSDP-wrapped module (Issue 1);
- fit's global_step counts OPTIMIZER steps, micro_step counts micro-batches (Issue 2).

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "launcher FSDP path requires CUDA")
class TestOfflineLaunchFSDP(unittest.TestCase):
    def test_fsdp_in_forward_path_and_optimizer_step_semantics(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29566")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.launch import build_offline_eagle3_runtime

        TTT, ACC, MAX_OPT_STEPS, N = 3, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="launch_fsdp_")
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)
        eagle3_model, target_head = fx.build_eagle3(workdir, ttt=TTT)

        def optimizer_factory(draft_module):
            return BF16Optimizer(
                draft_module,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=10,
            )

        trainer, loader = build_offline_eagle3_runtime(
            hidden_states_path=feat_dir,
            eagle3_model=eagle3_model,
            target_head=target_head,
            optimizer_factory=optimizer_factory,
            run_id="launch",
            output_dir=os.path.join(workdir, "out"),
            ttt_length=TTT,
            max_len=512,
            batch_size=1,
            accumulation_steps=ACC,
            num_epochs=3,
            max_steps=MAX_OPT_STEPS,
        )

        # Issue 1: the strategy runs forward through the FSDP-wrapped module
        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(
            module, FSDP, "strategy must hold the FSDP-wrapped module"
        )
        self.assertIsNotNone(trainer.core.backend.optimizer)

        step = trainer.fit(loader)

        # Issue 2: global_step == optimizer steps; micro_step == ACC * optimizer steps
        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(trainer.global_step, MAX_OPT_STEPS)
        self.assertEqual(trainer.micro_step, ACC * MAX_OPT_STEPS)

        # a checkpoint can be written through the FSDP FULL_STATE_DICT path
        ckpt = trainer.save_checkpoint(trainer.global_step)
        self.assertTrue(ckpt.checkpoint_uri.startswith("file://"))
        self.assertEqual(ckpt.global_step, MAX_OPT_STEPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
