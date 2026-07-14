# coding=utf-8
"""Launcher path (DFlash, online): build_online_runtime(strategy="dflash") end to end.

Interleaves a RolloutWorker with the policy adapter (HF DFlash target, no
SGLang) and FSDP training through the bounded mem:// stream. Proves the strategy
schema and tag drive a non-EAGLE3 model. GPU-only.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "DFlash online launcher path requires CUDA")
class TestDFlashOnlineLaunch(unittest.TestCase):
    def test_online_rollout_is_interleaved_with_fsdp_train(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29570")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge.inference.target_engine import get_target_engine
        from specforge.launch import build_online_runtime
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import assert_no_tensors

        HIDDEN, V, SEQ, ACC, MAX_OPT_STEPS, N = 64, fx.V, 16, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="dflash_online_")

        # trainable DFlash model + the saved tiny Qwen3 target it derived from
        dflash_model, width, target_dir, layers = fx.build_dflash(
            workdir,
            hidden=HIDDEN,
            vocab=V,
            block_size=4,
            num_anchors=8,
            attention_backend="sdpa",
        )
        self.assertEqual(width, HIDDEN)

        # HF DFlash target (no sglang) capturing the same layers the draft expects
        target = get_target_engine(
            target_dir,
            strategy="dflash",
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

        trainer = build_online_runtime(
            strategy="dflash",
            target_model=target,
            prompts=prompts,
            draft_model=dflash_model,
            optimizer_factory=optimizer_factory,
            run_id="dflash-online",
            output_dir=os.path.join(workdir, "out"),
            target_hidden_size=HIDDEN,  # benign for DFlash (no target distribution)
            batch_size=1,
            accumulation_steps=ACC,
            max_steps=MAX_OPT_STEPS,
        )

        # Build is metadata-only; rollout starts when the trainer requests a batch.
        self.assertEqual(trainer.dataflow_controller.sample_queue.depth(), 0)
        assert_no_tensors(trainer.dataflow_controller.status())

        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(module, FSDP)

        step = trainer.fit()

        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(trainer.micro_step, ACC * MAX_OPT_STEPS)
        self.assertEqual(trainer.rollout_stream.produced_count, ACC * MAX_OPT_STEPS)
        self.assertLessEqual(trainer.rollout_stream.peak_resident_samples, 1)
        self.assertTrue(
            all(torch.isfinite(p).all() for p in module.parameters()),
            "draft params became non-finite — loss was NaN/inf?",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
