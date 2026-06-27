# coding=utf-8
"""Launcher path (online): build_online_eagle3_runtime end to end, single rank.

Online analog of test_offline_launch_fsdp. Drives a RolloutWorker (HF target,
no sglang) to materialize SampleRefs into the mem:// store, then trains through
the queue with FSDP + gradient accumulation. Asserts:
- rollout produced one ref per prompt and the controller carries no tensors;
- the strategy runs forward through the FSDP-wrapped module;
- fit's global_step counts OPTIMIZER steps (micro_step = ACC * optimizer steps).

The old-vs-new bit-exact loss equivalence is covered by test_equiv_online_eagle3.
GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "online launcher path requires CUDA")
class TestOnlineLaunch(unittest.TestCase):
    def test_online_rollout_then_fsdp_train(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29568")

        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import assert_no_tensors
        from specforge.runtime.launch import build_online_eagle3_runtime

        H, V, SEQ, TTT, ACC, MAX_OPT_STEPS, N = fx.H, fx.V, 12, 3, 2, 2, 8
        workdir = tempfile.mkdtemp(prefix="online_launch_")

        # HF target (no sglang) + a fresh eagle3 draft model
        target, _dir, aux_ids = fx.build_hf_target(workdir, hidden=H, layers=8, vocab=V)
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        draft = AutoEagle3DraftModel.from_config(
            AutoDraftModelConfig.from_file(cfg),
            attention_backend="flex_attention",
            torch_dtype=torch.bfloat16,
        ).cuda()
        draft.load_vocab_mapping(vocab_path)
        draft.freeze_embedding()
        eagle3_model = OnlineEagle3Model(
            draft, length=TTT, attention_backend="flex_attention"
        ).cuda()

        # N metadata-only prompts (control plane carries no tensors)
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

        trainer, loader, workers, controller, drive_rollout = (
            build_online_eagle3_runtime(
                target_model=target,
                prompts=prompts,
                eagle3_model=eagle3_model,
                optimizer_factory=optimizer_factory,
                run_id="online-launch",
                output_dir=os.path.join(workdir, "out"),
                target_hidden_size=H,
                target_vocab_size=V,
                target_repr="logits",
                aux_hidden_state_layer_ids=tuple(aux_ids),
                batch_size=1,
                accumulation_steps=ACC,
                num_epochs=1,
                max_steps=MAX_OPT_STEPS,
            )
        )

        # Rollout produces exactly one SampleRef per prompt, on the queue.
        produced = drive_rollout()
        self.assertEqual(produced, N)
        self.assertEqual(controller.sample_queue.depth(), N)
        # control plane carries metadata only
        assert_no_tensors(controller.status())

        # FSDP must be in the forward path; optimizer exists.
        module = trainer.core.strategy.trainable_module()
        self.assertIsInstance(
            module, FSDP, "strategy must hold the FSDP-wrapped module"
        )
        self.assertIsNotNone(trainer.core.backend.optimizer)

        step = trainer.fit(loader)

        # global_step counts OPTIMIZER steps; micro_step counts micro-batches.
        self.assertEqual(step, MAX_OPT_STEPS)
        self.assertEqual(trainer.global_step, MAX_OPT_STEPS)
        self.assertEqual(trainer.micro_step, ACC * MAX_OPT_STEPS)


if __name__ == "__main__":
    unittest.main(verbosity=2)
