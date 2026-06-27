# coding=utf-8
"""M3 gate: checkpoint save -> resume restores weights and step (single rank).

Trains a few offline steps through the new TrainerController, saves a checkpoint,
reloads the draft state into a fresh model, and asserts the persisted draft
weights and global step round-trip exactly.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "checkpoint resume requires CUDA")
class TestCheckpointResume(unittest.TestCase):
    def test_checkpoint_resume(self):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29563")

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.data.preprocessing import OfflineEagle3Dataset
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.modeling.target import TargetHead
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.runtime.training.backend import (
            FSDPTrainingBackend,
            ParallelConfig,
        )
        from specforge.runtime.training.strategy import Eagle3TrainStrategy
        from specforge.runtime.training.trainer import TrainerController, TrainerCore

        TTT, BS, N = 3, 2, 6
        workdir = tempfile.mkdtemp(prefix="ckpt_resume_")
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(workdir, "target"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)
        out_dir = os.path.join(workdir, "out")

        draft_config = AutoDraftModelConfig.from_file(cfg)

        def build_model():
            dm = AutoEagle3DraftModel.from_config(
                draft_config,
                attention_backend="flex_attention",
                torch_dtype=torch.bfloat16,
            ).cuda()
            dm.load_vocab_mapping(vocab_path)
            dm.freeze_embedding()
            return OnlineEagle3Model(
                dm, length=TTT, attention_backend="flex_attention"
            ).cuda()

        head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
        ds = OfflineEagle3Dataset(
            sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir)), max_len=512
        )
        collate = DataCollatorWithPadding()

        def make_batches():
            out = []
            for s in range(0, N, BS):
                data = collate([ds[j] for j in range(s, s + BS)])
                out.append(
                    TrainBatch(
                        sample_ids=[str(j) for j in range(s, s + BS)],
                        strategy="eagle3",
                        tensors=dict(data),
                        metadata={"target_repr": "hidden_state", "ttt_length": TTT},
                    )
                )
            return out

        model = build_model()
        opt = BF16Optimizer(
            model.draft_model,
            lr=1e-3,
            max_grad_norm=0.5,
            warmup_ratio=0.0,
            total_steps=10,
        )
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)  # register module (no FSDP at 1 rank)
        backend.set_optimizer(opt)
        strategy = Eagle3TrainStrategy(model, target_head=head)
        core = TrainerCore(strategy, backend)
        ctrl = TrainerController(
            core, run_id="r", output_dir=out_dir, max_steps=3, num_epochs=2
        )
        step = ctrl.fit(make_batches())
        self.assertEqual(step, 3)
        wv = ctrl.save_checkpoint(step)

        # reload into a fresh model and compare persisted (non-embedding) weights
        ckpt = torch.load(
            os.path.join(wv.checkpoint_uri[len("file://") :], "training_state.pt"),
            map_location="cpu",
            weights_only=False,
        )
        self.assertEqual(ckpt["global_step"], 3)
        self.assertEqual(ckpt["strategy"], "eagle3")

        fresh = build_model()
        missing, unexpected = fresh.draft_model.load_state_dict(
            ckpt["draft_state_dict"], strict=False
        )
        self.assertEqual(unexpected, [])  # all persisted keys belong to the draft
        # every persisted weight must now match the trained model bit-for-bit
        trained = strategy.checkpoint_state_filter(backend.state_dict())
        fresh_sd = fresh.draft_model.state_dict()
        for k, v in trained.items():
            self.assertTrue(
                torch.equal(v.cpu(), fresh_sd[k].cpu()), msg=f"weight {k} mismatch"
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
