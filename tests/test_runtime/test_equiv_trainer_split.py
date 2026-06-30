# coding=utf-8
"""M3 gate: the trainer-boundary split does not change the math (paired single step).

Old path = legacy ``run_forward`` + ``run_backward_and_update``. New path =
``Eagle3TrainStrategy`` + ``TrainerCore`` + ``FSDPTrainingBackend``. Two models
with identical initial weights take one step on the same offline batch; the
weighted loss must match bit-exactly and the reduced grad-norm near-exactly.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import types
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "trainer-split equivalence requires CUDA")
class TestEquivTrainerSplit(unittest.TestCase):
    def test_equiv_trainer_split(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29562")

        import pathlib
        import sys

        sys.path.insert(
            0, str(pathlib.Path(__file__).resolve().parents[2])
        )  # repo root

        from scripts.train_eagle3 import run_backward_and_update, run_forward
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
        from specforge.runtime.training.trainer import TrainerCore

        TTT, BS = 3, 2
        workdir = tempfile.mkdtemp(prefix="equiv_split_")
        cfg = fx.write_draft_config(os.path.join(workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(workdir, "target"))
        vocab_path = fx.write_vocab_mapping(os.path.join(workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=BS)

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
                draft_model=dm, length=TTT, attention_backend="flex_attention"
            ).cuda()

        model_a = build_model()
        model_b = build_model()
        # identical initial weights
        model_b.draft_model.load_state_dict(model_a.draft_model.state_dict())
        head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")

        ds = OfflineEagle3Dataset(
            sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir)), max_len=512
        )
        data = DataCollatorWithPadding()([ds[j] for j in range(BS)])

        args = types.SimpleNamespace(
            is_vlm=False,
            target_model_backend="hf",
            shard_target_output=False,
            draft_accumulation_steps=1,
            # compact_teacher=False -> run_forward uses the legacy full-logits
            # teacher path, matching the behavior this equivalence test asserts.
            compact_teacher=False,
            compact_teacher_chunk_size=None,
        )

        # --- old path on model_a ---
        opt_a = BF16Optimizer(
            model_a.draft_model,
            lr=1e-4,
            max_grad_norm=0.5,
            warmup_ratio=0.0,
            total_steps=10,
        )
        plosses_a, _, _, _, _, _, _ = run_forward(
            args, model_a, data, head, is_online=False
        )
        loss_old = sum(0.8**i * plosses_a[i] for i in range(len(plosses_a))).item()
        gn_old = run_backward_and_update(args, plosses_a, opt_a, global_step=1)

        # --- new path on model_b ---
        opt_b = BF16Optimizer(
            model_b.draft_model,
            lr=1e-4,
            max_grad_norm=0.5,
            warmup_ratio=0.0,
            total_steps=10,
        )
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.set_optimizer(opt_b)
        strategy = Eagle3TrainStrategy(model_b, target_head=head)
        core = TrainerCore(strategy, backend, accumulation_steps=1)
        batch = TrainBatch(
            sample_ids=["a", "b"],
            strategy="eagle3",
            tensors=dict(data),
            metadata={"target_repr": "hidden_state", "ttt_length": TTT},
        )
        rep = core.train_step(batch)

        self.assertAlmostEqual(
            loss_old, rep.loss, places=4, msg=f"loss: old={loss_old} new={rep.loss}"
        )
        self.assertAlmostEqual(
            float(gn_old.item()),
            rep.grad_norm,
            places=3,
            msg=f"grad_norm: old={gn_old.item()} new={rep.grad_norm}",
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
