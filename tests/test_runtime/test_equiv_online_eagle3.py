# coding=utf-8
"""M2 gate: online EAGLE3 through PromptTask -> SampleRef -> TrainBatch matches.

Old path = legacy online ``run_forward`` (target.generate_eagle3_data -> eagle3
model). New path = PromptTask -> SGLangAdapter (via RolloutWorker) -> FeatureStore
-> FeatureDataLoader -> TrainBatch -> Eagle3TrainStrategy. Single rank, batch
size 1 (so old and new run the identical per-sample target forward): the weighted
loss must match near-exactly, and the controller never carries a tensor.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "online EAGLE3 equivalence requires CUDA")
class TestEquivOnlineEagle3(unittest.TestCase):
    def test_equiv_online_eagle3(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29565")

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.runtime.contracts import assert_no_tensors
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane import FeatureDataLoader, LocalFeatureStore
        from specforge.runtime.inference.capture import CaptureConfig
        from specforge.runtime.inference.rollout_worker import RolloutWorker
        from specforge.runtime.inference.sglang_adapter import SGLangAdapter
        from specforge.runtime.training.strategy import Eagle3TrainStrategy

        H, V, SEQ, TTT = fx.H, fx.V, 12, 3
        workdir = tempfile.mkdtemp(prefix="equiv_online_")
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
        eagle3_model.eval()

        torch.manual_seed(11)
        ids = torch.randint(0, V, (SEQ,)).tolist()

        # --- old online path (batch size 1, tp=1 so no TP sharding) ---
        @torch.no_grad()
        def old_loss():
            input_ids = torch.tensor([ids], device="cuda")
            attn = torch.ones_like(input_ids)
            loss_mask = torch.ones_like(input_ids)
            d = target.generate_eagle3_data(input_ids, attn, loss_mask)
            plosses, _, _, _, _, _, _ = eagle3_model(
                input_ids=d.input_ids,
                attention_mask=d.attention_mask,
                loss_mask=d.loss_mask,
                target=d.target,
                hidden_states=d.hidden_states,
            )
            return sum(0.8**i * plosses[i] for i in range(len(plosses))).item()

        # --- new dataflow path ---
        @torch.no_grad()
        def new_loss():
            ctrl = DataFlowController("online")
            ctrl.ingest_prompts(
                [{"payload": {"input_ids": ids, "loss_mask": [1] * SEQ}}]
            )
            store = LocalFeatureStore("online")
            adapter = SGLangAdapter(target, device="cuda")
            capture = CaptureConfig.from_strategy(
                required_features=Eagle3TrainStrategy.required_features,
                aux_hidden_state_layer_ids=tuple(aux_ids),
                target_repr="logits",
                target_hidden_size=H,
                target_vocab_size=V,
            )
            worker = RolloutWorker(ctrl, store, adapter, capture, run_id="online")
            worker.start()
            refs = worker.run_once(max_tasks=4)
            assert len(refs) == 1
            # controller holds only metadata
            assert_no_tensors(ctrl.status())
            for r in refs:
                assert_no_tensors(r)

            def cat_collate(feats):
                return {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}

            loader = FeatureDataLoader(
                store,
                ctrl.sample_queue,
                batch_size=1,
                collate_fn=cat_collate,
            )
            strategy = Eagle3TrainStrategy(eagle3_model, target_head=None)
            losses = []
            for batch in loader:
                self.assertEqual(batch.metadata["target_repr"], "logits")
                losses.append(strategy.forward_loss(batch).loss.item())
            return losses[0]

        old = old_loss()
        new = new_loss()
        self.assertAlmostEqual(
            old, new, places=3, msg=f"online loss: old={old} new={new}"
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
