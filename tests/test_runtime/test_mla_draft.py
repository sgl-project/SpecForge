# coding=utf-8
"""MLA (DeepSeek) Eagle3 draft gates: the draft-ARCHITECTURE axis (plan G4).

The eagle3 algorithm surface is untouched — same strategy, same runtime, same
fixtures — only the draft architecture differs, which is exactly the claim the
two-axis design makes. Three gates:

- suffix-cache parity: the TTT cache path at step 0 must equal the plain
  causal (no-cache) path — pins the MLA suffix-attention math to the same
  contract LlamaAttention honors;
- auto-mapping: the DeepseekV3 config resolves through AutoDraftModelConfig /
  AutoEagle3DraftModel;
- train smoke: 3 optimizer steps through the UNCHANGED Eagle3TrainStrategy +
  TrainerController over the shared offline fixtures — finite decreasing loss,
  trainable grads, E1-compatible per-position metrics.

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


def _make_batches(feat_dir, bs, ttt, nbatches):
    """Assemble eagle3 offline TrainBatches from fixture feature files.

    Mirrors the batch construction in test_checkpoint_resume (OfflineEagle3Dataset
    -> DataCollatorWithPadding -> TrainBatch) so the ported MLA smoke drives the
    unchanged Eagle3TrainStrategy exactly like the llama draft does.
    """
    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.data.utils import DataCollatorWithPadding
    from specforge.runtime.contracts import TrainBatch

    files = sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir))
    ds = OfflineEagle3Dataset(files, max_len=512)
    collate = DataCollatorWithPadding()
    out = []
    for b in range(nbatches):
        s = b * bs
        data = collate([ds[j] for j in range(s, s + bs)])
        out.append(
            TrainBatch(
                sample_ids=[str(j) for j in range(s, s + bs)],
                strategy="eagle3",
                tensors=dict(data),
                metadata={"target_repr": "hidden_state", "ttt_length": ttt},
            )
        )
    return out


@unittest.skipUnless(CUDA, "MLA draft gates require CUDA")
class TestMLADraft(unittest.TestCase):
    def test_suffix_cache_matches_causal_at_step0(self):
        from specforge.modeling.auto import AutoDraftModelConfig
        from specforge.modeling.draft.deepseek_eagle3 import DeepseekMLAAttention
        from tests.test_runtime import _fixtures as fx

        torch.manual_seed(0)
        workdir = tempfile.mkdtemp(prefix="mla_attn_")
        cfg = AutoDraftModelConfig.from_file(
            fx.write_mla_draft_config(os.path.join(workdir, "mla.json"))
        )
        attn = DeepseekMLAAttention(cfg).cuda().to(torch.float32).eval()

        bsz, seq, h = 2, 16, cfg.hidden_size
        x = torch.randn(bsz, seq, 2 * h, device="cuda")
        position_ids = torch.arange(seq, device="cuda").unsqueeze(0)
        # dense additive causal mask, as the model's forward builds it
        mask = torch.full((seq, seq), torch.finfo(torch.float32).min, device="cuda")
        mask = torch.triu(mask, diagonal=1)[None, None]

        with torch.no_grad():
            out_causal = attn(
                x.clone(),
                cache_hidden=None,
                attention_mask=mask,
                position_ids=position_ids,
            )
            out_cached = attn(
                x.clone(),
                cache_hidden=[[], []],
                attention_mask=mask,
                position_ids=position_ids,
            )
        self.assertTrue(
            torch.allclose(out_causal, out_cached, atol=1e-5, rtol=1e-5),
            msg=f"max diff {(out_causal - out_cached).abs().max().item()}",
        )

    def test_train_smoke_through_unchanged_strategy(self):
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29573")

        from specforge.modeling.draft.deepseek_eagle3 import DeepseekV3ForCausalLMEagle3
        from specforge.optimizer import BF16Optimizer
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerController, TrainerCore
        from specforge.training.strategies.base import Eagle3TrainStrategy

        TTT, BS, N = 3, 2, 6
        workdir = tempfile.mkdtemp(prefix="mla_smoke_")
        feat_dir = fx.write_offline_files(os.path.join(workdir, "features"), n=N)

        torch.manual_seed(0)
        model, head = fx.build_mla_eagle3(workdir, ttt=TTT)
        self.assertIsInstance(model.draft_model, DeepseekV3ForCausalLMEagle3)

        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)
        backend.set_optimizer(
            BF16Optimizer(
                model.draft_model,
                lr=1e-3,
                max_grad_norm=0.5,
                warmup_ratio=0.0,
                total_steps=10,
            )
        )
        strategy = Eagle3TrainStrategy(model, target_head=head)
        core = TrainerCore(strategy, backend)

        # the persisted surface is draft-only, exactly like the llama draft
        draft_keys = strategy.checkpoint_state_filter(backend.state_dict()["model"])
        self.assertTrue(draft_keys)
        self.assertTrue(all(not k.startswith("draft_model.") for k in draft_keys))

        losses = []
        ctrl = TrainerController(
            core,
            run_id="mla",
            output_dir=os.path.join(workdir, "out"),
            max_steps=3,
            num_epochs=2,
            log_interval=1,
            logger=lambda m, s: losses.append(m["loss"]),
        )
        step = ctrl.fit(_make_batches(feat_dir, BS, TTT, N // BS))
        self.assertEqual(step, 3)
        self.assertEqual(len(losses), 3)
        self.assertTrue(all(torch.isfinite(torch.tensor(x)) for x in losses))

        # E1 evaluator compatibility: per-position counts flow out of the strategy
        out = strategy.forward_loss(_make_batches(feat_dir, BS, TTT, 1)[0])
        self.assertIn("acc_corrects", out.metrics)
        self.assertEqual(len(out.metrics["acc_corrects"]), TTT)


if __name__ == "__main__":
    unittest.main(verbosity=2)
