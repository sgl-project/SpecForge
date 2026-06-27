# coding=utf-8
"""M1 exit gate: offline EAGLE3 through SampleRef -> TrainBatch is bit-exact.

Differential equivalence: the same tiny model + same offline feature files are
driven through (a) today's offline ``run_forward`` math and (b) the new
DataFlow path (OfflineManifestReader -> LocalFeatureStore -> FeatureDataLoader ->
TrainBatch -> Eagle3TrainStrategy.forward_loss). The per-batch weighted loss
must match bit-for-bit (tol=0) — this is a plumbing gate, not a correctness gate.

GPU-only (the EAGLE3 draft/flex-attention path requires CUDA). Run on the H200
box via rcli. Uses a synthetic tiny fixture — no model download.
"""

import json
import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


def _build_distributed_single_rank():
    import torch.distributed as dist

    if dist.is_available() and dist.is_initialized():
        return
    os.environ.setdefault("RANK", "0")
    os.environ.setdefault("WORLD_SIZE", "1")
    os.environ.setdefault("LOCAL_RANK", "0")
    os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
    os.environ.setdefault("MASTER_PORT", "29555")
    torch.cuda.set_device(0)
    from specforge.distributed import init_distributed

    init_distributed(timeout=10, tp_size=1, sp_ulysses_size=1, sp_ring_size=1)


TINY_DRAFT_CONFIG = {
    "architectures": ["LlamaForCausalLMEagle3"],
    "bos_token_id": 1,
    "eos_token_id": 2,
    "hidden_act": "silu",
    "hidden_size": 64,
    "initializer_range": 0.02,
    "intermediate_size": 128,
    "max_position_embeddings": 512,
    "model_type": "llama",
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "num_hidden_layers": 1,
    "pad_token_id": 0,
    "rms_norm_eps": 1e-5,
    "tie_word_embeddings": False,
    "torch_dtype": "bfloat16",
    "vocab_size": 256,
    "draft_vocab_size": 64,
}


def _write_target_head_dir(d, hidden, vocab):
    """Write a minimal HF-style dir so TargetHead.from_pretrained works."""
    from safetensors.torch import save_file

    cfg = {
        "architectures": ["LlamaForCausalLM"],
        "model_type": "llama",
        "hidden_size": hidden,
        "vocab_size": vocab,
        "num_hidden_layers": 1,
        "num_attention_heads": 4,
        "intermediate_size": 128,
    }
    with open(os.path.join(d, "config.json"), "w") as f:
        json.dump(cfg, f)
    lm_head = torch.randn(vocab, hidden, dtype=torch.float32)
    save_file({"lm_head.weight": lm_head}, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "model.safetensors.index.json"), "w") as f:
        json.dump(
            {"metadata": {}, "weight_map": {"lm_head.weight": "model.safetensors"}}, f
        )


def _write_vocab_mapping(path, vocab, draft_vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    draft_ids = torch.randperm(vocab, generator=g)[:draft_vocab].sort().values
    t2d = torch.zeros(vocab, dtype=torch.bool)
    t2d[draft_ids] = True
    d2t = draft_ids - torch.arange(draft_vocab)
    torch.save({"t2d": t2d, "d2t": d2t.to(torch.int64)}, path)


def _write_offline_files(d, n, seq, hidden, vocab, seed=0):
    g = torch.Generator().manual_seed(seed)
    for i in range(n):
        torch.save(
            {
                "input_ids": torch.randint(0, vocab, (seq,), generator=g),
                "loss_mask": torch.ones(seq, dtype=torch.long),
                # prepared offline features are bf16 (target ran in bf16)
                "hidden_state": torch.randn(1, seq, hidden, generator=g).to(
                    torch.bfloat16
                ),
                "aux_hidden_state": torch.randn(1, seq, 3 * hidden, generator=g).to(
                    torch.bfloat16
                ),
            },
            os.path.join(d, f"{i:04d}.ckpt"),
        )


@unittest.skipUnless(CUDA, "offline EAGLE3 equivalence requires CUDA")
class TestEquivOfflineEagle3(unittest.TestCase):
    def test_equiv_offline_eagle3(self):
        torch.manual_seed(0)
        torch.use_deterministic_algorithms(True, warn_only=True)
        _build_distributed_single_rank()

        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.data.preprocessing import OfflineEagle3Dataset
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.modeling.target import TargetHead
        from specforge.runtime.control_plane import DataFlowController
        from specforge.runtime.data_plane import (
            FeatureDataLoader,
            LocalFeatureStore,
            OfflineManifestReader,
        )
        from specforge.runtime.training.strategy import Eagle3TrainStrategy

        H, V, D, SEQ, N, BS, TTT = 64, 256, 64, 16, 4, 2, 3

        workdir = tempfile.mkdtemp(prefix="equiv_offline_")
        cfg_path = os.path.join(workdir, "draft.json")
        with open(cfg_path, "w") as f:
            json.dump(TINY_DRAFT_CONFIG, f)
        target_dir = os.path.join(workdir, "target")
        os.makedirs(target_dir)
        _write_target_head_dir(target_dir, H, V)
        vocab_path = os.path.join(workdir, "vocab_mapping.pt")
        _write_vocab_mapping(vocab_path, V, D)
        feat_dir = os.path.join(workdir, "features")
        os.makedirs(feat_dir)
        _write_offline_files(feat_dir, N, SEQ, H, V)

        # one shared model + head used by both paths
        draft_config = AutoDraftModelConfig.from_file(cfg_path)
        draft_model = AutoEagle3DraftModel.from_config(
            draft_config, attention_backend="flex_attention", torch_dtype=torch.bfloat16
        ).cuda()
        draft_model.load_vocab_mapping(vocab_path)
        draft_model.freeze_embedding()
        target_head = TargetHead.from_pretrained(
            target_dir, lm_head_key="lm_head.weight"
        )
        eagle3_model = OnlineEagle3Model(
            draft_model=draft_model, length=TTT, attention_backend="flex_attention"
        ).cuda()
        eagle3_model.eval()

        ploss_decay = 0.8

        @torch.no_grad()
        def old_path_losses():
            ds = OfflineEagle3Dataset(
                sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir)),
                max_len=512,
            )
            collate = DataCollatorWithPadding()
            losses = []
            for start in range(0, N, BS):
                if start + BS > N:
                    break
                batch = collate([ds[j] for j in range(start, start + BS)])
                input_ids, target, loss_mask = target_head.preprocess(
                    batch["input_ids"], batch["target"], batch["loss_mask"]
                )
                target = target_head(target.cuda())
                plosses, _, _, _, _, _, _ = eagle3_model(
                    input_ids=input_ids.cuda(),
                    attention_mask=batch["attention_mask"].cuda(),
                    loss_mask=loss_mask.cuda(),
                    target=target,
                    hidden_states=batch["hidden_state"].cuda(),
                )
                w = [ploss_decay**i for i in range(len(plosses))]
                losses.append(
                    sum(w[i] * plosses[i] for i in range(len(plosses))).item()
                )
            return losses

        @torch.no_grad()
        def new_path_losses():
            ctrl = DataFlowController("equiv-offline")
            ctrl.enqueue_offline_refs(
                OfflineManifestReader(
                    feat_dir, run_id="equiv-offline", ttt_length=TTT, max_len=512
                ).read()
            )
            store = LocalFeatureStore("equiv")
            loader = FeatureDataLoader(
                store,
                ctrl.sample_queue,
                batch_size=BS,
                collate_fn=DataCollatorWithPadding(),
                per_sample_transform=lambda raw: OfflineEagle3Dataset.process_data(
                    raw, 512
                ),
                drop_last=True,
            )
            strategy = Eagle3TrainStrategy(eagle3_model, target_head=target_head)
            losses = []
            for batch in loader:
                out = strategy.forward_loss(batch)
                losses.append(out.loss.item())
                ctrl.ack_train_refs("t0", batch.sample_ids)
            return losses

        old = old_path_losses()
        new = new_path_losses()
        self.assertEqual(len(old), len(new))
        self.assertEqual(len(old), N // BS)
        for i, (a, b) in enumerate(zip(old, new)):
            self.assertEqual(a, b, f"batch {i}: old={a} new={b} (must be bit-exact)")


if __name__ == "__main__":
    unittest.main(verbosity=2)
