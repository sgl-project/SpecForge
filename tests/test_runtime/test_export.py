# coding=utf-8
"""E gate: exporters round-trip a DataFlow checkpoint to HF / sglang formats.

Train a few tiny steps, save through the CheckpointManager layout, then:
to_hf output must reload through AutoDraftModel.from_pretrained with
bit-identical draft weights; to_sglang output must carry exactly the serving
key set (no trainer prefixes, no embeddings, t2d/d2t present).

GPU-only. Run on the H200 box via rcli.
"""

import os
import tempfile
import unittest

import torch

CUDA = torch.cuda.is_available()


@unittest.skipUnless(CUDA, "export round-trip requires CUDA")
class TestExporters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.manual_seed(0)
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29591")

        from specforge.core.eagle3 import OnlineEagle3Model
        from specforge.data.preprocessing import OfflineEagle3Dataset
        from specforge.data.utils import DataCollatorWithPadding
        from specforge.modeling.auto import AutoDraftModel, AutoDraftModelConfig
        from specforge.modeling.target.target_head import TargetHead
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerController, TrainerCore
        from specforge.training.strategies.base import Eagle3TrainStrategy

        TTT, BS, N = 3, 2, 4
        cls.workdir = tempfile.mkdtemp(prefix="export_gate_")
        cls.cfg_path = fx.write_draft_config(os.path.join(cls.workdir, "draft.json"))
        target_dir = fx.write_target_head_dir(os.path.join(cls.workdir, "target"))
        cls.vocab_path = fx.write_vocab_mapping(os.path.join(cls.workdir, "vm.pt"))
        feat_dir = fx.write_offline_files(os.path.join(cls.workdir, "features"), n=N)
        cls.out_dir = os.path.join(cls.workdir, "out")

        draft_config = AutoDraftModelConfig.from_file(cls.cfg_path)
        dm = AutoDraftModel.from_config(
            draft_config,
            attention_backend="flex_attention",
            torch_dtype=torch.bfloat16,
        ).cuda()
        dm.load_vocab_mapping(cls.vocab_path)
        dm.freeze_embedding()
        model = OnlineEagle3Model(
            dm, length=TTT, attention_backend="flex_attention"
        ).cuda()
        head = TargetHead.from_pretrained(target_dir, lm_head_key="lm_head.weight")
        ds = OfflineEagle3Dataset(
            sorted(os.path.join(feat_dir, f) for f in os.listdir(feat_dir)),
            max_len=512,
        )
        collate = DataCollatorWithPadding()
        batches = [
            TrainBatch(
                sample_ids=[str(j) for j in range(s, s + BS)],
                strategy="eagle3",
                tensors=dict(collate([ds[j] for j in range(s, s + BS)])),
                metadata={"target_repr": "hidden_state", "ttt_length": TTT},
            )
            for s in range(0, N, BS)
        ]

        opt = BF16Optimizer(
            dm, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
        )
        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)
        backend.set_optimizer(opt)
        strategy = Eagle3TrainStrategy(model, target_head=head)
        ctrl = TrainerController(
            TrainerCore(strategy, backend),
            run_id="exp",
            output_dir=cls.out_dir,
            max_steps=2,
        )
        ctrl.fit(batches)
        cls.ckpt = ctrl.save_checkpoint(ctrl.global_step)
        cls.trained_state = strategy.checkpoint_state_filter(
            backend.state_dict()["model"]
        )

    def test_to_hf_reloads_bit_identical(self):
        # the frozen embedding ships from a REAL source (the target), never
        # a random re-init: build a tiny embedding-bearing source dir.
        import json as _json

        from safetensors.torch import save_file

        from specforge.export import export_to_hf
        from specforge.modeling.auto import AutoDraftModel

        emb_src = os.path.join(self.workdir, "emb_src")
        os.makedirs(emb_src, exist_ok=True)
        emb = torch.randn(256, 64, dtype=torch.float32)
        save_file(
            {"model.embed_tokens.weight": emb},
            os.path.join(emb_src, "model.safetensors"),
        )
        with open(os.path.join(emb_src, "model.safetensors.index.json"), "w") as f:
            _json.dump(
                {
                    "metadata": {},
                    "weight_map": {"model.embed_tokens.weight": "model.safetensors"},
                },
                f,
            )

        out = export_to_hf(
            self.out_dir,  # run root: resolves via the `latest` pointer
            self.cfg_path,
            os.path.join(self.workdir, "hf_export"),
            embedding_source=emb_src,
        )
        reloaded = AutoDraftModel.from_pretrained(out, torch_dtype=torch.bfloat16)
        fresh_sd = reloaded.state_dict()
        # the exported embedding is the source's, not a random re-init
        self.assertTrue(
            torch.equal(
                fresh_sd["embed_tokens.weight"].float().cpu(),
                emb.to(torch.bfloat16).float(),
            )
        )
        for key, value in self.trained_state.items():
            self.assertTrue(
                torch.equal(value.cpu(), fresh_sd[key].cpu()),
                msg=f"weight {key} mismatch after to_hf round-trip",
            )

    def test_to_sglang_produces_exact_serving_keys(self):
        from safetensors.torch import load_file

        from specforge.export import export_to_sglang

        out = export_to_sglang(
            self.ckpt.checkpoint_uri[len("file://") :],  # checkpoint dir form
            self.cfg_path,
            os.path.join(self.workdir, "sglang_export"),
            vocab_mapping_path=self.vocab_path,
        )
        self.assertTrue(os.path.isfile(os.path.join(out, "config.json")))
        weight_file = os.path.join(out, "model.safetensors")
        self.assertTrue(os.path.isfile(weight_file))
        served = load_file(weight_file)

        for required in ("fc.weight", "norm.weight", "lm_head.weight", "t2d", "d2t"):
            self.assertIn(required, served)
        self.assertFalse(any(k.startswith("draft_model.") for k in served))
        self.assertFalse(any("embed" in k.lower() for k in served))
        # weights round-trip bit-identically
        for key, value in self.trained_state.items():
            self.assertTrue(
                torch.equal(value.cpu(), served[key].cpu()),
                msg=f"weight {key} mismatch in sglang export",
            )

    def test_weight_map_renames_are_applied(self):
        from safetensors.torch import load_file

        from specforge.export import export_to_sglang

        out = export_to_sglang(
            self.out_dir,
            self.cfg_path,
            os.path.join(self.workdir, "renamed_export"),
            weight_map={"norm.weight": "final_norm.weight"},
        )
        served = load_file(os.path.join(out, "model.safetensors"))
        self.assertIn("final_norm.weight", served)
        self.assertNotIn("norm.weight", served)

    def test_missing_required_serving_key_fails_loudly(self):
        from specforge.export.to_sglang import _serving_state

        with self.assertRaises(ValueError):
            _serving_state({"fc.weight": torch.zeros(1)}, {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
