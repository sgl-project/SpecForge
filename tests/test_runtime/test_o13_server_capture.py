# coding=utf-8
"""O1.3 gate: a LIVE SGLang server feeds eagle3 training with zero precomputed
features, through the reforward transport (server decodes; the in-process
capture engine extracts aux+target over [prompt + completion]).

What is pinned here:
- the live pipeline end-to-end: raw-input_ids HTTP decode on a real server
  (``--skip-tokenizer-init``) -> reforward capture -> a training-shaped
  ``Eagle3TargetOutput`` -> one real train step, no precomputed features;
- frozen-target determinism: the same request twice yields the same tokens,
  and the captured features over the same tokens are bit-identical;
- the loss region covers exactly the generated tokens (online convention).

Token-level decode parity against another engine is deliberately NOT asserted
on the random tiny model (near-uniform logits make argmax ties flaky); the
extraction-correctness gate (`test_extraction_vs_hf_reference`) covers feature
parity, and real-model decode parity belongs to the throughput spike runs.

OPT-IN: launches a real SGLang server (slow, GPU + sglang required). Enable
with SPECFORGE_RUN_SERVER_TESTS=1; the default suite skips it.
"""

import os
import subprocess
import sys
import tempfile
import time
import unittest

import torch

CUDA = torch.cuda.is_available()
ENABLED = os.environ.get("SPECFORGE_RUN_SERVER_TESTS") == "1"
PORT = 30987


@unittest.skipUnless(
    CUDA and ENABLED, "O1.3 server gate: set SPECFORGE_RUN_SERVER_TESTS=1 (GPU)"
)
class TestO13ServerCapture(unittest.TestCase):
    server = None
    workdir = None
    target_dir = None

    @classmethod
    def setUpClass(cls):
        from transformers import LlamaConfig, LlamaForCausalLM

        cls.workdir = tempfile.mkdtemp(prefix="o13_")
        cfg = LlamaConfig(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=8,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=256,
            max_position_embeddings=512,
            rms_norm_eps=1e-5,
            tie_word_embeddings=False,
        )
        torch.manual_seed(1234)
        model = LlamaForCausalLM(cfg).to(torch.bfloat16)
        cls.target_dir = os.path.join(cls.workdir, "target")
        model.save_pretrained(cls.target_dir)

        env = dict(os.environ, FLASHINFER_DISABLE_VERSION_CHECK="1")
        cls.server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                cls.target_dir,
                "--skip-tokenizer-init",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.3",
                "--port",
                str(PORT),
            ],
            stdout=open(os.path.join(cls.workdir, "server.log"), "w"),
            stderr=subprocess.STDOUT,
            env=env,
        )
        import requests

        deadline = time.time() + 300
        while time.time() < deadline:
            try:
                if requests.get(f"http://localhost:{PORT}/health", timeout=5).ok:
                    return
            except requests.RequestException:
                pass
            if cls.server.poll() is not None:
                break
            time.sleep(5)
        raise RuntimeError(
            f"sglang server did not become healthy; see {cls.workdir}/server.log"
        )

    @classmethod
    def tearDownClass(cls):
        if cls.server is not None and cls.server.poll() is None:
            cls.server.terminate()
            try:
                cls.server.wait(timeout=30)
            except subprocess.TimeoutExpired:
                cls.server.kill()

    def test_live_server_feeds_training(self):
        from tests.test_runtime import _fixtures as fx

        # the in-process capture engine reads the TP group at construction
        fx.build_single_rank_distributed(port="29574")

        from specforge.inference.target_engine import get_eagle3_target_model

        engine = get_eagle3_target_model(
            self.target_dir,
            backend="sglang_server",
            base_url=f"http://localhost:{PORT}",
            capture_backend="hf",
            torch_dtype=torch.bfloat16,
            device="cuda",
            max_new_tokens=6,
        )
        engine.set_aux_hidden_states_layers([1, 3, 4])
        self.assertTrue(engine.health())

        # two prompts, ragged lengths, right-padded
        rows = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14]]
        maxlen = max(len(r) for r in rows)
        input_ids = torch.zeros(2, maxlen, dtype=torch.long)
        attn = torch.zeros(2, maxlen, dtype=torch.long)
        for i, r in enumerate(rows):
            input_ids[i, : len(r)] = torch.tensor(r)
            attn[i, : len(r)] = 1
        loss_mask = torch.zeros_like(attn)

        out1 = engine.capture(
            input_ids=input_ids.cuda(), attention_mask=attn.cuda(), loss_mask=loss_mask
        )
        out2 = engine.capture(
            input_ids=input_ids.cuda(), attention_mask=attn.cuda(), loss_mask=loss_mask
        )

        # frozen target: the live stream is reproducible, features bit-identical
        self.assertTrue(torch.equal(out1.input_ids, out2.input_ids))
        self.assertTrue(torch.equal(out1.hidden_states, out2.hidden_states))

        # shapes: aux = 3 layers concat; sequences grew by the completion
        H = 64
        self.assertEqual(out1.hidden_states.shape[-1], 3 * H)
        self.assertGreater(out1.input_ids.shape[1], maxlen)
        # loss covers the generated region of each row, except the final
        # position (left-shifted target: no next-token teacher there)
        for i, r in enumerate(rows):
            row_loss = out1.loss_mask[i].squeeze(-1)
            self.assertEqual(int(row_loss[: len(r)].sum()), 0)
            self.assertGreater(int(row_loss.sum()), 0)
            last_real = int(out1.attention_mask[i].sum()) - 1
            self.assertEqual(int(row_loss[last_real]), 0)

        # zero precomputed features -> one real train step
        from specforge import (
            AutoDraftModelConfig,
            AutoEagle3DraftModel,
            OnlineEagle3Model,
        )
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import TrainBatch
        from specforge.training.backend import FSDPTrainingBackend, ParallelConfig
        from specforge.training.controller import TrainerCore
        from specforge.training.strategies.base import Eagle3TrainStrategy
        from tests.test_runtime import _fixtures as fx

        torch.manual_seed(0)
        draft_cfg = AutoDraftModelConfig.from_file(
            fx.write_draft_config(os.path.join(self.workdir, "draft.json"))
        )
        dm = AutoEagle3DraftModel.from_config(
            draft_cfg, attention_backend="sdpa", torch_dtype=torch.bfloat16
        ).cuda()
        dm.load_vocab_mapping(
            fx.write_vocab_mapping(os.path.join(self.workdir, "vm.pt"))
        )
        dm.freeze_embedding()
        model = OnlineEagle3Model(dm, length=3, attention_backend="sdpa").cuda()

        backend = FSDPTrainingBackend(ParallelConfig.from_distributed())
        backend.prepare_model(model, wrap=False)
        backend.set_optimizer(
            BF16Optimizer(
                dm, lr=1e-3, max_grad_norm=0.5, warmup_ratio=0.0, total_steps=10
            )
        )
        core = TrainerCore(Eagle3TrainStrategy(model, target_head=None), backend)
        batch = TrainBatch(
            sample_ids=["a", "b"],
            strategy="eagle3",
            tensors={
                "input_ids": out1.input_ids.cuda(),
                "attention_mask": out1.attention_mask.cuda(),
                "loss_mask": out1.loss_mask.cuda(),
                "target": out1.target.cuda(),
                "hidden_state": out1.hidden_states.cuda(),
            },
            metadata={"target_repr": "logits", "ttt_length": 3},
        )
        result = core.train_step(batch)
        self.assertTrue(torch.isfinite(torch.tensor(result.loss)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
