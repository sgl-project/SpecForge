# coding=utf-8
"""Server-capture gate: a LIVE patched SGLang server writes features into a
LIVE Mooncake store; the data plane consumes them zero-copy and trains.

What is pinned here:
- the whole zero-copy transport end-to-end: ``/generate`` with ``spec_capture``
  -> engine-side sink put (SpecForge key layout) -> ``SampleRef`` from
  ``meta_info`` only -> ``MooncakeFeatureStore.get`` reads the server-written
  bytes -> one real eagle3 train step (``target_repr="hidden_state"``, the
  frozen TargetHead recomputes logits);
- extraction correctness: the server-captured aux hidden states match an
  independent HF ``output_hidden_states=True`` forward at the configured
  layers, and ``lm_head(last_hidden)`` matches the HF logits (norm-placement
  agnostic), within the documented bf16 tolerance;
- strategy-agnosticism: the same server serves eagle3 (aux + last_hidden) and
  dflash (aux only) requests, named per strategy by the client schema.

OPT-IN: needs a GPU, sglang patched with
``patches/sglang/v0.5.14/spec-capture.patch`` (see
``scripts/apply_sglang_spec_capture_patch.sh``), the ``mooncake`` package, and
a reachable/spawnable ``mooncake_master``. Enable with
``SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1``.
"""

import importlib.util
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

import torch

CUDA = torch.cuda.is_available()
ENABLED = os.environ.get("SPECFORGE_RUN_SERVER_CAPTURE_TESTS") == "1"
PORT = 30989
AUX_LAYER_IDS = [1, 3, 4]
H, TOL = 64, 2e-2  # fixture hidden size; documented bf16 tolerance


def _patched_sglang() -> bool:
    return importlib.util.find_spec("sglang.srt.spec_capture_sink") is not None


def _mooncake_available() -> bool:
    return importlib.util.find_spec("mooncake.store") is not None


@unittest.skipUnless(
    CUDA and ENABLED,
    "server-capture gate: set SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1 (GPU)",
)
class TestServerCaptureGate(unittest.TestCase):
    server = None
    master = None
    workdir = None
    target_dir = None

    @classmethod
    def setUpClass(cls):
        if not _patched_sglang():
            raise unittest.SkipTest(
                "installed sglang lacks spec_capture_sink — apply "
                "patches/sglang/v0.5.14/spec-capture.patch "
                "(scripts/apply_sglang_spec_capture_patch.sh)"
            )
        if not _mooncake_available():
            raise unittest.SkipTest("mooncake package not installed")

        from transformers import LlamaConfig, LlamaForCausalLM

        cls.workdir = tempfile.mkdtemp(prefix="spec_capture_gate_")
        cfg = LlamaConfig(
            hidden_size=H,
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
        # save_pretrained writes a single un-indexed shard for tiny models;
        # TargetHead.load_weights locates lm_head.weight via the index file.
        index = os.path.join(cls.target_dir, "model.safetensors.index.json")
        if not os.path.exists(index):
            import json

            with open(index, "w") as f:
                json.dump(
                    {
                        "metadata": {},
                        "weight_map": {
                            "lm_head.weight": "model.safetensors",
                        },
                    },
                    f,
                )

        cls._ensure_mooncake_master()

        env = dict(
            os.environ,
            FLASHINFER_DISABLE_VERSION_CHECK="1",
            MOONCAKE_LOCAL_HOSTNAME=os.environ.get(
                "MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1"
            ),
        )
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
                "--chunked-prefill-size",
                "-1",
                "--enable-spec-capture",
                "--spec-capture-aux-layer-ids",
                *[str(i) for i in AUX_LAYER_IDS],
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
    def _ensure_mooncake_master(cls):
        """Use an externally provided master, else spawn one locally."""
        if os.environ.get("MOONCAKE_MASTER_SERVER_ADDR"):
            return
        binary = shutil.which("mooncake_master")
        if binary is None:
            raise unittest.SkipTest(
                "no MOONCAKE_MASTER_SERVER_ADDR in env and no mooncake_master "
                "binary on PATH"
            )
        cls.master = subprocess.Popen(
            [binary, "--enable-http-metadata-server=true"],
            stdout=open(os.path.join(cls.workdir, "mooncake_master.log"), "w"),
            stderr=subprocess.STDOUT,
        )
        time.sleep(3)
        if cls.master.poll() is not None:
            raise unittest.SkipTest(
                f"mooncake_master exited at startup; see "
                f"{cls.workdir}/mooncake_master.log"
            )
        os.environ.setdefault("MOONCAKE_MASTER_SERVER_ADDR", "127.0.0.1:50051")
        os.environ.setdefault(
            "MOONCAKE_METADATA_SERVER", "http://127.0.0.1:8080/metadata"
        )
        os.environ.setdefault("MOONCAKE_LOCAL_HOSTNAME", "127.0.0.1")
        os.environ.setdefault("MOONCAKE_PROTOCOL", "tcp")

    @classmethod
    def tearDownClass(cls):
        for proc in (cls.server, cls.master):
            if proc is not None and proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    proc.kill()

    # -- helpers ---------------------------------------------------------------
    def _store(self, store_id):
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        return MooncakeFeatureStore(
            store_id=store_id,
            setup_kwargs={
                "local_hostname": os.environ["MOONCAKE_LOCAL_HOSTNAME"],
                "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
                "global_segment_size": 1 << 28,
                "local_buffer_size": 1 << 28,
                "protocol": os.environ.get("MOONCAKE_PROTOCOL", "tcp"),
                "rdma_devices": "",
                "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            },
        )

    def _tasks(self, rows):
        from specforge.runtime.contracts import PromptTask

        return [
            PromptTask(
                task_id=f"t{i}",
                run_id="gate0",
                source_id="gate",
                payload={
                    "input_ids": list(r),
                    "loss_mask": [0] + [1] * (len(r) - 2) + [0],
                },
                max_length=len(r),
            )
            for i, r in enumerate(rows)
        ]

    def _hf_reference(self, rows):
        """Independent HF forward: per-row aux cat + full logits."""
        from transformers import AutoModelForCausalLM

        model = (
            AutoModelForCausalLM.from_pretrained(
                self.target_dir, torch_dtype=torch.bfloat16
            )
            .cuda()
            .eval()
        )
        aux, logits = [], []
        with torch.no_grad():
            for r in rows:
                ids = torch.tensor([r], dtype=torch.long, device="cuda")
                out = model(input_ids=ids, output_hidden_states=True, use_cache=False)
                # hidden_states[0]=embeddings; [k]=output of layer k-1, so
                # "capture layer id L" (sglang convention) == hidden_states[L+1]
                aux.append(
                    torch.cat(
                        [out.hidden_states[i + 1] for i in AUX_LAYER_IDS], dim=-1
                    ).cpu()
                )
                logits.append(out.logits.cpu())
        del model
        torch.cuda.empty_cache()
        return aux, logits

    # -- tests -------------------------------------------------------------------
    def test_eagle3_zero_copy_end_to_end(self):
        from specforge.inference.adapters.server_capture import (
            SGLangServerCaptureAdapter,
        )
        from specforge.inference.capture import FeatureContract
        from specforge.runtime.contracts import SampleRef

        rows = [[5, 6, 7, 8, 9, 10], [11, 12, 13, 14]]
        store = self._store("gate-eagle3")
        adapter = SGLangServerCaptureAdapter(
            f"http://localhost:{PORT}", store, run_id="gate0", strategy="eagle3"
        )
        contract = FeatureContract.from_strategy(
            required_features={
                "input_ids",
                "attention_mask",
                "loss_mask",
                "hidden_state",
                "target",
            },
            aux_hidden_state_layer_ids=tuple(AUX_LAYER_IDS),
            target_repr="hidden_state",
            target_hidden_size=H,
        )
        refs = adapter.produce_refs(self._tasks(rows), capture=contract)
        for ref in refs:
            self.assertIsInstance(
                ref, SampleRef, f"expected a ref, got failure: {ref}"
            )

        aux_ref, logits_ref = self._hf_reference(rows)
        from specforge.modeling.target.target_head import TargetHead

        head = TargetHead.from_pretrained(self.target_dir)

        fetched = []
        for i, ref in enumerate(refs):
            out, handle = store.get(ref)
            fetched.append(out)
            length = len(rows[i])
            self.assertEqual(
                out["hidden_state"].shape, (1, length, len(AUX_LAYER_IDS) * H)
            )
            self.assertEqual(out["target"].shape, (1, length, H))
            self.assertEqual(out["input_ids"].tolist(), [rows[i]])
            self.assertEqual(out["loss_mask"].shape, (1, length))
            # extraction correctness vs the independent HF forward
            torch.testing.assert_close(
                out["hidden_state"].float(),
                aux_ref[i].float(),
                rtol=TOL,
                atol=TOL,
            )
            with torch.no_grad():
                recomputed = head(out["target"].cuda()).cpu().float()
            torch.testing.assert_close(
                recomputed, logits_ref[i].float(), rtol=TOL, atol=TOL
            )
            store.release(handle)

        # one real train step from the server-captured features
        self._train_step(fetched, head)

    def _train_step(self, fetched, head):
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

        fx.build_single_rank_distributed(port="29576")
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
        core = TrainerCore(
            Eagle3TrainStrategy(model, target_head=head), backend
        )
        # batch_size=1 batches (ragged lengths), one step each
        for i, out in enumerate(fetched):
            batch = TrainBatch(
                sample_ids=[f"s{i}"],
                strategy="eagle3",
                tensors={
                    "input_ids": out["input_ids"].cuda(),
                    "attention_mask": out["attention_mask"].cuda(),
                    "loss_mask": out["loss_mask"].cuda(),
                    "target": out["target"].cuda(),
                    "hidden_state": out["hidden_state"].cuda(),
                },
                metadata={"target_repr": "hidden_state", "ttt_length": 3},
            )
            result = core.train_step(batch)
            self.assertTrue(torch.isfinite(torch.tensor(result.loss)))

    def test_dflash_capture_same_server(self):
        from specforge.inference.adapters.server_capture import (
            SGLangServerCaptureAdapter,
        )
        from specforge.inference.capture import FeatureContract
        from specforge.runtime.contracts import SampleRef

        rows = [[3, 1, 4, 1, 5]]
        store = self._store("gate-dflash")
        adapter = SGLangServerCaptureAdapter(
            f"http://localhost:{PORT}", store, run_id="gate1", strategy="dflash"
        )
        contract = FeatureContract.from_strategy(
            required_features={"input_ids", "hidden_states", "loss_mask"},
            aux_hidden_state_layer_ids=tuple(AUX_LAYER_IDS),
            target_repr="hidden_state",
            target_hidden_size=H,
        )
        (ref,) = adapter.produce_refs(self._tasks(rows), capture=contract)
        self.assertIsInstance(ref, SampleRef, f"expected a ref, got: {ref}")
        out, handle = store.get(ref)
        self.assertEqual(
            sorted(out), ["hidden_states", "input_ids", "loss_mask"]
        )
        self.assertEqual(
            out["hidden_states"].shape, (1, 5, len(AUX_LAYER_IDS) * H)
        )
        aux_ref, _ = self._hf_reference(rows)
        torch.testing.assert_close(
            out["hidden_states"].float(), aux_ref[0].float(), rtol=TOL, atol=TOL
        )
        store.release(handle)


if __name__ == "__main__":
    unittest.main(verbosity=2)
