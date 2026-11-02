"""Opt-in Qwen3.6 native-weight capture -> Mooncake -> DMTP optimizer step.

Run with SPECFORGE_DMTP_MODEL=/path/to/Qwen3.6-27B and the plugin on PYTHONPATH.
Uses GPU 1 for the target and GPU 2 for the trainer unless overridden. All
services are isolated on ephemeral ports and cleaned up after the test.
"""

import json
import os
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import torch

from tests.utils import terminate_process_trees


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@unittest.skipUnless(
    os.environ.get("SPECFORGE_DMTP_MODEL"),
    "set SPECFORGE_DMTP_MODEL for the native GPU test",
)
class DMTPLiveTest(unittest.TestCase):
    def test_capture_and_train(self):
        import requests

        from specforge.inference.adapters.server_capture import (
            ServerCaptureSchema,
            SGLangServerCaptureAdapter,
        )
        from specforge.inference.capture import CaptureConfig
        from specforge.runtime.contracts import PromptTask, SampleRef
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        model_path = os.environ["SPECFORGE_DMTP_MODEL"]
        work = Path(tempfile.mkdtemp(prefix="dmtp-live-", dir=".codex-remote"))
        print(f"DMTP_LOG_DIR={work}", flush=True)
        rpc, metadata, metrics, port = [free_port() for _ in range(4)]
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=os.environ.get("SPECFORGE_DMTP_SERVER_GPU", "1"),
            SGLANG_PLUGINS="specforge_spec_capture",
            SPECFORGE_SPEC_CAPTURE="1",
            SPECFORGE_MTP_MODEL_PATH=model_path,
            SGLANG_SPEC_CAPTURE_GPU_PUT="0",
            SGLANG_UNIFIED_RADIX_TREE_CORE_BACKEND="python",
            MOONCAKE_MASTER_SERVER_ADDR=f"127.0.0.1:{rpc}",
            MOONCAKE_METADATA_SERVER=f"http://127.0.0.1:{metadata}/metadata",
            MOONCAKE_LOCAL_HOSTNAME="127.0.0.1",
            MOONCAKE_PROTOCOL="tcp",
            NO_PROXY="localhost,127.0.0.1",
            no_proxy="localhost,127.0.0.1",
        )
        master_log = self.enterContext((work / "master.log").open("w"))
        server_log = self.enterContext((work / "server.log").open("w"))
        master = subprocess.Popen(
            [
                "mooncake_master",
                f"--rpc_port={rpc}",
                "--enable_http_metadata_server=true",
                f"--http_metadata_server_port={metadata}",
                f"--metrics_port={metrics}",
            ],
            stdout=master_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.addCleanup(terminate_process_trees, master, grace_s=5)
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "sglang.launch_server",
                "--model-path",
                model_path,
                "--skip-tokenizer-init",
                "--attention-backend",
                "triton",
                "--mem-fraction-static",
                "0.8",
                "--max-total-tokens",
                "1024",
                "--context-length",
                "1024",
                "--max-running-requests",
                "4",
                "--disable-cuda-graph",
                "--chunked-prefill-size",
                "-1",
                "--speculative-algorithm",
                "DFLASH",
                "--enable-return-aux-hidden-states",
                "--return-aux-hidden-state-layer-ids",
                "1",
                "--return-hidden-states-mode",
                "full",
                "--port",
                str(port),
            ],
            env=env,
            stdout=server_log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        self.addCleanup(terminate_process_trees, server, grace_s=10)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 900
        while time.monotonic() < deadline:
            if server.poll() is not None:
                self.fail((work / "server.log").read_text()[-8000:])
            try:
                if requests.get(url + "/health", timeout=2).ok:
                    break
            except requests.RequestException:
                pass
            time.sleep(2)
        else:
            self.fail("server startup timed out: " + str(work))

        store = MooncakeFeatureStore(
            store_id="dmtp-live",
            setup_kwargs={
                "local_hostname": "127.0.0.1",
                "metadata_server": env["MOONCAKE_METADATA_SERVER"],
                "master_server_addr": env["MOONCAKE_MASTER_SERVER_ADDR"],
                "protocol": "tcp",
                "rdma_devices": "",
                "global_segment_size": 1 << 28,
                "local_buffer_size": 1 << 28,
            },
        )
        self.addCleanup(store.close)
        schema = ServerCaptureSchema(
            aux_feature="hidden_states",
            last_hidden_feature="target_last_hidden_states",
            mtp_hidden_feature="mtp_hidden",
            passthrough=(
                ("input_ids", "input_ids", ()),
                ("loss_mask", "loss_mask", ()),
            ),
        )
        adapter = SGLangServerCaptureAdapter(
            url,
            store,
            run_id="dmtp-live",
            algorithm="dmtp",
            schema=schema,
            server_capture="plugin",
        )
        capture = CaptureConfig.from_strategy(
            {
                "input_ids",
                "loss_mask",
                "hidden_states",
                "target_last_hidden_states",
                "mtp_hidden",
            },
            [1],
            target_repr="hidden_state",
            target_hidden_size=5120,
        )
        # Different lengths in one batch exercise both target and MTP slicing.
        rows = [[5, 10, 20, 30, 40, 50, 60, 70], [9, 13, 17, 21, 25, 29]]
        tasks = [
            PromptTask(
                task_id=f"dmtp-{i}",
                run_id="dmtp-live",
                source_id="test",
                payload={"input_ids": row, "loss_mask": [1] * len(row)},
                max_length=len(row),
            )
            for i, row in enumerate(rows)
        ]
        refs = adapter.produce_refs(tasks, capture=capture)
        fetched = []
        for ref, row in zip(refs, rows, strict=True):
            self.assertIsInstance(ref, SampleRef, str(ref))
            tensors, handle = store.get(ref)
            fetched.append({k: v.clone() for k, v in tensors.items()})
            self.assertEqual(tuple(tensors["mtp_hidden"].shape), (1, len(row), 5120))
            self.assertEqual(tensors["mtp_hidden"][:, 0].abs().sum().item(), 0)
            self.assertTrue(torch.isfinite(tensors["mtp_hidden"]).all())
            store.release(handle)
        torch.save(fetched, work / "captured.pt")
        tokenized = work / "tokenized.jsonl"
        tokenized.write_text(
            "\n".join(
                json.dumps({"input_ids": row, "loss_mask": [1] * len(row)})
                for row in rows
            )
            + "\n"
        )
        offline = work / "offline"
        prepared = subprocess.run(
            [
                sys.executable,
                "scripts/prepare_dmtp_hidden_states.py",
                "--input-path",
                str(tokenized),
                "--output-path",
                str(offline),
                "--target-model-path",
                model_path,
                "--server-url",
                url,
                "--target-layer-ids",
                "1",
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=300,
        )
        self.assertEqual(prepared.returncode, 0, prepared.stdout + prepared.stderr)
        from specforge.algorithms.dmtp.data import build_reader, normalize
        from specforge.runtime.data_plane.feature_store import LocalFeatureStore

        reader = build_reader(
            str(offline), run_id="offline-test", ttt_length=1, max_len=32
        )
        file_store = LocalFeatureStore("dmtp-offline-test")
        for ref, expected in zip(reader, fetched, strict=True):
            raw, handle = file_store.get(ref)
            try:
                normalized = normalize(raw, 32)
                # BF16 target kernels can differ for batched and single-request
                # prefill. Compare whole-feature relative error; exact predictor
                # alignment against identical target inputs is checked below.
                actual = normalized["mtp_hidden"].float()
                reference = expected["mtp_hidden"].float()
                self.assertLess(
                    ((actual - reference).norm() / reference.norm()).item(), 0.03
                )
            finally:
                file_store.release(handle)
        # Release target memory before the independent native-weight oracle.
        terminate_process_trees(server, grace_s=10)
        self._train_and_compare(fetched, model_path, work)
        self._offline_cli(model_path, work)

    def _offline_cli(self, model_path, work):
        import yaml

        draft = json.loads((work / "draft-checkpoint" / "config.json").read_text())
        # Build a fresh draft through the real application configuration path.
        draft["architectures"] = ["DMTPDraftModel"]
        draft["auto_map"] = {"AutoModel": "dmtp.DMTPDraftModel"}
        draft_path = work / "draft-config.json"
        draft_path.write_text(json.dumps(draft))
        config = {
            "model": {
                "target_model_path": model_path,
                "draft_model_config": str(draft_path),
                "embedding_key": "model.language_model.embed_tokens.weight",
                "use_liger_kernel": False,
            },
            "data": {"hidden_states_path": str(work / "offline"), "max_length": 32},
            "training": {
                "strategy": "dmtp",
                "max_steps": 2,
                "num_epochs": 1,
                "batch_size": 1,
                "num_anchors": 2,
                "attention_backend": "sdpa",
                "dflash_teacher_metrics": False,
                "fused_logits_loss": False,
                "save_interval": 1,
                "log_interval": 1,
            },
            "tracking": {"report_to": "none"},
            "run_id": "dmtp-cli",
            "output_dir": str(work / "cli-output"),
        }
        config_path = work / "offline.yaml"
        config_path.write_text(yaml.safe_dump(config))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=os.environ.get("SPECFORGE_DMTP_TRAIN_GPU", "2"),
        )
        with (work / "cli.log").open("w") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "specforge.cli",
                    "train",
                    "--config",
                    str(config_path),
                ],
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=300,
            )
        self.assertEqual(result.returncode, 0, (work / "cli.log").read_text()[-8000:])
        self.assertTrue((work / "cli-output" / "dmtp-cli-latest").exists())
        print("DMTP_CLI_PASS steps=2", flush=True)

    def _train_and_compare(self, fetched, model_path, work):
        from transformers import Qwen3Config

        from specforge.algorithms.dmtp.model import OnlineDMTPModel
        from specforge.inference.mtp_capture import FrozenMTPCapture
        from specforge.modeling.draft.dmtp import DMTPDraftModel
        from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

        device = "cuda:" + os.environ.get("SPECFORGE_DMTP_TRAIN_GPU", "2")
        teacher = FrozenMTPCapture.from_pretrained(
            model_path, device=device, dtype=torch.bfloat16
        )
        for sample in fetched:
            ids = sample["input_ids"][0].to(device)
            hidden = sample["target_last_hidden_states"][0].to(device)
            actual = sample["mtp_hidden"][0].to(device)
            torch.testing.assert_close(
                actual, teacher(ids, hidden), atol=0.05, rtol=0.02
            )
            # Computing only each causal prefix must reproduce its boundary row.
            for n in (3, len(ids)):
                torch.testing.assert_close(
                    actual[n - 1],
                    teacher(ids[:n], hidden[:n])[-1],
                    atol=0.05,
                    rtol=0.02,
                )
        del teacher
        torch.cuda.empty_cache()
        cfg = Qwen3Config(
            vocab_size=248320,
            hidden_size=5120,
            intermediate_size=1024,
            num_attention_heads=8,
            num_key_value_heads=2,
            head_dim=64,
            num_hidden_layers=1,
            num_target_layers=64,
            layer_types=["full_attention"],
            block_size=3,
            dflash_config={"mask_token_id": 248070, "target_layer_ids": [1]},
        )
        cfg._attn_implementation = "sdpa"
        draft = DMTPDraftModel(cfg).to(device=device, dtype=torch.bfloat16)
        parts = TargetEmbeddingsAndHead.from_pretrained(
            model_path,
            embed_key="model.language_model.embed_tokens.weight",
            device=device,
            dtype=torch.bfloat16,
        )
        model = OnlineDMTPModel(
            draft,
            parts.lm_head,
            parts.embed_tokens,
            mask_token_id=248070,
            block_size=3,
            num_anchors=2,
            attention_backend="sdpa",
            teacher_metrics=False,
            fused_logits_loss=False,
        )
        optimizer = torch.optim.AdamW(draft.parameters(), lr=1e-4)
        before = draft.mtp_fusion.weight.detach().clone()
        sample = {
            k: v.to(device)
            for k, v in fetched[0].items()
            if k != "target_last_hidden_states"
        }
        loss, _, _ = model(**sample)
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(draft.mtp_fusion.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(before, draft.mtp_fusion.weight))
        draft.save_pretrained(work / "draft-checkpoint")
        print(
            f"DMTP_LIVE_PASS loss={loss.item():.6f}; checkpoint={work / 'draft-checkpoint'}",
            flush=True,
        )
