"""Opt-in live SGLang -> Mooncake -> packed online consumer lifecycle gate.

Use SPECFORGE_RUN_SERVER_CAPTURE_TESTS=1 and an isolated patched SGLang on
PYTHONPATH. CUDA_VISIBLE_DEVICES chooses the consumer; PACKING_TARGET_GPU
chooses the server. This gate owns and cleans up only its fixture processes.
"""

import json
import math
import numbers
import os
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import torch

from tests.test_runtime import test_server_capture_gate as gate


class TestOnlineSequencePackingGate(gate.TestServerCaptureGate):
    # Reuse the live fixture; its separate extraction tests remain in their
    # original module instead of being duplicated by this derived test case.
    test_eagle3_zero_copy_end_to_end = None
    test_dflash_capture_same_server = None
    test_dflash_capture_without_teacher_metrics_skips_last_hidden = None

    @classmethod
    def setUpClass(cls):
        gate.PORT = int(os.environ.get("PACKING_SERVER_PORT", "30992"))
        target_gpu = os.environ.get(
            "PACKING_TARGET_GPU", os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        )
        original_gpu = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = target_gpu
        try:
            super().setUpClass()
        finally:
            if original_gpu is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_gpu

    @classmethod
    def _ensure_mooncake_master(cls):
        rpc_port = os.environ.get("PACKING_MOONCAKE_RPC_PORT", "50192")
        metadata_port = os.environ.get("PACKING_MOONCAKE_METADATA_PORT", "8092")
        binary = shutil.which("mooncake_master")
        if binary is None:
            raise RuntimeError("live packing gate requires mooncake_master")
        cls.master = subprocess.Popen(
            [
                binary,
                "--enable-http-metadata-server=true",
                f"--rpc_port={rpc_port}",
                f"--http_metadata_server_port={metadata_port}",
                "--metrics_port=9092",
            ],
            stdout=open(os.path.join(cls.workdir, "mooncake_master.log"), "w"),
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        time.sleep(3)
        if cls.master.poll() is not None:
            raise RuntimeError("packing mooncake_master exited during startup")
        os.environ["MOONCAKE_MASTER_SERVER_ADDR"] = f"127.0.0.1:{rpc_port}"
        os.environ["MOONCAKE_METADATA_SERVER"] = (
            f"http://127.0.0.1:{metadata_port}/metadata"
        )
        os.environ["MOONCAKE_LOCAL_HOSTNAME"] = "127.0.0.1"
        os.environ["MOONCAKE_PROTOCOL"] = "tcp"

    @classmethod
    def _cleanup_processes(cls):
        super()._cleanup_processes()
        destination = os.environ.get("PACKING_ARTIFACT_DIR")
        if destination and cls.workdir:
            Path(destination).mkdir(parents=True, exist_ok=True)
            for source in Path(cls.workdir).glob("*.log"):
                shutil.copy2(source, Path(destination) / source.name)
            for source in Path(cls.workdir).glob("*.json"):
                if source.name.endswith("-result.json"):
                    shutil.copy2(source, Path(destination) / source.name)

    def _packing_store(self, run_id):
        from specforge.runtime.data_plane.mooncake_store import MooncakeFeatureStore

        return MooncakeFeatureStore(
            store_id=run_id,
            retain_on_release=True,
            setup_kwargs={
                "local_hostname": "127.0.0.1",
                "metadata_server": os.environ["MOONCAKE_METADATA_SERVER"],
                "global_segment_size": 1 << 28,
                "local_buffer_size": 1 << 28,
                "protocol": "tcp",
                "rdma_devices": "",
                "master_server_addr": os.environ["MOONCAKE_MASTER_SERVER_ADDR"],
            },
        )

    def _model(self, name, work):
        from safetensors.torch import load_file
        from torch import nn
        from transformers import Qwen3Config

        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel
        from specforge.modeling.draft.dflash import DFlashDraftModel
        from specforge.modeling.draft.dflash2 import DFlash2DraftModel
        from specforge.modeling.target.target_head import TargetHead
        from tests.test_runtime import _fixtures as fx

        if name == "eagle3":
            model, _ = fx.build_eagle3(str(work), ttt=3)
            return model, TargetHead.from_pretrained(self.target_dir)
        config = Qwen3Config(
            architectures=[
                "DFlash2DraftModel" if name == "dflash2" else "DFlashDraftModel"
            ],
            hidden_size=gate.H,
            intermediate_size=128,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_hidden_layers=1,
            num_target_layers=8,
            vocab_size=256,
            max_position_embeddings=512,
            layer_types=["full_attention"],
            dflash_config={
                "block_size": 4,
                "mask_token_id": 0,
                "target_layer_ids": gate.AUX_LAYER_IDS,
                "conv_group_size": 4,
                "conv_kernel_size": 2,
                "selector_rank": 4,
                "selector_top_k": 4,
            },
        )
        config._attn_implementation = "flex_attention"
        draft = (DFlash2DraftModel if name == "dflash2" else DFlashDraftModel)(config)
        weights = load_file(os.path.join(self.target_dir, "model.safetensors"))
        head = nn.Linear(gate.H, 256, bias=False)
        head.weight.data.copy_(weights["lm_head.weight"])
        head.requires_grad_(False)
        embeddings = nn.Embedding.from_pretrained(
            weights["model.embed_tokens.weight"], freeze=True
        )
        model = OnlineDFlashModel(
            draft,
            head,
            embeddings,
            mask_token_id=0,
            block_size=4,
            num_anchors=4,
            attention_backend="flex_attention",
        ).to(device="cuda", dtype=torch.bfloat16)
        return model, None

    def test_live_packed_online_training_and_durable_acks(self):
        from specforge.algorithms.builtin import builtin_algorithm_registry
        from specforge.algorithms.common.hidden_states_data import (
            PackedHiddenStatesCollator,
        )
        from specforge.algorithms.eagle3.data import DataCollatorWithPacking
        from specforge.inference.adapters.server_capture import (
            SGLangServerCaptureAdapter,
        )
        from specforge.inference.capture import CaptureConfig
        from specforge.launch import build_disagg_online_consumer
        from specforge.optimizer import BF16Optimizer
        from specforge.runtime.contracts import SampleRef
        from specforge.runtime.data_plane.streaming_ref_channel import (
            StreamingRefChannel,
        )
        from specforge.training.checkpoint import STATE_FILE
        from tests.test_runtime import _fixtures as fx

        fx.build_single_rank_distributed(port="29692")
        for name in ("eagle3", "dflash", "dflash2"):
            with self.subTest(architecture=name):
                algorithm_name = "dflash" if name == "dflash2" else name
                run_id = f"packing-online-{name}"
                work = Path(self.workdir) / name
                work.mkdir()
                store = self._packing_store(run_id)
                try:
                    adapter = SGLangServerCaptureAdapter(
                        f"http://127.0.0.1:{gate.PORT}",
                        store,
                        run_id=run_id,
                        algorithm=algorithm_name,
                        schema=gate._capture_schema(algorithm_name),
                    )
                    required = {"input_ids", "loss_mask"} | (
                        {"attention_mask", "hidden_state", "target"}
                        if name == "eagle3"
                        else {"hidden_states"}
                    )
                    contract = CaptureConfig.from_strategy(
                        required_features=required,
                        aux_hidden_state_layer_ids=tuple(gate.AUX_LAYER_IDS),
                        target_repr="hidden_state",
                        target_hidden_size=gate.H,
                    )
                    rows = [
                        list(range(3, 3 + length)) for length in (8, 24, 12, 16) * 2
                    ]
                    tasks = self._tasks(rows)
                    refs = list(adapter.produce_refs(tasks, capture=contract))
                    self.assertEqual(len(refs), 8)
                    self.assertTrue(
                        all(isinstance(ref, SampleRef) for ref in refs), repr(refs)
                    )
                    channel = StreamingRefChannel(str(work / "refs.jsonl"))
                    channel.publish_many(refs)
                    channel.close()
                    model, target_head = self._model(name, work)
                    logged = []
                    database = str(work / "consumer.sqlite")
                    trainer = build_disagg_online_consumer(
                        algorithm=builtin_algorithm_registry().resolve(algorithm_name),
                        feature_store=store,
                        channel=channel,
                        draft_model=model,
                        target_head=target_head,
                        optimizer_factory=lambda module: BF16Optimizer(
                            module,
                            lr=1e-3,
                            max_grad_norm=0.5,
                            warmup_ratio=0.0,
                            total_steps=2,
                        ),
                        run_id=run_id,
                        output_dir=str(work / "output"),
                        batch_size=2,
                        accumulation_steps=2,
                        max_steps=2,
                        sequence_packing=True,
                        save_interval=1,
                        log_interval=1,
                        metadata_db_path=database,
                        async_ack=False,
                        idle_timeout_s=60,
                        logger=lambda metrics, step: logged.append(
                            (dict(metrics), step)
                        ),
                    )
                    self.assertIsInstance(
                        trainer._loader.collate_fn,
                        (
                            DataCollatorWithPacking
                            if name == "eagle3"
                            else PackedHiddenStatesCollator
                        ),
                    )
                    self.assertEqual(trainer.fit(), 2)
                    self.assertEqual(trainer.micro_step, 4)
                    self.assertEqual(channel.consumer_quantum(), 4)
                    self.assertTrue(channel.consumer_stopped())
                    self.assertIsNone(channel.consumer_failure())
                    self.assertEqual([step for _, step in logged], [1, 2])
                    for metrics, _ in logged:
                        for key, value in metrics.items():
                            if isinstance(value, numbers.Real):
                                self.assertTrue(math.isfinite(value), f"{key}={value}")
                    with sqlite3.connect(database) as connection:
                        acked = [
                            row[0]
                            for row in connection.execute(
                                "SELECT sample_id FROM acked ORDER BY sample_id"
                            )
                        ]
                    self.assertEqual(acked, sorted(ref.sample_id for ref in refs))
                    for step in (1, 2):
                        self.assertTrue(
                            (
                                work / "output" / f"{run_id}-step{step}" / STATE_FILE
                            ).is_file()
                        )
                    result = {
                        "architecture": name,
                        "optimizer_steps": trainer.global_step,
                        "microsteps": trainer.micro_step,
                        "acked_samples": len(acked),
                        "sample_lengths": [len(row) for row in rows],
                        "consumer_quantum": channel.consumer_quantum(),
                        "logged_steps": [step for _, step in logged],
                    }
                    (Path(self.workdir) / f"{name}-result.json").write_text(
                        json.dumps(result, indent=2)
                    )
                    del trainer, model, target_head
                    torch.cuda.empty_cache()
                finally:
                    store.close()
