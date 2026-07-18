# coding=utf-8
"""CPU tests for the production DFlash fan-out launcher."""

import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from scripts import run_dflash_fanout as launcher
from specforge.config.production_fanout import ManifestError, load_manifest
from specforge.runtime import process_supervisor
from specforge.runtime.process_supervisor import (
    GPUReservationSet,
    LauncherError,
    ProcessSupervisor,
    RoleCommand,
    gpu_busy_reasons,
)


class FanoutManifestFixture(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="fanout-launcher-")
        self.model_dir = os.path.join(self.workdir, "model")
        os.makedirs(self.model_dir)
        self.draft_config = os.path.join(self.workdir, "draft.json")
        Path(self.draft_config).write_text(
            json.dumps(
                {
                    "hidden_size": 4096,
                    "block_size": 16,
                    "dflash_config": {"target_layer_ids": [1, 9, 17, 25, 33]},
                }
            ),
            encoding="utf-8",
        )
        self.train_data = os.path.join(self.workdir, "train.jsonl")
        Path(self.train_data).write_text("{}\n", encoding="utf-8")
        self.run_dir = os.path.join(self.workdir, "run")
        self.manifest_path = os.path.join(self.workdir, "manifest.json")

    def payload(self, *, variants=3):
        values = {
            "schema_version": 1,
            "run_id": "test-fanout",
            "capture": {
                "target_model_path": self.model_dir,
                "tokenizer_path": self.model_dir,
                "draft_config_path": self.draft_config,
                "train_data_path": self.train_data,
                "chat_template": "qwen",
                "is_preformatted": False,
                "max_length": 3072,
                "max_prompts": 4,
                "dataset_num_proc": 1,
                "capture_layer_ids": [1, 9, 17, 25, 33],
            },
            "server": {
                "python_executable": sys.executable,
                "url": "http://127.0.0.1:30000",
                "gpu": 4,
                "mem_fraction_static": 0.85,
                "readiness_timeout_s": 10.0,
                "readiness_poll_s": 0.1,
                "trust_remote_code": False,
            },
            "mooncake": {
                "mode": "managed",
                "master_executable": sys.executable,
                "local_hostname": "127.0.0.1",
                "metadata_server": "http://127.0.0.1:8080/metadata",
                "master_server_addr": "127.0.0.1:50051",
                "protocol": "tcp",
                "rdma_devices": "",
                "global_segment_size": 64 << 30,
                "client_buffer_size": 1 << 30,
                "metrics_port": 50052,
            },
            "training": {
                "python_executable": sys.executable,
                "batch_size": 2,
                "accumulation_steps": 1,
                "attention_backend": "sdpa",
                "flex_kernel_options": None,
                "linear_cross_entropy_backend": "torch",
                "max_grad_norm": 1.0,
                "checkpoint": {
                    "periodic": {
                        "step_interval": 2,
                        "sliding_window_size": 3,
                    },
                    "save_epoch_end": True,
                },
                "log_interval": 1,
                "mask_token_id": 151669,
                "embedding_key": None,
                "lm_head_key": None,
                "trust_remote_code": False,
            },
            "runtime": {
                "run_dir": self.run_dir,
                "gpu_lock_path_pattern": os.path.join(
                    self.workdir, "locks", "gpu-{gpu_id}.lock"
                ),
                "nvidia_smi_executable": sys.executable,
                "gpu_max_used_memory_mib": 1024,
                "gpu_poll_s": 60.0,
                "process_poll_s": 0.01,
                "termination_grace_s": 0.5,
                "kill_grace_s": 0.5,
                "master_readiness_timeout_s": 1.0,
                "master_readiness_poll_s": 0.01,
                "idle_timeout_s": 10.0,
                "consumer_registration_timeout_s": 10.0,
                "consumer_heartbeat_timeout_s": 3.0,
                "consumer_heartbeat_interval_s": 1.0,
                "finalize_timeout_s": 10.0,
                "gc_poll_s": 0.1,
            },
            "variants": [],
        }
        variant_values = (
            ("dflash-g7", 3, "dflash", 7.0),
            ("dflash-uniform", 5, "dflash", None),
            ("dpace-a05", 7, "dpace", None),
        )
        for index, (name, gpu, loss_type, gamma) in enumerate(
            variant_values[:variants]
        ):
            values["variants"].append(
                {
                    "subscription_id": name,
                    "seed": 42 + index,
                    "loss_type": loss_type,
                    "loss_decay_gamma": gamma,
                    "dpace_alpha": 0.5,
                    "num_anchors": 256,
                    "learning_rate": 0.0006,
                    "warmup_ratio": 0.04,
                    "gpu": gpu,
                }
            )
        return values

    def write_manifest(self, payload=None):
        Path(self.manifest_path).write_text(
            json.dumps(payload or self.payload()), encoding="utf-8"
        )
        return load_manifest(self.manifest_path)


class TestFanoutManifest(FanoutManifestFixture):
    def test_derives_all_mutable_paths_from_fresh_run_dir(self):
        manifest = self.write_manifest()
        self.assertEqual(manifest.training.batch_size, 2)
        self.assertEqual(len(manifest.variants), 3)
        self.assertEqual(
            manifest.window_registry_db_path,
            os.path.join(self.run_dir, "state", "windowed-capture.db"),
        )
        self.assertEqual(
            manifest.variant_output_dir(manifest.variants[0]),
            os.path.join(self.run_dir, "checkpoints", "dflash-g7"),
        )
        self.assertFalse(manifest.runtime.gpu_monitor.enabled)
        self.assertEqual(manifest.runtime.gpu_monitor.poll_s, 1.0)
        self.assertTrue(manifest.runtime.gpu_monitor.strict_process_ownership)
        self.assertEqual(
            manifest.gpu_samples_path,
            os.path.join(self.run_dir, "metrics", "gpu_samples.jsonl"),
        )
        self.assertEqual(
            manifest.gpu_summary_path,
            os.path.join(self.run_dir, "metrics", "gpu_summary.json"),
        )

    def test_gpu_monitor_config_is_strict_and_opt_in(self):
        payload = self.payload()
        payload["runtime"]["gpu_monitor"] = {
            "enabled": True,
            "poll_s": 0.5,
            "strict_process_ownership": False,
        }
        manifest = self.write_manifest(payload)
        self.assertTrue(manifest.runtime.gpu_monitor.enabled)
        self.assertEqual(manifest.runtime.gpu_monitor.poll_s, 0.5)
        self.assertFalse(manifest.runtime.gpu_monitor.strict_process_ownership)

        payload["runtime"]["gpu_monitor"]["poll_s"] = 0.0
        with self.assertRaisesRegex(ManifestError, "poll_s"):
            self.write_manifest(payload)

    def test_rejects_shared_gpu_and_non_divisible_prompt_count(self):
        payload = self.payload()
        payload["variants"][0]["gpu"] = payload["server"]["gpu"]
        with self.assertRaisesRegex(ManifestError, "GPU ids must be distinct"):
            self.write_manifest(payload)

        payload = self.payload()
        payload["capture"]["max_prompts"] = 3
        with self.assertRaisesRegex(ManifestError, "effective batch"):
            self.write_manifest(payload)

        payload = self.payload()
        payload["training"]["max_grad_norm"] = 0.0
        with self.assertRaisesRegex(ManifestError, "max_grad_norm"):
            self.write_manifest(payload)

        for field in ("step_interval", "sliding_window_size"):
            with self.subTest(field=field):
                payload = self.payload()
                payload["training"]["checkpoint"]["periodic"][field] = 0
                with self.assertRaisesRegex(ManifestError, field):
                    self.write_manifest(payload)

        payload = self.payload()
        payload["training"]["checkpoint"]["periodic"] = None
        manifest = self.write_manifest(payload)
        self.assertIsNone(manifest.training.checkpoint.periodic)

        for invalid in (1, 3073):
            with self.subTest(block_size=invalid):
                payload = self.payload()
                payload["variants"][0]["block_size"] = invalid
                with self.assertRaisesRegex(ManifestError, "block_size"):
                    self.write_manifest(payload)

    def test_flex_kernel_options_require_flex_attention(self):
        payload = self.payload()
        payload["training"]["flex_kernel_options"] = {"num_stages": 2}
        with self.assertRaisesRegex(ManifestError, "require attention_backend"):
            self.write_manifest(payload)

        payload["training"]["attention_backend"] = "flex_attention"
        manifest = self.write_manifest(payload)
        self.assertEqual(
            manifest.training.flex_kernel_options,
            {"num_stages": 2},
        )

        for invalid in (0, -1, True, "two", None, {"nested": "value"}):
            with self.subTest(num_stages=invalid):
                payload = self.payload()
                payload["training"]["attention_backend"] = "flex_attention"
                payload["training"]["flex_kernel_options"] = {"num_stages": invalid}
                with self.assertRaisesRegex(ManifestError, "num_stages"):
                    self.write_manifest(payload)

    def test_linear_cross_entropy_backend_defaults_to_torch(self):
        payload = self.payload()
        del payload["training"]["linear_cross_entropy_backend"]
        manifest = self.write_manifest(payload)
        self.assertEqual(manifest.training.linear_cross_entropy_backend, "torch")

    def test_compact_zero_weight_ce_rows_defaults_to_false(self):
        manifest = self.write_manifest(self.payload())
        self.assertFalse(manifest.training.compact_zero_weight_ce_rows)

    def test_compact_zero_weight_ce_rows_requires_liger(self):
        payload = self.payload()
        payload["training"]["compact_zero_weight_ce_rows"] = True
        with self.assertRaisesRegex(ManifestError, "requires.*liger"):
            self.write_manifest(payload)

    def test_draft_kernel_backend_defaults_to_torch(self):
        manifest = self.write_manifest(self.payload())
        self.assertEqual(manifest.training.draft_kernel_backend, "torch")

    def test_gradient_clip_backend_defaults_to_torch(self):
        manifest = self.write_manifest(self.payload())
        self.assertEqual(manifest.training.gradient_clip_backend, "torch")

    def test_windowed_delivery_defaults_and_variant_overrides(self):
        manifest = self.write_manifest(self.payload())
        self.assertEqual(manifest.runtime.delivery_mode, "async_window")
        self.assertEqual(manifest.runtime.consumer_prefetch_batches, 0)
        self.assertEqual(manifest.window_config(manifest.variants[0]), (2, 40, 8))

        payload = self.payload()
        payload["variants"][0].update(
            {"window_lookbehind": 1, "window_lookahead": 5, "max_prefetch": 3}
        )
        manifest = self.write_manifest(payload)
        self.assertEqual(manifest.runtime.delivery_mode, "async_window")
        self.assertEqual(manifest.window_config(manifest.variants[0]), (1, 5, 3))

    def test_broadcast_delivery_is_rejected(self):
        payload = self.payload()
        payload["runtime"]["delivery_mode"] = "broadcast"
        with self.assertRaisesRegex(ManifestError, "async_window"):
            self.write_manifest(payload)

    def test_windowed_prefetch_cannot_exceed_effective_lookahead(self):
        payload = self.payload()
        payload["variants"][0].update({"window_lookahead": 2, "max_prefetch": 4})
        with self.assertRaisesRegex(ManifestError, "effective lookahead"):
            self.write_manifest(payload)

    def test_checked_in_qwen_manifest_is_bsz2_256_anchor_1p3c(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "examples/disagg/qwen3_8b_dflash_fanout.production.json"
        )
        manifest = load_manifest(path)
        self.assertEqual(manifest.training.batch_size, 2)
        self.assertEqual(manifest.training.accumulation_steps, 1)
        self.assertEqual(manifest.training.attention_backend, "flex_attention")
        self.assertEqual(
            manifest.training.flex_kernel_options,
            {"num_stages": 2},
        )
        self.assertTrue(manifest.training.compact_zero_weight_ce_rows)
        self.assertEqual(manifest.training.draft_kernel_backend, "liger")
        self.assertEqual(manifest.training.gradient_clip_backend, "fused_adamw")
        self.assertEqual(manifest.training.linear_cross_entropy_backend, "liger")
        self.assertIsNone(manifest.training.checkpoint.periodic)
        self.assertTrue(manifest.training.checkpoint.save_epoch_end)
        self.assertEqual([value.num_anchors for value in manifest.variants], [256] * 3)
        self.assertTrue(manifest.runtime.gpu_monitor.enabled)
        self.assertEqual(manifest.runtime.gpu_monitor.poll_s, 1.0)
        self.assertEqual(
            [manifest.server.gpu, *(value.gpu for value in manifest.variants)],
            [4, 3, 5, 7],
        )

    def test_checkpoint_values_are_manifest_owned(self):
        manifest = self.write_manifest()
        periodic = manifest.training.checkpoint.periodic
        self.assertEqual(periodic.step_interval, 2)
        self.assertEqual(periodic.sliding_window_size, 3)
        self.assertTrue(manifest.training.checkpoint.save_epoch_end)

        payload = self.payload()
        payload["training"]["checkpoint"]["periodic"] = None
        manifest = self.write_manifest(payload)
        self.assertIsNone(manifest.training.checkpoint.periodic)


class TestRoleCommands(FanoutManifestFixture):
    def test_gpu_metrics_directory_only_exists_when_monitoring_is_enabled(self):
        manifest = self.write_manifest()
        launcher._prepare_run_directories(manifest)
        self.assertFalse(os.path.exists(manifest.metrics_dir))

    def test_enabled_gpu_monitor_prepares_metrics_directory(self):
        payload = self.payload()
        payload["runtime"]["gpu_monitor"] = {
            "enabled": True,
            "poll_s": 1.0,
            "strict_process_ownership": True,
        }
        manifest = self.write_manifest(payload)
        launcher._prepare_run_directories(manifest)
        self.assertTrue(os.path.isdir(manifest.metrics_dir))

    def test_builds_exactly_one_server_and_fixed_consumers(self):
        manifest = self.write_manifest()
        commands = launcher.build_role_commands(manifest, base_env={})
        roles = [command.role for command in commands]
        self.assertEqual(roles.count("target-server"), 1)
        self.assertEqual(
            roles,
            [
                "mooncake-master",
                "target-server",
                "producer",
                "cleanup",
                "consumer:dflash-g7",
                "consumer:dflash-uniform",
                "consumer:dpace-a05",
            ],
        )
        command_by_role = {command.role: command for command in commands}
        self.assertEqual(
            command_by_role["target-server"].env["SGLANG_SPEC_CAPTURE_TOKEN"],
            command_by_role["producer"].env["SGLANG_SPEC_CAPTURE_TOKEN"],
        )
        self.assertEqual(
            [
                command_by_role[f"consumer:{variant.subscription_id}"].env[
                    "CUDA_VISIBLE_DEVICES"
                ]
                for variant in manifest.variants
            ],
            ["3", "5", "7"],
        )
        flattened = " ".join(value for command in commands for value in command.argv)
        self.assertNotIn("gate", flattened.lower())
        self.assertNotIn("receipt", flattened.lower())
        self.assertNotIn("audit", flattened.lower())

    def test_managed_mooncake_uses_manifest_endpoints(self):
        payload = self.payload()
        payload["mooncake"].update(
            {
                "metadata_server": "http://127.0.0.1:18187/metadata",
                "master_server_addr": "127.0.0.1:52107",
                "metrics_port": 52108,
            }
        )
        manifest = self.write_manifest(payload)
        commands = launcher.build_role_commands(manifest, base_env={})
        master = next(
            command for command in commands if command.role == "mooncake-master"
        )

        self.assertIn("--rpc_address=127.0.0.1", master.argv)
        self.assertIn("--rpc_port=52107", master.argv)
        self.assertIn("--http_metadata_server_host=127.0.0.1", master.argv)
        self.assertIn("--http_metadata_server_port=18187", master.argv)
        self.assertIn("--metrics_port=52108", master.argv)

    def test_dry_run_has_no_filesystem_or_gpu_side_effects(self):
        payload = self.payload()
        payload["runtime"]["gpu_monitor"] = {
            "enabled": True,
            "poll_s": 1.0,
            "strict_process_ownership": True,
        }
        self.write_manifest(payload)
        output = io.StringIO()
        with (
            mock.patch.object(
                launcher, "gpu_busy_reasons", side_effect=AssertionError("GPU probe")
            ),
            mock.patch.object(
                launcher, "GpuMonitor", side_effect=AssertionError("NVML init")
            ),
        ):
            result = launcher.run_launcher(
                self.manifest_path,
                dry_run=True,
                wait_for_gpus_enabled=True,
                output=output,
            )
        self.assertEqual(result, 0)
        self.assertFalse(os.path.exists(self.run_dir))
        self.assertIn("consumer:dflash-g7", output.getvalue())


class TestProducerPrompts(FanoutManifestFixture):
    @staticmethod
    def _processed_row(token: int):
        return {
            "input_ids": torch.tensor([[token, token + 1, token + 2]]),
            "loss_mask": torch.tensor([[0, 1, 1]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }

    def test_reads_only_until_enough_eligible_heterogeneous_records(self):
        records = [
            {"conversations": [{"role": "user", "content": "one"}]},
            {
                "conversations": [],
                "input_line_index": 2,
            },
            {
                "conversations": [{"role": "user", "content": "three"}],
                "status": "success",
            },
            {
                "conversations": [{"role": "user", "content": "four"}],
                "tools": [{"type": "function"}],
            },
            {
                "conversations": [{"role": "user", "content": "five"}],
                "row_index": 5,
                "input_line_index": 5,
            },
        ]
        Path(self.train_data).write_text(
            "".join(f"{json.dumps(record)}\n" for record in records)
            + "{this sentinel must not be parsed\n",
            encoding="utf-8",
        )
        manifest = self.write_manifest(self.payload(variants=1))
        batch_sizes = []

        def fake_build(*, dataset, **kwargs):
            batch_sizes.append(len(dataset))
            self.assertEqual(dataset.column_names, ["conversations", "tools"])
            self.assertNotIn("cache_dir", kwargs)
            self.assertTrue(
                all(
                    value is None or isinstance(value, str)
                    for value in dataset["tools"]
                )
            )
            eligible = [row for row in dataset if row["conversations"]]
            return [self._processed_row(index + 10) for index, _ in enumerate(eligible)]

        with mock.patch(
            "specforge.data.preprocessing.build_eagle3_dataset",
            side_effect=fake_build,
        ):
            prompts = launcher._producer_prompts(manifest, tokenizer=object())

        self.assertEqual(batch_sizes, [4, 1])
        self.assertEqual(len(prompts), 4)
        self.assertEqual(
            [prompt["task_id"] for prompt in prompts],
            [f"prompt-{index:08d}" for index in range(4)],
        )

    def test_preformatted_input_projects_only_text(self):
        payload = self.payload(variants=1)
        payload["capture"]["is_preformatted"] = True
        records = [
            {"text": f"formatted-{index}", "extra_field": index} for index in range(4)
        ]
        Path(self.train_data).write_text(
            "".join(f"{json.dumps(record)}\n" for record in records),
            encoding="utf-8",
        )
        manifest = self.write_manifest(payload)

        def fake_build(*, dataset, **kwargs):
            self.assertEqual(dataset.column_names, ["text"])
            self.assertTrue(kwargs["is_preformatted"])
            return [self._processed_row(index) for index in range(len(dataset))]

        with mock.patch(
            "specforge.data.preprocessing.build_eagle3_dataset",
            side_effect=fake_build,
        ):
            prompts = launcher._producer_prompts(manifest, tokenizer=object())

        self.assertEqual(len(prompts), 4)

    def test_pretokenized_input_bypasses_dataset_preprocessing(self):
        payload = self.payload(variants=1)
        payload["capture"]["is_pretokenized"] = True
        payload["capture"]["max_prompts"] = 2
        payload["training"]["batch_size"] = 1
        rows = [
            {
                "input_ids": list(range(64)),
                "loss_mask": [1] * 64,
                "ignored": "not forwarded",
            },
            {
                "input_ids": list(range(64, 128)),
                "loss_mask": [1] * 64,
            },
        ]
        Path(self.train_data).write_text(
            "".join(f"{json.dumps(row)}\n" for row in rows), encoding="utf-8"
        )
        manifest = self.write_manifest(payload)

        with mock.patch("specforge.data.preprocessing.build_eagle3_dataset") as build:
            prompts = launcher._producer_prompts(manifest, tokenizer=None)

        build.assert_not_called()
        self.assertEqual(
            [row["payload"] for row in prompts],
            [
                {"input_ids": row["input_ids"], "loss_mask": row["loss_mask"]}
                for row in rows
            ],
        )

    def test_pretokenized_input_rejects_misaligned_or_nonbinary_rows(self):
        payload = self.payload(variants=1)
        payload["capture"]["is_pretokenized"] = True
        payload["capture"]["max_prompts"] = 2
        payload["training"]["batch_size"] = 1
        manifest = self.write_manifest(payload)

        for row, message in (
            ({"input_ids": [1, 2], "loss_mask": [1]}, "aligned lengths"),
            ({"input_ids": [1, 2], "loss_mask": [1, 2]}, "only 0/1"),
        ):
            with self.subTest(message=message):
                Path(self.train_data).write_text(
                    json.dumps(row) + "\n", encoding="utf-8"
                )
                with self.assertRaisesRegex(ValueError, message):
                    launcher._producer_prompts(manifest, tokenizer=None)

    def test_filters_shared_inputs_for_largest_variant_block_size(self):
        payload = self.payload()
        for variant, block_size in zip(payload["variants"], (4, 8, 16)):
            variant["block_size"] = block_size
        manifest = self.write_manifest(payload)
        Path(self.train_data).write_text(
            "".join(
                f"{json.dumps({'conversations': [{'role': 'user', 'content': str(index)}]})}\n"
                for index in range(4)
            ),
            encoding="utf-8",
        )

        def fake_build(*, dataset, **kwargs):
            self.assertEqual(kwargs["minimum_valid_tokens"], 32)
            return [self._processed_row(index) for index in range(len(dataset))]

        with mock.patch(
            "specforge.data.preprocessing.build_eagle3_dataset",
            side_effect=fake_build,
        ):
            prompts = launcher._producer_prompts(manifest, tokenizer=object())

        self.assertEqual(len(prompts), 4)


class TestConsumerModelBuild(FanoutManifestFixture):
    def test_forwards_prefetch_depth_to_windowed_consumer(self):
        payload = self.payload(variants=1)
        payload["runtime"].update(
            {"delivery_mode": "async_window", "consumer_prefetch_batches": 1}
        )
        manifest = self.write_manifest(payload)
        registry = mock.Mock()
        registry.wait_initialized.return_value = {
            "run_id": manifest.run_id,
            "contract_digest": "contract-digest",
            "total_samples": manifest.capture.max_prompts,
        }
        control = mock.Mock()
        runtime = mock.Mock()
        runtime.run.return_value = 2
        runtime.accounting_snapshot.return_value = {}
        captured = {}

        def build_runtime(**kwargs):
            captured.update(kwargs)
            return runtime

        with (
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture."
                "SQLiteWindowedCaptureRegistry",
                return_value=registry,
            ),
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture_runtime."
                "start_windowed_consumer_control",
                return_value=control,
            ),
            mock.patch.object(
                launcher,
                "_windowed_capture_contract",
                return_value=(object(), "contract-digest"),
            ),
            mock.patch.object(
                launcher,
                "_build_consumer_model",
                return_value=SimpleNamespace(block_size=16),
            ),
            mock.patch.object(launcher, "_fanout_store", return_value=object()),
            mock.patch(
                "specforge.launch.build_disagg_online_windowed_consumer",
                side_effect=build_runtime,
            ),
        ):
            launcher.run_consumer(manifest, manifest.variants[0].subscription_id)

        self.assertEqual(captured["loader_prefetch_batches"], 1)
        self.assertEqual(captured["registry_db_path"], manifest.window_registry_db_path)
        self.assertIs(captured["consumer_control"], control)
        self.assertEqual(captured["lookbehind"], 2)
        self.assertEqual(captured["lookahead"], 40)
        self.assertEqual(captured["prefetch_depth"], 8)
        self.assertEqual(
            captured["max_outstanding"],
            manifest.runtime.max_outstanding_per_consumer,
        )
        self.assertEqual(captured["contract_digest"], "contract-digest")

    def test_forwards_gradient_clip_backend_to_optimizer_factory(self):
        payload = self.payload(variants=1)
        payload["training"]["gradient_clip_backend"] = "fused_adamw"
        manifest = self.write_manifest(payload)
        control = mock.Mock()
        model = SimpleNamespace(block_size=16)
        runtime = mock.Mock()
        runtime.run.return_value = 1
        runtime.accounting_snapshot.return_value = {}
        registry = mock.Mock()
        registry.wait_initialized.return_value = {
            "run_id": manifest.run_id,
            "contract_digest": "contract-digest",
            "total_samples": manifest.capture.max_prompts,
        }

        def build_runtime(**kwargs):
            kwargs["optimizer_factory"](object())
            return runtime

        with (
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture."
                "SQLiteWindowedCaptureRegistry",
                return_value=registry,
            ),
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture_runtime."
                "start_windowed_consumer_control",
                return_value=control,
            ),
            mock.patch.object(
                launcher,
                "_windowed_capture_contract",
                return_value=(object(), "contract-digest"),
            ),
            mock.patch.object(launcher, "_build_consumer_model", return_value=model),
            mock.patch.object(launcher, "_fanout_store", return_value=object()),
            mock.patch(
                "specforge.launch.build_disagg_online_windowed_consumer",
                side_effect=build_runtime,
            ),
            mock.patch("specforge.optimizer.BF16Optimizer") as optimizer,
        ):
            launcher.run_consumer(manifest, manifest.variants[0].subscription_id)

        self.assertEqual(optimizer.call_args.kwargs["adamw_backend"], "fused")
        runtime.trainer.save_checkpoint.assert_called_once_with(1)

    def test_pre_runtime_failure_reports_terminal_consumer_state(self):
        manifest = self.write_manifest(self.payload(variants=1))
        control = mock.Mock()
        registry = mock.Mock()
        registry.wait_initialized.return_value = {
            "run_id": manifest.run_id,
            "contract_digest": "contract-digest",
            "total_samples": manifest.capture.max_prompts,
        }
        failure = RuntimeError("consumer model failed")

        with (
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture."
                "SQLiteWindowedCaptureRegistry",
                return_value=registry,
            ),
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture_runtime."
                "start_windowed_consumer_control",
                return_value=control,
            ),
            mock.patch.object(
                launcher,
                "_windowed_capture_contract",
                return_value=(object(), "contract-digest"),
            ),
            mock.patch.object(launcher, "_build_consumer_model", side_effect=failure),
            mock.patch(
                "specforge.launch.build_disagg_online_windowed_consumer"
            ) as build,
        ):
            with self.assertRaisesRegex(RuntimeError, "consumer model failed"):
                launcher.run_consumer(manifest, manifest.variants[0].subscription_id)

        control.fail.assert_called_once_with(failure)
        control.close.assert_called_once_with()
        registry.close.assert_called_once_with()
        build.assert_not_called()

    def test_registry_preflight_mismatch_closes_registry_before_registration(self):
        manifest = self.write_manifest(self.payload(variants=1))
        registry = mock.Mock()
        registry.wait_initialized.return_value = {
            "run_id": "wrong-run",
            "contract_digest": "contract-digest",
            "total_samples": manifest.capture.max_prompts,
        }

        with (
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture."
                "SQLiteWindowedCaptureRegistry",
                return_value=registry,
            ),
            mock.patch(
                "specforge.runtime.data_plane.windowed_capture_runtime."
                "start_windowed_consumer_control"
            ) as start_control,
            mock.patch.object(
                launcher,
                "_windowed_capture_contract",
                return_value=(object(), "contract-digest"),
            ),
            mock.patch(
                "specforge.launch.build_disagg_online_windowed_consumer"
            ) as build,
        ):
            with self.assertRaisesRegex(RuntimeError, "preflight mismatch"):
                launcher.run_consumer(
                    manifest,
                    manifest.variants[0].subscription_id,
                )

        registry.close.assert_called_once_with()
        start_control.assert_not_called()
        build.assert_not_called()

    def test_forwards_flex_kernel_options_from_manifest(self):
        payload = self.payload(variants=1)
        payload["training"]["attention_backend"] = "flex_attention"
        payload["training"]["flex_kernel_options"] = {"num_stages": 2}
        payload["training"]["compact_zero_weight_ce_rows"] = True
        payload["training"]["draft_kernel_backend"] = "liger"
        payload["training"]["linear_cross_entropy_backend"] = "liger"
        manifest = self.write_manifest(payload)
        draft_config = SimpleNamespace(dflash_config={})
        draft_model = mock.Mock(block_size=16)
        draft_model.to.return_value = draft_model
        draft_model.config.dflash_config = {}
        target = SimpleNamespace(lm_head=object(), embed_tokens=object())
        built_model = object()

        with (
            mock.patch("torch.cuda.device_count", return_value=1),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(mask_token_id=151669),
            ),
            mock.patch(
                "transformers.AutoConfig.from_pretrained", return_value=draft_config
            ),
            mock.patch(
                "specforge.modeling.draft.dflash.DFlashDraftModel",
                return_value=draft_model,
            ) as draft_factory,
            mock.patch(
                "specforge.modeling.target.target_utils.TargetEmbeddingsAndHead.from_pretrained",
                return_value=target,
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model.OnlineDFlashModel",
                return_value=built_model,
            ) as model_factory,
            mock.patch(
                "specforge.ops.dflash_kernels.validate_dflash_draft_kernel_backend"
            ) as validate_draft_kernels,
            mock.patch(
                "specforge.ops.fused_linear_cross_entropy.validate_liger_installation"
            ) as validate_liger,
        ):
            result = launcher._build_consumer_model(
                manifest,
                manifest.variants[0],
            )

        self.assertIs(result, built_model)
        from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

        supported_parameters = set(inspect.signature(OnlineDFlashModel).parameters)
        self.assertLessEqual(
            set(model_factory.call_args.kwargs),
            supported_parameters,
        )
        self.assertEqual(
            model_factory.call_args.kwargs["flex_kernel_options"],
            {"num_stages": 2},
        )
        self.assertEqual(
            model_factory.call_args.kwargs["draft_kernel_backend"],
            "liger",
        )
        self.assertEqual(
            model_factory.call_args.kwargs["linear_cross_entropy_backend"],
            "liger",
        )
        self.assertTrue(model_factory.call_args.kwargs["compact_zero_weight_ce_rows"])
        draft_factory.assert_called_once_with(
            draft_config,
            draft_kernel_backend="liger",
        )
        validate_draft_kernels.assert_called_once_with("liger")
        validate_liger.assert_called_once_with()

    def test_variant_block_size_overrides_shared_draft_config(self):
        payload = self.payload(variants=1)
        payload["variants"][0]["block_size"] = 4
        manifest = self.write_manifest(payload)
        draft_config = SimpleNamespace(block_size=16, dflash_config={})
        draft_model = mock.Mock()
        draft_model.to.return_value = draft_model
        target = SimpleNamespace(lm_head=object(), embed_tokens=object())
        built_model = object()

        def build_draft(config, *, draft_kernel_backend):
            self.assertEqual(draft_kernel_backend, "torch")
            draft_model.block_size = config.block_size
            draft_model.config = config
            return draft_model

        with (
            mock.patch("torch.cuda.device_count", return_value=1),
            mock.patch(
                "transformers.AutoTokenizer.from_pretrained",
                return_value=SimpleNamespace(mask_token_id=151669),
            ),
            mock.patch(
                "transformers.AutoConfig.from_pretrained", return_value=draft_config
            ),
            mock.patch(
                "specforge.modeling.draft.dflash.DFlashDraftModel",
                side_effect=build_draft,
            ),
            mock.patch(
                "specforge.modeling.target.target_utils.TargetEmbeddingsAndHead.from_pretrained",
                return_value=target,
            ),
            mock.patch(
                "specforge.algorithms.common.dflash_family_model.OnlineDFlashModel",
                return_value=built_model,
            ) as model_factory,
        ):
            result = launcher._build_consumer_model(
                manifest,
                manifest.variants[0],
            )

        self.assertIs(result, built_model)
        self.assertEqual(draft_config.block_size, 4)
        self.assertEqual(model_factory.call_args.kwargs["block_size"], 4)


class TestGPUResources(FanoutManifestFixture):
    def test_nvidia_smi_requires_no_compute_process_and_low_memory(self):
        responses = {
            "--query-gpu=index,uuid,memory.used": "3, GPU-a, 0\n4, GPU-b, 900\n",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory": (
                "GPU-b, 1234, python, 800\n"
            ),
        }

        def fake_run(argv, **kwargs):
            query = next(value for value in argv if value.startswith("--query-"))
            return subprocess.CompletedProcess(argv, 0, responses[query], "")

        reasons = gpu_busy_reasons(
            "nvidia-smi", [3, 4], max_used_memory_mib=1024, run=fake_run
        )
        self.assertEqual(len(reasons), 1)
        self.assertIn("pid=1234", reasons[0])
        reasons = gpu_busy_reasons(
            "nvidia-smi", [3, 4], max_used_memory_mib=500, run=fake_run
        )
        self.assertEqual(len(reasons), 2)
        self.assertIn("900 MiB", reasons[1])

    def test_wait_returns_immediately_after_first_free_poll(self):
        output = io.StringIO()
        sleep = mock.Mock()
        with mock.patch.object(
            process_supervisor,
            "gpu_busy_reasons",
            side_effect=[("GPU 3 busy",), ()],
        ) as probe:
            process_supervisor.wait_for_free_gpus(
                "nvidia-smi",
                [3],
                max_used_memory_mib=1024,
                wait=True,
                poll_s=60.0,
                output=output,
                sleep=sleep,
            )
        self.assertEqual(probe.call_count, 2)
        sleep.assert_called_once_with(60.0)

    def test_lock_is_exclusive_even_within_one_process(self):
        pattern = os.path.join(self.workdir, "locks", "gpu-{gpu_id}.lock")
        first = GPUReservationSet([3], lock_path_pattern=pattern)
        second = GPUReservationSet([3], lock_path_pattern=pattern)
        first.acquire()
        with self.assertRaisesRegex(LauncherError, "reserved by another launcher"):
            second.acquire()
        first.close()
        second.acquire()
        second.close()

    def test_lock_reuses_read_only_file_created_by_another_user(self):
        pattern = os.path.join(self.workdir, "locks", "gpu-{gpu_id}.lock")
        path = pattern.format(gpu_id=3)
        os.makedirs(os.path.dirname(path))
        Path(path).touch(mode=0o444)
        os.chmod(path, 0o444)

        reservation = GPUReservationSet([3], lock_path_pattern=pattern)
        reservation.acquire()
        reservation.close()


class _FakeReservationSet:
    instances = []

    def __init__(self, gpu_ids, *, lock_path_pattern):
        self.gpu_ids = tuple(sorted(gpu_ids))
        self.acquired = False
        self.closed = False
        self.__class__.instances.append(self)

    def acquire(self):
        self.acquired = True

    def close(self):
        self.closed = True


class _FakeSupervisor:
    instances = []
    monitor_error = None

    def __init__(self, **kwargs):
        self.children = {}
        self.events = []
        self.__class__.instances.append(self)

    def request_shutdown(self, signum, frame=None):
        self.events.append(("signal", signum))

    def start(self, command):
        self.events.append(("start", command.role))
        child = SimpleNamespace(command=command, pgid=10_000 + len(self.children))
        self.children[command.role] = child
        return child

    def wait_for_tcp(self, *args, **kwargs):
        self.events.append(("wait-tcp", args[0]))

    def wait_for_http(self, *args, **kwargs):
        self.events.append(("wait-http", args[0]))

    def monitor(self, roles, *, health_check=None):
        self.events.append(("monitor", tuple(roles)))
        if health_check is not None:
            health_check()
            self.events.append(("health-check",))
        if self.monitor_error is not None:
            raise self.monitor_error

    def stop_roles(self, roles):
        self.events.append(("stop", tuple(roles)))

    def consume_shutdown_request(self):
        self.events.append(("consume-shutdown",))

    def wait_for_role(self, role, **kwargs):
        self.events.append(("wait-role", role))

    def shutdown(self):
        self.events.append(("shutdown",))


class _FakeGpuMonitor:
    instances = []

    def __init__(self, assignments, sample_path, summary_path, **kwargs):
        self.assignments = tuple(assignments)
        self.sample_path = sample_path
        self.summary_path = summary_path
        self.kwargs = kwargs
        self.started = False
        self.stopped = False
        self.process_groups = {}
        self.health_checks = 0
        self.__class__.instances.append(self)

    def start(self):
        self.started = True
        return True

    def register_process_group(self, role, pgid):
        self.process_groups[role] = pgid

    def raise_if_ownership_violated(self):
        self.health_checks += 1

    def stop(self):
        self.stopped = True
        return {"status": "ok"}


class TestLauncherOrchestration(FanoutManifestFixture):
    def setUp(self):
        super().setUp()
        self.write_manifest()
        _FakeReservationSet.instances.clear()
        _FakeSupervisor.instances.clear()
        _FakeSupervisor.monitor_error = None
        _FakeGpuMonitor.instances.clear()

    def run_launcher(self):
        with (
            mock.patch.object(launcher, "GPUReservationSet", _FakeReservationSet),
            mock.patch.object(launcher, "ProcessSupervisor", _FakeSupervisor),
            mock.patch.object(launcher, "wait_for_free_gpus"),
            mock.patch.object(launcher, "_prepare_run_directories"),
        ):
            return launcher.run_launcher(
                self.manifest_path,
                dry_run=False,
                wait_for_gpus_enabled=True,
                output=io.StringIO(),
            )

    def test_success_starts_dependencies_then_consumers_then_producer(self):
        self.assertEqual(self.run_launcher(), 0)
        events = _FakeSupervisor.instances[0].events
        starts = [event[1] for event in events if event[0] == "start"]
        self.assertEqual(
            starts,
            [
                "mooncake-master",
                "target-server",
                "consumer:dflash-g7",
                "consumer:dflash-uniform",
                "consumer:dpace-a05",
                "producer",
            ],
        )
        self.assertNotIn("cleanup", starts)
        self.assertEqual(events[-1], ("shutdown",))
        self.assertTrue(_FakeReservationSet.instances[0].closed)

    def test_enabled_gpu_monitor_tracks_cuda_roles_and_stops(self):
        payload = self.payload()
        payload["runtime"]["gpu_monitor"] = {
            "enabled": True,
            "poll_s": 0.25,
            "strict_process_ownership": True,
        }
        self.write_manifest(payload)
        with (
            mock.patch.object(launcher, "GPUReservationSet", _FakeReservationSet),
            mock.patch.object(launcher, "ProcessSupervisor", _FakeSupervisor),
            mock.patch.object(launcher, "GpuMonitor", _FakeGpuMonitor),
            mock.patch.object(launcher, "wait_for_free_gpus"),
            mock.patch.object(launcher, "_prepare_run_directories"),
        ):
            result = launcher.run_launcher(
                self.manifest_path,
                dry_run=False,
                wait_for_gpus_enabled=True,
                output=io.StringIO(),
            )

        self.assertEqual(result, 0)
        monitor = _FakeGpuMonitor.instances[0]
        self.assertTrue(monitor.started)
        self.assertTrue(monitor.stopped)
        self.assertEqual(monitor.kwargs["poll_s"], 0.25)
        self.assertEqual(
            [
                (value.gpu, value.logical_role, value.process_role)
                for value in monitor.assignments
            ],
            [
                (4, "producer", "target-server"),
                (3, "consumer:dflash-g7", "consumer:dflash-g7"),
                (5, "consumer:dflash-uniform", "consumer:dflash-uniform"),
                (7, "consumer:dpace-a05", "consumer:dpace-a05"),
            ],
        )
        self.assertEqual(
            set(monitor.process_groups),
            {
                "target-server",
                "consumer:dflash-g7",
                "consumer:dflash-uniform",
                "consumer:dpace-a05",
            },
        )
        self.assertEqual(monitor.health_checks, 1)

    def test_enabled_gpu_monitor_stops_after_worker_failure(self):
        payload = self.payload()
        payload["runtime"]["gpu_monitor"] = {
            "enabled": True,
            "poll_s": 0.25,
            "strict_process_ownership": True,
        }
        self.write_manifest(payload)
        _FakeSupervisor.monitor_error = LauncherError(
            "injected consumer failure", role="consumer:dflash-g7"
        )
        with (
            mock.patch.object(launcher, "GPUReservationSet", _FakeReservationSet),
            mock.patch.object(launcher, "ProcessSupervisor", _FakeSupervisor),
            mock.patch.object(launcher, "GpuMonitor", _FakeGpuMonitor),
            mock.patch.object(launcher, "wait_for_free_gpus"),
            mock.patch.object(launcher, "_prepare_run_directories"),
        ):
            with self.assertRaisesRegex(LauncherError, "injected consumer failure"):
                launcher.run_launcher(
                    self.manifest_path,
                    dry_run=False,
                    wait_for_gpus_enabled=True,
                    output=io.StringIO(),
                )

        self.assertTrue(_FakeGpuMonitor.instances[0].stopped)
        self.assertTrue(_FakeReservationSet.instances[0].closed)

    def test_worker_failure_stops_writers_then_runs_cleanup(self):
        root = LauncherError("injected consumer failure", role="consumer:dflash-g7")
        _FakeSupervisor.monitor_error = root
        with self.assertRaisesRegex(LauncherError, "injected consumer failure"):
            self.run_launcher()
        events = _FakeSupervisor.instances[0].events
        stop_index = next(i for i, event in enumerate(events) if event[0] == "stop")
        consume_index = events.index(("consume-shutdown",))
        cleanup_index = events.index(("start", "cleanup"))
        wait_index = events.index(("wait-role", "cleanup"))
        self.assertLess(stop_index, consume_index)
        self.assertLess(consume_index, cleanup_index)
        self.assertLess(cleanup_index, wait_index)
        self.assertEqual(events[-1], ("shutdown",))
        self.assertTrue(_FakeReservationSet.instances[0].closed)


class TestProcessSupervisor(unittest.TestCase):
    def setUp(self):
        self.workdir = tempfile.mkdtemp(prefix="process-supervisor-")

    def command(self, role, code, *, persistent=False):
        return RoleCommand(
            role=role,
            argv=(sys.executable, "-c", code),
            env=dict(os.environ),
            log_path=os.path.join(self.workdir, f"{role}.log"),
            persistent=persistent,
        )

    def supervisor(self):
        return ProcessSupervisor(
            termination_grace_s=0.5,
            kill_grace_s=0.5,
            poll_s=0.01,
            cwd=self.workdir,
            output=io.StringIO(),
        )

    def test_success_keeps_persistent_role_until_controlled_shutdown(self):
        supervisor = self.supervisor()
        persistent = supervisor.start(
            self.command("server", "import time; time.sleep(30)", persistent=True)
        )
        supervisor.start(self.command("worker", "raise SystemExit(0)"))
        supervisor.monitor(["worker"])
        self.assertIsNone(persistent.process.poll())
        supervisor.shutdown()
        self.assertIsNotNone(persistent.process.poll())

    def test_worker_failure_reaps_sleeping_peer(self):
        supervisor = self.supervisor()
        peer = supervisor.start(self.command("peer", "import time; time.sleep(30)"))
        supervisor.start(self.command("failed", "raise SystemExit(3)"))
        with self.assertRaisesRegex(LauncherError, "exited with 3"):
            supervisor.monitor(["peer", "failed"])
        supervisor.shutdown()
        self.assertIsNotNone(peer.process.poll())

    def test_shutdown_reaps_descendant_that_escapes_the_process_group(self):
        supervisor = self.supervisor()
        pid_path = os.path.join(self.workdir, "escaped.pid")
        code = (
            "import pathlib, subprocess, sys, time; "
            "child = subprocess.Popen("
            "[sys.executable, '-c', 'import time; time.sleep(30)'], "
            "start_new_session=True); "
            f"pathlib.Path({pid_path!r}).write_text(str(child.pid)); "
            "time.sleep(30)"
        )
        supervisor.start(self.command("parent", code))
        deadline = time.monotonic() + 2.0
        while not os.path.exists(pid_path) and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertTrue(os.path.exists(pid_path))
        escaped_pid = int(Path(pid_path).read_text(encoding="utf-8"))

        try:
            supervisor.shutdown()
        finally:
            try:
                os.kill(escaped_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        snapshot = ProcessSupervisor._proc_snapshot()
        self.assertTrue(escaped_pid not in snapshot or snapshot[escaped_pid][0] == "Z")

    def test_monitor_runs_external_health_check(self):
        supervisor = self.supervisor()
        supervisor.start(self.command("worker", "import time; time.sleep(30)"))

        def fail_health_check():
            raise LauncherError("GPU ownership changed", role="gpu-monitor")

        with self.assertRaisesRegex(LauncherError, "GPU ownership changed"):
            supervisor.monitor(["worker"], health_check=fail_health_check)
        supervisor.shutdown()

    def test_start_aborts_if_shutdown_arrives_during_pre_start_check(self):
        popen = mock.Mock()
        supervisor = ProcessSupervisor(
            termination_grace_s=0.5,
            kill_grace_s=0.5,
            poll_s=0.01,
            cwd=self.workdir,
            output=io.StringIO(),
            popen=popen,
        )

        def request_shutdown(_command):
            supervisor.request_shutdown(signal.SIGTERM)

        supervisor._pre_start_check = request_shutdown
        with self.assertRaisesRegex(LauncherError, "received signal"):
            supervisor.start(self.command("worker", "raise SystemExit(0)"))
        popen.assert_not_called()
        supervisor.shutdown()

    def test_wait_for_role_keeps_group_managed_and_observes_shutdown(self):
        supervisor = self.supervisor()
        child = supervisor.start(self.command("worker", "raise SystemExit(0)"))
        supervisor.wait_for_role("worker", timeout_s=1.0)
        self.assertFalse(child.log_handle.closed)
        supervisor.shutdown()
        self.assertTrue(child.log_handle.closed)

        supervisor = self.supervisor()
        supervisor.start(self.command("sleeping", "import time; time.sleep(30)"))
        supervisor.request_shutdown(signal.SIGTERM)
        self.assertEqual(supervisor.consume_shutdown_request(), signal.SIGTERM)
        self.assertIsNone(supervisor.consume_shutdown_request())
        supervisor.request_shutdown(signal.SIGTERM)
        with self.assertRaisesRegex(LauncherError, "received signal"):
            supervisor.wait_for_role("sleeping", timeout_s=10.0)
        supervisor.shutdown()


if __name__ == "__main__":
    unittest.main(verbosity=2)
