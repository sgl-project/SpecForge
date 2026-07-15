# coding=utf-8
"""Dependency-light contracts for the YAML-driven unified launcher."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from urllib.error import HTTPError

from pydantic import ValidationError

from specforge.cli import _config_for_role, main
from specforge.config import Config, apply_overrides
from specforge.launch_plan import (
    CommandSpec,
    LaunchPlan,
    ReadinessSpec,
    ServiceSpec,
    _http_ready,
    build_launch_plan,
    run_commands,
)
from specforge.training.capture_contract import ServerCaptureContract


def _config(*, mode="local_colocated", nproc=1, nnodes=1):
    raw = {
        "model": {"target_model_path": "target", "draft_model_config": "draft"},
        "data": {"prompts_path": "prompts.jsonl"},
        "training": {"strategy": "dflash", "max_steps": 1},
        "deployment": {
            "mode": mode,
            "trainer": {
                "nnodes": nnodes,
                "nproc_per_node": nproc,
                **({"master_addr": "trainer-0"} if nnodes > 1 else {}),
            },
        },
    }
    if mode == "disaggregated":
        raw["deployment"]["disaggregated"] = {
            "control_dir": "/shared/attempt-1",
            "backend": "mooncake",
            "server_urls": ["http://capture:30000"],
        }
    return Config.model_validate(raw)


def _offline_disaggregated_config(*, backend="mooncake", producer_segment_size=None):
    raw = _config(mode="disaggregated").model_dump()
    raw["data"] = {"hidden_states_path": "features"}
    raw["deployment"]["disaggregated"]["backend"] = backend
    raw["deployment"]["disaggregated"]["server_urls"] = []
    raw["training"]["server_urls"] = []
    if backend == "shared_dir":
        raw["deployment"]["disaggregated"]["store_root"] = "/shared/features"
    if producer_segment_size is not None:
        raw["deployment"]["disaggregated"][
            "producer_segment_size"
        ] = producer_segment_size
    return Config.model_validate(raw)


def _managed_config(control_dir, *, nproc=2, servers=None):
    raw = _config(mode="disaggregated", nproc=nproc).model_dump()
    raw["deployment"]["disaggregated"]["server_urls"] = []
    raw["training"]["server_urls"] = []
    raw["deployment"]["disaggregated"]["managed_local"] = {
        "trainer_cuda_visible_devices": [str(i + 4) for i in range(nproc)],
        "mooncake": {
            "rpc_port": 35551,
            "metadata_port": 35880,
            "metrics_port": 35903,
            "global_segment_size_bytes": 4096,
            "local_buffer_size_bytes": 1024,
        },
        "capture_servers": servers
        or [
            {
                "port": 30000,
                "cuda_visible_devices": ["0", "1"],
                "tp_size": 2,
            }
        ],
    }
    raw["deployment"]["disaggregated"]["control_dir"] = control_dir
    return Config.model_validate(raw)


CAPTURE_CONTRACT = ServerCaptureContract(
    method="dflash",
    aux_layer_ids=(1, 9, 17, 25, 33),
    target_hidden_size=4096,
    target_vocab_size=151936,
    draft_vocab_size=151936,
)


def _managed_plan(root):
    log_dir = os.path.join(root, "logs")
    return LaunchPlan(
        "managed_supervisor",
        "both",
        commands=(
            CommandSpec("producer", ("producer",)),
            CommandSpec("consumer", ("consumer",)),
        ),
        services=(
            ServiceSpec(
                CommandSpec("mooncake", ("mooncake",)),
                ReadinessSpec(
                    "mooncake",
                    "http://127.0.0.1:35880/metadata?key=health",
                    1,
                    tcp_host="127.0.0.1",
                    tcp_port=35551,
                ),
                os.path.join(log_dir, "mooncake.log"),
                0,
            ),
            ServiceSpec(
                CommandSpec("capture-server-0", ("capture",)),
                ReadinessSpec("http", "http://127.0.0.1:30000/health", 1),
                os.path.join(log_dir, "capture.log"),
                1,
            ),
        ),
        managed_root=root,
        managed_ports=(35551, 35880, 35903, 30000),
        shutdown_grace_s=0.01,
    )


MOONCAKE_ENV = {
    "MOONCAKE_METADATA_SERVER": "http://metadata:8080/metadata",
    "MOONCAKE_MASTER_SERVER_ADDR": "master:50051",
}


class _FakeProcess:
    _next_pid = 41000

    def __init__(self, poll_results, *, label=None, events=None):
        self.poll_results = list(poll_results)
        self.status = None
        self.terminated = False
        self.killed = False
        self.label = label
        self.events = events
        self.pid = self._next_pid
        type(self)._next_pid += 1

    def poll(self):
        if self.status is not None:
            return self.status
        if not self.poll_results:
            return None
        result = self.poll_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result is not None:
            self.status = result
        return result

    def terminate(self):
        self.terminated = True
        self.status = -15
        if self.events is not None:
            self.events.append(f"terminate:{self.label}")

    def kill(self):
        self.killed = True
        self.status = -9
        if self.events is not None:
            self.events.append(f"kill:{self.label}")

    def wait(self, timeout=None):
        return self.status


class LaunchPlanTest(unittest.TestCase):
    def test_local_multi_rank_self_launches_torchrun(self):
        plan = build_launch_plan(
            _config(nproc=4),
            config_path="run.yaml",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        self.assertEqual(plan.kind, "command")
        self.assertEqual(
            plan.commands[0].argv,
            (
                "torchrun",
                "--standalone",
                "--nproc_per_node",
                "4",
                "specforge",
                "train",
                "--config",
                "run.yaml",
                "--role",
                "all",
            ),
        )

    def test_production_command_uses_python_module_entrypoints(self):
        plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("training.batch_size=2",),
            env={},
        )
        self.assertEqual(
            plan.commands[0].argv,
            (
                sys.executable,
                "-m",
                "torch.distributed.run",
                "--standalone",
                "--nproc_per_node",
                "2",
                "--module",
                "specforge.cli",
                "train",
                "--config",
                "run.yaml",
                "--role",
                "all",
                "training.batch_size=2",
            ),
        )
        self.assertEqual(plan.commands[0].argv.count("training.batch_size=2"), 1)

    def test_existing_torchrun_becomes_a_worker_without_nested_launch(self):
        distributed = {
            "RANK": "1",
            "WORLD_SIZE": "4",
            "LOCAL_RANK": "1",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        plan = build_launch_plan(
            _config(nproc=4), config_path="run.yaml", env=distributed
        )
        self.assertEqual(plan.kind, "worker")
        self.assertEqual(plan.role, "all")
        self.assertEqual(plan.commands, ())

    def test_existing_torchrun_must_match_yaml_world_size(self):
        distributed = {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with self.assertRaisesRegex(ValueError, "does not match YAML"):
            build_launch_plan(_config(nproc=4), config_path="run.yaml", env=distributed)

    def test_disaggregated_auto_supervises_both_roles_on_one_node(self):
        plan = build_launch_plan(
            _config(mode="disaggregated", nproc=2),
            config_path="run.yaml",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env=MOONCAKE_ENV,
        )
        self.assertEqual((plan.kind, plan.role), ("supervisor", "both"))
        self.assertEqual(
            [item.label for item in plan.commands], ["producer", "consumer"]
        )
        self.assertEqual(plan.commands[0].argv[-2:], ("--role", "producer"))
        self.assertEqual(
            plan.commands[1].argv[:4],
            ("torchrun", "--standalone", "--nproc_per_node", "2"),
        )
        self.assertEqual(
            plan.commands[0].env["DISAGG_REF_CHANNEL"],
            "/shared/attempt-1/refs.jsonl",
        )
        self.assertEqual(
            plan.commands[1].env["DISAGG_DB"],
            "/shared/attempt-1/consumer.sqlite",
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")
        self.assertEqual(
            plan.commands[1].env["DISAGG_CLIENT_BUFFER_SIZE"], str(256 << 20)
        )

    def test_consumer_state_dir_keeps_refs_shared_and_state_node_local(self):
        raw = _config(mode="disaggregated", nproc=2).model_dump()
        raw["deployment"]["disaggregated"][
            "consumer_state_dir"
        ] = "/local/attempt-state"
        plan = build_launch_plan(
            Config.model_validate(raw),
            config_path="run.yaml",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env=MOONCAKE_ENV,
        )

        for command in plan.commands:
            self.assertEqual(
                command.env["DISAGG_REF_CHANNEL"],
                "/shared/attempt-1/refs.jsonl",
            )
            self.assertEqual(
                command.env["DISAGG_DB"],
                "/local/attempt-state/consumer.sqlite",
            )
            self.assertEqual(
                command.env["DISAGG_INBOX_DIR"],
                "/local/attempt-state/inboxes",
            )

    def test_consumer_state_dir_rejects_unsupported_topologies(self):
        cfg = _config(mode="disaggregated")
        invalid_cases = {
            "offline": (
                lambda raw: raw.update(data={"hidden_states_path": "features"}),
                "valid only for online disaggregated",
            ),
            "multi-node trainer": (
                lambda raw: raw["deployment"]["trainer"].update(
                    nnodes=2, master_addr="trainer-0"
                ),
                "requires deployment.trainer.nnodes=1",
            ),
            "explicit metadata db": (
                lambda raw: raw["training"].update(
                    metadata_db_path="/other/consumer.sqlite"
                ),
                "conflicts with",
            ),
            "surrounding whitespace": (
                lambda raw: raw["deployment"]["disaggregated"].update(
                    consumer_state_dir=" /local/attempt-state"
                ),
                "surrounding whitespace",
            ),
        }
        for name, (mutate, message) in invalid_cases.items():
            with self.subTest(case=name):
                raw = cfg.model_dump()
                raw["deployment"]["disaggregated"][
                    "consumer_state_dir"
                ] = "/local/attempt-state"
                mutate(raw)
                with self.assertRaisesRegex(ValidationError, message):
                    Config.model_validate(raw)

    def test_disaggregated_roles_are_independently_selectable(self):
        producer = build_launch_plan(
            _config(mode="disaggregated", nproc=4),
            config_path="run.yaml",
            requested_role="producer",
            env=MOONCAKE_ENV,
        )
        self.assertEqual((producer.kind, producer.role), ("worker", "producer"))

        consumer = build_launch_plan(
            _config(mode="disaggregated", nproc=4),
            config_path="run.yaml",
            requested_role="consumer",
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env=MOONCAKE_ENV,
        )
        self.assertEqual((consumer.kind, consumer.role), ("command", "consumer"))
        self.assertIn("--nproc_per_node", consumer.commands[0].argv)

    def test_managed_local_schema_round_trips_and_validates_resources(self):
        cfg = _managed_config("/fresh/managed-attempt")
        self.assertEqual(Config.model_validate(cfg.model_dump()), cfg)
        managed = cfg.deployment.disaggregated.managed_local
        self.assertEqual(managed.trainer_cuda_visible_devices, ["4", "5"])
        self.assertEqual(managed.capture_servers[0].cuda_visible_devices, ["0", "1"])

        invalid_cases = {
            "trainer count": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"].update(
                    trainer_cuda_visible_devices=["4"]
                ),
                "count must equal",
            ),
            "overlapping devices": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"].update(
                    trainer_cuda_visible_devices=["0", "5"]
                ),
                "must be disjoint",
            ),
            "tp mismatch": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"][
                    "capture_servers"
                ][0].update(tp_size=1),
                "tp_size must equal",
            ),
            "duplicate capture port": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"].update(
                    capture_servers=[
                        {
                            "port": 30000,
                            "cuda_visible_devices": ["0"],
                            "tp_size": 1,
                        },
                        {
                            "port": 30000,
                            "cuda_visible_devices": ["1"],
                            "tp_size": 1,
                        },
                    ]
                ),
                "ports must be unique",
            ),
            "mooncake port collision": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"][
                    "mooncake"
                ].update(rpc_port=30000),
                "must not overlap",
            ),
            "invalid local hostname": (
                lambda raw: raw["deployment"]["disaggregated"]["managed_local"][
                    "mooncake"
                ].update(local_hostname=" 127.0.0.1"),
                "surrounding whitespace",
            ),
        }
        for name, (mutate, message) in invalid_cases.items():
            with self.subTest(case=name):
                raw = cfg.model_dump()
                mutate(raw)
                with self.assertRaisesRegex(ValidationError, message):
                    Config.model_validate(raw)

    def test_managed_local_rejects_external_and_nonlocal_modes(self):
        cfg = _managed_config("/fresh/managed-attempt")
        invalid_cases = {
            "server urls": (
                lambda raw: (
                    raw["deployment"]["disaggregated"].update(
                        server_urls=["http://external:30000"]
                    ),
                    raw["training"].update(server_urls=["http://external:30000"]),
                ),
                "derives capture server URLs",
            ),
            "mooncake endpoint": (
                lambda raw: raw["deployment"]["disaggregated"].update(
                    mooncake_metadata_server="http://external/metadata"
                ),
                "derives Mooncake endpoints",
            ),
            "offline": (
                lambda raw: raw.update(data={"hidden_states_path": "features"}),
                "online capture only",
            ),
            "multi-node": (
                lambda raw: raw["deployment"]["trainer"].update(
                    nnodes=2, master_addr="trainer-0"
                ),
                "nnodes=1",
            ),
            "resume": (
                lambda raw: raw["training"].update(resume_from="checkpoint"),
                "does not support resume",
            ),
            "persisted role": (
                lambda raw: raw["training"].update(role="producer"),
                "persisted training role",
            ),
            "VLM": (
                lambda raw: (
                    raw["training"].update(strategy="eagle3"),
                    raw["model"].update(
                        input_modality="qwen2_5_vl",
                        vocab_mapping_path="vocab-map.pt",
                    ),
                ),
                "requires deployment_mode=local_colocated",
            ),
            "SGLang DP": (
                lambda raw: raw["model"].update(sglang_enable_dp_attention=True),
                "do not support SGLang DP",
            ),
            "capture context too short": (
                lambda raw: raw["model"].update(
                    sglang_context_length=raw["data"]["max_length"] + 6
                ),
                r"data.max_length \+ 7",
            ),
        }
        for name, (mutate, message) in invalid_cases.items():
            with self.subTest(case=name):
                raw = cfg.model_dump()
                mutate(raw)
                with self.assertRaisesRegex(ValidationError, message):
                    Config.model_validate(raw)

    def test_managed_local_accepts_minimum_explicit_capture_context(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _managed_config(os.path.join(root, "attempt"))
            raw = cfg.model_dump()
            raw["data"]["max_length"] = 128
            raw["model"]["sglang_context_length"] = 135
            validated = Config.model_validate(raw)
            with mock.patch(
                "specforge.training.capture_contract.resolve_server_capture_contract",
                return_value=CAPTURE_CONTRACT,
            ):
                plan = build_launch_plan(validated, config_path="run.yaml", env={})

        argv = plan.services[1].command.argv
        self.assertEqual(argv[argv.index("--context-length") + 1], "135")

    def test_managed_local_plan_owns_mooncake_and_multiple_capture_servers(self):
        servers = [
            {
                "port": 30000,
                "cuda_visible_devices": ["0", "1"],
                "tp_size": 2,
            },
            {
                "port": 30001,
                "cuda_visible_devices": ["2", "3"],
                "tp_size": 2,
                "attention_backend": "triton",
            },
        ]
        with tempfile.TemporaryDirectory() as root:
            control_dir = os.path.join(root, "attempt")
            cfg = _managed_config(control_dir, servers=servers)
            raw = cfg.model_dump()
            raw["model"].update(
                torch_dtype="float32",
                cache_dir="/models/cache",
                sglang_attention_backend="fa3",
                sglang_enable_nccl_nvls=True,
                sglang_enable_symm_mem=True,
                sglang_enable_torch_compile=True,
                sglang_max_running_requests=64,
                sglang_max_total_tokens=8192,
            )
            cfg = Config.model_validate(raw)
            with mock.patch(
                "specforge.training.capture_contract.resolve_server_capture_contract",
                return_value=CAPTURE_CONTRACT,
            ):
                plan = build_launch_plan(
                    cfg,
                    config_path="run.yaml",
                    worker_prefix=("specforge",),
                    torchrun_prefix=("torchrun",),
                    env={},
                )

        self.assertEqual((plan.kind, plan.role), ("managed_supervisor", "both"))
        self.assertEqual(
            [service.command.label for service in plan.services],
            ["mooncake", "capture-server-0", "capture-server-1"],
        )
        self.assertEqual([service.phase for service in plan.services], [0, 1, 1])
        self.assertEqual(plan.managed_ports, (35551, 35880, 35903, 30000, 30001))
        mooncake = plan.services[0]
        self.assertEqual(
            mooncake.command.argv,
            (
                "mooncake_master",
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=127.0.0.1",
                "--rpc_port=35551",
                "--http_metadata_server_port=35880",
                "--metrics_port=35903",
            ),
        )
        self.assertEqual(mooncake.readiness.kind, "mooncake")
        for index, service in enumerate(plan.services[1:]):
            argv = service.command.argv
            self.assertIn("--enable-spec-capture", argv)
            self.assertEqual(argv[argv.index("--dtype") + 1], "float32")
            self.assertEqual(argv[argv.index("--download-dir") + 1], "/models/cache")
            self.assertEqual(argv[argv.index("--spec-capture-method") + 1], "dflash")
            layer_index = argv.index("--spec-capture-aux-layer-ids")
            self.assertEqual(
                argv[layer_index + 1 : layer_index + 6],
                ("1", "9", "17", "25", "33"),
            )
            self.assertEqual(
                service.command.env["CUDA_VISIBLE_DEVICES"],
                "0,1" if index == 0 else "2,3",
            )
            self.assertEqual(
                argv[argv.index("--attention-backend") + 1],
                "fa3" if index == 0 else "triton",
            )
            for flag in (
                "--enable-nccl-nvls",
                "--enable-symm-mem",
                "--enable-torch-compile",
            ):
                self.assertIn(flag, argv)
            self.assertEqual(argv[argv.index("--max-running-requests") + 1], "64")
            self.assertEqual(argv[argv.index("--max-total-tokens") + 1], "8192")
            self.assertEqual(argv[argv.index("--context-length") + 1], "2055")
        producer, consumer = plan.commands
        expected_urls = "http://127.0.0.1:30000,http://127.0.0.1:30001"
        self.assertEqual(producer.env["DISAGG_SERVER_URLS"], expected_urls)
        self.assertEqual(producer.env["CUDA_VISIBLE_DEVICES"], "")
        self.assertEqual(consumer.env["CUDA_VISIBLE_DEVICES"], "4,5")
        self.assertEqual(consumer.env["MOONCAKE_MASTER_SERVER_ADDR"], "127.0.0.1:35551")
        rendered = json.loads(plan.render())
        self.assertEqual(len(rendered["services"]), 3)

    def test_multiserver_example_yaml_builds_the_managed_plan(self):
        path = (
            Path(__file__).resolve().parents[2]
            / "examples/configs/qwen3.6-27b-dflash-multiserver-disaggregated.yaml"
        )
        cfg = Config.from_file(str(path))
        with (
            mock.patch(
                "specforge.training.capture_contract.resolve_server_capture_contract",
                return_value=CAPTURE_CONTRACT,
            ),
            mock.patch("specforge.launch_plan.os.path.exists", return_value=False),
        ):
            plan = build_launch_plan(cfg, config_path=str(path), env={})

        self.assertEqual((plan.kind, plan.role), ("managed_supervisor", "both"))
        self.assertEqual(
            [
                service.command.env.get("CUDA_VISIBLE_DEVICES")
                for service in plan.services
            ],
            ["", "0,1", "2,3"],
        )
        self.assertEqual(plan.commands[1].env["CUDA_VISIBLE_DEVICES"], "4,5")
        for service in plan.services[1:]:
            argv = service.command.argv
            self.assertEqual(argv[argv.index("--context-length") + 1], "2055")

    def test_managed_local_plan_rejects_split_or_existing_runtime(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _managed_config(os.path.join(root, "attempt"))
            with self.assertRaisesRegex(ValueError, "only --role auto"):
                build_launch_plan(
                    cfg,
                    config_path="run.yaml",
                    requested_role="producer",
                    env={},
                )
            distributed = {
                "RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
            with self.assertRaisesRegex(ValueError, "existing torchrun"):
                build_launch_plan(cfg, config_path="run.yaml", env=distributed)

            os.makedirs(cfg.deployment.disaggregated.control_dir)
            with self.assertRaisesRegex(ValueError, "fresh control_dir"):
                build_launch_plan(cfg, config_path="run.yaml", env={})

    def test_managed_local_role_children_become_workers_without_recursion(self):
        with tempfile.TemporaryDirectory() as root:
            control_dir = os.path.join(root, "attempt")
            cfg = _managed_config(control_dir)
            with mock.patch(
                "specforge.training.capture_contract.resolve_server_capture_contract",
                return_value=CAPTURE_CONTRACT,
            ):
                parent = build_launch_plan(
                    cfg,
                    config_path="run.yaml",
                    worker_prefix=("specforge",),
                    torchrun_prefix=("torchrun",),
                    env={},
                )
            os.makedirs(os.path.join(control_dir, "logs"))

            producer = build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="producer",
                env=parent.commands[0].env,
            )
            self.assertEqual((producer.kind, producer.role), ("worker", "producer"))

            consumer_env = {
                **parent.commands[1].env,
                "RANK": "0",
                "WORLD_SIZE": "2",
                "LOCAL_RANK": "0",
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": "29500",
            }
            consumer = build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="consumer",
                env=consumer_env,
            )
            self.assertEqual((consumer.kind, consumer.role), ("worker", "consumer"))

            for role in ("producer", "consumer"):
                with self.subTest(projected_role=role):
                    projected = _config_for_role(cfg, role)
                    self.assertEqual(projected.training.role, role)
                    self.assertIsNone(projected.deployment.disaggregated.managed_local)

    def test_online_disaggregation_forces_server_owned_client_segments(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["producer_segment_size"] = 4096
        config = Config.model_validate(raw)
        plan = build_launch_plan(
            config,
            config_path="run.yaml",
            env={
                **MOONCAKE_ENV,
                "DISAGG_CLIENT_SEGMENT_SIZE": "8192",
            },
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")
        self.assertEqual(plan.commands[1].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")

    def test_offline_mooncake_uses_role_specific_segment_ownership(self):
        config = _offline_disaggregated_config(producer_segment_size=4096)
        plan = build_launch_plan(
            config,
            config_path="run.yaml",
            env=MOONCAKE_ENV,
        )
        self.assertEqual(plan.commands[0].env["DISAGG_CLIENT_SEGMENT_SIZE"], "4096")
        self.assertEqual(plan.commands[1].env["DISAGG_CLIENT_SEGMENT_SIZE"], "0")

    def test_offline_mooncake_producer_requires_a_positive_segment(self):
        config = _offline_disaggregated_config()
        with self.assertRaisesRegex(ValueError, "positive.*producer_segment_size"):
            build_launch_plan(
                config,
                config_path="run.yaml",
                requested_role="producer",
                env=MOONCAKE_ENV,
            )

        fallback = build_launch_plan(
            config,
            config_path="run.yaml",
            requested_role="producer",
            env={**MOONCAKE_ENV, "MOONCAKE_GLOBAL_SEGMENT_SIZE": "8192"},
        )
        self.assertEqual(fallback.worker_env["DISAGG_CLIENT_SEGMENT_SIZE"], "8192")

    def test_offline_shared_directory_does_not_set_mooncake_segments(self):
        plan = build_launch_plan(
            _offline_disaggregated_config(backend="shared_dir"),
            config_path="run.yaml",
            env={},
        )
        for command in plan.commands:
            self.assertNotIn("DISAGG_CLIENT_SEGMENT_SIZE", command.env)
            self.assertNotIn("DISAGG_CLIENT_BUFFER_SIZE", command.env)

    def test_capture_server_urls_are_required_only_for_producers(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["server_urls"] = []
        raw["training"]["server_urls"] = []
        cfg = Config.model_validate(raw)
        with self.assertRaisesRegex(ValueError, "requires server URLs"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="producer",
                env=MOONCAKE_ENV,
            )
        consumer = build_launch_plan(
            cfg,
            config_path="run.yaml",
            requested_role="consumer",
            env=MOONCAKE_ENV,
        )
        self.assertEqual(consumer.role, "consumer")

    def test_multi_node_auto_fails_instead_of_remote_spawning(self):
        cfg = _config(mode="disaggregated", nproc=2, nnodes=2)
        with self.assertRaisesRegex(ValueError, "one trainer node only"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                node_rank=0,
                env=MOONCAKE_ENV,
            )

    def test_multi_node_producer_ignores_scheduler_node_rank(self):
        plan = build_launch_plan(
            _config(mode="disaggregated", nproc=2, nnodes=2),
            config_path="run.yaml",
            requested_role="producer",
            env={**MOONCAKE_ENV, "NODE_RANK": "99"},
        )
        self.assertEqual((plan.kind, plan.role), ("worker", "producer"))

    def test_multi_node_consumer_requires_rank_before_database_checks(self):
        with self.assertRaisesRegex(ValueError, "requires --node-rank"):
            build_launch_plan(
                _config(mode="disaggregated", nproc=2, nnodes=2),
                config_path="run.yaml",
                requested_role="consumer",
                env=MOONCAKE_ENV,
            )

    def test_partial_distributed_environment_fails_before_launch(self):
        with self.assertRaisesRegex(ValueError, "environment is incomplete"):
            build_launch_plan(_config(), config_path="run.yaml", env={"RANK": "0"})

    def test_multi_rank_torchrun_rejects_duplicate_producers(self):
        distributed = {
            "RANK": "0",
            "WORLD_SIZE": "2",
            "LOCAL_RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": "29500",
        }
        with self.assertRaisesRegex(ValueError, "direct single process"):
            build_launch_plan(
                _config(mode="disaggregated", nproc=2),
                config_path="run.yaml",
                requested_role="producer",
                env={**distributed, **MOONCAKE_ENV},
            )

    def test_online_database_freshness_and_resume_are_checked(self):
        with tempfile.TemporaryDirectory() as root:
            cfg = _config(mode="disaggregated")
            raw = cfg.model_dump()
            raw["deployment"]["disaggregated"]["control_dir"] = root
            cfg = Config.model_validate(raw)
            database = os.path.join(root, "consumer.sqlite")
            open(database, "a", encoding="utf-8").close()
            with self.assertRaisesRegex(ValueError, "fresh attempt path"):
                build_launch_plan(
                    cfg,
                    config_path="run.yaml",
                    requested_role="consumer",
                    env=MOONCAKE_ENV,
                )

            raw = cfg.model_dump()
            raw["training"]["resume_from"] = os.path.join(root, "latest")
            resumed = Config.model_validate(raw)
            plan = build_launch_plan(
                resumed,
                config_path="run.yaml",
                env=MOONCAKE_ENV,
            )
            self.assertEqual(plan.role, "consumer")

            os.remove(database)
            with self.assertRaisesRegex(ValueError, "retained metadata database"):
                build_launch_plan(
                    resumed,
                    config_path="run.yaml",
                    requested_role="consumer",
                    env=MOONCAKE_ENV,
                )

    def test_new_deployment_surface_rejects_legacy_conflicts(self):
        raw = _config().model_dump()
        raw["training"]["deployment_mode"] = "disaggregated"
        with self.assertRaisesRegex(ValidationError, "conflicts"):
            Config.model_validate(raw)

    def test_online_disaggregated_schema_rejects_shared_directory_store(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["deployment"]["disaggregated"]["backend"] = "shared_dir"
        raw["deployment"]["disaggregated"]["store_root"] = "/shared/features"
        with self.assertRaisesRegex(ValidationError, "backend=mooncake"):
            Config.model_validate(raw)

    def test_deployment_server_url_override_updates_assembly_view(self):
        cfg = _config(mode="disaggregated")
        updated = apply_overrides(
            cfg,
            ["deployment.disaggregated.server_urls=[http://new-capture:31000]"],
        )
        self.assertEqual(
            updated.deployment.disaggregated.server_urls,
            ["http://new-capture:31000"],
        )
        self.assertEqual(
            updated.training.server_urls,
            ["http://new-capture:31000"],
        )

        legacy_updated = apply_overrides(
            updated,
            ["training.server_urls=[http://legacy-name:32000]"],
        )
        self.assertEqual(
            legacy_updated.deployment.disaggregated.server_urls,
            ["http://legacy-name:32000"],
        )

    def test_deployment_aliases_round_trip_and_survive_unrelated_overrides(self):
        cfg = _config(mode="disaggregated")
        self.assertEqual(Config.model_validate(cfg.model_dump()), cfg)
        updated = apply_overrides(cfg, ["training.batch_size=2"])
        self.assertEqual(updated.training.server_urls, cfg.training.server_urls)
        self.assertEqual(
            updated.deployment.disaggregated.server_urls,
            cfg.deployment.disaggregated.server_urls,
        )

    def test_conflicting_server_url_aliases_fail_loudly(self):
        raw = _config(mode="disaggregated").model_dump()
        raw["training"]["server_urls"] = ["http://legacy:30000"]
        with self.assertRaisesRegex(ValidationError, "server_urls conflicts"):
            Config.model_validate(raw)

    def test_cli_plan_is_json_and_does_not_execute(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "run.json")
            raw = _config(nproc=2).model_dump(mode="json")
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(raw, stream)
            with (
                mock.patch.dict(os.environ, {}, clear=True),
                mock.patch("specforge.launch_plan.run_commands") as run,
                mock.patch("builtins.print") as output,
            ):
                self.assertEqual(main(["train", "-c", path, "--plan"]), 0)
            run.assert_not_called()
            rendered = output.call_args.args[0]
            payload = json.loads(rendered)
            self.assertEqual(payload["kind"], "command")
            self.assertEqual(set(payload), {"kind", "role", "commands", "worker_env"})
            self.assertNotIn("--plan", payload["commands"][0]["argv"])

    def test_cli_producer_projection_ignores_consumer_only_settings(self):
        with tempfile.TemporaryDirectory() as root:
            path = os.path.join(root, "run.json")
            raw = _config(mode="disaggregated").model_dump(mode="json")
            raw["training"]["metadata_db_path"] = os.path.join(root, "consumer.sqlite")
            raw["profiling"]["enabled"] = True
            with open(path, "w", encoding="utf-8") as stream:
                json.dump(raw, stream)

            with (
                mock.patch.dict(os.environ, MOONCAKE_ENV, clear=True),
                mock.patch("specforge.cli._train", return_value=0) as train,
            ):
                self.assertEqual(
                    main(["train", "-c", path, "--role", "producer"]),
                    0,
                )

            projected = train.call_args.args[0]
            self.assertEqual(projected.training.role, "producer")
            self.assertIsNone(projected.training.metadata_db_path)
            self.assertFalse(projected.profiling.enabled)

    def test_plan_redacts_secret_overrides(self):
        plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("tracking.wandb_key=top-secret",),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        rendered = plan.render()
        self.assertNotIn("top-secret", rendered)
        self.assertIn("tracking.wandb_key=<redacted>", rendered)

        url_plan = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=("tracking.mlflow_tracking_uri=https://user:pass@mlflow.test/a",),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        )
        rendered = url_plan.render()
        self.assertNotIn("user:pass", rendered)
        self.assertIn("<redacted>@mlflow.test", rendered)

        malformed_port = build_launch_plan(
            _config(nproc=2),
            config_path="run.yaml",
            overrides=(
                "tracking.mlflow_tracking_uri=https://user:pass@mlflow.test:bad/a",
            ),
            worker_prefix=("specforge",),
            torchrun_prefix=("torchrun",),
            env={},
        ).render()
        self.assertNotIn("user:pass", malformed_port)

    def test_explicit_both_rejects_consumer_only_resume(self):
        cfg = _config(mode="disaggregated")
        raw = cfg.model_dump()
        raw["training"]["resume_from"] = "/checkpoint/latest"
        cfg = Config.model_validate(raw)
        with self.assertRaisesRegex(ValueError, "--role consumer"):
            build_launch_plan(
                cfg,
                config_path="run.yaml",
                requested_role="both",
                env=MOONCAKE_ENV,
            )

    def test_managed_supervisor_starts_services_by_phase_and_stops_in_reverse(self):
        events = []
        mooncake = _FakeProcess([None], label="mooncake", events=events)
        capture = _FakeProcess([None], label="capture", events=events)
        producer = _FakeProcess([0], label="producer", events=events)
        consumer = _FakeProcess([0], label="consumer", events=events)
        processes = iter((mooncake, capture, producer, consumer))
        with tempfile.TemporaryDirectory() as parent:
            plan = _managed_plan(os.path.join(parent, "attempt"))

            def ready(service, _process, started):
                events.append(f"ready:{service.command.label}")
                if service.command.label == "mooncake":
                    self.assertEqual(len(started), 1)
                else:
                    self.assertEqual(len(started), 2)

            with mock.patch(
                "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
            ):
                status = run_commands(
                    plan,
                    popen=lambda *_args, **_kwargs: next(processes),
                    managed_preflight=lambda _plan: None,
                    readiness_waiter=ready,
                )

        self.assertEqual(status, 0)
        self.assertEqual(events[:2], ["ready:mooncake", "ready:capture-server-0"])
        self.assertLess(
            events.index("terminate:capture"), events.index("terminate:mooncake")
        )

    def test_managed_supervisor_service_failure_stops_both_roles(self):
        mooncake = _FakeProcess([None])
        capture = _FakeProcess([7])
        producer = _FakeProcess([None])
        consumer = _FakeProcess([None])
        processes = iter((mooncake, capture, producer, consumer))
        with tempfile.TemporaryDirectory() as parent:
            plan = _managed_plan(os.path.join(parent, "attempt"))
            with mock.patch(
                "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
            ):
                status = run_commands(
                    plan,
                    popen=lambda *_args, **_kwargs: next(processes),
                    managed_preflight=lambda _plan: None,
                    readiness_waiter=lambda *_args: None,
                )
        self.assertEqual(status, 7)
        self.assertTrue(producer.terminated)
        self.assertTrue(consumer.terminated)
        self.assertTrue(mooncake.terminated)

    def test_managed_supervisor_allows_a_clean_producer_to_finish_early(self):
        mooncake = _FakeProcess([None])
        capture = _FakeProcess([None])
        producer = _FakeProcess([0])
        consumer = _FakeProcess([None, 0])
        processes = iter((mooncake, capture, producer, consumer))
        with tempfile.TemporaryDirectory() as parent:
            plan = _managed_plan(os.path.join(parent, "attempt"))
            with mock.patch(
                "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
            ):
                status = run_commands(
                    plan,
                    popen=lambda *_args, **_kwargs: next(processes),
                    managed_preflight=lambda _plan: None,
                    readiness_waiter=lambda *_args: None,
                )

        self.assertEqual(status, 0)
        self.assertFalse(consumer.terminated)
        self.assertTrue(mooncake.terminated)

    def test_managed_supervisor_signal_cleans_roles_and_services(self):
        handlers = {}
        events = []

        def set_signal(signum, handler):
            previous = handlers.get(signum, signal.SIG_DFL)
            handlers[signum] = handler
            return previous

        class SignalProcess(_FakeProcess):
            def __init__(self):
                super().__init__([None])
                self.sent = False

            def poll(self):
                if not self.sent:
                    self.sent = True
                    handlers[signal.SIGTERM](signal.SIGTERM, None)
                return super().poll()

        mooncake = _FakeProcess([None], label="mooncake", events=events)
        capture = _FakeProcess([None], label="capture", events=events)
        producer = SignalProcess()
        consumer = _FakeProcess([None])
        processes = iter((mooncake, capture, producer, consumer))
        with tempfile.TemporaryDirectory() as parent:
            plan = _managed_plan(os.path.join(parent, "attempt"))
            with (
                mock.patch(
                    "specforge.launch_plan.signal.signal", side_effect=set_signal
                ),
                mock.patch(
                    "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
                ),
            ):
                status = run_commands(
                    plan,
                    popen=lambda *_args, **_kwargs: next(processes),
                    managed_preflight=lambda _plan: None,
                    readiness_waiter=lambda *_args: None,
                )

        self.assertEqual(status, 128 + signal.SIGTERM)
        self.assertTrue(producer.terminated)
        self.assertTrue(consumer.terminated)
        self.assertLess(
            events.index("terminate:capture"), events.index("terminate:mooncake")
        )

    def test_managed_supervisor_cleans_services_after_readiness_failure(self):
        mooncake = _FakeProcess([None])
        capture = _FakeProcess([None])
        processes = iter((mooncake, capture))
        with tempfile.TemporaryDirectory() as parent:
            plan = _managed_plan(os.path.join(parent, "attempt"))

            def ready(service, *_args):
                if service.command.label.startswith("capture"):
                    raise TimeoutError("capture timeout")

            with (
                mock.patch(
                    "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
                ),
                self.assertRaisesRegex(TimeoutError, "capture timeout"),
            ):
                run_commands(
                    plan,
                    popen=lambda *_args, **_kwargs: next(processes),
                    managed_preflight=lambda _plan: None,
                    readiness_waiter=ready,
                )
        self.assertTrue(mooncake.terminated)
        self.assertTrue(capture.terminated)

    def test_managed_preflight_rejects_an_occupied_port_before_spawning(self):
        with tempfile.TemporaryDirectory() as parent:
            root = os.path.join(parent, "attempt")
            port = 30000
            plan = _managed_plan(root)
            plan = LaunchPlan(
                **{
                    **plan.__dict__,
                    "managed_ports": (port,),
                }
            )
            port_probe = mock.MagicMock()
            port_probe.__enter__.return_value = port_probe
            port_probe.bind.side_effect = OSError("occupied")
            with (
                mock.patch(
                    "specforge.launch_plan.shutil.which",
                    return_value="/usr/bin/mooncake_master",
                ),
                mock.patch(
                    "specforge.launch_plan.importlib.util.find_spec",
                    return_value=object(),
                ),
                mock.patch(
                    "specforge.launch_plan.socket.socket", return_value=port_probe
                ),
                self.assertRaisesRegex(RuntimeError, "is unavailable"),
            ):
                run_commands(plan, popen=mock.Mock())

    def test_mooncake_readiness_accepts_missing_key_but_rejects_server_errors(self):
        readiness = ReadinessSpec(
            "mooncake",
            "http://127.0.0.1:35880/metadata?key=missing",
            1,
            tcp_host="127.0.0.1",
            tcp_port=35551,
        )
        for status, expected in ((404, True), (500, False), (503, False)):
            with (
                self.subTest(status=status),
                mock.patch(
                    "specforge.launch_plan.urllib_request.urlopen",
                    side_effect=HTTPError(
                        readiness.url, status, "response", None, None
                    ),
                ),
            ):
                self.assertEqual(_http_ready(readiness), expected)

    def test_supervisor_terminates_a_sibling_after_child_failure(self):
        producer = _FakeProcess([None])
        consumer = _FakeProcess([7])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with mock.patch(
            "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
        ):
            status = run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertEqual(status, 7)
        self.assertTrue(producer.terminated)

    def test_clean_producer_exit_does_not_cancel_a_running_consumer(self):
        producer = _FakeProcess([0])
        consumer = _FakeProcess([None, 0])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with mock.patch("specforge.launch_plan.time.sleep"):
            status = run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertEqual(status, 0)
        self.assertFalse(consumer.terminated)

    def test_keyboard_interrupt_cleans_up_every_child(self):
        producer = _FakeProcess([KeyboardInterrupt()])
        consumer = _FakeProcess([None])
        processes = iter((producer, consumer))
        plan = LaunchPlan(
            "supervisor",
            "both",
            commands=(
                CommandSpec("producer", ("producer",)),
                CommandSpec("consumer", ("consumer",)),
            ),
        )
        with (
            mock.patch(
                "specforge.launch_plan.os.killpg", side_effect=ProcessLookupError
            ),
            self.assertRaises(KeyboardInterrupt),
        ):
            run_commands(plan, popen=lambda *_args, **_kwargs: next(processes))
        self.assertTrue(producer.terminated)
        self.assertTrue(consumer.terminated)

    @unittest.skipUnless(
        hasattr(os, "killpg") and hasattr(signal, "SIGTERM"),
        "requires POSIX process groups",
    )
    def test_sigterm_cleans_the_real_child_process_group(self):
        child_code = """
import os
import subprocess
import sys
import time

grandchild = subprocess.Popen(
    [sys.executable, "-c", "import time; time.sleep(60)"]
)
with open(sys.argv[1], "w", encoding="utf-8") as stream:
    stream.write(f"{os.getpid()} {grandchild.pid}")
time.sleep(60)
"""
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        with tempfile.TemporaryDirectory() as root:
            marker = os.path.join(root, "children.txt")
            runner_code = f"""
import sys
from specforge.launch_plan import CommandSpec, LaunchPlan, run_commands

plan = LaunchPlan(
    "command",
    "all",
    commands=(
        CommandSpec(
            "child",
            (sys.executable, "-c", {child_code!r}, {marker!r}),
        ),
    ),
)
raise SystemExit(run_commands(plan))
"""
            runner = subprocess.Popen(
                [sys.executable, "-c", runner_code],
                cwd=repo_root,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            child_pid = None
            try:
                deadline = time.monotonic() + 10
                while time.monotonic() < deadline:
                    if os.path.exists(marker):
                        with open(marker, encoding="utf-8") as stream:
                            raw = stream.read().strip()
                        if len(raw.split()) == 2:
                            child_pid = int(raw.split()[0])
                            break
                    if runner.poll() is not None:
                        break
                    time.sleep(0.02)
                self.assertIsNotNone(child_pid, "managed child never became ready")
                self.assertEqual(os.getpgid(child_pid), child_pid)

                os.kill(runner.pid, signal.SIGTERM)
                stdout, stderr = runner.communicate(timeout=10)
                self.assertEqual(
                    runner.returncode,
                    128 + signal.SIGTERM,
                    f"stdout={stdout!r} stderr={stderr!r}",
                )

                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    try:
                        os.killpg(child_pid, 0)
                    except ProcessLookupError:
                        break
                    time.sleep(0.02)
                else:
                    self.fail(f"managed process group {child_pid} survived SIGTERM")
            finally:
                if runner.poll() is None:
                    runner.kill()
                    runner.wait(timeout=5)
                if child_pid is not None:
                    try:
                        os.killpg(child_pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_signal_exit_status_is_normalized(self):
        process = _FakeProcess([-15])
        plan = LaunchPlan(
            "command",
            "consumer",
            commands=(CommandSpec("consumer", ("consumer",)),),
        )
        self.assertEqual(
            run_commands(plan, popen=lambda *_args, **_kwargs: process),
            143,
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)
