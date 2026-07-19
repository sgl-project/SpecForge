# coding=utf-8
"""Pure launch planning and process supervision for ``specforge train``."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Literal, Mapping, Optional, Sequence
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import urlsplit, urlunsplit

from specforge.config import SGLANG_CAPTURE_CONTEXT_HEADROOM, Config

if TYPE_CHECKING:
    from specforge.algorithms.registry import AlgorithmRegistration

LaunchRole = Literal["auto", "all", "producer", "consumer", "both"]
PlanKind = Literal["worker", "command", "supervisor", "managed_supervisor"]
ReadinessKind = Literal["http", "mooncake"]

_DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
_MANAGED_CHILD_ENV = "SPECFORGE_MANAGED_LOCAL_CHILD"
_SECRET_NAMES = (
    "auth_token",
    "password",
    "secret",
    "credential",
    "wandb_key",
    "swanlab_key",
)


class _ForwardedSignal(BaseException):
    def __init__(self, signum: int):
        self.signum = signum


def _redacted(value: str) -> str:
    name = None
    raw = value
    if "=" in value:
        name, raw = value.split("=", 1)
        if any(fragment in name.lower() for fragment in _SECRET_NAMES):
            return f"{name}=<redacted>"
    try:
        parsed = urlsplit(raw)
    except ValueError:
        parsed = None
    if parsed is not None and parsed.scheme and parsed.hostname and parsed.username:
        hostname = parsed.hostname
        if ":" in hostname and not hostname.startswith("["):
            hostname = f"[{hostname}]"
        try:
            port = parsed.port
        except ValueError:
            port = None
        if port is not None:
            hostname = f"{hostname}:{port}"
        raw = urlunsplit(
            (
                parsed.scheme,
                f"<redacted>@{hostname}",
                parsed.path,
                parsed.query,
                parsed.fragment,
            )
        )
    return f"{name}={raw}" if name is not None else raw


def _redacted_env(values: Mapping[str, str]) -> dict[str, str]:
    return {
        name: (
            "<redacted>"
            if any(fragment in name.lower() for fragment in _SECRET_NAMES)
            else _redacted(value)
        )
        for name, value in sorted(values.items())
    }


@dataclass(frozen=True)
class CommandSpec:
    label: str
    argv: tuple[str, ...]
    env: Mapping[str, str] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "label": self.label,
            "argv": [_redacted(value) for value in self.argv],
            "env": _redacted_env(self.env),
        }


@dataclass(frozen=True)
class ReadinessSpec:
    kind: ReadinessKind
    url: str
    timeout_s: float
    tcp_host: Optional[str] = None
    tcp_port: Optional[int] = None

    def as_dict(self) -> dict:
        return {
            "kind": self.kind,
            "url": _redacted(self.url),
            "timeout_s": self.timeout_s,
            "tcp_host": self.tcp_host,
            "tcp_port": self.tcp_port,
        }


@dataclass(frozen=True)
class ServiceSpec:
    command: CommandSpec
    readiness: ReadinessSpec
    log_path: str
    phase: int

    def as_dict(self) -> dict:
        return {
            "command": self.command.as_dict(),
            "readiness": self.readiness.as_dict(),
            "log_path": self.log_path,
            "phase": self.phase,
        }


@dataclass(frozen=True)
class LaunchPlan:
    kind: PlanKind
    role: Literal["all", "producer", "consumer", "both"]
    commands: tuple[CommandSpec, ...] = ()
    worker_env: Mapping[str, str] = field(default_factory=dict)
    services: tuple[ServiceSpec, ...] = ()
    managed_root: Optional[str] = None
    managed_ports: tuple[int, ...] = ()
    # SIGTERM-trapped workers run Mooncake drains, checkpoint flushes, and
    # failure-sentinel publication inside this window before SIGKILL.
    shutdown_grace_s: float = 30.0

    def render(self) -> str:
        payload = {
            "kind": self.kind,
            "role": self.role,
            "commands": [command.as_dict() for command in self.commands],
            "worker_env": _redacted_env(self.worker_env),
        }
        if self.kind == "managed_supervisor":
            payload.update(
                {
                    "services": [service.as_dict() for service in self.services],
                    "managed_root": self.managed_root,
                    "managed_ports": list(self.managed_ports),
                    "shutdown_grace_s": self.shutdown_grace_s,
                }
            )
        return json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )


def _distributed_state(env: Mapping[str, str]) -> bool:
    present = [name for name in _DIST_ENV if name in env]
    if present and len(present) != len(_DIST_ENV):
        missing = [name for name in _DIST_ENV if name not in env]
        raise ValueError(
            f"distributed environment is incomplete; present={present}, "
            f"missing={missing}"
        )
    return bool(present)


def _resolve_role(
    cfg: Config, requested: LaunchRole, *, distributed: bool
) -> Literal["all", "producer", "consumer", "both"]:
    disaggregated = cfg.deployment.mode == "disaggregated"
    if requested == "auto":
        legacy_role = cfg.training.role
        if disaggregated and legacy_role in ("producer", "consumer"):
            requested = legacy_role
        elif disaggregated and cfg.training.resume_from:
            requested = "consumer"
        else:
            requested = "both" if disaggregated else "all"

    if disaggregated and requested == "all":
        raise ValueError("disaggregated training does not support --role all")
    if not disaggregated and requested != "all":
        raise ValueError("--role producer/consumer/both requires disaggregated mode")
    if requested == "both" and cfg.training.resume_from:
        raise ValueError(
            "--role both cannot resume a disaggregated producer; use " "--role consumer"
        )
    if distributed and requested == "both":
        raise ValueError(
            "an existing torchrun environment requires an explicit "
            "--role consumer for disaggregated training"
        )
    return requested


def _resolved_node_rank(
    cfg: Config, node_rank: Optional[int], env: Mapping[str, str]
) -> Optional[int]:
    if node_rank is None:
        node_rank = cfg.deployment.trainer.node_rank
    if node_rank is None and env.get("NODE_RANK"):
        node_rank = int(env["NODE_RANK"])
    if node_rank is not None and not 0 <= node_rank < cfg.deployment.trainer.nnodes:
        raise ValueError(
            f"node_rank={node_rank} must be in [0, " f"{cfg.deployment.trainer.nnodes})"
        )
    return node_rank


def _offline_producer_segment_size(
    configured: Optional[int], base_env: Mapping[str, str]
) -> Optional[int]:
    if configured is not None:
        return configured
    raw = base_env.get("DISAGG_CLIENT_SEGMENT_SIZE")
    source = "DISAGG_CLIENT_SEGMENT_SIZE"
    if not raw:
        raw = base_env.get("MOONCAKE_GLOBAL_SEGMENT_SIZE")
        source = "MOONCAKE_GLOBAL_SEGMENT_SIZE"
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(f"{source} must be a positive integer, got {raw!r}") from exc
    if value <= 0:
        raise ValueError(f"{source} must be positive for an offline producer")
    return value


def _disaggregated_env(
    cfg: Config,
    base_env: Mapping[str, str],
    *,
    role: Literal["producer", "consumer"],
) -> dict[str, str]:
    deployment = cfg.deployment.disaggregated
    if deployment is None:
        # Legacy configs continue to consume their existing environment contract.
        return {}

    control_dir = Path(deployment.control_dir)
    consumer_state_dir = Path(deployment.consumer_state_dir or control_dir)
    values: dict[str, str] = {
        "DISAGG_BACKEND": deployment.backend,
        "DISAGG_STORE_ID": deployment.store_id or cfg.run_id,
    }
    if cfg.mode == "online":
        if deployment.backend != "mooncake":
            raise ValueError("online disaggregated training requires Mooncake")
        # Online feature objects are allocated by the external capture server.
        # SpecForge roles only read or publish references to those objects.
        values["DISAGG_CLIENT_SEGMENT_SIZE"] = "0"
        values.update(
            {
                "DISAGG_REF_CHANNEL": str(control_dir / "refs.jsonl"),
                "DISAGG_DB": str(consumer_state_dir / "consumer.sqlite"),
                # SQLite/WAL stays on rank 0's local filesystem.  Inboxes are
                # ordinary append-only channels and must remain visible to
                # ranks on every trainer node.
                "DISAGG_INBOX_DIR": str(
                    (
                        control_dir
                        if cfg.deployment.trainer.nnodes > 1
                        else consumer_state_dir
                    )
                    / "inboxes"
                ),
            }
        )
    else:
        values["DISAGG_MANIFEST"] = str(control_dir / "manifest.json")
        if deployment.backend == "mooncake":
            if role == "producer":
                segment_size = _offline_producer_segment_size(
                    deployment.producer_segment_size, base_env
                )
                if segment_size is None:
                    raise ValueError(
                        "offline Mooncake producer requires a positive "
                        "deployment.disaggregated.producer_segment_size or "
                        "MOONCAKE_GLOBAL_SEGMENT_SIZE"
                    )
                values["DISAGG_CLIENT_SEGMENT_SIZE"] = str(segment_size)
            else:
                values["DISAGG_CLIENT_SEGMENT_SIZE"] = "0"
    if deployment.store_root:
        values["DISAGG_STORE_ROOT"] = deployment.store_root
    if deployment.backend == "mooncake":
        values["DISAGG_CLIENT_BUFFER_SIZE"] = str(deployment.client_buffer_size)

    optional_values = {
        "MOONCAKE_METADATA_SERVER": deployment.mooncake_metadata_server,
        "MOONCAKE_MASTER_SERVER_ADDR": deployment.mooncake_master_server_addr,
        "MOONCAKE_LOCAL_HOSTNAME": deployment.mooncake_local_hostname,
        "MOONCAKE_PROTOCOL": deployment.mooncake_protocol,
        "MOONCAKE_RDMA_DEVICES": deployment.mooncake_rdma_devices,
        "DISAGG_IDLE_TIMEOUT": deployment.idle_timeout_s,
        "DISAGG_PEER_WAIT_TIMEOUT": deployment.peer_wait_timeout_s,
        "DISAGG_PRODUCER_HOLD_S": deployment.producer_hold_s,
    }
    for name, configured in optional_values.items():
        value = base_env.get(name, configured)
        if value is not None:
            values[name] = str(value)

    if deployment.backend == "mooncake":
        required = ("MOONCAKE_METADATA_SERVER", "MOONCAKE_MASTER_SERVER_ADDR")
        missing = [
            name for name in required if not values.get(name) and not base_env.get(name)
        ]
        if missing:
            raise ValueError(
                "Mooncake endpoints must be provided by deployment config or "
                f"environment: {missing}"
            )
    return values


def _managed_local_environment(cfg: Config) -> dict[str, str]:
    deployment = cfg.deployment.disaggregated
    assert deployment is not None and deployment.managed_local is not None
    managed = deployment.managed_local
    mooncake = managed.mooncake
    server_urls = [
        f"http://127.0.0.1:{server.port}" for server in managed.capture_servers
    ]
    values = {
        "MOONCAKE_METADATA_SERVER": (
            f"http://127.0.0.1:{mooncake.metadata_port}/metadata"
        ),
        "MOONCAKE_MASTER_SERVER_ADDR": f"127.0.0.1:{mooncake.rpc_port}",
        "MOONCAKE_LOCAL_HOSTNAME": mooncake.local_hostname,
        "MOONCAKE_PROTOCOL": mooncake.protocol,
        "DISAGG_SERVER_URLS": ",".join(server_urls),
    }
    if mooncake.rdma_devices:
        values["MOONCAKE_RDMA_DEVICES"] = mooncake.rdma_devices
    return values


def _managed_local_services(
    cfg: Config,
    *,
    algorithm: "AlgorithmRegistration",
) -> tuple[ServiceSpec, ...]:
    from specforge.training.capture_contract import resolve_server_capture_contract

    deployment = cfg.deployment.disaggregated
    assert deployment is not None and deployment.managed_local is not None
    managed = deployment.managed_local
    mooncake = managed.mooncake
    control_dir = Path(deployment.control_dir)
    log_dir = control_dir / "logs"
    shared_env = _managed_local_environment(cfg)
    capture_context_length = cfg.model.sglang_context_length or (
        cfg.data.max_length + SGLANG_CAPTURE_CONTEXT_HEADROOM
    )

    mooncake_service = ServiceSpec(
        command=CommandSpec(
            "mooncake",
            (
                "mooncake_master",
                "--enable_http_metadata_server=true",
                "--http_metadata_server_host=127.0.0.1",
                f"--rpc_port={mooncake.rpc_port}",
                f"--http_metadata_server_port={mooncake.metadata_port}",
                f"--metrics_port={mooncake.metrics_port}",
            ),
            {"CUDA_VISIBLE_DEVICES": ""},
        ),
        readiness=ReadinessSpec(
            "mooncake",
            shared_env["MOONCAKE_METADATA_SERVER"] + "?key=specforge-health-check",
            mooncake.startup_timeout_s,
            tcp_host="127.0.0.1",
            tcp_port=mooncake.rpc_port,
        ),
        log_path=str(log_dir / "mooncake.log"),
        phase=0,
    )

    contract = resolve_server_capture_contract(cfg, algorithm=algorithm)
    capture_services = []
    for index, server in enumerate(managed.capture_servers):
        argv = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            cfg.model.target_model_path,
            "--dtype",
            cfg.model.torch_dtype,
        ]
        if cfg.model.trust_remote_code:
            argv.append("--trust-remote-code")
        if cfg.model.cache_dir:
            argv.extend(("--download-dir", cfg.model.cache_dir))
        argv.extend(
            [
                "--skip-tokenizer-init",
                "--tp-size",
                str(server.tp_size),
                "--context-length",
                str(capture_context_length),
                "--mem-fraction-static",
                str(
                    server.mem_fraction_static
                    if server.mem_fraction_static is not None
                    else cfg.model.sglang_mem_fraction_static
                ),
                "--ep-size",
                str(cfg.model.sglang_ep_size),
                "--chunked-prefill-size",
                "-1",
                "--enable-spec-capture",
                "--spec-capture-method",
                contract.method,
                "--spec-capture-aux-layer-ids",
                *[str(layer) for layer in contract.aux_layer_ids],
                "--host",
                "127.0.0.1",
                "--port",
                str(server.port),
                "--attention-backend",
                server.attention_backend or cfg.model.sglang_attention_backend,
            ]
        )
        for enabled, flag in (
            (cfg.model.sglang_enable_nccl_nvls, "--enable-nccl-nvls"),
            (cfg.model.sglang_enable_symm_mem, "--enable-symm-mem"),
            (cfg.model.sglang_enable_torch_compile, "--enable-torch-compile"),
        ):
            if enabled:
                argv.append(flag)
        for value, flag in (
            (cfg.model.sglang_max_running_requests, "--max-running-requests"),
            (cfg.model.sglang_max_total_tokens, "--max-total-tokens"),
            (cfg.model.sglang_dp_size, "--dp-size"),
            (cfg.model.sglang_moe_a2a_backend, "--moe-a2a-backend"),
            (cfg.model.sglang_moe_runner_backend, "--moe-runner-backend"),
            (cfg.model.sglang_page_size, "--page-size"),
            (cfg.model.sglang_quantization, "--quantization"),
            (
                cfg.model.sglang_fp4_gemm_runner_backend,
                "--fp4-gemm-runner-backend",
            ),
            (
                cfg.model.sglang_mamba_radix_cache_strategy,
                "--mamba-radix-cache-strategy",
            ),
            (cfg.model.sglang_max_mamba_cache_size, "--max-mamba-cache-size"),
            (
                cfg.model.sglang_swa_full_tokens_ratio,
                "--swa-full-tokens-ratio",
            ),
            (
                cfg.model.sglang_mamba_full_memory_ratio,
                "--mamba-full-memory-ratio",
            ),
        ):
            if value is not None:
                argv.extend((flag, str(value)))
        service_env = {
            **shared_env,
            "CUDA_VISIBLE_DEVICES": ",".join(server.cuda_visible_devices),
            "FLASHINFER_DISABLE_VERSION_CHECK": "1",
            "MOONCAKE_GLOBAL_SEGMENT_SIZE": str(mooncake.global_segment_size_bytes),
            "MOONCAKE_LOCAL_BUFFER_SIZE": str(mooncake.local_buffer_size_bytes),
        }
        capture_services.append(
            ServiceSpec(
                command=CommandSpec(
                    f"capture-server-{index}", tuple(argv), service_env
                ),
                readiness=ReadinessSpec(
                    "http",
                    f"http://127.0.0.1:{server.port}/health",
                    server.startup_timeout_s,
                ),
                log_path=str(log_dir / f"capture-server-{index}.log"),
                phase=1,
            )
        )
    return (mooncake_service, *capture_services)


def _worker_argv(
    command_prefix: Sequence[str],
    config_path: str,
    role: Literal["all", "producer", "consumer"],
    overrides: Sequence[str],
) -> list[str]:
    return [
        *command_prefix,
        "train",
        "--config",
        config_path,
        "--role",
        role,
        *overrides,
    ]


def _trainer_command(
    cfg: Config,
    *,
    config_path: str,
    role: Literal["all", "consumer"],
    overrides: Sequence[str],
    worker_prefix: Sequence[str],
    torchrun_prefix: Sequence[str],
    distributed_entry: Sequence[str],
    node_rank: Optional[int],
    env: Mapping[str, str],
) -> CommandSpec:
    topology = cfg.deployment.trainer
    worker = _worker_argv(worker_prefix, config_path, role, overrides)
    world_size = topology.nnodes * topology.nproc_per_node
    cfg.validate_world_size(world_size)
    if world_size == 1:
        return CommandSpec(role, tuple(worker), env)
    if topology.nnodes == 1:
        launch = [
            *torchrun_prefix,
            "--standalone",
            "--nproc_per_node",
            str(topology.nproc_per_node),
        ]
    else:
        if node_rank is None:
            raise ValueError(
                "multi-node training requires --node-rank on each trainer node"
            )
        launch = [
            *torchrun_prefix,
            "--nnodes",
            str(topology.nnodes),
            "--node_rank",
            str(node_rank),
            "--master_addr",
            str(topology.master_addr),
            "--master_port",
            str(topology.master_port),
            "--nproc_per_node",
            str(topology.nproc_per_node),
        ]
    worker_args = worker[len(worker_prefix) :]
    return CommandSpec(role, tuple([*launch, *distributed_entry, *worker_args]), env)


def _validate_consumer_database(
    cfg: Config,
    *,
    role: Literal["all", "producer", "consumer", "both"],
    launch_env: Mapping[str, str],
    base_env: Mapping[str, str],
    distributed: bool,
    node_rank: Optional[int],
) -> None:
    if (
        cfg.mode != "online"
        or cfg.deployment.mode != "disaggregated"
        or role not in ("consumer", "both")
    ):
        return
    database = launch_env.get("DISAGG_DB") or base_env.get("DISAGG_DB")
    deployment = cfg.deployment.disaggregated
    if not database and deployment is not None:
        state_dir = deployment.consumer_state_dir or deployment.control_dir
        database = str(Path(state_dir) / "consumer.sqlite")
    if not database:
        raise ValueError("online disaggregated consumer requires a metadata database")
    state_owner = int(base_env["RANK"]) == 0 if distributed else node_rank in (None, 0)
    if cfg.training.resume_from:
        if state_owner and not os.path.exists(database):
            raise ValueError(
                "consumer resume requires the retained metadata database: "
                f"{database}"
            )
        return
    stale = [
        path
        for path in (database, f"{database}-wal", f"{database}-shm")
        if os.path.exists(path)
    ]
    if state_owner and stale:
        raise ValueError(
            "online consumer metadata database must use a fresh attempt path; "
            f"found {stale}"
        )


def _validate_capture_urls(
    cfg: Config,
    *,
    role: Literal["all", "producer", "consumer", "both"],
    base_env: Mapping[str, str],
) -> None:
    if (
        cfg.mode != "online"
        or cfg.deployment.mode != "disaggregated"
        or role not in ("producer", "both")
    ):
        return
    deployment = cfg.deployment.disaggregated
    if deployment is not None and deployment.managed_local is not None:
        return
    if deployment is not None and deployment.server_urls:
        return
    if base_env.get("DISAGG_SERVER_URLS") or base_env.get("DISAGG_SERVER_URL"):
        return
    raise ValueError(
        "online disaggregated producer requires server URLs in "
        "deployment.disaggregated.server_urls or DISAGG_SERVER_URL(S)"
    )


def build_launch_plan(
    cfg: Config,
    *,
    algorithm: Optional["AlgorithmRegistration"] = None,
    config_path: str,
    overrides: Sequence[str] = (),
    requested_role: LaunchRole = "auto",
    node_rank: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    worker_prefix: Optional[Sequence[str]] = None,
    torchrun_prefix: Optional[Sequence[str]] = None,
) -> LaunchPlan:
    """Resolve one validated config into a side-effect-free process plan."""
    if cfg.mode == "online":
        if cfg.deployment.mode != "disaggregated":
            raise ValueError(
                "online launch planning requires disaggregated producer/consumer "
                "mode; colocated online training is no longer supported"
            )
        if cfg.model.target_backend != "sglang":
            raise ValueError(
                "online launch planning requires an external SGLang capture server"
            )
    base_env = os.environ if env is None else env
    distributed = _distributed_state(base_env)
    deployment = cfg.deployment.disaggregated
    managed_local = deployment.managed_local if deployment is not None else None
    if managed_local is not None and algorithm is None:
        raise ValueError(
            "managed_local launch planning requires a resolved algorithm registration"
        )
    managed_child = base_env.get(_MANAGED_CHILD_ENV) == "1"
    if managed_local is not None:
        if managed_child:
            if requested_role not in ("producer", "consumer"):
                raise ValueError(
                    "managed_local child workers require an explicit producer "
                    "or consumer role"
                )
            if not Path(deployment.control_dir, "logs").is_dir():
                raise ValueError(
                    "managed_local child worker requires an active supervisor "
                    f"control_dir: {deployment.control_dir}"
                )
        else:
            if distributed:
                raise ValueError("managed_local cannot run inside an existing torchrun")
            if requested_role not in ("auto", "both"):
                raise ValueError(
                    "managed_local supports only --role auto or --role both"
                )
            if node_rank is not None:
                raise ValueError("managed_local does not accept --node-rank")
            if os.path.exists(deployment.control_dir):
                raise ValueError(
                    "managed_local requires a fresh control_dir, but it already "
                    f"exists: {deployment.control_dir}"
                )
    role = _resolve_role(cfg, requested_role, distributed=distributed)
    topology = cfg.deployment.trainer
    if role == "both" and topology.nnodes > 1:
        raise ValueError(
            "automatic disaggregated --role both supports one trainer node only; "
            "launch producer and each consumer node explicitly"
        )
    resolved_rank = (
        _resolved_node_rank(cfg, node_rank, base_env)
        if role in ("all", "consumer", "both")
        else None
    )
    producer_env: dict[str, str] = {}
    consumer_env: dict[str, str] = {}
    if cfg.deployment.mode == "disaggregated":
        role_base_env = base_env
        managed_environment: dict[str, str] = {}
        if managed_local is not None:
            managed_environment = _managed_local_environment(cfg)
            role_base_env = {**base_env, **managed_environment}
        if role in ("producer", "both"):
            producer_env = _disaggregated_env(cfg, role_base_env, role="producer")
        if role in ("consumer", "both"):
            consumer_env = _disaggregated_env(cfg, role_base_env, role="consumer")
        if managed_local is not None:
            producer_env.update(managed_environment)
            producer_env["CUDA_VISIBLE_DEVICES"] = ""
            producer_env[_MANAGED_CHILD_ENV] = "1"
            consumer_env.update(managed_environment)
            consumer_env["CUDA_VISIBLE_DEVICES"] = ",".join(
                managed_local.trainer_cuda_visible_devices
            )
            consumer_env[_MANAGED_CHILD_ENV] = "1"

    _validate_capture_urls(cfg, role=role, base_env=base_env)
    _validate_consumer_database(
        cfg,
        role=role,
        launch_env=consumer_env,
        base_env=base_env,
        distributed=distributed,
        node_rank=resolved_rank,
    )
    if distributed:
        assert role != "both"
        if role == "producer" and int(base_env["WORLD_SIZE"]) > 1:
            raise ValueError(
                "producer must be a direct single process, not a multi-rank "
                "torchrun worker"
            )
        if role != "producer":
            actual_world_size = int(base_env["WORLD_SIZE"])
            expected_world_size = topology.nnodes * topology.nproc_per_node
            if actual_world_size != expected_world_size:
                raise ValueError(
                    f"torchrun WORLD_SIZE={actual_world_size} does not match YAML "
                    "deployment.trainer topology "
                    f"({topology.nnodes} * {topology.nproc_per_node} = "
                    f"{expected_world_size})"
                )
            cfg.validate_world_size(actual_world_size)
        worker_env = producer_env if role == "producer" else consumer_env
        return LaunchPlan("worker", role, worker_env=worker_env)

    worker_prefix = tuple(worker_prefix or (sys.executable, "-m", "specforge.cli"))
    use_module_entry = torchrun_prefix is None
    torchrun_prefix = tuple(
        torchrun_prefix or (sys.executable, "-m", "torch.distributed.run")
    )
    distributed_entry = (
        ("--module", "specforge.cli") if use_module_entry else worker_prefix
    )
    if role == "producer":
        return LaunchPlan("worker", role, worker_env=producer_env)
    if role in ("all", "consumer"):
        command = _trainer_command(
            cfg,
            config_path=config_path,
            role=role,
            overrides=overrides,
            worker_prefix=worker_prefix,
            torchrun_prefix=torchrun_prefix,
            distributed_entry=distributed_entry,
            node_rank=resolved_rank,
            env=consumer_env,
        )
        if command.argv[: len(worker_prefix)] == worker_prefix:
            return LaunchPlan("worker", role, worker_env=consumer_env)
        if deployment is not None:
            return LaunchPlan(
                "command",
                role,
                commands=(command,),
                shutdown_grace_s=deployment.shutdown_grace_s,
            )
        return LaunchPlan("command", role, commands=(command,))

    producer = CommandSpec(
        "producer",
        tuple(_worker_argv(worker_prefix, config_path, "producer", overrides)),
        producer_env,
    )
    consumer = _trainer_command(
        cfg,
        config_path=config_path,
        role="consumer",
        overrides=overrides,
        worker_prefix=worker_prefix,
        torchrun_prefix=torchrun_prefix,
        distributed_entry=distributed_entry,
        node_rank=resolved_rank,
        env=consumer_env,
    )
    if managed_local is not None:
        return LaunchPlan(
            "managed_supervisor",
            "both",
            commands=(producer, consumer),
            services=_managed_local_services(cfg, algorithm=algorithm),
            managed_root=deployment.control_dir,
            managed_ports=(
                managed_local.mooncake.rpc_port,
                managed_local.mooncake.metadata_port,
                managed_local.mooncake.metrics_port,
                *[server.port for server in managed_local.capture_servers],
            ),
            shutdown_grace_s=managed_local.shutdown_grace_s,
        )
    return LaunchPlan(
        "supervisor",
        "both",
        commands=(producer, consumer),
        shutdown_grace_s=deployment.shutdown_grace_s,
    )


def _process_group_exists(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except (AttributeError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def _signal_process_group(process: subprocess.Popen, signum: int) -> bool:
    try:
        os.killpg(process.pid, signum)
    except (AttributeError, ProcessLookupError, PermissionError):
        return False
    return True


def _terminate_processes(
    processes: Sequence[subprocess.Popen],
    *,
    grace_s: float = 5.0,
    exited_group_leaders: Sequence[subprocess.Popen] = (),
) -> None:
    """Terminate child sessions, including descendants of a dead group leader."""
    tracked_groups: set[int] = set()
    for process in processes:
        if process.poll() is None:
            if _signal_process_group(process, signal.SIGTERM):
                tracked_groups.add(process.pid)
            else:
                process.terminate()
    # A leader killed directly by the OOM killer can already have a terminal
    # poll status while its children remain in the detached session. The PID is
    # passed only from the poll that observed that failure, minimizing reuse risk.
    for process in exited_group_leaders:
        if _signal_process_group(process, signal.SIGTERM):
            tracked_groups.add(process.pid)
    deadline = time.monotonic() + grace_s
    while time.monotonic() < deadline and (
        any(process.poll() is None for process in processes)
        or any(_process_group_exists(pgid) for pgid in tracked_groups)
    ):
        time.sleep(0.05)
    killed_groups: set[int] = set()
    for process in processes:
        if process.poll() is None:
            if _signal_process_group(process, signal.SIGKILL):
                killed_groups.add(process.pid)
            else:
                process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass
    for pgid in tracked_groups - killed_groups:
        if _process_group_exists(pgid):
            try:
                os.killpg(pgid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError, PermissionError):
                pass


def _managed_preflight(plan: LaunchPlan) -> None:
    if plan.managed_root is None:
        raise ValueError("managed supervisor plan is missing managed_root")
    if os.path.exists(plan.managed_root):
        raise ValueError(
            f"managed_local requires a fresh control_dir: {plan.managed_root}"
        )
    if shutil.which("mooncake_master") is None:
        raise RuntimeError("managed_local requires mooncake_master on PATH")
    try:
        mooncake_available = importlib.util.find_spec("mooncake.store") is not None
    except ModuleNotFoundError:
        mooncake_available = False
    if not mooncake_available:
        raise RuntimeError("managed_local requires the mooncake Python package")
    try:
        patched_sglang = (
            importlib.util.find_spec("sglang.srt.spec_capture_sink") is not None
        )
    except ModuleNotFoundError:
        patched_sglang = False
    if not patched_sglang:
        raise RuntimeError(
            "managed_local requires patched SGLang spec capture; run "
            "scripts/apply_sglang_spec_capture_patch.sh"
        )

    for port in plan.managed_ports:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError as exc:
                raise RuntimeError(
                    f"managed_local port 127.0.0.1:{port} is unavailable: {exc}"
                ) from exc


def _http_ready(readiness: ReadinessSpec) -> bool:
    try:
        with urllib_request.urlopen(readiness.url, timeout=1.0) as response:
            status = getattr(response, "status", 200)
            return readiness.kind == "mooncake" or 200 <= status < 300
    except urllib_error.HTTPError as exc:
        # Mooncake returns a 4xx missing-key response for this read-only probe.
        # A 5xx means the service is listening but not yet healthy.
        return readiness.kind == "mooncake" and 400 <= exc.code < 500
    except (OSError, urllib_error.URLError):
        return False


def _readiness_satisfied(readiness: ReadinessSpec) -> bool:
    if not _http_ready(readiness):
        return False
    if readiness.kind != "mooncake":
        return True
    assert readiness.tcp_host is not None and readiness.tcp_port is not None
    try:
        with socket.create_connection(
            (readiness.tcp_host, readiness.tcp_port), timeout=1.0
        ):
            return True
    except OSError:
        return False


def _wait_for_service(
    service: ServiceSpec,
    process: subprocess.Popen,
    started: Sequence[tuple[ServiceSpec, subprocess.Popen]],
) -> None:
    deadline = time.monotonic() + service.readiness.timeout_s
    while time.monotonic() < deadline:
        for active_service, active_process in started:
            status = active_process.poll()
            if status is not None:
                raise RuntimeError(
                    f"managed service {active_service.command.label!r} exited "
                    f"with status {status}; see {active_service.log_path}"
                )
        if _readiness_satisfied(service.readiness):
            return
        time.sleep(0.25)
    if process.poll() is not None:
        raise RuntimeError(
            f"managed service {service.command.label!r} exited during startup; "
            f"see {service.log_path}"
        )
    raise TimeoutError(
        f"managed service {service.command.label!r} did not become ready after "
        f"{service.readiness.timeout_s:.1f}s; see {service.log_path}"
    )


def _spawn_command(
    command: CommandSpec,
    *,
    popen: Callable[..., subprocess.Popen],
    stdout=None,
    stderr=None,
) -> subprocess.Popen:
    child_env = os.environ.copy()
    child_env.update(command.env)
    kwargs = {"env": child_env, "start_new_session": True}
    if stdout is not None:
        kwargs["stdout"] = stdout
    if stderr is not None:
        kwargs["stderr"] = stderr
    return popen(command.argv, **kwargs)


def _spawn_service(
    service: ServiceSpec,
    *,
    popen: Callable[..., subprocess.Popen],
) -> subprocess.Popen:
    with open(service.log_path, "x", encoding="utf-8") as log:
        return _spawn_command(
            service.command,
            popen=popen,
            stdout=log,
            stderr=subprocess.STDOUT,
        )


def _stop_services(
    started: Sequence[tuple[ServiceSpec, subprocess.Popen]], *, grace_s: float
) -> None:
    phases = sorted({service.phase for service, _ in started}, reverse=True)
    for phase in phases:
        processes = [process for service, process in started if service.phase == phase]
        running = []
        exited_group_leaders = []
        for process in processes:
            if process.poll() is None:
                running.append(process)
            else:
                # Readiness can fail because the service leader already died
                # while its capture/Mooncake descendants remain in the detached
                # session. Preserve that just-observed PID as a group-cleanup
                # target instead of silently skipping the entire process tree.
                exited_group_leaders.append(process)
        _terminate_processes(
            running,
            grace_s=grace_s,
            exited_group_leaders=exited_group_leaders,
        )


def _normalized_status(status: int, *, unexpected_clean: bool = False) -> int:
    if status < 0:
        return 128 - status
    if status == 0 and unexpected_clean:
        return 1
    return status


def run_commands(
    plan: LaunchPlan,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
    managed_preflight: Optional[Callable[[LaunchPlan], None]] = None,
    readiness_waiter: Optional[
        Callable[
            [
                ServiceSpec,
                subprocess.Popen,
                Sequence[tuple[ServiceSpec, subprocess.Popen]],
            ],
            None,
        ]
    ] = None,
) -> int:
    """Execute workers and, when requested, their local capture services."""
    if plan.kind == "worker":
        raise ValueError("worker plans execute inside specforge.cli")
    processes: list[subprocess.Popen] = []
    services: list[tuple[ServiceSpec, subprocess.Popen]] = []
    previous_handlers = {}
    managed_preflight = managed_preflight or _managed_preflight
    readiness_waiter = readiness_waiter or _wait_for_service

    def forward_signal(signum, _frame):
        # The first signal starts orderly process-group teardown. Ignore later
        # INT/TERM/HUP deliveries until the finally block restores the caller's
        # handlers; otherwise a second Ctrl-C can interrupt cleanup and orphan
        # detached capture-server grandchildren.
        for installed in previous_handlers:
            signal.signal(installed, signal.SIG_IGN)
        raise _ForwardedSignal(signum)

    try:
        managed_signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, "SIGHUP"):
            managed_signals.append(signal.SIGHUP)
        for signum in managed_signals:
            try:
                previous_handlers[signum] = signal.signal(signum, forward_signal)
            except ValueError:
                # Unit tests or embedding applications may call from a non-main thread.
                for installed, handler in previous_handlers.items():
                    signal.signal(installed, handler)
                previous_handlers.clear()
                break
        if plan.kind == "managed_supervisor":
            managed_preflight(plan)
            assert plan.managed_root is not None
            Path(plan.managed_root, "logs").mkdir(parents=True, exist_ok=False)
            phases = sorted({service.phase for service in plan.services})
            for phase in phases:
                current_phase = [
                    service for service in plan.services if service.phase == phase
                ]
                for service in current_phase:
                    services.append((service, _spawn_service(service, popen=popen)))
                for service, process in services[-len(current_phase) :]:
                    readiness_waiter(service, process, tuple(services))
        for command in plan.commands:
            processes.append(_spawn_command(command, popen=popen))
        remaining = set(range(len(processes)))
        first_failure = 0
        while remaining:
            for index in tuple(remaining):
                status = processes[index].poll()
                if status is None:
                    continue
                remaining.remove(index)
                if status != 0 and first_failure == 0:
                    first_failure = _normalized_status(status)
                    _terminate_processes(
                        [processes[item] for item in remaining],
                        grace_s=plan.shutdown_grace_s,
                        exited_group_leaders=(processes[index],),
                    )
                    remaining.clear()
                    break
            if plan.kind == "managed_supervisor" and remaining and not first_failure:
                for service, process in services:
                    status = process.poll()
                    if status is None:
                        continue
                    first_failure = _normalized_status(status, unexpected_clean=True)
                    _terminate_processes(
                        [processes[item] for item in remaining],
                        grace_s=plan.shutdown_grace_s,
                        exited_group_leaders=(process,),
                    )
                    remaining.clear()
                    break
            if remaining:
                time.sleep(0.05)
        if plan.kind == "managed_supervisor":
            _stop_services(services, grace_s=plan.shutdown_grace_s)
            services.clear()
        return first_failure
    except _ForwardedSignal as received:
        _terminate_processes(processes, grace_s=plan.shutdown_grace_s)
        _stop_services(services, grace_s=plan.shutdown_grace_s)
        services.clear()
        return 128 + received.signum
    except BaseException:
        _terminate_processes(processes, grace_s=plan.shutdown_grace_s)
        _stop_services(services, grace_s=plan.shutdown_grace_s)
        services.clear()
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


__all__ = [
    "CommandSpec",
    "LaunchPlan",
    "LaunchRole",
    "ReadinessSpec",
    "ServiceSpec",
    "build_launch_plan",
    "run_commands",
]
