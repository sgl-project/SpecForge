# coding=utf-8
"""Pure launch planning and process supervision for ``specforge train``."""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Literal, Mapping, Optional, Sequence
from urllib.parse import urlsplit, urlunsplit

from specforge.config import Config

LaunchRole = Literal["auto", "all", "producer", "consumer", "both"]
PlanKind = Literal["worker", "command", "supervisor"]

_DIST_ENV = ("RANK", "WORLD_SIZE", "LOCAL_RANK", "MASTER_ADDR", "MASTER_PORT")
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
class LaunchPlan:
    kind: PlanKind
    role: Literal["all", "producer", "consumer", "both"]
    commands: tuple[CommandSpec, ...] = ()
    worker_env: Mapping[str, str] = field(default_factory=dict)

    def render(self) -> str:
        return json.dumps(
            {
                "kind": self.kind,
                "role": self.role,
                "commands": [command.as_dict() for command in self.commands],
                "worker_env": _redacted_env(self.worker_env),
            },
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
    disaggregated = cfg.training.deployment_mode == "disaggregated"
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
                "DISAGG_DB": str(control_dir / "consumer.sqlite"),
                "DISAGG_INBOX_DIR": str(control_dir / "inboxes"),
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
        or cfg.training.deployment_mode != "disaggregated"
        or role not in ("consumer", "both")
    ):
        return
    database = (
        cfg.training.metadata_db_path
        or launch_env.get("DISAGG_DB")
        or base_env.get("DISAGG_DB")
    )
    if not database:
        raise ValueError("online disaggregated consumer requires a metadata database")
    state_owner = int(base_env["RANK"]) == 0 if distributed else node_rank in (None, 0)
    if cfg.training.resume_from:
        if not os.path.exists(database):
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
        or cfg.training.deployment_mode != "disaggregated"
        or role not in ("producer", "both")
    ):
        return
    if cfg.training.server_urls:
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
    config_path: str,
    overrides: Sequence[str] = (),
    requested_role: LaunchRole = "auto",
    node_rank: Optional[int] = None,
    env: Optional[Mapping[str, str]] = None,
    worker_prefix: Optional[Sequence[str]] = None,
    torchrun_prefix: Optional[Sequence[str]] = None,
) -> LaunchPlan:
    """Resolve one validated config into a side-effect-free process plan."""
    base_env = os.environ if env is None else env
    distributed = _distributed_state(base_env)
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
    if cfg.training.deployment_mode == "disaggregated":
        if role in ("producer", "both"):
            producer_env = _disaggregated_env(cfg, base_env, role="producer")
        if role in ("consumer", "both"):
            consumer_env = _disaggregated_env(cfg, base_env, role="consumer")

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
    return LaunchPlan("supervisor", "both", commands=(producer, consumer))


def _terminate_processes(processes: Sequence[subprocess.Popen]) -> None:
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except (AttributeError, ProcessLookupError):
                process.terminate()
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and any(
        process.poll() is None for process in processes
    ):
        time.sleep(0.05)
    for process in processes:
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (AttributeError, ProcessLookupError):
                process.kill()
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


def run_commands(
    plan: LaunchPlan,
    *,
    popen: Callable[..., subprocess.Popen] = subprocess.Popen,
) -> int:
    """Execute a command or supervise both disaggregated roles."""
    if plan.kind == "worker":
        raise ValueError("worker plans execute inside specforge.cli")
    processes: list[subprocess.Popen] = []
    previous_handlers = {}

    def forward_signal(signum, _frame):
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
        for command in plan.commands:
            child_env = os.environ.copy()
            child_env.update(command.env)
            processes.append(popen(command.argv, env=child_env, start_new_session=True))
        remaining = set(range(len(processes)))
        first_failure = 0
        while remaining:
            for index in tuple(remaining):
                status = processes[index].poll()
                if status is None:
                    continue
                remaining.remove(index)
                if status != 0 and first_failure == 0:
                    first_failure = 128 - status if status < 0 else status
                    _terminate_processes([processes[item] for item in remaining])
                    remaining.clear()
                    break
            if remaining:
                time.sleep(0.05)
        return first_failure
    except _ForwardedSignal as received:
        _terminate_processes(processes)
        return 128 + received.signum
    except BaseException:
        _terminate_processes(processes)
        raise
    finally:
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)


__all__ = [
    "CommandSpec",
    "LaunchPlan",
    "LaunchRole",
    "build_launch_plan",
    "run_commands",
]
