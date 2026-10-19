"""Core seams the TPU plugin patches. Each entry says why it exists."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from typing import Any

import torch

from specforge_tpu import hooks, producer, shapes


def _pick_ports(count: int) -> list[int]:
    sockets = [socket.socket(socket.AF_INET, socket.SOCK_STREAM) for _ in range(count)]
    try:
        for sock in sockets:
            sock.bind(("", 0))
        return [sock.getsockname()[1] for sock in sockets]
    finally:
        for sock in sockets:
            sock.close()


def tpu_launch_env(nproc_per_node: int) -> dict[str, str]:
    """Env every rank of a single-host torch_tpu process group must share."""
    ports = _pick_ports(nproc_per_node)
    return {
        "TORCH_TPU_SLICEBUILDER_ADDRESSES": ",".join(f"localhost:{p}" for p in ports),
        "TORCH_TPU_DEVICES_PER_HOST": str(nproc_per_node),
    }


def _trainer_command(original: Callable[..., Any], cfg: Any, **kwargs: Any) -> Any:
    # torch_tpu ranks rendezvous through slicebuilder ports that the parent
    # must pick once; torchrun alone only shares MASTER_ADDR/PORT.
    command = original(cfg, **kwargs)
    topology = cfg.deployment.trainer
    if topology.nnodes != 1:
        raise NotImplementedError("specforge_tpu supports single-host trainers only")
    if topology.nproc_per_node == 1:
        return command
    env: Mapping[str, Any] = {
        **command.env,
        **tpu_launch_env(topology.nproc_per_node),
    }
    return type(command)(command.label, command.argv, env)


def _resolve_rank_topology() -> None:
    # Any torch.tpu call that touches the PjRt client (e.g. accelerate's
    # set_seed before init_distributed) needs the topology of the whole group.
    if "WORLD_SIZE" not in os.environ:
        return
    from torch_tpu._internal.utils import hardware

    world_size = int(os.environ["WORLD_SIZE"])
    os.environ.setdefault("TORCH_TPU_TOPOLOGY", hardware.get_tpu_topology(world_size))


def _bind_local_device(original: Callable[[str], int], device_type: str) -> int:
    # torch_tpu exposes one device per process (device_count() == 1), so the
    # core set_device(LOCAL_RANK) would be out of range.
    if device_type != "tpu":
        return original(device_type)
    return int(os.environ.get("LOCAL_RANK", "0"))


def _get_local_device() -> torch.device:
    # One TPU core per torchrun rank: every rank sees its core as tpu:0.
    return torch.device("tpu", 0)


def apply() -> None:
    # Registers torch.compile(backend="tpu").
    import torch_tpu._internal.compile  # noqa: F401

    _resolve_rank_topology()

    # First: later imports bind get_local_device by name.
    hooks.replace("specforge.utils.get_local_device", _get_local_device)

    from specforge import distributed

    distributed._DISTRIBUTED_BACKENDS["tpu"] = "tpu_dist"
    hooks.around("specforge.distributed._bind_local_device", _bind_local_device)
    hooks.around("specforge.launch_plan._trainer_command", _trainer_command)
    hooks.around(
        "specforge.algorithms.common.hidden_states_data._padded_collator",
        shapes.padded_collator,
    )
    hooks.around(
        "specforge.training.strategies.base._cpu_max_valid_anchors",
        shapes.shape_only_max_valid_anchors,
    )
    hooks.around(
        "specforge.inference.adapters.server_capture.SGLangServerCaptureAdapter.__init__",
        producer.adapter_init,
    )
