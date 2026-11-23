"""Core seams the TPU plugin patches. Each entry says why it exists."""

from __future__ import annotations

import os
import socket
from collections.abc import Callable, Mapping
from functools import partial
from importlib import import_module
from typing import Any

import torch
from specforge_tpu import shapes


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
    import_module("torch_tpu._internal.compile")

    from specforge import distributed, launch_plan, utils
    from specforge.algorithms.common import batch_shapes

    # Resolve the backends, wrappers and bucket configuration before changing core.
    backends = distributed._DISTRIBUTED_BACKENDS
    bind_device = partial(_bind_local_device, distributed._bind_local_device)
    trainer_command = partial(_trainer_command, launch_plan._trainer_command)
    buckets = shapes.seq_buckets()
    _resolve_rank_topology()

    utils.get_local_device = _get_local_device
    backends["tpu"] = "tpu_dist"
    distributed._bind_local_device = bind_device
    launch_plan._trainer_command = trainer_command
    if buckets:
        batch_shapes.padded_sequence_length = partial(
            shapes.bucket_for, buckets=buckets
        )
    batch_shapes.anchor_capacity = shapes.anchor_capacity
