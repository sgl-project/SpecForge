import os
from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH


def init_distributed(timeout: int = 10, tp_size: int = 1):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))

    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        timeout=timedelta(minutes=timeout),
    )
    torch.cuda.set_device(local_rank)
    print(f"Rank {rank} (local_rank {local_rank}) is bound to device cuda:{local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    print(
        f"world size: {world_size}, dp size: {dp_size}, tp size: {tp_size}, local rank: {local_rank}"
    )
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=["dp", "tp"]
    )
    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")
    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _DP_GROUP = dp_group


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()


def print_rank_info():
    global_rank = dist.get_rank()
    device_mesh = get_device_mesh()
    if device_mesh is not None:
        coord = device_mesh.get_coordinate()
        if coord is not None:
            dp_rank, tp_rank = coord
            dp_size, tp_size = device_mesh.mesh.shape
        else:
            dp_rank = tp_rank = 0
            dp_size = tp_size = 1
    else:
        dp_rank = tp_rank = 0
        dp_size = tp_size = 1

    print(
        f"[Global Rank {global_rank}] DP Rank {dp_rank}/{dp_size}, TP Rank {tp_rank}/{tp_size}"
    )


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
