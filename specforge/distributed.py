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
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    local_rank = dist.get_rank() % torch.cuda.device_count()
    torch.cuda.set_device(local_rank)
    print_with_rank(f"bind to device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=["dp", "tp"]
    )
    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")
    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _DP_GROUP = dp_group
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()


def shard_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    rank = dist.get_rank(process_group)
    size = dist.get_world_size(process_group)
    return tensor.chunk(size, dim=dim)[rank].contiguous()


def gather_tensor(
    tensor: torch.Tensor, process_group: dist.ProcessGroup = None, dim: int = -1
) -> torch.Tensor:
    size = dist.get_world_size(process_group)
    obj_list = [torch.empty_like(tensor) for _ in range(size)]
    dist.all_gather(obj_list, tensor, group=process_group)
    gather_tensor = torch.cat(obj_list, dim=dim)
    return gather_tensor


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
