from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.utils import print_on_rank0, print_with_rank

_DEVICE_MESH = None
_TARGET_TP_DEVICE_MESH = None
_DRAFT_DP_DEVICE_MESH = None
_TARGET_TP_GROUP = None
_TARGET_DP_GROUP = None
_DRAFT_TP_GROUP = None
_DRAFT_DP_GROUP = None


def get_target_tp_group():
    global _TARGET_TP_GROUP
    return _TARGET_TP_GROUP


def get_target_tp_size():
    global _TARGET_TP_GROUP
    return dist.get_world_size(_TARGET_TP_GROUP)


def get_target_tp_rank():
    global _TARGET_TP_GROUP
    return dist.get_rank(_TARGET_TP_GROUP)


def get_target_dp_group():
    global _TARGET_DP_GROUP
    return _TARGET_DP_GROUP


def get_target_dp_size():
    global _TARGET_DP_GROUP
    return dist.get_world_size(_TARGET_DP_GROUP)


def get_target_dp_rank():
    global _TARGET_DP_GROUP
    return dist.get_rank(_TARGET_DP_GROUP)


def get_draft_tp_group():
    global _DRAFT_TP_GROUP
    return _DRAFT_TP_GROUP


def get_draft_tp_size():
    global _DRAFT_TP_GROUP
    return dist.get_world_size(_DRAFT_TP_GROUP)


def get_draft_tp_rank():
    global _DRAFT_TP_GROUP
    return dist.get_rank(_DRAFT_TP_GROUP)


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP


def get_draft_dp_size():
    global _DRAFT_DP_GROUP
    return dist.get_world_size(_DRAFT_DP_GROUP)


def get_draft_dp_rank():
    global _DRAFT_DP_GROUP
    return dist.get_rank(_DRAFT_DP_GROUP)


def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_target_tp_device_mesh():
    global _TARGET_TP_DEVICE_MESH
    return _TARGET_TP_DEVICE_MESH


def init_distributed(
    timeout: int = 10, target_tp_size: int = 1, draft_tp_size: int = 1
):
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
    target_dp_size = world_size // target_tp_size
    draft_dp_size = world_size // draft_tp_size
    assert (
        world_size == target_tp_size * target_dp_size
    ), "world size must be divisible by target tp size"
    assert (
        world_size == draft_tp_size * draft_dp_size
    ), "world size must be divisible by draft tp size"
    target_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (target_dp_size, target_tp_size),
        mesh_dim_names=["target_dp", "target_tp"],
    )
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (draft_dp_size, draft_tp_size), mesh_dim_names=["draft_dp", "draft_tp"]
    )
    print_on_rank0(f"target device mesh: {target_device_mesh}")
    print_on_rank0(f"draft device mesh: {draft_device_mesh}")
    global _TARGET_TP_GROUP, _TARGET_DP_GROUP, _DRAFT_TP_GROUP, _DRAFT_DP_GROUP, _TARGET_TP_DEVICE_MESH
    _TARGET_TP_GROUP = target_device_mesh.get_group("target_tp")
    _TARGET_DP_GROUP = target_device_mesh.get_group("target_dp")
    _DRAFT_TP_GROUP = draft_device_mesh.get_group("draft_tp")
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _TARGET_TP_DEVICE_MESH = dist.DeviceMesh.from_group(
        _TARGET_TP_GROUP, device_type="cuda"
    )


def destroy_distributed():
    global _TARGET_TP_GROUP, _TARGET_DP_GROUP, _DRAFT_TP_GROUP, _DRAFT_DP_GROUP
    dist.destroy_process_group(_TARGET_TP_GROUP)
    dist.destroy_process_group(_TARGET_DP_GROUP)
    dist.destroy_process_group(_DRAFT_TP_GROUP)
    dist.destroy_process_group(_DRAFT_DP_GROUP)
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
    tp_group = get_target_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
