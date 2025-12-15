from datetime import timedelta
from typing import Optional, Any

import torch
import torch.distributed as dist
from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

from specforge.utils import print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None

def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def get_draft_dp_group():
    global _DRAFT_DP_GROUP
    return _DRAFT_DP_GROUP

def get_draft_sp_group():
    global _DRAFT_SP_GROUP
    return _DRAFT_SP_GROUP

def get_device_mesh():
    global _DEVICE_MESH
    return _DEVICE_MESH


def get_tp_device_mesh():
    global _TP_DEVICE_MESH
    return _TP_DEVICE_MESH


def get_dp_device_mesh():
    global _DP_DEVICE_MESH
    return _DP_DEVICE_MESH



def get_sp_ulysses_group():
    global _SP_ULYSSES_GROUP
    return _SP_ULYSSES_GROUP


def get_sp_ring_group():
    global _SP_RING_GROUP
    return _SP_RING_GROUP


def init_distributed(timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1):
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
    assert world_size == tp_size * dp_size, f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )
    total_sp_size = sp_ulysses_size * sp_ring_size
    dist.barrier()

    # 确保 Draft DP 大小计算正确
    draft_dp_size = world_size // total_sp_size
    assert world_size == draft_dp_size * total_sp_size, \
        f"World size ({world_size}) cannot be evenly divided by total SP size ({total_sp_size})"

    print_with_rank(f"Initializing Draft Groups: Draft_DP={draft_dp_size}, SP={total_sp_size}")
    draft_dp_size = world_size // sp_ulysses_size // sp_ring_size

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, \
        _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP

    for i in range(draft_dp_size):
        start_rank = i * total_sp_size
        end_rank = start_rank + total_sp_size
        ranks = list(range(start_rank, end_rank))
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            _DRAFT_SP_GROUP = group

        # 2. 创建 Draft DP Group (Data Parallel for Draft Model)
        # 逻辑：跨越 SP 组取 rank
        # 例如：[0, 4], [1, 5], [2, 6], [3, 7] ...
    for i in range(total_sp_size):
        ranks = [j * total_sp_size + i for j in range(draft_dp_size)]
        group = dist.new_group(ranks)
        if dist.get_rank() in ranks:
            _DRAFT_DP_GROUP = group

    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")


    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _DP_GROUP = dp_group
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group



def destroy_distributed():
    global _TP_GROUP, _DP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DRAFT_DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group(_SP_ULYSSES_GROUP)
    dist.destroy_process_group(_SP_RING_GROUP)
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


def all_gather_tensor(local_tensor: torch.Tensor, group: Optional[dist.ProcessGroup] = None, async_op: bool = False):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(output_shape, dtype=local_tensor.dtype, device=local_tensor.device)
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output

class Gather(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: Any,
        group: dist.ProcessGroup,
        local_tensor: torch.Tensor,
        gather_dim: int,
        grad_scaler: bool = True,
        async_op=False,
    ) -> torch.Tensor:
        ctx.group = group
        ctx.gather_dim = gather_dim
        ctx.grad_scaler = grad_scaler
        ctx.async_op = async_op

        sp_world_size = dist.get_world_size(group=group)
        ctx.sp_world_size = sp_world_size

        sp_rank = dist.get_rank(group=group)
        ctx.sp_rank = sp_rank

        local_shape = list(local_tensor.size())
        split_size = local_shape[0]
        part_size = local_shape[gather_dim]  # store original size
        ctx.part_size = part_size

        output = all_gather_tensor(local_tensor, group, async_op)
        return torch.cat(output.split(split_size, dim=0), dim=gather_dim)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> Any:
        if ctx.grad_scaler:
            grad_output = grad_output * ctx.sp_world_size
        return (
            None,
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[ctx.sp_rank].contiguous(),
            None,
            None,
            None,
            None,
        )

def gather_outputs_and_unpad(
    x: torch.Tensor,
    gather_dim: int,
    grad_scaler: bool = True,
    group: Optional[dist.ProcessGroup] = None,
):
    """
    Gather a tensor across a process group and optionally unpad its padded elements.

    Args:
        x (Tensor): Input tensor to gather.
        gather_dim (int): Dimension along which to gather across ranks.
        unpad_dim (int, optional): Dimension from which to remove padding. If None, no unpadding.
        padding_size (int): Number of padding elements to remove on `unpad_dim`. Defaults to 0.
        grad_scaler (bool): Whether to apply gradient scaling during gather. Defaults to True.
        group (ProcessGroup, optional): Process group for gathering. If None, uses
            `get_ulysses_sequence_parallel_group()`. If still None, returns `x` unchanged.

    Returns:
        Tensor: The gathered tensor, with padding removed if requested.
    """
    if not group:
        group = get_draft_sp_group()
    if torch.distributed.get_world_size(group) == 1:
        return x
    x = Gather.apply(group, x, gather_dim, grad_scaler)
    return x


def is_tp_rank_0():
    """Return True if current process is rank 0 in its TP group."""
    tp_group = get_tp_group()
    if tp_group is None:
        return True
    return dist.get_rank(group=tp_group) == 0
