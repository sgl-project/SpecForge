from datetime import timedelta
from typing import Any, Optional

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


def get_sp_rank():
    """Return this rank's position within its SP group."""
    global _DRAFT_SP_GROUP
    if _DRAFT_SP_GROUP is None:
        return 0
    return dist.get_rank(_DRAFT_SP_GROUP)


def get_draft_dp_rank():
    """Return this rank's DP index (across SP groups)."""
    global _DRAFT_DP_GROUP
    if _DRAFT_DP_GROUP is None:
        return 0
    return dist.get_rank(_DRAFT_DP_GROUP)


def get_sp_size():
    """Return the SP group size."""
    global _DRAFT_SP_GROUP
    if _DRAFT_SP_GROUP is None:
        return 1
    return dist.get_world_size(_DRAFT_SP_GROUP)


def get_draft_dp_size():
    """Return the number of DP groups."""
    global _DRAFT_DP_GROUP
    if _DRAFT_DP_GROUP is None:
        return 1
    return dist.get_world_size(_DRAFT_DP_GROUP)


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
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
    dp_size = world_size // tp_size
    assert (
        world_size == tp_size * dp_size
    ), f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        "cuda", (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size*sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        "cuda",
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")

    sp_ulysses_group = PROCESS_GROUP.ULYSSES_PG
    sp_ring_group = PROCESS_GROUP.RING_PG
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type="cuda")

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DP_GROUP = dp_group
    _DRAFT_DP_GROUP = draft_device_mesh.get_group("draft_dp")
    _DRAFT_SP_GROUP = draft_device_mesh.get_group("sp")
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type="cuda")


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP, _SP_ULYSSES_GROUP, _SP_RING_GROUP, _DRAFT_DP_GROUP
    if _TP_GROUP is not None:
        dist.destroy_process_group(_TP_GROUP)
    if _DP_GROUP is not None:
        dist.destroy_process_group(_DP_GROUP)
    if _SP_ULYSSES_GROUP is not None:
        dist.destroy_process_group(_SP_ULYSSES_GROUP)
    if _SP_RING_GROUP is not None:
        dist.destroy_process_group(_SP_RING_GROUP)
    if _DRAFT_DP_GROUP is not None:
        dist.destroy_process_group(_DRAFT_DP_GROUP)
    if _DRAFT_SP_GROUP is not None:
        dist.destroy_process_group(_DRAFT_SP_GROUP)
    dist.destroy_process_group()


def init_distributed_from_subgroup(
    local_group: dist.ProcessGroup,
    tp_size: int = 1,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
) -> None:
    """Initialize TP/DP/SP process groups from an existing subgroup.

    Used in NCCL transfer mode where a global process group is
    initialized first, and local (rollout / train) groups are created
    as subgroups via ``dist.new_group()``.

    This sets the same module-level globals as ``init_distributed()``
    but derives them from *local_group* instead of the default group.

    Args:
        local_group:      A subgroup created by ``dist.new_group(ranks=...)``.
        tp_size:          Tensor-parallel degree.
        sp_ulysses_size:  Ulysses SP degree.
        sp_ring_size:     Ring SP degree.
    """
    local_world_size = dist.get_world_size(local_group)
    local_rank = dist.get_rank(local_group)
    dp_size = local_world_size // tp_size

    assert local_world_size == tp_size * dp_size, (
        f"local world size must be divisible by tp size, "
        f"now {local_world_size=}, {tp_size=}"
    )

    sp_size = sp_ulysses_size * sp_ring_size
    assert local_world_size % sp_size == 0, (
        f"local world size ({local_world_size}) must be divisible by "
        f"SP size ({sp_size})"
    )

    # Get the global ranks that belong to this subgroup
    global_rank = dist.get_rank()
    global_world_size = dist.get_world_size()

    # Collect all global ranks in this subgroup via all_gather
    rank_tensor = torch.tensor([global_rank], dtype=torch.long, device="cuda")
    gathered = [
        torch.zeros(1, dtype=torch.long, device="cuda") for _ in range(local_world_size)
    ]
    dist.all_gather(gathered, rank_tensor, group=local_group)
    subgroup_global_ranks = sorted([t.item() for t in gathered])

    # Build TP groups: ranks within each TP stripe
    tp_groups = []
    for dp_idx in range(dp_size):
        tp_ranks = [subgroup_global_ranks[dp_idx * tp_size + t] for t in range(tp_size)]
        tp_groups.append(dist.new_group(ranks=tp_ranks, use_local_synchronization=True))

    # Build DP groups: ranks across TP stripes at same tp_rank
    dp_groups = []
    for tp_idx in range(tp_size):
        dp_ranks = [subgroup_global_ranks[d * tp_size + tp_idx] for d in range(dp_size)]
        dp_groups.append(dist.new_group(ranks=dp_ranks, use_local_synchronization=True))

    # SP groups
    draft_dp_size = local_world_size // sp_size
    draft_dp_groups = []
    draft_sp_groups = []
    for ddp_idx in range(draft_dp_size):
        sp_ranks = [
            subgroup_global_ranks[ddp_idx * sp_size + s] for s in range(sp_size)
        ]
        draft_sp_groups.append(
            dist.new_group(ranks=sp_ranks, use_local_synchronization=True)
        )
    for sp_idx in range(sp_size):
        ddp_ranks = [
            subgroup_global_ranks[d * sp_size + sp_idx] for d in range(draft_dp_size)
        ]
        draft_dp_groups.append(
            dist.new_group(ranks=ddp_ranks, use_local_synchronization=True)
        )

    # Pick the groups this rank belongs to
    tp_rank = local_rank % tp_size
    dp_rank = local_rank // tp_size
    sp_rank = local_rank % sp_size
    draft_dp_rank = local_rank // sp_size

    my_tp_group = tp_groups[dp_rank]
    my_dp_group = dp_groups[tp_rank]
    my_draft_dp_group = draft_dp_groups[sp_rank]
    my_draft_sp_group = draft_sp_groups[draft_dp_rank]

    # Set yunchang SP groups if SP is used
    if sp_size > 1:
        set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, local_rank, local_world_size)

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH
    global _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _TP_GROUP = my_tp_group
    _DP_GROUP = my_dp_group
    _DRAFT_DP_GROUP = my_draft_dp_group
    _DRAFT_SP_GROUP = my_draft_sp_group
    _SP_ULYSSES_GROUP = PROCESS_GROUP.ULYSSES_PG if sp_size > 1 else my_draft_sp_group
    _SP_RING_GROUP = PROCESS_GROUP.RING_PG if sp_size > 1 else my_draft_sp_group
    _TP_DEVICE_MESH = dist.DeviceMesh.from_group(my_tp_group, device_type="cuda")
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(my_dp_group, device_type="cuda")
    _DEVICE_MESH = None  # 2D mesh not available in subgroup mode


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


def all_gather_tensor(
    local_tensor: torch.Tensor,
    group: Optional[dist.ProcessGroup] = None,
    async_op: bool = False,
):
    sp_world_size = dist.get_world_size(group=group)
    output_shape = list(local_tensor.shape)
    output_shape[0] = output_shape[0] * sp_world_size
    output = torch.empty(
        output_shape, dtype=local_tensor.dtype, device=local_tensor.device
    )
    dist.all_gather_into_tensor(output, local_tensor, group=group, async_op=async_op)
    return output


# Adapted from https://github.com/volcengine/verl/blob/a0e8e4472b8b472409defb0c8fcc5162301450af/verl/utils/ulysses.py#L194
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
            grad_output.split(ctx.part_size, dim=ctx.gather_dim)[
                ctx.sp_rank
            ].contiguous(),
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
