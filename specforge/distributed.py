import os
from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.utils import get_device_type, print_with_rank

_DEVICE_MESH = None
_TP_DEVICE_MESH = None
_TP_GROUP = None
_DP_DEVICE_MESH = None
_DP_GROUP = None
_DRAFT_DP_GROUP = None
_DRAFT_SP_GROUP = None
_SP_ULYSSES_GROUP = None
_SP_RING_GROUP = None

_DISTRIBUTED_BACKENDS = {
    "cpu": "gloo",
    "cuda": "nccl",
    "npu": "hccl",
    "xpu": "xccl",
}


def _distributed_backend(device_type: str) -> str:
    try:
        return _DISTRIBUTED_BACKENDS[device_type]
    except KeyError:
        raise ValueError(
            f"unsupported distributed device type {device_type!r}; "
            f"supported: {sorted(_DISTRIBUTED_BACKENDS)}"
        ) from None


def _device_module(device_type: str):
    """Return the active accelerator module, importing torch-npu lazily."""
    if device_type == "npu" and not hasattr(torch, "npu"):
        try:
            import torch_npu  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "SPECFORGE_DEVICE=npu requires a compatible torch-npu package"
            ) from exc
    module = getattr(torch, device_type, None)
    if module is None:
        raise RuntimeError(
            f"PyTorch does not expose the requested {device_type!r} device module"
        )
    return module


def _load_yunchang_globals():
    """Import sequence-parallel globals only inside trainer initialization."""

    from yunchang.globals import PROCESS_GROUP, set_seq_parallel_pg

    return PROCESS_GROUP, set_seq_parallel_pg


def _bind_local_device(device_type: str) -> int:
    """Bind this torchrun rank to one visible CUDA/NPU device."""
    if device_type == "cpu":
        return 0
    module = _device_module(device_type)
    if not module.is_available():
        raise RuntimeError(f"requested {device_type!r} accelerator is not available")
    count = int(module.device_count())
    if count <= 0:
        raise RuntimeError(f"requested {device_type!r} accelerator has no devices")
    if dist.is_initialized():
        rank_fallback = dist.get_rank() % count
    else:
        # ``torchrun`` supplies LOCAL_RANK, while ``mp.spawn``-style launchers
        # commonly supply only RANK before the default process group exists.
        rank_fallback = int(os.environ.get("RANK", "0")) % count
    local_rank = int(os.environ.get("LOCAL_RANK", rank_fallback))
    if not 0 <= local_rank < count:
        raise ValueError(
            f"LOCAL_RANK={local_rank} is outside the {count} visible "
            f"{device_type} devices"
        )
    module.set_device(local_rank)
    return local_rank


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


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
        tp_size(int): The degree of tensor parallelism
    """
    device_type = get_device_type()
    backend = _distributed_backend(device_type)
    # HCCL requires the process to bind its local NPU before process-group
    # initialization; doing the same for NCCL also removes ambiguous rank/device
    # inference on heterogeneous hosts.
    local_rank = _bind_local_device(device_type)
    # Yunchang probes the active CUDA device at import, which crashes
    # NPU-only torch builds. Only USP needs it, so load it lazily when SP
    # sizes exceed 1. For SP=1, the getters are populated from the singleton
    # DeviceMesh group below; publishing None would make torch.distributed
    # operations silently use the default WORLD group instead.
    if sp_ulysses_size * sp_ring_size > 1:
        process_group, set_seq_parallel_pg = _load_yunchang_globals()
    else:
        process_group, set_seq_parallel_pg = None, None

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout))
    print_with_rank(f"bind to {device_type} device {local_rank}")

    world_size = dist.get_world_size()
    dp_size = world_size // tp_size
    assert (
        world_size == tp_size * dp_size
    ), f"world size must be divisible by tp size, now {world_size=}, {(tp_size * dp_size)=} "

    device_mesh = dist.device_mesh.init_device_mesh(
        device_type, (dp_size, tp_size), mesh_dim_names=("dp", "tp")
    )

    assert (
        world_size % (sp_ulysses_size * sp_ring_size) == 0
    ), f"World size ({world_size}) cannot be evenly divided by total SP size ({sp_ulysses_size*sp_ring_size})"

    draft_dp_size = world_size // (sp_ulysses_size * sp_ring_size)
    draft_device_mesh = dist.device_mesh.init_device_mesh(
        device_type,
        (draft_dp_size, sp_ulysses_size * sp_ring_size),
        mesh_dim_names=("draft_dp", "sp"),
    )
    if set_seq_parallel_pg is not None:
        set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, dist.get_rank(), world_size)

    print_with_rank(f"device mesh: {device_mesh}")
    tp_group = device_mesh.get_group("tp")
    dp_group = device_mesh.get_group("dp")
    draft_dp_group = draft_device_mesh.get_group("draft_dp")
    draft_sp_group = draft_device_mesh.get_group("sp")

    if process_group is not None:
        sp_ulysses_group = process_group.ULYSSES_PG
        sp_ring_group = process_group.RING_PG
    else:
        # Both SP dimensions are 1 here. Reuse the per-rank singleton SP group
        # so callers always receive a real group whose world size is 1.
        sp_ulysses_group = draft_sp_group
        sp_ring_group = draft_sp_group
    # we need to create a 1D submesh
    tp_device_mesh = dist.DeviceMesh.from_group(tp_group, device_type=device_type)

    global _TP_GROUP, _DP_GROUP, _DEVICE_MESH, _TP_DEVICE_MESH, _DP_DEVICE_MESH, _SP_RING_GROUP, _SP_ULYSSES_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    _DEVICE_MESH = device_mesh
    _TP_GROUP = tp_group
    _TP_DEVICE_MESH = tp_device_mesh
    _SP_ULYSSES_GROUP = sp_ulysses_group
    _SP_RING_GROUP = sp_ring_group
    _DP_GROUP = dp_group
    _DRAFT_DP_GROUP = draft_dp_group
    _DRAFT_SP_GROUP = draft_sp_group
    _DP_DEVICE_MESH = dist.DeviceMesh.from_group(dp_group, device_type=device_type)


def destroy_distributed(*, abort: bool = False):
    """Tear down every cached process group and device mesh.

    ``abort=True`` (rank-local failure path) aborts communicators instead of the
    collective destroy, so this rank exits with its traceback instead of hanging
    behind peers stuck in a collective; torchrun then terminates the peers.
    """
    global _DEVICE_MESH, _TP_DEVICE_MESH, _TP_GROUP
    global _DP_DEVICE_MESH, _DP_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP
    global _SP_ULYSSES_GROUP, _SP_RING_GROUP
    # Teardown must never crash the process. Several handles can alias the same
    # underlying group (e.g. DP and draft-DP when there is no sequence
    # parallelism), and degenerate single-rank SP groups (created when
    # sp_ulysses_size == 1 or sp_ring_size == 1) are not registered in torch's
    # process-group map and would raise on destroy. Clean up each distinct,
    # valid group at most once.
    default_group = dist.group.WORLD if dist.is_initialized() else None
    seen = set()
    for group in (
        _TP_GROUP,
        _DP_GROUP,
        _SP_ULYSSES_GROUP,
        _SP_RING_GROUP,
        _DRAFT_DP_GROUP,
        _DRAFT_SP_GROUP,
        default_group,  # may alias the DP group; the seen-set dedups it
    ):
        if group is None or id(group) in seen:
            continue
        seen.add(id(group))
        try:
            abort_group = getattr(group, "abort", None)
            if abort and callable(abort_group):
                abort_group()
            else:
                dist.destroy_process_group(group)
        except Exception:
            # Group not registered (e.g. degenerate single-rank SP group) or
            # already destroyed.
            pass

    # Process-group and DeviceMesh objects are invalid after teardown. Keeping
    # them reachable makes a later single-process load look initialized while
    # collectives fail against stale handles.
    _DEVICE_MESH = None
    _TP_DEVICE_MESH = None
    _TP_GROUP = None
    _DP_DEVICE_MESH = None
    _DP_GROUP = None
    _DRAFT_DP_GROUP = None
    _DRAFT_SP_GROUP = None
    _SP_ULYSSES_GROUP = None
    _SP_RING_GROUP = None
