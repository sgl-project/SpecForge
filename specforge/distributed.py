import os
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Any

import torch
import torch.distributed as dist

from specforge.utils import get_device_type, print_with_rank


@dataclass(frozen=True)
class Topology:
    """The one record of this job's shape: its ranks, sizes and process groups.

    :func:`init_distributed` builds it once, reading each rank once, and its
    consumers receive it explicitly. ``Topology()`` is one process with no
    groups, which is what a test builds; a capture producer has no topology.
    """

    rank: int = 0
    world_size: int = 1
    #: The local accelerator this rank is bound to; 0 on CPU.
    local_rank: int = 0
    tp_size: int = 1
    sp_ulysses_size: int = 1
    sp_ring_size: int = 1
    tp_rank: int = 0
    dp_rank: int = 0
    draft_dp_rank: int = 0
    #: Rank in the draft sequence-parallel group.
    sp_rank: int = 0
    device_type: str = "cpu"
    world_group: Any = field(default=None, compare=False, repr=False)
    tp_group: Any = field(default=None, compare=False, repr=False)
    dp_group: Any = field(default=None, compare=False, repr=False)
    draft_dp_group: Any = field(default=None, compare=False, repr=False)
    draft_sp_group: Any = field(default=None, compare=False, repr=False)
    sp_ulysses_group: Any = field(default=None, compare=False, repr=False)
    sp_ring_group: Any = field(default=None, compare=False, repr=False)

    def __post_init__(self):
        sizes = {
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "sp_ulysses_size": self.sp_ulysses_size,
            "sp_ring_size": self.sp_ring_size,
        }
        below_one = sorted(name for name, size in sizes.items() if size < 1)
        if below_one:
            raise ValueError(f"topology sizes must be positive: {below_one}")
        if not 0 <= self.rank < self.world_size:
            raise ValueError(
                f"rank {self.rank} is not a rank of a world of {self.world_size}"
            )
        if self.world_size % (self.tp_size * self.sp_size):
            raise ValueError(
                f"a world of {self.world_size} is not divided by tp_size "
                f"{self.tp_size} x sp_size {self.sp_size}"
            )

    @property
    def sp_size(self) -> int:
        return self.sp_ulysses_size * self.sp_ring_size

    @property
    def dp_size(self) -> int:
        return self.world_size // self.tp_size

    @property
    def draft_dp_size(self) -> int:
        return self.world_size // self.sp_size

    @property
    def data_parallel_size(self) -> int:
        """Ranks that receive different samples."""
        return self.world_size // (self.tp_size * self.sp_size)

    def layout(self) -> dict:
        """The sizes that describe the job, for records."""
        return {
            "world_size": self.world_size,
            "tp_size": self.tp_size,
            "sp_ulysses_size": self.sp_ulysses_size,
            "sp_ring_size": self.sp_ring_size,
        }


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


def init_distributed(
    timeout: int = 10, tp_size: int = 1, sp_ulysses_size: int = 1, sp_ring_size: int = 1
) -> Topology:
    """Initialize distributed training and return the job's topology.

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
    # sizes exceed 1. For SP=1, the SP groups are the singleton DeviceMesh
    # group below; a None group would make torch.distributed operations
    # silently use the default WORLD group instead.
    if sp_ulysses_size * sp_ring_size > 1:
        process_group, set_seq_parallel_pg = _load_yunchang_globals()
    else:
        process_group, set_seq_parallel_pg = None, None

    dist.init_process_group(backend=backend, timeout=timedelta(minutes=timeout))
    try:
        return _build_topology(
            device_type,
            local_rank,
            tp_size=tp_size,
            sp_ulysses_size=sp_ulysses_size,
            sp_ring_size=sp_ring_size,
            process_group=process_group,
            set_seq_parallel_pg=set_seq_parallel_pg,
        )
    except BaseException:
        # No topology reaches the caller, so nothing else can tear the default
        # group down; a retry in this process would find it initialized.
        dist.destroy_process_group()
        raise


def _build_topology(
    device_type,
    local_rank,
    *,
    tp_size,
    sp_ulysses_size,
    sp_ring_size,
    process_group,
    set_seq_parallel_pg,
) -> Topology:
    """The groups and meshes of an initialized world, created in a fixed order."""
    print_with_rank(f"bind to {device_type} device {local_rank}")

    rank = dist.get_rank()
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
        set_seq_parallel_pg(sp_ulysses_size, sp_ring_size, rank, world_size)

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

    return Topology(
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
        tp_rank=dist.get_rank(tp_group),
        dp_rank=dist.get_rank(dp_group),
        draft_dp_rank=dist.get_rank(draft_dp_group),
        sp_rank=dist.get_rank(draft_sp_group),
        device_type=device_type,
        world_group=dist.group.WORLD,
        tp_group=tp_group,
        dp_group=dp_group,
        draft_dp_group=draft_dp_group,
        draft_sp_group=draft_sp_group,
        sp_ulysses_group=sp_ulysses_group,
        sp_ring_group=sp_ring_group,
    )


def destroy_distributed(topology: Topology, *, abort: bool = False) -> None:
    """Tear down every process group ``topology`` holds.

    ``abort=True`` (rank-local failure path) aborts communicators instead of the
    collective destroy, so this rank exits with its traceback instead of hanging
    behind peers stuck in a collective; torchrun then terminates the peers.
    """
    # Teardown must never crash the process. Several handles can alias the same
    # underlying group (e.g. DP and draft-DP when there is no sequence
    # parallelism), and degenerate single-rank SP groups (created when
    # sp_ulysses_size == 1 or sp_ring_size == 1) are not registered in torch's
    # process-group map and would raise on destroy. Clean up each distinct,
    # valid group at most once.
    seen = set()
    for group in (
        topology.tp_group,
        topology.dp_group,
        topology.sp_ulysses_group,
        topology.sp_ring_group,
        topology.draft_dp_group,
        topology.draft_sp_group,
        topology.world_group,  # may alias the DP group; the seen-set dedups it
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
