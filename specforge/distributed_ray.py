"""
Ray-aware torch.distributed initialization utilities.

Unlike the torchrun-based distributed.py, this module initializes
process groups by explicitly setting MASTER_ADDR / MASTER_PORT env vars,
so each Ray Actor can bring up its own distributed backend without the
torchrun launcher.
"""

import os
import socket
from contextlib import closing
from datetime import timedelta

import torch
import torch.distributed as dist


def get_free_port() -> int:
    """
    Find an unused TCP port on localhost.
    Called in the orchestrator process to obtain a rendezvous port for
    torch.distributed.init_process_group.
    """
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(("", 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]


def _set_dist_env(
    rank: int, world_size: int, master_addr: str, master_port: int
) -> None:
    """Set the environment variables expected by torch.distributed."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # LOCAL_RANK: Each Ray Actor is allocated exactly 1 GPU via
    # @ray.remote(num_gpus=1).  Ray sets CUDA_VISIBLE_DEVICES to expose
    # only that single GPU, so torch.cuda.device_count() == 1 inside the
    # actor.  Therefore LOCAL_RANK is always 0.
    os.environ["LOCAL_RANK"] = "0"


# ─────────────────────────────────────────────────────────────────────────
# Global process group (cross-group NCCL for disaggregated mode)
# ─────────────────────────────────────────────────────────────────────────


def init_global_distributed(
    global_rank: int,
    global_world_size: int,
    master_addr: str,
    master_port: int,
    timeout_minutes: int = 20,
) -> None:
    """Initialize a GLOBAL NCCL process group spanning ALL actors.

    Must be called by every actor (rollout + train) before any local
    subgroup initialization.  After this call, ``dist.get_rank()``
    returns the globally unique rank.

    Args:
        global_rank:       Unique rank across all actors (rollout + train).
        global_world_size: Total number of actors.
        master_addr:       Shared rendezvous address.
        master_port:       Shared rendezvous port.
        timeout_minutes:   NCCL timeout.
    """
    _set_dist_env(global_rank, global_world_size, master_addr, master_port)
    torch.cuda.set_device(0)  # Each actor sees exactly 1 GPU
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout_minutes),
    )


def init_rollout_subgroup(
    rollout_ranks: list,
    tp_size: int,
) -> None:
    """Create local subgroups for a RolloutWorker after global init.

    Uses use_local_synchronization=True so only rollout ranks need to
    participate — no global barrier with train ranks.
    """
    from specforge.distributed import init_distributed_from_subgroup

    rollout_group = dist.new_group(
        ranks=rollout_ranks,
        use_local_synchronization=True,
    )
    init_distributed_from_subgroup(
        local_group=rollout_group,
        tp_size=tp_size,
        sp_ulysses_size=1,
        sp_ring_size=1,
    )


def init_train_subgroup(
    train_ranks: list,
    tp_size: int,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
) -> None:
    """Create local subgroups for a TrainWorker after global init.

    Uses use_local_synchronization=True so only train ranks need to
    participate — no global barrier with rollout ranks.
    """
    from specforge.distributed import init_distributed_from_subgroup

    train_group = dist.new_group(
        ranks=train_ranks,
        use_local_synchronization=True,
    )
    init_distributed_from_subgroup(
        local_group=train_group,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
    )


# ─────────────────────────────────────────────────────────────────────────
# Legacy per-group initialization (used by Ray object store transfer mode)
# ─────────────────────────────────────────────────────────────────────────


def init_rollout_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp_size: int,
    timeout_minutes: int = 20,
) -> None:
    """
    Initialize the torch.distributed process group for a RolloutWorker.

    Only the TP group is needed on the rollout side.  The function
    delegates to specforge.distributed.init_distributed with sp=1 so that
    only the TP/DP device meshes are created – no SP groups.

    Args:
        rank:            Global rank of this worker in the rollout world.
        world_size:      Total number of rollout workers (== rollout_num_gpus).
        master_addr:     IP / hostname of rank-0 in the rollout group.
        master_port:     TCP port for rendezvous.
        tp_size:         Tensor-parallel degree (1 = no TP).
        timeout_minutes: NCCL timeout in minutes.
    """
    _set_dist_env(rank, world_size, master_addr, master_port)
    from specforge.distributed import init_distributed

    init_distributed(
        timeout=timeout_minutes,
        tp_size=tp_size,
        sp_ulysses_size=1,
        sp_ring_size=1,
    )


def init_train_distributed(
    rank: int,
    world_size: int,
    master_addr: str,
    master_port: int,
    tp_size: int,
    sp_ulysses_size: int = 1,
    sp_ring_size: int = 1,
    timeout_minutes: int = 20,
) -> None:
    """
    Initialize the torch.distributed process groups for a TrainWorker.

    Fully equivalent to calling specforge.distributed.init_distributed()
    from a torchrun context, but populates the required env vars first
    so that no external launcher is needed.

    Args:
        rank:             Global rank of this TrainWorker.
        world_size:       Total number of TrainWorkers (== train_num_gpus).
        master_addr:      IP / hostname of rank-0 in the training group.
        master_port:      TCP port for rendezvous.
        tp_size:          Tensor-parallel degree for the draft model.
        sp_ulysses_size:  Ulysses sequence-parallel degree.
        sp_ring_size:     Ring sequence-parallel degree.
        timeout_minutes:  NCCL timeout in minutes.
    """
    _set_dist_env(rank, world_size, master_addr, master_port)
    from specforge.distributed import init_distributed

    init_distributed(
        timeout=timeout_minutes,
        tp_size=tp_size,
        sp_ulysses_size=sp_ulysses_size,
        sp_ring_size=sp_ring_size,
    )
