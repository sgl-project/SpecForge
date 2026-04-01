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


def _set_dist_env(rank: int, world_size: int, master_addr: str, master_port: int) -> None:
    """Set the environment variables expected by torch.distributed."""
    os.environ["MASTER_ADDR"] = master_addr
    os.environ["MASTER_PORT"] = str(master_port)
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    # LOCAL_RANK: in a Ray Actor, CUDA_VISIBLE_DEVICES exposes exactly
    # the GPUs allocated to this actor, so we always map to device index 0
    # within the visible set.  init_distributed() in distributed.py also
    # recomputes local_rank as rank % device_count(), which gives the same
    # result when each actor owns exactly one GPU (device_count == 1).
    os.environ["LOCAL_RANK"] = str(rank % max(torch.cuda.device_count(), 1))


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
    # Reuse the existing init_distributed logic; sp sizes default to 1 so
    # no SP process groups are created.
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

    After this call, ALL global variables in specforge.distributed
    (_TP_GROUP, _DP_GROUP, _DRAFT_DP_GROUP, _DRAFT_SP_GROUP,
    _SP_ULYSSES_GROUP, _SP_RING_GROUP, etc.) are correctly set and the
    helper functions (get_tp_group, get_dp_group, …) return usable objects.

    Constraints (same as init_distributed):
        world_size == dp_size * tp_size
        world_size % (sp_ulysses_size * sp_ring_size) == 0

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
