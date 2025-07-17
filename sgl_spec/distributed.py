from datetime import timedelta

import torch
import torch.distributed as dist

_TP_GROUP = None
_DP_GROUP = None


def get_tp_group():
    global _TP_GROUP
    return _TP_GROUP


def get_dp_group():
    global _DP_GROUP
    return _DP_GROUP


def init_distributed(timeout: int = 10, tp_size: int = 1):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())
    _create_tp_group(tp_size)


def _create_tp_group(tp_size):
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size % tp_size == 0, "world size must be divisible by tp size"

    num_tp_groups = world_size // tp_size
    tp_ranks = [
        list(range(i * tp_size, (i + 1) * tp_size)) for i in range(num_tp_groups)
    ]

    tp_group_of_current_rank = None
    for ranks in tp_ranks:
        tp_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            tp_group_of_current_rank = tp_group

    global _TP_GROUP
    _TP_GROUP = tp_group_of_current_rank

    return _TP_GROUP
