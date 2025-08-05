
from datetime import timedelta

import torch
import torch.distributed as dist

from specforge.utils import print_with_rank,detect_communication_backend

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
        tp_size(int): The degree of tensor parallelism
    """
    backend_name=detect_communication_backend()
    dist.init_process_group(backend=backend_name, timeout=timedelta(minutes=timeout))

    # initialize sub groups
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Set CUDA device only if CUDA is available
    if torch.cuda.is_available():
        local_rank = rank % torch.cuda.device_count()
        torch.cuda.set_device(local_rank)
        print_with_rank(f"Bind to CUDA device {local_rank}")
    else:
        print_with_rank("CUDA not available, running on CPU")

    dp_size = world_size // tp_size
    assert world_size == tp_size * dp_size, "world size must be divisible by tp size"
    global _TP_GROUP, _DP_GROUP

    # create tp group
    tp_ranks = [list(range(i * tp_size, (i + 1) * tp_size)) for i in range(dp_size)]
    for ranks in tp_ranks:
        tp_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            _TP_GROUP = tp_group

    # create dp group
    dp_ranks = [list(range(i, world_size, tp_size)) for i in range(tp_size)]
    for ranks in dp_ranks:
        dp_group = dist.new_group(ranks=ranks)
        if rank in ranks:
            _DP_GROUP = dp_group


def destroy_distributed():
    global _TP_GROUP, _DP_GROUP
    dist.destroy_process_group(_TP_GROUP)
    dist.destroy_process_group(_DP_GROUP)
    dist.destroy_process_group()
