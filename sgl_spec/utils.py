import json
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import PretrainedConfig


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@torch.no_grad()
def padding(tensor, left=True):
    zeropadding = torch.zeros_like(tensor[:, -1:])
    if left:
        tensor = torch.cat((zeropadding, tensor[:, :-1]), dim=1)
    else:
        tensor = torch.cat((tensor[:, 1:], zeropadding), dim=1)
    return tensor


def init_distributed(timeout: int = 10):
    """Initialize distributed training.

    Args:
        timeout(int): Timeout for collective communication in minutes
    """
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)
