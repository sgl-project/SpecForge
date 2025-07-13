import os
from omegaconf import OmegaConf
from contextlib import contextmanager
import torch.distributed as dist
import torch

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

def init_distributed():
    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(dist.get_rank())
