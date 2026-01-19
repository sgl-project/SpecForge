"""
Distributed initialization utilities for SGLang backend.

This module provides functions to initialize torch distributed for SGLang backend,
similar to how train_eagle3.py does it via specforge.distributed.
"""

import logging
import os
from datetime import timedelta
from typing import Optional

import torch
import torch.distributed as dist

import specforge.distributed as specforge_dist

logger = logging.getLogger(__name__)


def init_sglang_distributed(
    tp_size: int = 1,
    timeout: int = 20,
) -> int:
    """
    Initialize torch distributed for SGLang backend.
    
    This sets up the distributed environment and creates the TP group
    needed by SGLang's model runner. For TP > 1, this should be launched
    via torchrun.
    
    Args:
        tp_size: Tensor parallel size
        timeout: Timeout for distributed initialization in minutes
    
    Returns:
        The local rank (tp_rank)
    """
    if tp_size <= 1:
        _init_single_gpu_distributed(timeout)
        return 0
    
    if dist.is_initialized():
        tp_rank = dist.get_rank()
        logger.info(f"Distributed already initialized, rank={tp_rank}")
        _ensure_tp_group_set(tp_size)
        return tp_rank
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    if world_size != tp_size:
        raise ValueError(
            f"For TP={tp_size}, launch with: torchrun --standalone --nproc_per_node={tp_size} ...\n"
            f"Got WORLD_SIZE={world_size}"
        )
    
    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    
    _setup_tp_group(tp_size)
    
    logger.info(
        f"Initialized distributed for SGLang: rank={local_rank}, "
        f"world_size={world_size}, tp_size={tp_size}"
    )
    return local_rank


def _init_single_gpu_distributed(timeout: int = 20) -> None:
    """Initialize minimal distributed environment for single GPU."""
    if dist.is_initialized():
        return
    
    if "RANK" not in os.environ:
        os.environ["RANK"] = "0"
    if "WORLD_SIZE" not in os.environ:
        os.environ["WORLD_SIZE"] = "1"
    if "LOCAL_RANK" not in os.environ:
        os.environ["LOCAL_RANK"] = "0"
    if "MASTER_ADDR" not in os.environ:
        os.environ["MASTER_ADDR"] = "localhost"
    if "MASTER_PORT" not in os.environ:
        os.environ["MASTER_PORT"] = "29500"
    
    torch.cuda.set_device(0)
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout))
    
    _setup_tp_group(tp_size=1)
    logger.info("Initialized single GPU distributed environment")


def _setup_tp_group(tp_size: int) -> None:
    """Set up the TP group in specforge.distributed module."""
    world_size = dist.get_world_size()
    
    if tp_size == world_size:
        specforge_dist._TP_GROUP = dist.group.WORLD
    else:
        num_tp_groups = world_size // tp_size
        for i in range(num_tp_groups):
            ranks = list(range(i * tp_size, (i + 1) * tp_size))
            group = dist.new_group(ranks)
            if dist.get_rank() in ranks:
                specforge_dist._TP_GROUP = group
    
    logger.debug(f"Set up TP group with tp_size={tp_size}")


def _ensure_tp_group_set(tp_size: int) -> None:
    """Ensure TP group is set if distributed is already initialized."""
    if specforge_dist._TP_GROUP is None:
        _setup_tp_group(tp_size)


def destroy_sglang_distributed() -> None:
    """Clean up distributed process groups."""
    if dist.is_initialized():
        dist.destroy_process_group()
        specforge_dist._TP_GROUP = None
        logger.info("Destroyed distributed process groups")
