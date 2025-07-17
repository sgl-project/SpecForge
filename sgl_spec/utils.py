import os
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


def load_config_from_file(config_path: str):
    with open(config_path, "r") as f:
        config = json.load(f)

    return PretrainedConfig.from_dict(config)

def save_training_checkpoint(output_dir: str,
                             epoch: int,
                             draft_model,
                             optimizer,
                             scheduler,
                             args):
    """只在 rank0 上写磁盘"""
    if dist.get_rank() != 0:
        return
    ckpt_dir = os.path.join(output_dir, f"checkpoint-{epoch}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # 1) 保存模型权重（沿用你原来的写法）
    draft_model.save_pretrained(ckpt_dir)

    # 2) 保存训练状态
    torch.save(
        {
            "epoch": epoch,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "args": vars(args),          # 方便以后完全复现实验
        },
        os.path.join(ckpt_dir, "training_states.pt")
    )
    print_on_rank0(f"[CHECKPOINT] saved to {ckpt_dir}")


def load_training_checkpoint(ckpt_dir: str,
                             draft_model,
                             optimizer,
                             scheduler,
                             args):
    """所有 rank 都执行；返回下一 epoch 的编号"""
    # 1) 加载模型权重
    draft_model = type(draft_model).from_pretrained(ckpt_dir).cuda().to(torch.bfloat16)

    # 2) 加载训练状态
    states = torch.load(os.path.join(ckpt_dir, "training_states.pt"),
                        map_location="cpu")
    optimizer.load_state_dict(states["optimizer"])
    scheduler.load_state_dict(states["scheduler"])
    next_epoch = states["epoch"] + 1
    print_on_rank0(f"[RESUME] 从 epoch {next_epoch} 继续")
    return draft_model, optimizer, scheduler, next_epoch