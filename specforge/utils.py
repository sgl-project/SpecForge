import json
import netrc
import os
import re
import sys
from contextlib import contextmanager
from datetime import timedelta

import torch
import torch.distributed as dist
from transformers import PretrainedConfig


def detect_communication_backend() -> str:
    """
    Detect the best communication backend for distributed training,
    based on the platform and hardware availability.

    Returns:
        backend_name (str): One of 'nccl', 'gloo', or 'mpi'
    """
    if sys.platform == 'darwin':
        backend_name = 'gloo'  # macOS does not support NCCL
    elif torch.cuda.is_available():
        backend_name = 'nccl'  # Recommended for GPU-based training
    else:
        backend_name = 'gloo'  # CPU-only training (Linux/Windows)
    print(f"[INFO] Selected backend: {backend_name} on platform {sys.platform} (CUDA: {torch.cuda.is_available()})")
    return backend_name


def detect_attention_impl(prefer="flex_attention") -> str:
    """
    Detect the best available attention implementation given the device.

    Args:
        prefer (str): preferred backend, one of ["flex_attention", "flash", "sdpa"].
                      Will try to use it if supported, otherwise fallback.

    Returns:
        str: selected attention implementation ("flex_attention", "flash", "sdpa").
    """
    device = (
        "cuda" if torch.cuda.is_available()
        else "hpu" if getattr(torch.backends, "hpu", None) and torch.backends.hpu.is_available()
        else "cpu" if torch.backends.mkldnn.is_available()
        else "mps"
    )

    # FlexAttention
    if prefer == "flex_attention":
        if device in ["cuda", "cpu", "hpu"]:
            print(f"[INFO] Using FlexAttention on {device}")
            return "flex_attention"
        else:
            print(f"[WARN] FlexAttention not supported on {device}, falling back to SDPA")
            return "sdpa"

    # FlashAttention, only for cuda, delayed
    # if prefer == "flash":
    #     if device == "cuda":
    #         print(f"[INFO] Using FlashAttention on CUDA")
    #         return "flash"
    #     else:
    #         print(f"[WARN] FlashAttention only works on CUDA, falling back to SDPA")
    #         return "sdpa"

    # default PyTorch SDPA
    print(f"[INFO] Using PyTorch SDPA attention on {device}")
    return "sdpa"


def detect_device() -> torch.device:
    """
    Detect the best available device (CUDA, MPS, or CPU) for PyTorch computation.

    Returns:
        torch.device: The selected device.
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        reason = "CUDA is available"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps:0")
        reason = "MPS (Apple Silicon GPU) is available"
    else:
        device = torch.device("cpu")
        reason = "falling back to CPU"

    print(f"[INFO] Using device: {device} ({reason})")

    return device


def validate_wandb_args(parser, args):
    if not args.wandb:
        return
    if args.wandb_key is not None:
        return

    if "WANDB_API_KEY" in os.environ:
        args.wandb_key = os.environ["WANDB_API_KEY"]
        return

    # Check ~/.netrc file for wandb credentials
    try:
        netrc_path = os.path.expanduser("~/.netrc")
        if os.path.exists(netrc_path):
            netrc_file = netrc.netrc(netrc_path)
            # Check for api.wandb.ai machine
            if "api.wandb.ai" in netrc_file.hosts:
                login, account, password = netrc_file.authenticators("api.wandb.ai")
                if password:
                    args.wandb_key = password
                    return True
    except (FileNotFoundError, netrc.NetrcParseError):
        pass

    if args.wandb_key is None:
        parser.error(
            "When --wandb is enabled, you must provide a wandb API key via one of:\n"
            "  1. --wandb-key argument\n"
            "  2. WANDB_API_KEY environment variable\n"
            "  3. wandb login api-key"
        )


@contextmanager
def rank_0_priority():
    rank = dist.get_rank()

    if rank == 0:
        yield
        dist.barrier()
    else:
        dist.barrier()
        yield


@contextmanager
def default_torch_dtype(dtype: torch.dtype):
    current_dtype = torch.get_default_dtype()
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(current_dtype)


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


def print_with_rank(message):
    print(f"rank {dist.get_rank()}: {message}")


PREFIX_CHECKPOINT_DIR = "epoch"
_re_checkpoint = re.compile(r"^" + PREFIX_CHECKPOINT_DIR + r"_(\d+)$")


def get_last_checkpoint(folder):
    content = os.listdir(folder)
    checkpoints = [
        path
        for path in content
        if _re_checkpoint.search(path) is not None
           and os.path.isdir(os.path.join(folder, path))
    ]
    if len(checkpoints) == 0:
        return
    return os.path.join(
        folder,
        max(checkpoints, key=lambda x: int(_re_checkpoint.search(x).groups()[0])),
    )
