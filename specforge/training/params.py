"""Local storage access shared by gradient scaling and FP32 master updates."""

import torch
from torch.distributed.tensor import DTensor


def local_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Return a shard view without gathering or changing a DTensor's placement.

    Callers explicitly reduce local gradient norms over the owning shard group;
    mixing DTensor reductions with that collective would count shards twice.
    Resolve the view at use time since FSDP replaces storage around forward.
    """
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor
