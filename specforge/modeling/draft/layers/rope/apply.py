"""Apply full or partial rotation to query/key tensors independently."""

import torch
from torch import Tensor


def apply_rotary(hidden: Tensor, position_embeddings: tuple[Tensor, Tensor]) -> Tensor:
    """Rotate [batch, heads, sequence, dim] using [batch, sequence, rotary_dim].

    Queries and keys may have different head counts or positions; callers select
    their embeddings separately. Dimensions after rotary_dim pass through.
    """
    cos, sin = position_embeddings
    if hidden.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("expected rank-4 query/key and matching rank-3 cosine/sine")
    dim = cos.shape[-1]
    if dim < 2 or dim % 2 or dim > hidden.shape[-1]:
        raise ValueError("rotary dimension must be positive, even, and fit the head")
    rotated, tail = hidden[..., :dim], hidden[..., dim:]
    first, second = rotated.chunk(2, dim=-1)
    half_rotated = torch.cat((-second, first), dim=-1)
    output = rotated * cos.unsqueeze(1) + half_rotated * sin.unsqueeze(1)
    return torch.cat((output, tail), dim=-1)
