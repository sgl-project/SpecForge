"""Apply full or partial rotation to query/key tensors independently."""

import torch
from torch import Tensor


def _rotate_half(hidden: Tensor) -> Tensor:
    first = hidden[..., : hidden.shape[-1] // 2]
    second = hidden[..., hidden.shape[-1] // 2 :]
    return torch.cat((-second, first), dim=-1)


def apply_rotary(
    hidden: Tensor,
    position_embeddings: tuple[Tensor, Tensor],
    *,
    interleaved: bool = False,
) -> Tensor:
    """Rotate [batch, heads, sequence, dim] using [batch, sequence, rotary_dim].

    Queries and keys may have different head counts or positions; callers select
    their embeddings separately. Tables and arithmetic use FP32, with one final
    cast as in SGLang. Dimensions after rotary_dim pass through.
    """
    cos, sin = position_embeddings
    if hidden.ndim != 4 or cos.ndim != 3 or sin.shape != cos.shape:
        raise ValueError("expected rank-4 query/key and matching rank-3 cosine/sine")
    if cos.shape[0] not in (1, hidden.shape[0]) or cos.shape[1] != hidden.shape[2]:
        raise ValueError(
            "RoPE batch must broadcast to hidden and sequence lengths must match"
        )
    dim = cos.shape[-1]
    if dim < 2 or dim % 2 or dim > hidden.shape[-1]:
        raise ValueError("rotary dimension must be positive, even, and fit the head")
    dtype = hidden.dtype
    hidden, cos, sin = hidden.float(), cos.float(), sin.float()
    if interleaved:
        rotated, tail = hidden[..., :dim], hidden[..., dim:]
        cos, sin = cos[..., : dim // 2].unsqueeze(1), sin[..., : dim // 2].unsqueeze(1)
        first, second = rotated[..., ::2], rotated[..., 1::2]
        output = torch.stack(
            (first * cos - second * sin, second * cos + first * sin), dim=-1
        ).flatten(-2)
        return torch.cat((output, tail), dim=-1).to(dtype)
    if dim == hidden.shape[-1]:
        # Preserve HF's operation order and transposed-view layout for full heads.
        cos, sin = cos.unsqueeze(1), sin.unsqueeze(1)
        return ((hidden * cos) + (_rotate_half(hidden) * sin)).to(dtype)
    rotated, tail = hidden[..., :dim], hidden[..., dim:]
    first, second = rotated.chunk(2, dim=-1)
    half_rotated = torch.cat((-second, first), dim=-1)
    output = rotated * cos.unsqueeze(1) + half_rotated * sin.unsqueeze(1)
    return torch.cat((output, tail), dim=-1).to(dtype)
