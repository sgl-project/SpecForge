"""Intersect backend-independent causal/window rules with algorithm visibility."""

import torch
from torch.nn.attention.flex_attention import BlockMask

from .flex_attention import compile_friendly_create_block_mask
from .flex_attention_backend import flex_attention_backend


def resolve_visibility(
    mask: torch.Tensor | BlockMask | None,
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    causal: bool,
    window: int | None,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    flex: bool = False,
) -> torch.Tensor | BlockMask | None:
    if (query_positions is None) != (key_positions is None):
        raise ValueError("Supply both query_positions and key_positions")
    if query_positions is not None:
        if query_positions.ndim != 2 or key_positions.ndim != 2:
            raise ValueError("Visibility positions must have shape [batch, sequence]")
        if (
            query_positions.shape[-1] != q.shape[-2]
            or key_positions.shape[-1] != k.shape[-2]
        ):
            raise ValueError(
                "Visibility positions must cover all queries and keys, including cache"
            )
        if query_positions.shape[0] not in (1, q.shape[0]) or key_positions.shape[
            0
        ] not in (1, q.shape[0]):
            raise ValueError("Visibility positions must broadcast to the query batch")
    if flex and mask is not None and not isinstance(mask, BlockMask):
        raise TypeError("FlexAttention requires a BlockMask")
    if not flex and isinstance(mask, BlockMask):
        raise TypeError("BlockMask requires the FlexAttention backend")
    if not causal and window is None:
        return mask
    q_len, k_len = q.shape[-2], k.shape[-2]

    def visible(b, h, qi, ki):
        if query_positions is None:
            distance = qi + k_len - q_len - ki
        else:
            # create_block_mask evaluates padded indices as well.
            qp = query_positions[b % query_positions.shape[0], qi.clamp(max=q_len - 1)]
            kp = key_positions[b % key_positions.shape[0], ki.clamp(max=k_len - 1)]
            distance = qp - kp
        allowed = (qi < q_len) & (ki < k_len)
        if causal:
            allowed = allowed & (distance >= 0)
        if window is not None:
            allowed = allowed & (distance.abs() < window)
        return allowed

    if flex:
        old_mod = None if mask is None else mask.mask_mod

        def combined(b, h, qi, ki):
            allowed = visible(b, h, qi, ki)
            return allowed if old_mod is None else allowed & old_mod(b, h, qi, ki)

        block_size = (
            mask.BLOCK_SIZE
            if mask is not None
            else (256, 128) if flex_attention_backend() == "FLASH" else 128
        )
        return compile_friendly_create_block_mask(
            combined,
            B=q.shape[0],
            H=None if mask is None else mask.shape[1],
            Q_LEN=q_len,
            KV_LEN=k_len,
            device=q.device,
            BLOCK_SIZE=block_size,
        )
    b = torch.arange(q.shape[0], device=q.device)[:, None, None, None]
    qi = torch.arange(q_len, device=q.device)[None, None, :, None]
    ki = torch.arange(k_len, device=q.device)[None, None, None, :]
    allowed = visible(b, 0, qi, ki)
    if mask is None:
        return allowed
    if mask.dtype == torch.bool:
        return mask & allowed
    return mask.masked_fill(~allowed, float("-inf"))


def prepare_tensor_mask(
    mask: torch.Tensor | None,
    dtype: torch.dtype,
    *,
    eager: bool,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Stabilize empty rows, and report validity per head before output mixing.

    Finite negative attention biases are valid scores. -inf and the conventional
    finfo.min sentinel are the only additive values treated as excluded keys.
    """
    if mask is None:
        return None, None
    if mask.ndim != 4:
        raise ValueError("Eager/SDPA masks must have shape [batch, heads, query, key]")
    if mask.dtype != torch.bool and not mask.is_floating_point():
        raise TypeError("Attention masks must be boolean or floating point")
    allowed = mask if mask.dtype == torch.bool else mask > torch.finfo(mask.dtype).min
    valid = allowed.any(-1, keepdim=True)
    if eager and mask.dtype == torch.bool:
        mask = torch.zeros_like(mask, dtype=dtype).masked_fill(
            ~mask, torch.finfo(dtype).min
        )
    elif mask.dtype != torch.bool:
        # Avoid softmax(-inf, ..., -inf), including its undefined backward.
        mask = mask.masked_fill(~valid, 0)
    return mask, valid
