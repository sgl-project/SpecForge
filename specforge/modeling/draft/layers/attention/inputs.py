"""Shared shape, layout, context and joint-RoPE checks for dense projections."""

import torch
from torch import Tensor

from .masks import AttentionLayout


def validate_attention_inputs(
    hidden: Tensor,
    layout: AttentionLayout,
    position_embeddings: tuple[Tensor, Tensor] | None,
    target_context: Tensor | None,
    *,
    input_size: int,
    num_heads: int,
    rotary_dim: int,
) -> int:
    """Validate context-then-query inputs and return the context length."""
    if hidden.ndim != 3 or hidden.shape[-1] != input_size:
        raise ValueError("hidden must have shape [batch, query_length, input_size]")
    batch, length = hidden.shape[:2]
    if target_context is not None and (
        target_context.ndim != 3
        or target_context.shape[0] != batch
        or target_context.shape[-1] != hidden.shape[-1]
        or target_context.device != hidden.device
        or not target_context.is_floating_point()
        or (
            target_context.dtype != hidden.dtype
            and not (
                torch.is_autocast_enabled(hidden.device.type)
                and hidden.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and target_context.dtype
                in (torch.float32, torch.float16, torch.bfloat16)
            )
        )
    ):
        raise ValueError(
            "target_context must match hidden batch, input width, and device; "
            "different floating dtypes require autocast"
        )
    context_length = 0 if target_context is None else target_context.shape[1]
    keys = context_length + length
    if not isinstance(layout, AttentionLayout):
        raise TypeError("attention requires AttentionLayout")
    if (
        layout.query_positions.shape[0] not in (1, batch)
        or layout.query_positions.shape[1] != length
        or layout.key_positions.shape[1] != keys
        or layout.query_positions.device != hidden.device
        or layout.visibility.shape[1] not in (1, num_heads)
    ):
        raise ValueError(
            "layout must describe the query and context-then-query KV tensors"
        )
    if rotary_dim:
        if position_embeddings is None or len(position_embeddings) != 2:
            raise ValueError("rotary attention requires a joint cosine/sine pair")
        for value in position_embeddings:
            if (
                value.ndim != 3
                or value.shape[0] not in (1, batch)
                or value.shape[1:] != (keys, rotary_dim)
                or value.device != hidden.device
                or not value.is_floating_point()
            ):
                raise ValueError(
                    "RoPE tensors must describe all context-then-query positions"
                )
    elif position_embeddings is not None:
        raise ValueError("disabled RoPE requires position_embeddings=None")
    return context_length
