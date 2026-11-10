"""Projection-independent input contracts for draft attention components."""

import torch
from torch import Tensor

from specforge.config.draft.attention import AttentionSpec

from .backends import AttentionBackend, PreparedAttention


def validate_hidden(
    hidden: Tensor, context: Tensor | None, input_size: int
) -> tuple[int, int, int]:
    if hidden.ndim != 3 or hidden.shape[-1] != input_size:
        raise ValueError("hidden must have shape [batch, query_length, input_size]")
    if context is not None and (
        context.ndim != 3
        or context.shape[0] != hidden.shape[0]
        or context.shape[-1] != input_size
        or context.device != hidden.device
        or not context.is_floating_point()
        or (
            context.dtype != hidden.dtype
            and not (
                torch.is_autocast_enabled(hidden.device.type)
                and hidden.dtype in (torch.float32, torch.float16, torch.bfloat16)
                and context.dtype in (torch.float32, torch.float16, torch.bfloat16)
            )
        )
    ):
        raise ValueError(
            "target_context must match hidden batch, input width, and device; "
            "different floating dtypes require autocast"
        )
    return hidden.shape[0], hidden.shape[1], 0 if context is None else context.shape[1]


def validate_rotary(
    embeddings: tuple[Tensor, Tensor] | None,
    hidden: Tensor,
    key_length: int,
    rotary_dim: int,
) -> None:
    if not rotary_dim:
        if embeddings is not None:
            raise ValueError("disabled RoPE requires position_embeddings=None")
        return
    if embeddings is None or len(embeddings) != 2:
        raise ValueError("rotary attention requires a joint cosine/sine pair")
    for value in embeddings:
        if (
            value.ndim != 3
            or value.shape[0] not in (1, hidden.shape[0])
            or value.shape[1:] != (key_length, rotary_dim)
            or value.device != hidden.device
            or value.dtype != torch.float32
        ):
            raise ValueError(
                "RoPE tensors must describe all context-then-query positions and use FP32"
            )


def validate_attention_inputs(
    hidden: Tensor,
    attention: PreparedAttention,
    position_embeddings: tuple[Tensor, Tensor] | None,
    target_context: Tensor | None,
    *,
    input_size: int,
    config: AttentionSpec,
    backend: AttentionBackend,
    rotary_dim: int,
) -> tuple[int, int, int]:
    batch, length, context_length = validate_hidden(hidden, target_context, input_size)
    if not isinstance(attention, PreparedAttention):
        raise TypeError("attention requires PreparedAttention")
    layout = attention.layout
    keys = context_length + length
    if (
        attention.batch_size != batch
        or attention.num_heads != config.num_attention_heads
        or attention.backend is not backend
        or attention.is_causal != config.is_causal
        or attention.sliding_window != config.sliding_window
        or layout.query_positions.shape[1] != length
        or layout.key_positions.shape[1] != keys
        or layout.query_positions.device != hidden.device
    ):
        raise ValueError(
            "prepared attention must match the layer and context-then-query tensors"
        )
    validate_rotary(position_embeddings, hidden, keys, rotary_dim)
    return batch, length, context_length
