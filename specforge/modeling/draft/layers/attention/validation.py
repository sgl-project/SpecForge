"""Projection-independent input contracts for draft attention components."""

import torch
from torch import Tensor


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
        or context.dtype != hidden.dtype
    ):
        raise ValueError(
            "target_context must match hidden batch, input width, dtype, and device"
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
