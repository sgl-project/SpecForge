"""Execution adapters for explicit Q/K/V. No projection weights."""

from typing import TYPE_CHECKING, Callable, Optional

import torch
from torch.nn.attention.flex_attention import BlockMask
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from .flex_attention import compile_friendly_flex_attention
from .flex_attention_backend import flex_attention_backend

if TYPE_CHECKING:
    from ..base import DenseAttention


def prepare_eager_mask(
    attention_mask: Optional[torch.Tensor],
    dtype: torch.dtype,
) -> tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Convert a boolean allow-mask to eager's additive representation."""

    if attention_mask is None or attention_mask.dtype != torch.bool:
        return attention_mask, None

    valid_queries = attention_mask.any(dim=-1, keepdim=True)
    # A finite minimum keeps eager softmax stable. Fully masked query rows are
    # explicitly zeroed after attention so they cannot average forbidden values.
    additive_mask = torch.zeros_like(attention_mask, dtype=dtype)
    additive_mask.masked_fill_(~attention_mask, torch.finfo(dtype).min)
    return additive_mask, valid_queries


def dense_attention(
    module: "DenseAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Run the configured backend and report eager's fully masked queries."""

    implementation = module.config._attn_implementation
    valid_queries = None
    if implementation == "flex_attention":
        kernel_options = dict(kwargs.pop("kernel_options", None) or {})
        backend = flex_attention_backend()
        if backend is not None:
            kernel_options["BACKEND"] = backend

        attn_output = compile_friendly_flex_attention(
            q,
            k,
            v,
            block_mask=attention_mask,
            enable_gqa=True,
            scale=module.scaling,
            kernel_options=kernel_options or None,
        ).transpose(1, 2)
        attn_weights = None
    else:
        attn_fn: Callable = eager_attention_forward
        if implementation == "eager":
            attention_mask, valid_queries = prepare_eager_mask(attention_mask, q.dtype)
        else:
            attn_fn = ALL_ATTENTION_FUNCTIONS[implementation]
        attn_output, attn_weights = attn_fn(
            module,
            q,
            k,
            v,
            attention_mask,
            dropout=0.0 if not module.training else module.attention_dropout,
            scaling=module.scaling,
            sliding_window=module.sliding_window,
            **kwargs,
        )
        if valid_queries is not None and attn_weights is not None:
            attn_weights = attn_weights.masked_fill(~valid_queries, 0)
    return attn_output, attn_weights, valid_queries
