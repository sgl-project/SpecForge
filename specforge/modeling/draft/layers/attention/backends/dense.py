"""Execution adapters for explicit Q/K/V. No projection or indexer weights."""

from typing import TYPE_CHECKING, Callable, Optional

from torch.nn.attention.flex_attention import BlockMask

if TYPE_CHECKING:
    from ..base import DenseAttention

import torch
from transformers.models.qwen3.modeling_qwen3 import (
    ALL_ATTENTION_FUNCTIONS,
    eager_attention_forward,
)

from .flex_attention import compile_friendly_flex_attention
from .flex_attention_backend import flex_attention_backend
from .visibility import prepare_tensor_mask, resolve_visibility


def dense_attention(
    module: "DenseAttention",
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor | BlockMask | None,
    *,
    query_positions: torch.Tensor | None = None,
    key_positions: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    implementation = module.config._attn_implementation
    causal = kwargs.pop("is_causal", module.is_causal)
    if not isinstance(causal, bool):
        raise ValueError("is_causal must be a boolean")
    valid_queries = None
    native_sdpa_causal = (
        implementation == "sdpa"
        and causal
        and module.sliding_window is None
        and attention_mask is None
        and query_positions is None
        and key_positions is None
        and (q.shape[-2] == k.shape[-2] or q.shape[-2] == 1)
    )
    if implementation in ("eager", "sdpa", "flex_attention") and not native_sdpa_causal:
        attention_mask = resolve_visibility(
            attention_mask,
            q,
            k,
            causal=causal,
            window=module.sliding_window,
            query_positions=query_positions,
            key_positions=key_positions,
            flex=implementation == "flex_attention",
        )
    elif implementation.startswith("flash_attention"):
        # Flash's native right-aligned causal/window support is suitable for
        # contiguous sequences. Arbitrary logical layouts need Eager/SDPA/Flex.
        if query_positions is not None or key_positions is not None:
            raise ValueError(
                "Explicit visibility positions require Eager, SDPA or FlexAttention"
            )
        if attention_mask is not None:
            if attention_mask.ndim != 2:
                raise ValueError("FlashAttention only supports a 2D padding mask")
            valid_queries = attention_mask[:, None, -q.shape[-2] :, None].bool()
            valid_queries = valid_queries & attention_mask.any(-1)[:, None, None, None]
    elif implementation != "sdpa":
        raise ValueError(f"Unsupported dense attention backend: {implementation}")
    if implementation == "flex_attention":
        kernel_options = dict(kwargs.pop("kernel_options", None) or {})
        backend = flex_attention_backend()
        if backend is not None:
            kernel_options["BACKEND"] = backend

        result = compile_friendly_flex_attention(
            q,
            k,
            v,
            block_mask=attention_mask,
            enable_gqa=True,
            scale=module.scaling,
            kernel_options=kernel_options or None,
            return_lse=attention_mask is not None,
        )
        if attention_mask is not None:
            attn_output, lse = result
            valid_queries = (~torch.isneginf(lse)).unsqueeze(-1)
        else:
            attn_output = result
        attn_output = attn_output.transpose(1, 2)
        attn_weights = None
    else:
        attn_fn: Callable = eager_attention_forward
        if implementation in ("eager", "sdpa"):
            attention_mask, valid_queries = prepare_tensor_mask(
                attention_mask, q.dtype, eager=implementation == "eager"
            )
        if implementation != "eager":
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
            is_causal=causal,
            **kwargs,
        )
        if valid_queries is not None and attn_weights is not None:
            attn_weights = attn_weights.masked_fill(~valid_queries, 0)
    if valid_queries is not None:
        # Zero each head before o_proj; otherwise an empty head leaks forbidden
        # values even when another head has visible keys for the same query.
        attn_output = attn_output.masked_fill(~valid_queries.transpose(1, 2), 0)
    return attn_output, attn_weights, valid_queries
