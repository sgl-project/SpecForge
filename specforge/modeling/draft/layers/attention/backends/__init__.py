"""Projection-independent runtime attention execution."""

from ..masks import PaddingMask
from .base import (
    AttentionBackend,
    AttentionPreparationScope,
    AttentionResult,
    PreparedAttention,
)
from .dense import EagerAttentionBackend, SDPAAttentionBackend
from .eager import eager_attention
from .flash import FlashAttentionBackend
from .flex import FlexAttentionBackend
from .sdpa import sdpa_attention

_BACKENDS = {
    "eager": EagerAttentionBackend(),
    "sdpa": SDPAAttentionBackend(),
    "flex_attention": FlexAttentionBackend(),
    "fa": FlashAttentionBackend(),
}


def resolve_attention_backend(backend: str | AttentionBackend) -> AttentionBackend:
    if isinstance(backend, AttentionBackend):
        return backend
    if backend == "usp":
        raise NotImplementedError("USP is not implemented for canonical attention")
    if not isinstance(backend, str) or backend not in _BACKENDS:
        raise ValueError(f"unsupported attention backend {backend!r}")
    return _BACKENDS[backend]


__all__ = [
    "AttentionBackend",
    "AttentionPreparationScope",
    "AttentionResult",
    "PreparedAttention",
    "EagerAttentionBackend",
    "SDPAAttentionBackend",
    "FlexAttentionBackend",
    "FlashAttentionBackend",
    "PaddingMask",
    "resolve_attention_backend",
    "eager_attention",
    "sdpa_attention",
]


def attention_backend(name):
    if name == "eager":
        return eager_attention
    if name == "sdpa":
        return sdpa_attention
    if name == "flex_attention":
        from .flex import flex_attention

        return flex_attention
    raise ValueError(f"unsupported attention backend: {name!r}")


def execute_attention(function, query, key, value, *, attention_mask, **options):
    """Return activations and active query rows, including fully masked sparse rows."""
    from torch import Tensor

    if not isinstance(attention_mask, Tensor):
        return function(
            query,
            key,
            value,
            attention_mask=attention_mask,
            return_valid=True,
            **options,
        )
    output = function(query, key, value, attention_mask=attention_mask, **options)
    return output, attention_mask.any(dim=-1).any(dim=1)
