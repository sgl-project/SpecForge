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
]
