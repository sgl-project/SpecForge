"""Runtime attention backends and preparation scoped to one model forward."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

import torch
from torch import Tensor

from specforge.config.draft._validation import boolean, positive_int

from ..masks import AttentionLayout


@dataclass(frozen=True, eq=False)
class AttentionResult:
    output: Tensor
    active_queries: Tensor
    lse: Tensor | None = None

    def mask_output(self, output: Tensor) -> Tensor:
        """Apply after output projection so its bias cannot revive inactive queries."""
        return output.masked_fill(~self.active_queries.unsqueeze(-1), 0)


class AttentionBackend(ABC):
    def validate(self, *, head_dim: int, value_head_dim: int, dropout_p: float) -> None:
        """Check static execution limits before parameters are allocated.

        dropout_p is the effective probability: zero for inference.
        Device and tensor dtype checks remain in attend().
        """

    @abstractmethod
    def prepare(
        self,
        layout: AttentionLayout,
        *,
        batch_size: int,
        num_heads: int,
        is_causal: bool,
        sliding_window: int | None,
    ) -> tuple[object, Tensor]:
        """Return backend mask and active queries; do not allocate learned state."""

    @abstractmethod
    def attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        mask: object,
        scale: float,
        dropout_p: float,
        training: bool,
        return_lse: bool,
    ) -> tuple[Tensor, Tensor | None]:
        """Return attention output and optional pre-dropout logsumexp [B, H, Q]."""


@dataclass(frozen=True, eq=False)
class PreparedAttention:
    layout: AttentionLayout
    backend: AttentionBackend
    mask: object
    active_queries: Tensor
    batch_size: int
    num_heads: int
    is_causal: bool
    sliding_window: int | None

    def attend(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        *,
        scale: float,
        dropout_p: float,
        training: bool,
        return_lse: bool = False,
    ) -> AttentionResult:
        if any(t.ndim != 4 for t in (query, key, value)):
            raise ValueError("query/key/value must have rank four")
        if (
            query.shape[:3]
            != (self.batch_size, self.num_heads, self.layout.query_positions.shape[1])
            or key.shape[0] != self.batch_size
            or key.shape[2] != self.layout.key_positions.shape[1]
            or key.shape[-1] != query.shape[-1]
            or value.shape[:3] != key.shape[:3]
            or key.shape[1] == 0
            or self.num_heads % key.shape[1]
            or any(
                t.device != self.layout.query_positions.device
                or not t.is_floating_point()
                for t in (query, key, value)
            )
            or key.dtype != query.dtype
        ):
            raise ValueError(
                "prepared attention must match query/key/value dimensions, dtype, and device"
            )
        # Norm/RoPE may promote Q/K under autocast. All backends share this boundary.
        output, lse = self.backend.attend(
            query,
            key,
            value.to(query.dtype),
            mask=self.mask,
            scale=scale,
            dropout_p=dropout_p,
            training=training,
            return_lse=return_lse,
        )
        active = self.active_queries
        if lse is not None:
            # Flex returns LSE to identify empty rows without an O(Q*K) reduction.
            active = active & ~torch.isneginf(lse).all(dim=1)
        return AttentionResult(output, active, lse if return_lse else None)


class AttentionPreparationScope:
    """Use a fresh `with AttentionPreparationScope()` before each model forward.

    Layout tensors/callables are keyed by identity; equal node causal/window
    semantics share a prepared object. Closing the scope drops the cache, while
    returned objects remain alive for forward and checkpoint recomputation.
    """

    def __init__(self):
        self._cache = None

    def __enter__(self):
        if self._cache is not None:
            raise RuntimeError("attention preparation is already active")
        self._cache = []
        return self

    def __exit__(self, *exc):
        self._cache = None

    def prepare(
        self,
        backend: AttentionBackend,
        layout: AttentionLayout,
        *,
        batch_size: int,
        num_heads: int,
        is_causal: bool,
        sliding_window: int | None,
    ) -> PreparedAttention:
        if self._cache is None:
            raise RuntimeError(
                "prepare attention inside a per-forward preparation scope"
            )
        boolean("is_causal", is_causal)
        if sliding_window is not None:
            positive_int("sliding_window", sliding_window)
        layout.validate_dimensions(batch_size, num_heads)
        for entry in self._cache:
            if (
                entry.backend is backend
                and entry.layout.query_positions is layout.query_positions
                and entry.layout.key_positions is layout.key_positions
                and entry.layout.visibility is layout.visibility
                and entry.layout.active_queries is layout.active_queries
                and entry.batch_size == batch_size
                and entry.num_heads == num_heads
                and entry.is_causal == is_causal
                and entry.sliding_window == sliding_window
            ):
                return entry
        mask, active = backend.prepare(
            layout,
            batch_size=batch_size,
            num_heads=num_heads,
            is_causal=is_causal,
            sliding_window=sliding_window,
        )
        prepared = PreparedAttention(
            layout,
            backend,
            mask,
            active,
            batch_size,
            num_heads,
            is_causal,
            sliding_window,
        )
        self._cache.append(prepared)
        return prepared
