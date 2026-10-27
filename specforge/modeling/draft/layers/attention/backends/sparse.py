"""Typed extension boundary for sparse execution, not an executable backend."""

from dataclasses import dataclass
from typing import Protocol, TypeVar

import torch

from ...config import AttentionConfig
from ..base import AttentionContext
from ..selection import SparseSelection

Inputs = TypeVar("Inputs", contravariant=True)


@dataclass(frozen=True)
class SparseAttentionResult:
    """Values [B, Q, H, Dv] before output projection; validity [B, Q, H].

    Empty heads must have zero values. The attention module must also zero
    queries with no valid heads after output projection, including its bias.
    """

    values: torch.Tensor
    valid_queries: torch.Tensor


class SparseAttentionBackend(Protocol[Inputs]):
    """A backend specializes Inputs to its own typed KV representation.

    Explicit GQA and latent/compressed KV need not share a tensor signature.
    Inputs must describe each source's storage, logical positions, validity and
    cache mapping. The adapter translates context visibility into those sources,
    and rejects unsupported masks, source combinations, layouts or backward.
    It does not own projections, indexer parameters, cache mutation or losses.
    No dense expansion/fallback is implied by implementing this protocol.
    """

    def __call__(
        self,
        inputs: Inputs,
        selection: SparseSelection,
        context: AttentionContext,
        options: AttentionConfig,
    ) -> SparseAttentionResult: ...
