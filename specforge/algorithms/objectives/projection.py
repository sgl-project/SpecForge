"""Projection adapters expose computation, never loss or training policy."""

from collections.abc import Callable
from dataclasses import dataclass, field

from torch import nn


@dataclass(frozen=True)
class Projection:
    project: Callable
    chunk_safe: bool = field(default=True, kw_only=True)
    recompute_safe: bool = field(default=True, kw_only=True)


@dataclass(frozen=True)
class LinearProjection(Projection):
    """Explicit opt-in for a plain, untransformed linear output."""

    head: nn.Module

    def weight_and_bias(self):
        from specforge.modeling.draft.bindings import BoundLMHead

        if type(self.head) is nn.Linear:
            return self.head.weight, self.head.bias
        if type(self.head) is BoundLMHead:
            return self.head.bound_weight(), None
        raise TypeError("Linear projection requires a plain linear or bound head")


@dataclass(frozen=True)
class CorrectedProjection(Projection):
    """Base logits plus a compact suffix correction over complete blocks."""

    project_parts: Callable
    block_size: int
    suffix_start: int
