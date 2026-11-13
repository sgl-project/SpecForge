"""Declarative loss terms and optional, non-overlapping joint ranges."""

import math
from dataclasses import dataclass, replace

from torch import Tensor

from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class LossTerm:
    output: str = "logits"
    coefficient: float = 1.0
    loss: TokenLoss | None = None
    mask: Tensor | None = None
    weighting: PositionWeighting | None = None
    denominator: Tensor | None = None

    def __post_init__(self):
        if not self.output or not math.isfinite(self.coefficient):
            raise ValueError("Loss term requires an output and a finite coefficient")
        if self.denominator is not None:
            if self.denominator.ndim != 0:
                raise ValueError("Loss denominator must be scalar")
            if self.denominator.requires_grad:
                raise ValueError(
                    "Differentiable denominators require a separate reduction implementation"
                )

    def select(self, units):
        return replace(self, mask=None if self.mask is None else self.mask[units])


@dataclass(frozen=True)
class LossPlan:
    terms: tuple[LossTerm, ...] = (LossTerm(),)
    joint_groups: tuple[tuple[int, ...], ...] = ()

    def __post_init__(self):
        if not self.terms:
            raise ValueError("Loss plan must contain at least one term")
        seen = set()
        for group in self.joint_groups:
            if len(group) < 2:
                raise ValueError("Joint ranges must contain at least two terms")
            for index in group:
                if type(index) is not int or not 0 <= index < len(self.terms):
                    raise ValueError("Joint range references an unknown term")
                if index in seen:
                    raise ValueError("Joint ranges must not overlap")
                seen.add(index)

    @property
    def groups(self):
        grouped = {index for group in self.joint_groups for index in group}
        return self.joint_groups + tuple(
            (index,) for index in range(len(self.terms)) if index not in grouped
        )

    @property
    def shared_reduction(self):
        return all(
            term.mask is None and term.weighting is None and term.denominator is None
            for term in self.terms
        )

    def select(self, units):
        return replace(self, terms=tuple(term.select(units) for term in self.terms))
