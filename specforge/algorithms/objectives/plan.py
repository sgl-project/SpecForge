"""Declarative loss terms and optional, non-overlapping joint ranges."""

import math
from dataclasses import dataclass, replace

from torch import Tensor

from .losses import PositionWeighting, TokenLoss


@dataclass(frozen=True)
class LossTerm:
    """One mathematical loss with optional supervision/reduction overrides.

    Attributes:
        output: Named prediction output or value produced by the chunk computation.
        coefficient: Multiplier applied after the term's statistics are reduced.
        loss: Explicit loss, or None/default to inherit the runtime selection.
        mask: Optional [units, positions] supervision mask.
        weighting: Optional position-weighting rule.
        denominator: Optional fixed scalar denominator for the whole prediction
            set, not each chunk. It must not require gradients.
    """

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
        """Return a term with its mask indexed along the unit dimension.

        Args:
            units: A slice or tensor index selecting complete projection units.

        Returns:
            A new LossTerm; scalar coefficients and the fixed denominator are kept.
        """
        return replace(self, mask=None if self.mask is None else self.mask[units])


@dataclass(frozen=True)
class LossPlan:
    """Loss definitions and optional ranges permitted to execute jointly.

    Attributes:
        terms: Nonempty ordered tuple of loss terms.
        joint_groups: Non-overlapping tuples of at least two term indices.
            These permit joint execution but do not require a particular kernel
            or imply shared masks/denominators. Unlisted terms run independently.
    """

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
        """Return declared joint ranges followed by ungrouped singleton tuples."""
        grouped = {index for group in self.joint_groups for index in group}
        return self.joint_groups + tuple(
            (index,) for index in range(len(self.terms)) if index not in grouped
        )

    @property
    def shared_reduction(self):
        """Return whether every term inherits the prediction's reduction inputs."""
        return all(
            term.mask is None and term.weighting is None and term.denominator is None
            for term in self.terms
        )

    def select(self, units):
        """Return a plan whose per-term masks select the given complete units.

        Args:
            units: Slice or tensor index applied along the unit dimension.

        Returns:
            A new LossPlan preserving coefficients, denominators and joint ranges.
        """
        return replace(
            self,
            terms=tuple(term.select(units) for term in self.terms),
        )
