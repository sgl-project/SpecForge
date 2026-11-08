"""Normalization semantics, including the stored weight parameterization."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import finite_number
from .base import NormSpec


@dataclass(frozen=True, kw_only=True)
class RMSNormSpec(NormSpec):
    type: ClassVar[str] = "rms"

    eps: float
    weight_offset: int

    def __post_init__(self):
        finite_number("eps", self.eps)
        if self.eps <= 0:
            raise ValueError("eps must be positive")
        if type(self.weight_offset) is not int or self.weight_offset not in (0, 1):
            raise ValueError("weight_offset must be 0 (w) or 1 (1 + w)")
