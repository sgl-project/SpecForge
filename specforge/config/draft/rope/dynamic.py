"""Dynamic NTK scaling with an explicit original context length."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_int, positive_number
from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class DynamicNTKRoPESpec(RotarySpec):
    rope_type: ClassVar[str] = "dynamic"
    factor: float
    original_max_position_embeddings: int

    def __post_init__(self):
        super().__post_init__()
        positive_number("factor", self.factor)
        positive_int(
            "original_max_position_embeddings", self.original_max_position_embeddings
        )

    def rotary_dim(self, head_dim: int) -> int:
        dim = super().rotary_dim(head_dim)
        if dim <= 2:
            raise ValueError("dynamic NTK requires a rotary dimension greater than two")
        return dim

    def scaled_theta(self, head_dim: int) -> float:
        """SGLang fixes the effective length at factor * original_max_positions."""
        dim = self.rotary_dim(head_dim)
        length = self.factor * self.original_max_position_embeddings
        stretch = self.factor * length / self.original_max_position_embeddings - (
            self.factor - 1
        )
        return self.rope_theta * stretch ** (dim / (dim - 2))
