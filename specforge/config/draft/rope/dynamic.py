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

    def scaled_theta(self, head_dim: int, seq_len: int) -> float:
        dim = self.rotary_dim(head_dim)
        positive_int("seq_len", seq_len)
        if seq_len <= self.original_max_position_embeddings:
            return self.rope_theta
        stretch = self.factor * seq_len / self.original_max_position_embeddings - (
            self.factor - 1
        )
        return self.rope_theta * stretch ** (dim / (dim - 2))
