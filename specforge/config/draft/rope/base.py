"""RoPE settings are structural; backends do not choose their defaults."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._validation import boolean, positive_int, positive_number


@dataclass(frozen=True, kw_only=True)
class RopeSpec(ABC):
    """Base for explicit rotary-position settings, including disabled RoPE."""

    @abstractmethod
    def rotary_dim(self, head_dim: int) -> int:
        """Validate and derive the number of dimensions to rotate."""


@dataclass(frozen=True, kw_only=True)
class RotarySpec(RopeSpec):
    rope_theta: float
    partial_rotary_factor: float
    interleaved: bool

    def __post_init__(self):
        boolean("interleaved", self.interleaved)
        positive_number("rope_theta", self.rope_theta)
        positive_number("partial_rotary_factor", self.partial_rotary_factor)
        if self.partial_rotary_factor > 1:
            raise ValueError("partial_rotary_factor must be in (0, 1]")

    def rotary_dim(self, head_dim: int) -> int:
        positive_int("head_dim", head_dim)
        dim = int(head_dim * self.partial_rotary_factor)
        if dim < 2 or dim % 2:
            raise ValueError(
                "partial_rotary_factor must produce a positive even rotary dimension"
            )
        return dim
