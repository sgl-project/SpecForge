"""Three-axis rotary sections, measured in inverse-frequency pairs."""

from dataclasses import dataclass
from typing import ClassVar

from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class MultiAxisRoPESpec(RotarySpec):
    rope_type: ClassVar[str] = "mrope"
    mrope_section: tuple[int, ...]

    def __post_init__(self):
        super().__post_init__()
        if type(self.mrope_section) is not tuple or len(self.mrope_section) != 3:
            raise ValueError("mrope_section must contain exactly three sections")
        if any(type(size) is not int or size < 0 for size in self.mrope_section):
            raise ValueError("mrope_section must contain nonnegative integers")
        if sum(self.mrope_section) == 0:
            raise ValueError("mrope_section must contain at least one rotary pair")

    def rotary_dim(self, head_dim: int) -> int:
        dim = super().rotary_dim(head_dim)
        if 2 * sum(self.mrope_section) != dim:
            raise ValueError(
                "twice the sum of mrope_section must equal the rotary dimension"
            )
        return dim
