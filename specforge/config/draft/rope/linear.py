"""Linear position scaling."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_number
from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class LinearRoPESpec(RotarySpec):
    rope_type: ClassVar[str] = "linear"
    factor: float

    def __post_init__(self):
        super().__post_init__()
        positive_number("factor", self.factor)
