"""muP divides loaded head weights in training dtype, not output logits."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_number
from .base import TargetLogitSpec


@dataclass(frozen=True, kw_only=True)
class MuPLogitTransformSpec(TargetLogitSpec):
    type: ClassVar[str] = "mup"
    multiplier: float

    def __post_init__(self):
        positive_number("multiplier", self.multiplier)
