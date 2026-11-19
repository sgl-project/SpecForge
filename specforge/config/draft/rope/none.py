"""An explicit request to disable rotary embedding."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_int
from .base import RopeSpec


@dataclass(frozen=True, kw_only=True)
class NoRoPESpec(RopeSpec):
    rope_type: ClassVar[str] = "none"

    def rotary_dim(self, head_dim: int) -> int:
        positive_int("head_dim", head_dim)
        return 0
