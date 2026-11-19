"""Resolved YaRN settings, including MLA's full-logit scaling."""

import math
from dataclasses import dataclass
from typing import ClassVar

from .._validation import boolean, finite_number, positive_int, positive_number
from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class YaRNSpec(RotarySpec):
    rope_type: ClassVar[str] = "yarn"
    factor: float
    original_max_position_embeddings: int
    attention_factor: float
    beta_fast: float
    beta_slow: float
    truncate: bool
    mscale_all_dim: float

    def __post_init__(self):
        super().__post_init__()
        if self.rope_theta <= 1:
            raise ValueError("YaRN rope_theta must exceed 1")
        for name in ("factor", "attention_factor", "beta_fast", "beta_slow"):
            positive_number(name, getattr(self, name))
        positive_int(
            "original_max_position_embeddings", self.original_max_position_embeddings
        )
        boolean("truncate", self.truncate)
        finite_number("mscale_all_dim", self.mscale_all_dim)
        if self.mscale_all_dim < 0:
            raise ValueError("mscale_all_dim must be nonnegative")

    @property
    def mla_qk_scale_multiplier(self) -> float:
        """Multiplier for MLA's full QK score, separate from rotary cos/sin."""
        if self.factor <= 1:
            return 1.0
        return (1.0 + 0.1 * self.mscale_all_dim * math.log(self.factor)) ** 2
