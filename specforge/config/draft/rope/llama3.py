"""Llama3 wavelength-based rotary scaling with explicit frequency boundaries."""

from dataclasses import dataclass
from typing import ClassVar

from .._validation import positive_int, positive_number
from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class Llama3RoPESpec(RotarySpec):
    rope_type: ClassVar[str] = "llama3"
    factor: float
    low_freq_factor: float
    high_freq_factor: float
    original_max_position_embeddings: int

    def __post_init__(self):
        super().__post_init__()
        positive_number("factor", self.factor)
        positive_number("low_freq_factor", self.low_freq_factor)
        positive_number("high_freq_factor", self.high_freq_factor)
        positive_int(
            "original_max_position_embeddings", self.original_max_position_embeddings
        )
        # Equal boundaries are the step transition supported by the existing
        # EAGLE rotary implementation; an inverted transition is not supported.
        if self.high_freq_factor < self.low_freq_factor:
            raise ValueError("high_freq_factor must be at least low_freq_factor")
