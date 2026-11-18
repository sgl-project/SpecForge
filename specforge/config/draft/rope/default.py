"""Unscaled rotary frequencies, with explicit full or partial rotation."""

from dataclasses import dataclass
from typing import ClassVar

from .base import RotarySpec


@dataclass(frozen=True, kw_only=True)
class DefaultRoPESpec(RotarySpec):
    rope_type: ClassVar[str] = "default"
