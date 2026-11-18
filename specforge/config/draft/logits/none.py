"""Target logits need no extra transform."""

from dataclasses import dataclass
from typing import ClassVar

from .base import TargetLogitSpec


@dataclass(frozen=True, kw_only=True)
class NoLogitTransformSpec(TargetLogitSpec):
    type: ClassVar[str] = "none"
