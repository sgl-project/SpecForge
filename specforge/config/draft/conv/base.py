"""Base schema for conv configuration nodes."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class ConvSpec:
    """Base for sublayer convolution schemas."""

    capabilities: ClassVar[frozenset[str]] = frozenset()
