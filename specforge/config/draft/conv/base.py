"""Base schema for conv configuration nodes."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ConvSpec:
    """Base for sublayer convolution schemas."""

    def capabilities(self) -> frozenset[str]:
        return frozenset()
