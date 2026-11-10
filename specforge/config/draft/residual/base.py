"""Base schema for residual configuration nodes."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class ResidualSpec:
    """Base for residual-stream schemas."""

    def capabilities(self) -> frozenset[str]:
        return frozenset()
