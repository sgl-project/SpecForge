"""Base schema for residual configuration nodes."""

from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class ResidualSpec:
    """Base for residual-stream schemas."""

    capabilities: ClassVar[frozenset[str]] = frozenset()
