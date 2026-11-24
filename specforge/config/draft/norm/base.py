"""Base schema for norm configuration nodes."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class NormSpec:
    """Base for model-level normalization schemas."""
