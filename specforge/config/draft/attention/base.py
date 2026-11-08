"""Base schema for attention configuration nodes."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class AttentionSpec:
    """Base for attention schemas with different projection/state semantics."""
