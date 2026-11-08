"""Base schema for mlp configuration nodes."""

from dataclasses import dataclass


@dataclass(frozen=True, kw_only=True)
class FFNSpec:
    """Base for dense and expert feed-forward schemas."""
