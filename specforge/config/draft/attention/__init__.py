"""Attention structure; runtime backends and model dimensions live elsewhere."""

from .base import AttentionSpec
from .gqa import GQASpec, MHASpec

__all__ = ["AttentionSpec", "GQASpec", "MHASpec"]
