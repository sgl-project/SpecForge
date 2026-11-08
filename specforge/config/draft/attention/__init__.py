"""Attention structure; runtime backends and model dimensions live elsewhere."""

from .base import AttentionSpec
from .gqa import GQASpec, MHASpec
from .mla import MLASpec

__all__ = ["AttentionSpec", "GQASpec", "MHASpec", "MLASpec"]
