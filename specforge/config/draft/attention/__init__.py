"""Attention structure; runtime backends and model dimensions live elsewhere."""

from .base import AttentionSpec
from .gqa import GQASpec, MHASpec
from .kda import KDASpec
from .mla import MLASpec

__all__ = ["AttentionSpec", "GQASpec", "KDASpec", "MHASpec", "MLASpec"]
