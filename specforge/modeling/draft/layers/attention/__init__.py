"""Attention structures and their independent compute backends."""

from .gqa import GQAAttention
from .mla import MLAAttention

__all__ = ["GQAAttention", "MLAAttention"]
