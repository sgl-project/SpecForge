"""Convolution inside a decoder layer, independent of an algorithm name."""

from .base import ConvSpec
from .grouped_causal import GroupedCausalConvSpec

__all__ = ["ConvSpec", "GroupedCausalConvSpec"]
