"""Normalization semantics, including the stored weight parameterization."""

from .base import NormSpec
from .rms import RMSNormSpec

__all__ = ["NormSpec", "RMSNormSpec"]
