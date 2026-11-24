"""Model-wide residual streams; layers do not select separate layouts."""

from .base import ResidualSpec
from .standard import StandardResidualSpec

__all__ = ["ResidualSpec", "StandardResidualSpec"]
