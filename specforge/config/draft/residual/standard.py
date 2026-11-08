"""Model-wide residual streams; layers do not select separate layouts."""

from dataclasses import dataclass
from typing import ClassVar

from .base import ResidualSpec


@dataclass(frozen=True, kw_only=True)
class StandardResidualSpec(ResidualSpec):
    type: ClassVar[str] = "standard"
    capabilities: ClassVar[frozenset[str]] = frozenset({"single_stream"})
