"""Configuration interface shared by architecture specs, with no model hooks."""

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass

from .block import BlockProtocol


@dataclass(frozen=True, kw_only=True)
class ArchitectureSpec(ABC):
    @property
    @abstractmethod
    def protocol(self) -> BlockProtocol | None:
        """Return the fixed block protocol, or None for a step-based architecture."""

    @abstractmethod
    def validate_target(self, target: Mapping) -> None:
        """Check this architecture's target-derived fields against the text config."""
