"""Base schema for attention configuration nodes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(ABC):
    """Base for attention schemas with different projection/state semantics."""

    capabilities: ClassVar[frozenset[str]] = frozenset()

    @property
    @abstractmethod
    def rotary_input_dim(self) -> int | None:
        """Projection subspace that can use RoPE, or None when not applicable."""
