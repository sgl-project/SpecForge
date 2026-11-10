"""Base schema for attention configuration nodes."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from ..rope import RopeSpec


@dataclass(frozen=True, kw_only=True)
class AttentionSpec(ABC):
    """Base for attention schemas with different projection/state semantics."""

    def capabilities(self) -> frozenset[str]:
        return frozenset()

    # None explicitly inherits model-level RoPE; NoRoPESpec disables rotation.
    rope_parameters: RopeSpec | None

    def __post_init__(self):
        if self.rope_parameters is not None and not isinstance(
            self.rope_parameters, RopeSpec
        ):
            raise ValueError("attention.rope_parameters must be a RopeSpec or None")

    def effective_rope(self, model_rope: RopeSpec) -> RopeSpec:
        if not isinstance(model_rope, RopeSpec):
            raise ValueError("model rope_parameters must be a RopeSpec")
        return model_rope if self.rope_parameters is None else self.rope_parameters

    @property
    @abstractmethod
    def rotary_input_dim(self) -> int | None:
        """Projection subspace that can use RoPE, or None when not applicable."""
