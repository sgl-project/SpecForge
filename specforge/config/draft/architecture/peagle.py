"""P-EAGLE composes auxiliary fusion, a learned feature mask, and a layer stack."""

from dataclasses import dataclass

from .._validation import boolean
from .base import ArchitectureSpec
from .eagle_fusion import AuxiliaryFeatureSpec, attention_input_width


@dataclass(frozen=True, kw_only=True)
class PEagleSpec(ArchitectureSpec):
    target_features: AuxiliaryFeatureSpec
    norm_before_residual: bool

    def __post_init__(self):
        if type(self.target_features) is not AuxiliaryFeatureSpec:
            raise ValueError("target_features must be an AuxiliaryFeatureSpec")
        boolean("norm_before_residual", self.norm_before_residual)

    @property
    def protocol(self) -> None:
        # Parallel rollout length and its masks belong to runtime preparation.
        return None

    @property
    def mask_hidden_shape(self) -> tuple[int, int, int]:
        return (1, 1, self.target_features.input_width)

    def attention_input_width(self, hidden_size: int, layer_index: int) -> int:
        return attention_input_width(hidden_size, layer_index)
