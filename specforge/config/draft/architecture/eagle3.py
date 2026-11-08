"""EAGLE3's auxiliary fusion and output normalization contract."""

from dataclasses import dataclass

from .._validation import boolean
from .base import ArchitectureSpec
from .eagle_fusion import (
    AuxiliaryFeatureSpec,
    attention_input_width,
    validate_auxiliary_fusion,
)


@dataclass(frozen=True, kw_only=True)
class Eagle3Spec(ArchitectureSpec):
    target_features: AuxiliaryFeatureSpec
    normalize_features: bool
    norm_output: bool

    def __post_init__(self):
        if type(self.target_features) is not AuxiliaryFeatureSpec:
            raise ValueError("target_features must be an AuxiliaryFeatureSpec")
        boolean("normalize_features", self.normalize_features)
        boolean("norm_output", self.norm_output)

    @property
    def protocol(self) -> None:
        # Expansion depth is a runtime setting; vocabulary mappings are tensors
        # in the checkpoint, not a second copy of the vocabulary in this node.
        return None

    def attention_input_width(self, hidden_size: int, layer_index: int) -> int:
        width = attention_input_width(hidden_size, layer_index)
        if layer_index != 0:
            raise ValueError("EAGLE3 has exactly one decoder layer")
        return width


def validate_eagle3(config) -> None:
    if config.num_hidden_layers != 1:
        raise ValueError("EAGLE3 requires exactly one decoder layer")
    validate_auxiliary_fusion(config)
