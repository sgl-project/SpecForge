"""Resolved target feature selection; no target-family configuration objects."""

from dataclasses import dataclass

from .._validation import positive_int
from .block import AnchoredBlockSpec


@dataclass(frozen=True, kw_only=True)
class FinalHiddenFeatureSpec:
    """Target final hidden states after its final normalization."""

    hidden_size: int

    def __post_init__(self):
        positive_int("target_features.hidden_size", self.hidden_size)


@dataclass(frozen=True, kw_only=True)
class TargetFeatureSpec:
    mask_token_id: int
    target_layer_ids: tuple[int, ...]

    def __post_init__(self):
        if type(self.mask_token_id) is not int or self.mask_token_id < 0:
            raise ValueError("mask_token_id must be a nonnegative integer")
        if type(self.target_layer_ids) is not tuple or not self.target_layer_ids:
            raise ValueError("target_layer_ids must be a nonempty tuple")
        if any(type(layer) is not int or layer < 0 for layer in self.target_layer_ids):
            raise ValueError("target_layer_ids must contain nonnegative integers")

    def input_width(self, target_hidden_size: int) -> int:
        positive_int("target_hidden_size", target_hidden_size)
        return len(self.target_layer_ids) * target_hidden_size


def validate_anchored_inputs(block, target_features) -> None:
    if type(block) is not AnchoredBlockSpec:
        raise ValueError("block must be an AnchoredBlockSpec")
    if type(target_features) is not TargetFeatureSpec:
        raise ValueError("target_features must be a TargetFeatureSpec")
