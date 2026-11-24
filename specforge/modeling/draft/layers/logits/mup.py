"""Fold muP division into the target head in training dtype."""

from torch import Tensor

from specforge.config.draft.logits import MuPLogitTransformSpec

from ..registry import register_component_builder
from .base import TargetLogitTransform


class MuPLogitTransform(TargetLogitTransform):
    def transform_weight_(self, weight: Tensor) -> None:
        # Dividing output logits instead changes low-precision rounding.
        weight.div_(self.config.multiplier)


register_component_builder("logits", MuPLogitTransformSpec, MuPLogitTransform)
