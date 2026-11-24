"""Use the target head weight without a logit transform."""

from torch import Tensor

from specforge.config.draft.logits import NoLogitTransformSpec

from ..registry import register_component_builder
from .base import TargetLogitTransform


class NoLogitTransform(TargetLogitTransform):
    def transform_weight_(self, weight: Tensor) -> None:
        pass


register_component_builder("logits", NoLogitTransformSpec, NoLogitTransform)
