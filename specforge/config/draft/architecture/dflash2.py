"""DFlash2 composes shared inputs with its own selector and logit settings."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._validation import positive_int, positive_number
from .base import ArchitectureSpec
from .block import AnchoredBlockSpec
from .features import TargetFeatureSpec, validate_anchored_inputs


@dataclass(frozen=True, kw_only=True)
class DFlash2Spec(ArchitectureSpec):
    block: AnchoredBlockSpec
    target_features: TargetFeatureSpec
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: float | None

    def __post_init__(self):
        validate_anchored_inputs(self.block, self.target_features)
        positive_int("selector_rank", self.selector_rank)
        positive_int("selector_top_k", self.selector_top_k)
        positive_number("output_multiplier", self.output_multiplier)
        if self.final_logit_softcapping is not None:
            positive_number("final_logit_softcapping", self.final_logit_softcapping)

    def validate_target(self, target: Mapping) -> None:
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> AnchoredBlockSpec:
        return self.block
