"""DFlash composes the anchored protocol and target-feature selection."""

from collections.abc import Mapping
from dataclasses import dataclass

from .base import ArchitectureSpec
from .block import AnchoredBlockSpec
from .features import TargetFeatureSpec, validate_anchored_inputs


@dataclass(frozen=True, kw_only=True)
class DFlashSpec(ArchitectureSpec):
    block: AnchoredBlockSpec
    target_features: TargetFeatureSpec

    def __post_init__(self):
        validate_anchored_inputs(self.block, self.target_features)

    def validate_target(self, target: Mapping) -> None:
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> AnchoredBlockSpec:
        return self.block
