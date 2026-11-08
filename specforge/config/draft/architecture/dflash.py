"""DFlash composes the anchored protocol and target-feature selection."""

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

    @property
    def protocol(self) -> AnchoredBlockSpec:
        return self.block
