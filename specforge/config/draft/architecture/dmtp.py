"""DMTP's external first candidate, frozen predictor features and input fusion."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._validation import positive_int
from .base import ArchitectureSpec
from .block import ExternalFirstBlockSpec
from .features import TargetFeatureSpec


@dataclass(frozen=True, kw_only=True)
class MTPFeatureSpec:
    """Post-normalization MTP state at A, whose prediction supplies T1.

    Width is explicit and may differ from the draft's width. The feature
    provider supplies this state; the draft never executes the frozen MTP.
    """

    hidden_size: int

    def __post_init__(self):
        positive_int("mtp_features.hidden_size", self.hidden_size)

    @property
    def token_offset(self) -> int:
        """Feature position relative to the first model slot T1."""
        return -1

    def fusion_input_width(self, draft_hidden_size: int) -> int:
        positive_int("draft_hidden_size", draft_hidden_size)
        return draft_hidden_size + self.hidden_size


@dataclass(frozen=True, kw_only=True)
class DMTPSpec(ArchitectureSpec):
    block: ExternalFirstBlockSpec
    target_features: TargetFeatureSpec
    mtp_features: MTPFeatureSpec

    def __post_init__(self):
        if type(self.block) is not ExternalFirstBlockSpec:
            raise ValueError("block must be ExternalFirstBlockSpec")
        if type(self.target_features) is not TargetFeatureSpec:
            raise ValueError("target_features must be TargetFeatureSpec")
        if type(self.mtp_features) is not MTPFeatureSpec:
            raise ValueError("mtp_features must be MTPFeatureSpec")

    def validate_target(self, target: Mapping) -> None:
        """Check target context; external MTP width belongs to its provider."""
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> ExternalFirstBlockSpec:
        return self.block

    @property
    def context_end_offset(self) -> int:
        """Exclusive target-context end relative to T1: context stops before A."""
        return -1
