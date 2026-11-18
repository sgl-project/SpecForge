"""MTP input fusion followed by an ordinary causal decoder stack."""

from collections.abc import Mapping
from dataclasses import dataclass

from .base import ArchitectureSpec
from .features import FinalHiddenFeatureSpec


@dataclass(frozen=True, kw_only=True)
class MTPSpec(ArchitectureSpec):
    """Normalize embedding and target feature separately, concatenate, project.

    The embedding at t and the target's final feature at t-1 predict token t+1.
    All fusion norms use the model's norm_config. Layer components and depth
    are declared independently; native origin does not select an implementation.
    """

    target_features: FinalHiddenFeatureSpec

    def __post_init__(self):
        if type(self.target_features) is not FinalHiddenFeatureSpec:
            raise ValueError("target_features must be a FinalHiddenFeatureSpec")

    def validate_target(self, target: Mapping) -> None:
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> None:
        return None

    @property
    def feature_token_offset(self) -> int:
        return -1

    @property
    def prediction_token_offset(self) -> int:
        return 1


def validate_mtp(config) -> None:
    if "single_stream" not in config.residual_config.capabilities():
        raise ValueError("MTP fusion requires single_stream residuals")
    if config.draft_vocab_size != config.vocab_size:
        raise ValueError("MTP requires draft_vocab_size == vocab_size")
    for name, attention in config.attention_configs.items():
        if "causal" not in attention.capabilities():
            raise ValueError(f"attention_configs.{name}: MTP requires causal attention")
