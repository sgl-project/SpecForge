"""DSpine's inter-layer injection structure, independent of training policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._validation import positive_int
from .base import ArchitectureSpec
from .block import AnchoredBlockSpec
from .constraints import validate_target_context
from .features import TargetFeatureSpec, validate_anchored_inputs

if TYPE_CHECKING:
    from ..model import DraftConfig


@dataclass(frozen=True, kw_only=True)
class AdjacentInjectionSpec:
    transfer_rank: int
    message_size: int

    def __post_init__(self):
        positive_int("injection.transfer_rank", self.transfer_rank)
        positive_int("injection.message_size", self.message_size)


@dataclass(frozen=True, kw_only=True)
class DSpineSpec(ArchitectureSpec):
    block: AnchoredBlockSpec
    target_features: TargetFeatureSpec
    injection: AdjacentInjectionSpec

    def __post_init__(self):
        validate_anchored_inputs(self.block, self.target_features)
        if type(self.injection) is not AdjacentInjectionSpec:
            raise ValueError("injection must be an AdjacentInjectionSpec")

    @property
    def protocol(self) -> AnchoredBlockSpec:
        return self.block


def validate_dspine(config: DraftConfig) -> None:
    validate_target_context(config)
    rank = config.spec_config.injection.transfer_rank
    if rank > min(config.hidden_size, config.vocab_size - 1):
        raise ValueError(
            "injection.transfer_rank must not exceed min(hidden_size, vocab_size - 1)"
        )
    if "single_stream" not in config.residual_config.capabilities:
        raise ValueError("DSpine adjacent injection requires single_stream residuals")
    for name, attention in config.attention_configs.items():
        # A component without an explicit causal contract cannot feed injection.
        if getattr(attention, "is_causal", None) is not True:
            raise ValueError(
                f"attention_configs.{name}: DSpine requires causal attention"
            )
