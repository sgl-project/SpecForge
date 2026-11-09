"""DSpark composes its draft-only protocol and optional prediction heads."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._validation import boolean, positive_int
from .base import ArchitectureSpec
from .block import DraftOnlyBlockSpec
from .features import TargetFeatureSpec


@dataclass(frozen=True, kw_only=True)
class MarkovHeadSpec:
    type: str
    rank: int

    def __post_init__(self):
        if type(self.type) is not str or self.type not in {"vanilla", "gated", "rnn"}:
            raise ValueError("markov.type must be vanilla, gated, or rnn")
        positive_int("markov.rank", self.rank)


@dataclass(frozen=True, kw_only=True)
class ConfidenceHeadSpec:
    with_markov: bool

    def __post_init__(self):
        boolean("confidence.with_markov", self.with_markov)

    def input_width(self, hidden_size: int, markov: MarkovHeadSpec | None) -> int:
        positive_int("hidden_size", hidden_size)
        if self.with_markov:
            if type(markov) is not MarkovHeadSpec:
                raise ValueError("confidence.with_markov requires a Markov head")
            return hidden_size + markov.rank
        return hidden_size


@dataclass(frozen=True, kw_only=True)
class DSparkSpec(ArchitectureSpec):
    block: DraftOnlyBlockSpec
    target_features: TargetFeatureSpec
    markov: MarkovHeadSpec | None
    confidence: ConfidenceHeadSpec | None

    def __post_init__(self):
        if type(self.block) is not DraftOnlyBlockSpec:
            raise ValueError("block must be a DraftOnlyBlockSpec")
        if type(self.target_features) is not TargetFeatureSpec:
            raise ValueError("target_features must be a TargetFeatureSpec")
        if self.markov is not None and type(self.markov) is not MarkovHeadSpec:
            raise ValueError("markov must be a MarkovHeadSpec or None")
        if self.confidence is not None:
            if type(self.confidence) is not ConfidenceHeadSpec:
                raise ValueError("confidence must be a ConfidenceHeadSpec or None")
            if self.confidence.with_markov and self.markov is None:
                raise ValueError("confidence.with_markov requires a Markov head")

    def validate_target(self, target: Mapping) -> None:
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> DraftOnlyBlockSpec:
        return self.block
