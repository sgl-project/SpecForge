"""Domino's label alignment and GRU correction, independent of its objective."""

from collections.abc import Mapping
from dataclasses import dataclass

from .._validation import boolean, positive_int
from .base import ArchitectureSpec
from .block import BlockProtocol
from .features import TargetFeatureSpec


@dataclass(frozen=True, kw_only=True)
class DominoBlockSpec(BlockProtocol):
    shift_label: bool

    def __post_init__(self):
        super().__post_init__()
        if self.block_size < 2:
            raise ValueError("Domino block_size must be at least 2")
        boolean("shift_label", self.shift_label)

    @property
    def label_offset(self) -> int:
        return int(self.shift_label)

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        # Shifted training also supervises the last slot, whose next-token
        # prediction is not used by the existing block sampler.
        return tuple(range(1 - self.label_offset, self.block_size))

    @property
    def candidate_positions(self) -> tuple[int, ...]:
        """Output slots used to propose the block's draft tokens at serving time."""
        start = 1 - self.label_offset
        return tuple(range(start, start + self.num_draft_tokens))

    @property
    def num_draft_tokens(self) -> int:
        return self.block_size - 1

    @property
    def num_external_candidates(self) -> int:
        return 0


@dataclass(frozen=True, kw_only=True)
class GRUHeadSpec:
    hidden_size: int
    projection_size: int

    def __post_init__(self):
        positive_int("gru.hidden_size", self.hidden_size)
        positive_int("gru.projection_size", self.projection_size)

    def input_width(self, draft_hidden_size: int) -> int:
        """Width of concatenated decoder and GRU states entering the projection."""
        positive_int("draft_hidden_size", draft_hidden_size)
        return draft_hidden_size + self.hidden_size


@dataclass(frozen=True, kw_only=True)
class DominoSpec(ArchitectureSpec):
    block: DominoBlockSpec
    target_features: TargetFeatureSpec
    gru: GRUHeadSpec
    pure_draft_prefix_len: int

    def __post_init__(self):
        if type(self.block) is not DominoBlockSpec:
            raise ValueError("block must be a DominoBlockSpec")
        if type(self.target_features) is not TargetFeatureSpec:
            raise ValueError("target_features must be a TargetFeatureSpec")
        if type(self.gru) is not GRUHeadSpec:
            raise ValueError("gru must be a GRUHeadSpec")
        if (
            type(self.pure_draft_prefix_len) is not int
            or self.pure_draft_prefix_len < 0
        ):
            raise ValueError("pure_draft_prefix_len must be a nonnegative integer")
        if self.suffix_start >= self.block.block_size:
            raise ValueError("pure_draft_prefix_len must leave a nonempty GRU suffix")

    @property
    def suffix_start(self) -> int:
        return self.pure_draft_prefix_len + 1 - self.block.label_offset

    def validate_target(self, target: Mapping) -> None:
        self.target_features.validate_target(target)

    @property
    def protocol(self) -> DominoBlockSpec:
        return self.block
