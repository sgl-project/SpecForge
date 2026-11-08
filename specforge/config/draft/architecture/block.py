"""Token counts and prediction positions belong to the protocol, not callers."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from .._validation import positive_int


@dataclass(frozen=True, kw_only=True)
class BlockProtocol(ABC):
    block_size: int

    def __post_init__(self):
        positive_int("block_size", self.block_size)

    @property
    def num_model_slots(self) -> int:
        return self.block_size

    @property
    @abstractmethod
    def prediction_positions(self) -> tuple[int, ...]:
        """Model slots supervised as draft predictions."""

    @property
    @abstractmethod
    def num_draft_tokens(self) -> int:
        """Number of predictions, derived without materializing their positions."""

    @property
    @abstractmethod
    def num_external_candidates(self) -> int:
        """Candidates supplied by another predictor, excluding the anchor."""

    @property
    def verify_window(self) -> int:
        return 1 + self.num_draft_tokens + self.num_external_candidates


@dataclass(frozen=True, kw_only=True)
class AnchoredBlockSpec(BlockProtocol):
    """DFlash/DFlash2: [anchor, draft predictions...]."""

    def __post_init__(self):
        super().__post_init__()
        if self.block_size < 2:
            raise ValueError("anchored block_size must be at least 2")

    @property
    def num_draft_tokens(self) -> int:
        return self.block_size - 1

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        return tuple(range(1, self.block_size))

    @property
    def num_external_candidates(self) -> int:
        return 0


@dataclass(frozen=True, kw_only=True)
class DraftOnlyBlockSpec(BlockProtocol):
    """DSpark's draft-only layout: every model slot predicts a candidate."""

    @property
    def num_draft_tokens(self) -> int:
        return self.block_size

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        return tuple(range(self.block_size))

    @property
    def num_external_candidates(self) -> int:
        return 0


@dataclass(frozen=True, kw_only=True)
class ExternalFirstBlockSpec(BlockProtocol):
    """DMTP: [external T1, draft T2...]; verification also includes the anchor."""

    def __post_init__(self):
        super().__post_init__()
        if self.block_size < 2:
            raise ValueError("external-first block_size must be at least 2")

    @property
    def num_draft_tokens(self) -> int:
        return self.block_size - 1

    @property
    def prediction_positions(self) -> tuple[int, ...]:
        return tuple(range(1, self.block_size))

    @property
    def num_external_candidates(self) -> int:
        return 1
