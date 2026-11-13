"""Loss execution policy, independent of supervision and model structure."""

from dataclasses import dataclass

from .base import check_options


@dataclass(frozen=True)
class LossExecution:
    # None preserves the protocol's chunk size; zero explicitly disables chunking.
    """Execution policy independent of mathematical loss and algorithm structure.

    Attributes:
        chunk_size: Units per chunk. None inherits the protocol default;
            zero disables chunking.
        checkpoint: Recompute eligible chunk projections/losses during backward.
        fused: Allow applicable registered optimized implementations.
        trim: Skip units without supervision when the loss contract permits it.
        teacher_chunk_size: Optional positive vocabulary tile size for compact
            offline teacher projection; None uses ordinary teacher projection.
    """
    chunk_size: int | None = None
    checkpoint: bool = True
    fused: bool = False
    trim: bool = False
    teacher_chunk_size: int | None = None

    def __post_init__(self):
        check_options(
            zero_counts={"chunk_size": self.chunk_size},
            counts={"teacher_chunk_size": self.teacher_chunk_size},
        )
        for name in ("checkpoint", "fused", "trim"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
