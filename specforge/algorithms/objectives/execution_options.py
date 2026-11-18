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
        fused: None selects a default-enabled optimization wherever one applies
            and the reference formula elsewhere, including off CUDA/ROCm and
            without the optional kernel packages. True requires optimized
            execution: every registered implementation is considered and a
            categorical loss that cannot be optimized raises. False always
            evaluates the reference.
        trim: Skip units without supervision when the loss contract permits it.
        teacher_chunk_size: Optional positive vocabulary tile size for compact
            offline teacher projection; None uses ordinary teacher projection.
    """
    chunk_size: int | None = None
    checkpoint: bool = True
    fused: bool | None = None
    trim: bool = False
    teacher_chunk_size: int | None = None

    def __post_init__(self):
        check_options(
            zero_counts={"chunk_size": self.chunk_size},
            counts={"teacher_chunk_size": self.teacher_chunk_size},
        )
        for name in ("checkpoint", "trim"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.fused is not None and type(self.fused) is not bool:
            raise ValueError("fused must be boolean or None")
