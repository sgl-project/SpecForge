"""Loss execution policy, independent of supervision and model structure."""

from dataclasses import dataclass

from .base import check_options


@dataclass(frozen=True)
class LossExecution:
    # None preserves the protocol's chunk size; zero explicitly disables chunking.
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
