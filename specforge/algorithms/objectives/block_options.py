"""Memory policy and token-loss options shared by block objectives."""

from dataclasses import dataclass

from .base import check_options


@dataclass
class BlockLossOptions:
    num_anchors: int = 512
    loss_decay_gamma: float | None = None
    objective_chunk_blocks: int = 128
    objective_checkpointing: bool = True
    fused_logits_loss: bool = True

    def __post_init__(self):
        check_options(
            counts={"num_anchors": self.num_anchors},
            positive={"loss_decay_gamma": self.loss_decay_gamma},
            zero_counts={"objective_chunk_blocks": self.objective_chunk_blocks},
        )
        for name in ("objective_checkpointing", "fused_logits_loss"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")


@dataclass
class DFlashLossOptions(BlockLossOptions):
    loss_type: str = "dflash"
    dpace_alpha: float = 0.5
    lk_loss_type: str | None = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    metric_top_k: int = 16
    teacher_metrics: bool = True

    def __post_init__(self):
        from specforge.core.position_weights import DPACE_LOSS_TYPES

        super().__post_init__()
        if self.loss_type not in {"dflash", *DPACE_LOSS_TYPES}:
            raise ValueError(f"unknown loss_type: {self.loss_type!r}")
        if self.lk_loss_type not in (None, "alpha", "lambda", "tv"):
            raise ValueError(f"unknown lk_loss_type: {self.lk_loss_type!r}")
        check_options(
            probabilities={"dpace_alpha": self.dpace_alpha},
            counts={"metric_top_k": self.metric_top_k},
        )
        if type(self.teacher_metrics) is not bool:
            raise ValueError("teacher_metrics must be boolean")
