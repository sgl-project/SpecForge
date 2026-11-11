"""Token losses and position weights, independent of algorithm execution."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from specforge.core import fused_loss

from .base import check_options

_DISTILLATION = {
    "kl": fused_loss.kl_div_loss,
    "reverse_kl": fused_loss.reverse_kl_div_loss,
    "js": fused_loss.js_div_loss,
    "tv": fused_loss.tv_loss,
    "nla": fused_loss.neg_log_acceptance_loss,
    "lk": fused_loss.lk_hybrid_loss,
}


@dataclass(frozen=True)
class TokenLoss:
    """Override the base loss without replacing an algorithm's auxiliary losses."""

    kind: str = "default"

    def __post_init__(self):
        if self.kind not in {"default", "ce", "soft_ce", *_DISTILLATION}:
            raise ValueError(f"unknown token loss: {self.kind!r}")

    @property
    def requires_teacher(self):
        return self.kind not in ("default", "ce")

    def __call__(self, logits, labels, *, teacher=None, default=None):
        if self.kind == "default" and default is not None:
            return default
        if self.kind in ("default", "ce"):
            return F.cross_entropy(
                logits.to(torch.promote_types(logits.dtype, torch.float32)).reshape(
                    -1, logits.shape[-1]
                ),
                labels.reshape(-1),
                reduction="none",
            ).reshape(labels.shape)
        if teacher is None or teacher.shape != logits.shape:
            raise ValueError("distillation requires aligned teacher and draft logits")
        if self.kind == "soft_ce":
            return -(teacher.float().softmax(-1) * logits.float().log_softmax(-1)).sum(
                -1
            )
        return _DISTILLATION[self.kind](logits, teacher)


@dataclass(frozen=True)
class PositionWeighting:
    kind: str = "uniform"
    gamma: float | None = None

    def __post_init__(self):
        if self.kind not in ("uniform", "exponential"):
            raise ValueError(f"unknown position weighting: {self.kind!r}")
        check_options(positive={"gamma": self.gamma})
        if (self.kind == "exponential") != (self.gamma is not None):
            raise ValueError("gamma is required only for exponential weighting")

    def __call__(self, weights, positions):
        if self.kind == "uniform":
            return weights
        return weights * torch.exp(-positions.float().clamp_min(0) / self.gamma)
