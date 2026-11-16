"""Weight and compose per-token losses without choosing their implementation.

Callers obtain CE/L1 from ``core.fused_loss`` and pass explicit weights for
each term. Position weights, model-specific normalization, distributed
reductions, projection and checkpointing stay with the caller. Selector and
confidence objectives are composed separately by the model wrappers.
"""

from dataclasses import dataclass
from typing import NamedTuple, Optional

import torch


@dataclass(frozen=True)
class TokenObjectiveConfig:
    """Internal mapping of existing model loss settings, not execution policy.

    The hard-label term is CE (None/alpha), one-hot TV (tv), or an adaptive
    CE/TV mix (lambda). Teacher L1 is a separate distribution-matching term.
    """

    hard_ce_weight: float = 1.0
    teacher_l1_weight: float = 0.0
    hard_label_mode: Optional[str] = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0

    def __post_init__(self) -> None:
        if self.hard_ce_weight < 0 or self.teacher_l1_weight < 0:
            raise ValueError("token objective coefficients must be nonnegative")
        if self.hard_label_mode not in {None, "alpha", "tv", "lambda"}:
            raise ValueError(f"unknown hard-label mode {self.hard_label_mode!r}")

    @property
    def requires_teacher_distribution(self) -> bool:
        """Objective requirement; confidence/diagnostics may also need teacher data."""
        return self.teacher_l1_weight > 0


class TokenObjectiveChunkTerms(NamedTuple):
    """Additive numerators; the caller owns each term's normalization."""

    ce_num: torch.Tensor
    hard_tv_num: torch.Tensor
    teacher_l1_num: torch.Tensor


def token_objective_chunk_terms(
    config: TokenObjectiveConfig,
    *,
    cross_entropy: torch.Tensor,
    hard_weights: torch.Tensor,
    teacher_l1: Optional[torch.Tensor] = None,
    teacher_weights: Optional[torch.Tensor] = None,
) -> TokenObjectiveChunkTerms:
    """Weight aligned per-token CE/L1 tensors, shaped ``[..., positions]``.

    Weights include the caller's masks and any position/anchor scaling.
    Teacher weights are explicit: they need not match hard-label weights.
    No normalization, backend dispatch or logits computation occurs here.
    """
    ce_num = (cross_entropy * hard_weights).sum()
    hard_tv_num = ce_num.new_zeros(())
    if config.hard_label_mode in {"tv", "lambda"}:
        hard_tv_num = ((1.0 - cross_entropy.neg().exp()) * hard_weights).sum()
    teacher_l1_num = ce_num.new_zeros(())
    if config.requires_teacher_distribution:
        if teacher_l1 is None or teacher_weights is None:
            raise ValueError(
                "teacher L1 objective requires teacher_l1 and teacher_weights"
            )
        teacher_l1_num = (teacher_l1 * teacher_weights).sum()
    return TokenObjectiveChunkTerms(
        ce_num=ce_num,
        hard_tv_num=hard_tv_num,
        teacher_l1_num=teacher_l1_num,
    )


def lk_kl_weight(
    config: TokenObjectiveConfig,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Detached LK-lambda CE coefficient after full-objective reduction."""
    if config.hard_label_mode != "lambda":
        return None
    acceptance = probability_num / probability_den.clamp_min(1.0)
    return config.kl_scale * torch.exp(-config.kl_decay * acceptance.detach())


def compose_token_objective(
    config: TokenObjectiveConfig,
    *,
    ce_num: torch.Tensor,
    hard_tv_num: Optional[torch.Tensor] = None,
    teacher_l1_num: Optional[torch.Tensor] = None,
    probability_num: Optional[torch.Tensor] = None,
    probability_den: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Combine reduced terms; apply method-specific denominators at the caller.

    Terms may share a denominator applied afterwards (DFlash/DSpark), or be
    individually normalized before composition when the objective requires it.
    LK's probability statistics must cover all chunks, not individual slices.
    """
    hard_num = ce_num
    if config.hard_label_mode in {"tv", "lambda"}:
        if hard_tv_num is None:
            raise ValueError("hard-label TV/lambda requires hard_tv_num")
        hard_num = hard_tv_num
        if config.hard_label_mode == "lambda":
            if probability_num is None or probability_den is None:
                raise ValueError("LK-lambda requires reduced probability statistics")
            weight = lk_kl_weight(config, probability_num, probability_den)
            hard_num = weight * ce_num + (1.0 - weight) * hard_tv_num
    if config.hard_ce_weight == 1.0 and config.teacher_l1_weight == 0.0:
        return hard_num
    if teacher_l1_num is None:
        if config.requires_teacher_distribution:
            raise ValueError("teacher L1 objective requires teacher_l1_num")
        teacher_l1_num = ce_num.new_zeros(())
    return config.hard_ce_weight * hard_num + config.teacher_l1_weight * teacher_l1_num
