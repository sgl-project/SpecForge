# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License").
"""Shared token objective for block-parallel draft models.

The objective is architecture independent: given draft logits aligned with
hard labels (and optionally the frozen target's logits at the same positions)
it returns one weighted numerator per objective slice plus the additive
telemetry each wrapper logs. ``OnlineDFlashModel`` and ``OnlineDSparkModel``
both bind it to their own prediction output; they differ only in how they
produce the aligned logits, labels and masks, and in which extra terms
(selector, confidence head) they add around it.

Three pieces compose:

* :class:`TokenObjectiveConfig` declares the objective: hard-label CE/TV/LK
  mode, the teacher L1 distillation weight, and the position-weight mode. It
  also declares its data needs (:attr:`TokenObjectiveConfig.requires_teacher_distribution`)
  so callers derive capture/transport requirements from the objective.
* :func:`token_objective_chunk_terms` computes the additive numerators for one
  objective slice; wrappers call it inside their ``checkpointed_chunk_reduce``
  chunk functions next to their own terms.
* :func:`compose_token_objective` turns the reduced numerators into the token
  loss numerator (CE, hard TV, LK-lambda mix, teacher L1 weighting).

:func:`compute_token_objective` wires the three together for callers that do
not need custom chunk terms.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import partial
from typing import Dict, NamedTuple, Optional, Tuple

import torch
import torch.nn.functional as F

from specforge.core import fused_loss
from specforge.core.chunking import checkpointed_chunk_reduce

HARD_LABEL_MODES = frozenset({None, "alpha", "tv", "lambda"})
DPACE_POSITION_WEIGHT_MODES = frozenset(
    {
        "dpace",
        "dpace-cumulative-confidence-only",
        "dpace-continuation-value-only",
    }
)
POSITION_WEIGHT_MODES = frozenset({"static"}) | DPACE_POSITION_WEIGHT_MODES
#: ``"1"`` routes the teacher TV/L1 distance through the fused Triton kernel,
#: ``"0"`` keeps the FP32 eager distance. Unset: fused for BF16 logits only.
FUSED_TEACHER_TV_ENV = "SPECFORGE_TOKEN_OBJECTIVE_FUSED_TV"
#: Legacy DSpark spelling of :data:`FUSED_TEACHER_TV_ENV`; still honored.
LEGACY_FUSED_TEACHER_TV_ENV = "SPECFORGE_DSPARK_FUSED_LOSS"


@dataclass(frozen=True)
class TokenObjectiveConfig:
    """Declarative token objective.

    The token loss numerator is ``hard_ce_weight * hard + teacher_l1_weight *
    L1`` where ``hard`` is the hard-label term selected by ``hard_label_mode``
    (``None``/``"alpha"``: CE; ``"tv"``: one-hot total variation ``1 - q``;
    ``"lambda"``: an acceptance-adaptive mix of the two) and ``L1`` is the
    full-vocabulary distance between the draft and frozen-target distributions
    (``2 x TV``). ``position_weight_mode`` ``"static"`` uses the caller's
    weights as given; the D-PACE modes derive detached dynamic weights from the
    draft's probability on the label.
    """

    hard_ce_weight: float = 1.0
    teacher_l1_weight: float = 0.0
    hard_label_mode: Optional[str] = None
    kl_scale: float = 1.0
    kl_decay: float = 1.0
    position_weight_mode: str = "static"
    dpace_alpha: float = 0.5

    def __post_init__(self) -> None:
        if self.hard_ce_weight < 0:
            raise ValueError("hard_ce_weight must be >= 0")
        if self.teacher_l1_weight < 0:
            raise ValueError("teacher_l1_weight must be >= 0")
        if self.hard_label_mode not in HARD_LABEL_MODES:
            raise ValueError(
                "hard_label_mode must be one of None, 'alpha', 'lambda', or 'tv'"
            )
        if self.position_weight_mode not in POSITION_WEIGHT_MODES:
            raise ValueError(
                f"position_weight_mode={self.position_weight_mode!r}; must be one "
                f"of {sorted(POSITION_WEIGHT_MODES)}"
            )
        if not 0.0 <= self.dpace_alpha <= 1.0:
            raise ValueError(f"dpace_alpha must be in [0, 1], got {self.dpace_alpha}")

    @property
    def requires_teacher_distribution(self) -> bool:
        """Whether the objective reads the frozen target's distribution."""

        return self.teacher_l1_weight > 0

    @property
    def requires_hard_tv(self) -> bool:
        """Whether the one-hot TV numerator feeds the composed objective."""

        return self.hard_label_mode in {"lambda", "tv"}

    @property
    def is_dpace(self) -> bool:
        return self.position_weight_mode in DPACE_POSITION_WEIGHT_MODES

    @property
    def uses_default_weights(self) -> bool:
        """``True`` when the objective is exactly the hard-label term."""

        return self.hard_ce_weight == 1.0 and self.teacher_l1_weight == 0.0


class TokenObjectiveChunkTerms(NamedTuple):
    """Additive numerators for one objective slice (flat tensors only)."""

    ce_num: torch.Tensor
    hard_tv_num: torch.Tensor
    teacher_l1_num: torch.Tensor
    probability_num: torch.Tensor
    loss_den: torch.Tensor


class TokenObjectiveChunk(NamedTuple):
    """One slice's additive terms plus the per-token tensors wrappers reuse."""

    terms: TokenObjectiveChunkTerms
    #: Per-token negative log draft probability on the label.
    neg_log_q: torch.Tensor
    #: ``exp(-neg_log_q)``.
    target_probability: torch.Tensor
    #: Effective loss weights after position weighting (D-PACE, anchor scale).
    loss_weights: torch.Tensor
    #: Per-token TV distance to the teacher distribution, or ``None``.
    teacher_tv: Optional[torch.Tensor]


class TokenObjectiveResult(NamedTuple):
    """Composed token objective after chunk reduction."""

    loss_num: torch.Tensor
    hard_num: torch.Tensor
    ce_num: torch.Tensor
    hard_tv_num: torch.Tensor
    teacher_l1_num: torch.Tensor
    probability_num: torch.Tensor
    lk_kl_weight: Optional[torch.Tensor]


def position_decay_weights(positions: torch.Tensor, gamma: float) -> torch.Tensor:
    """Exponential decay ``exp(-positions / gamma)`` over integer positions."""

    return torch.exp(-positions.float() / gamma)


def dpace_position_weights(
    prob: torch.Tensor,
    binary_mask: torch.Tensor,
    binary_mask_b: torch.Tensor,
    mode: str,
    alpha: float,
) -> torch.Tensor:
    """Compute D-PACE position weights (the caller detaches ``prob``).

    ``prob`` is the draft probability on the label at each draft position.
    Invalid positions are treated as multiplicative no-ops inside prefix
    products and excluded from suffix sums; the caller still multiplies the
    returned weights by ``binary_mask`` before reduction.
    """

    smooth = (1.0 - alpha) * prob + alpha
    smooth = torch.where(binary_mask_b, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)

    if mode == "dpace-cumulative-confidence-only":
        return prefix

    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * binary_mask, dims=[-1]), dim=-1),
        dims=[-1],
    )

    if mode == "dpace":
        return suffix
    if mode == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(f"unknown D-PACE loss_type {mode!r}")


def _use_fused_teacher_tv(draft_logits: torch.Tensor) -> bool:
    default = "1" if draft_logits.dtype == torch.bfloat16 else "0"
    value = os.environ.get(FUSED_TEACHER_TV_ENV)
    if value is None:
        value = os.environ.get(LEGACY_FUSED_TEACHER_TV_ENV, default)
    return value == "1"


def teacher_tv_distance(
    draft_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-position ``(tv, l1)`` between the draft and teacher distributions.

    ``l1 = 2 * tv`` for normalized softmax pairs, and ``1 - tv`` is the expected
    rejection-sampling acceptance. Gradients flow into ``draft_logits`` only.
    BF16 logits default to the fused Triton kernel (no ``[..., V]`` probability
    tensor); other dtypes use the FP32 eager distance unless
    :data:`FUSED_TEACHER_TV_ENV` overrides the choice.
    """

    vocab_size = draft_logits.shape[-1]
    lead_shape = draft_logits.shape[:-1]
    if _use_fused_teacher_tv(draft_logits):
        tv_per_token = fused_loss.fused_tv_loss(
            draft_logits.reshape(lead_shape[0], -1, vocab_size),
            teacher_logits.reshape(lead_shape[0], -1, vocab_size),
        ).reshape(lead_shape)
        return tv_per_token, 2.0 * tv_per_token
    with torch.no_grad():
        teacher_probabilities = torch.softmax(teacher_logits.float(), dim=-1)
    draft_probabilities = torch.softmax(draft_logits.float(), dim=-1)
    l1_per_token = (draft_probabilities - teacher_probabilities).abs().sum(dim=-1)
    return 0.5 * l1_per_token, l1_per_token


def token_objective_chunk_terms(
    config: TokenObjectiveConfig,
    *,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    logits: Optional[torch.Tensor] = None,
    neg_log_q: Optional[torch.Tensor] = None,
    loss_weights: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
) -> TokenObjectiveChunk:
    """Compute the additive token-objective terms for one aligned slice.

    ``labels`` is ``[..., K]``; ``logits`` is ``[..., K, V]`` aligned with it.
    ``neg_log_q`` (``[..., K]``) may replace ``logits`` when a fused head
    already produced the per-token CE; teacher terms then need ``logits``.
    ``loss_mask`` marks supervised positions (float or bool) and scales the
    probability telemetry. In ``"static"`` mode ``loss_weights`` (default:
    ``loss_mask``) carries the caller's fixed position weights; in the D-PACE
    modes it must be ``None`` and the weights are
    ``loss_mask * dpace * sequence_anchor_scale`` with the anchor-balanced
    denominator. ``teacher_logits`` (``[..., K, V]``, detached) enables the
    teacher TV/L1 distance; it is required when the config distills.
    """

    if neg_log_q is None:
        if logits is None:
            raise ValueError("token objective needs logits or neg_log_q")
        neg_log_q = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            labels.reshape(-1),
            reduction="none",
        ).reshape_as(labels)
    target_probability = torch.exp(-neg_log_q)

    if config.is_dpace:
        if loss_weights is not None:
            raise ValueError("D-PACE position weighting derives its own loss_weights")
        if sequence_anchor_scale is None:
            raise ValueError(
                "precomputed full-sequence sequence_anchor_scale is required "
                "for D-PACE chunk reduction"
            )
        with torch.no_grad():
            dpace_weights = dpace_position_weights(
                target_probability.detach(),
                loss_mask,
                loss_mask > 0,
                config.position_weight_mode,
                config.dpace_alpha,
            )
        loss_weights = loss_mask * dpace_weights
        valid_anchors = (loss_mask > 0).any(dim=-1)
        loss_weights = loss_weights * sequence_anchor_scale
        # Each valid anchor contributes 1 / A_b to the denominator, so
        # reducing all chunks yields the number of valid sequences. This
        # preserves D-PACE's total credit mass while balancing sequences
        # with different numbers of sampled anchors.
        loss_den = (
            valid_anchors.to(loss_mask.dtype) * sequence_anchor_scale.squeeze(-1)
        ).sum()
    else:
        if loss_weights is None:
            loss_weights = loss_mask
        loss_den = loss_weights.sum()

    ce_num = (neg_log_q * loss_weights).sum()
    if config.requires_hard_tv:
        hard_tv_num = ((1.0 - target_probability) * loss_weights).sum()
    else:
        hard_tv_num = ce_num.new_zeros(())

    teacher_tv = None
    teacher_l1_num = ce_num.new_zeros(())
    if teacher_logits is not None:
        if logits is None:
            raise ValueError(
                "the teacher distribution term needs the full objective logits"
            )
        teacher_tv, teacher_l1 = teacher_tv_distance(logits, teacher_logits)
        if config.teacher_l1_weight > 0:
            teacher_l1_num = (teacher_l1 * loss_weights).sum()
    elif config.requires_teacher_distribution:
        raise ValueError(
            "the token objective distills from the teacher distribution "
            "(teacher_l1_weight > 0) but no teacher_logits were given"
        )
    probability_num = (target_probability.detach() * loss_mask).sum()

    return TokenObjectiveChunk(
        terms=TokenObjectiveChunkTerms(
            ce_num=ce_num,
            hard_tv_num=hard_tv_num,
            teacher_l1_num=teacher_l1_num,
            probability_num=probability_num,
            loss_den=loss_den,
        ),
        neg_log_q=neg_log_q,
        target_probability=target_probability,
        loss_weights=loss_weights,
        teacher_tv=teacher_tv,
    )


def lk_kl_weight(
    config: TokenObjectiveConfig,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return the LK-lambda CE weight, or ``None`` for other hard-label modes."""

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
) -> TokenObjectiveResult:
    """Compose the reduced numerators into the token objective numerator.

    ``hard_tv_num`` is required for the ``"tv"`` and ``"lambda"`` modes,
    ``probability_num``/``probability_den`` for ``"lambda"``, and
    ``teacher_l1_num`` when ``teacher_l1_weight > 0``.
    """

    mode = config.hard_label_mode
    kl_weight = None
    if mode is None or mode == "alpha":
        # With a one-hot target distribution, LK-alpha is exactly NLL/CE.
        hard_num = ce_num
    elif mode == "tv":
        if hard_tv_num is None:
            raise ValueError("hard_label_mode='tv' needs hard_tv_num")
        hard_num = hard_tv_num
    elif mode == "lambda":
        if hard_tv_num is None or probability_num is None or probability_den is None:
            raise ValueError(
                "hard_label_mode='lambda' needs hard_tv_num, probability_num "
                "and probability_den"
            )
        kl_weight = lk_kl_weight(config, probability_num, probability_den)
        hard_num = kl_weight * ce_num + (1.0 - kl_weight) * hard_tv_num
    else:  # defensive: the config validates the mode.
        raise ValueError(f"unknown hard_label_mode {mode!r}")

    if config.uses_default_weights:
        loss_num = hard_num
    else:
        if teacher_l1_num is None:
            if config.teacher_l1_weight > 0:
                raise ValueError("teacher_l1_weight > 0 needs teacher_l1_num")
            teacher_l1_num = ce_num.new_zeros(())
        loss_num = (
            config.hard_ce_weight * hard_num + config.teacher_l1_weight * teacher_l1_num
        )

    zero = ce_num.new_zeros(())
    return TokenObjectiveResult(
        loss_num=loss_num,
        hard_num=hard_num,
        ce_num=ce_num,
        hard_tv_num=zero if hard_tv_num is None else hard_tv_num,
        teacher_l1_num=zero if teacher_l1_num is None else teacher_l1_num,
        probability_num=zero if probability_num is None else probability_num,
        lk_kl_weight=kl_weight,
    )


def _standalone_chunk_terms(
    config: TokenObjectiveConfig,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: Optional[torch.Tensor],
    teacher_logits: Optional[torch.Tensor],
    sequence_anchor_scale: Optional[torch.Tensor],
) -> Tuple[torch.Tensor, ...]:
    chunk = token_objective_chunk_terms(
        config,
        labels=labels,
        loss_mask=loss_mask,
        logits=logits,
        loss_weights=loss_weights,
        teacher_logits=teacher_logits,
        sequence_anchor_scale=sequence_anchor_scale,
    )
    return tuple(chunk.terms)


def compute_token_objective(
    config: TokenObjectiveConfig,
    *,
    logits: torch.Tensor,
    labels: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_weights: Optional[torch.Tensor] = None,
    teacher_logits: Optional[torch.Tensor] = None,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
    chunk_size: int = 0,
    dim: int = 1,
) -> Tuple[
    torch.Tensor, TokenObjectiveResult, Dict[str, Tuple[torch.Tensor, torch.Tensor]]
]:
    """Token objective over already-aligned logits, chunked along ``dim``.

    Returns ``(loss, result, ratio_metrics)``: ``loss`` is the composed
    numerator over the position-weight denominator; ``ratio_metrics`` holds
    ``(numerator, denominator)`` pairs under the keys the DFlash wrapper logs
    (``ce_loss`` or ``lk_loss``, ``target_probability``, ``l1_loss`` when the
    objective distills, ``objective/lk_kl_weight`` for LK-lambda).
    """

    ce_num, hard_tv_num, teacher_l1_num, probability_num, loss_den = (
        checkpointed_chunk_reduce(
            partial(_standalone_chunk_terms, config),
            logits,
            labels,
            loss_mask,
            loss_weights,
            teacher_logits,
            sequence_anchor_scale,
            chunk_size=chunk_size,
            dim=dim,
        )
    )
    probability_den = loss_mask.sum()
    result = compose_token_objective(
        config,
        ce_num=ce_num,
        hard_tv_num=hard_tv_num,
        teacher_l1_num=teacher_l1_num,
        probability_num=probability_num,
        probability_den=probability_den,
    )
    denominator_floor = torch.finfo(loss_den.dtype).tiny
    loss = result.loss_num / loss_den.clamp_min(denominator_floor)
    ratio_metrics = {
        "lk_loss" if config.hard_label_mode is not None else "ce_loss": (
            result.hard_num.detach(),
            loss_den.detach(),
        ),
        "target_probability": (probability_num.detach(), probability_den.detach()),
    }
    if config.requires_teacher_distribution:
        ratio_metrics["l1_loss"] = (result.teacher_l1_num.detach(), loss_den.detach())
    if result.lk_kl_weight is not None:
        ratio_metrics["objective/lk_kl_weight"] = (
            (result.lk_kl_weight * probability_den).detach(),
            probability_den.detach(),
        )
    return loss, result, ratio_metrics


__all__ = [
    "DPACE_POSITION_WEIGHT_MODES",
    "FUSED_TEACHER_TV_ENV",
    "HARD_LABEL_MODES",
    "LEGACY_FUSED_TEACHER_TV_ENV",
    "POSITION_WEIGHT_MODES",
    "TokenObjectiveChunk",
    "TokenObjectiveChunkTerms",
    "TokenObjectiveConfig",
    "TokenObjectiveResult",
    "compose_token_objective",
    "compute_token_objective",
    "dpace_position_weights",
    "lk_kl_weight",
    "position_decay_weights",
    "teacher_tv_distance",
    "token_objective_chunk_terms",
]
