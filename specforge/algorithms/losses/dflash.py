"""Losses: dflash."""

from typing import Optional

import torch

from specforge.algorithms.losses.blocks import (
    logits_loss_terms,
    project_logits,
    selector_token_terms,
)
from specforge.algorithms.losses.dflash_terms import DFlashObjectiveTerms, SelectorTerms
from specforge.core.position_weights import DPACE_LOSS_TYPES, dpace_weights


def dflash_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    logits = project_logits(model, hidden)
    transform = getattr(model, "transform_unary_logits", None)
    is_identity = getattr(model, "unary_logits_transform_is_identity", None)
    # The loss normalizes in FP32; an identity transform needs no FP32 copy.
    if transform is not None and not (is_identity is not None and is_identity()):
        return transform(logits)
    return logits


@torch.no_grad()
def dflash_loss_weights(
    options,
    target_probability: torch.Tensor,
    weight_mask: torch.Tensor,
    *,
    slot_positions: torch.Tensor,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Detached token weights; slot positions refer to the original full block.

    Training can omit anchor slot 0 while diagnostics retain it. D-PACE
    uses full-sequence anchor counts; callers keep their own reductions.
    """
    if options.loss_type == "dflash":
        if options.loss_decay_gamma is None or options.loss_decay_gamma <= 0:
            return weight_mask
        decay_weights = torch.exp(
            -(slot_positions - 1).clamp(min=0).float() / options.loss_decay_gamma
        )
        return weight_mask * decay_weights

    if options.loss_type in DPACE_LOSS_TYPES:
        if sequence_anchor_scale is None:
            raise ValueError(
                "precomputed full-sequence sequence_anchor_scale is required "
                "for D-PACE weights"
            )
        position_weights = dpace_weights(
            target_probability,
            weight_mask,
            alpha=options.dpace_alpha,
            mode=options.loss_type,
        )
        return weight_mask * position_weights * sequence_anchor_scale

    raise ValueError(f"unknown loss_type {options.loss_type!r}")


def dflash_objective_chunk_terms(
    model,
    options,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    predecessor_ids: torch.Tensor,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
) -> DFlashObjectiveTerms:
    """Return additive objective/metric terms, optionally omitting anchor slot 0."""

    block_size = hidden.shape[2]
    candidate_selector = getattr(model, "candidate_selector", None)
    terms = logits_loss_terms(
        options,
        dflash_logits(model, hidden),
        target_ids,
        top_k=(
            candidate_selector.top_k
            if (
                getattr(model, "candidate_selector", None) is not None
                and getattr(options, "selector_loss_alpha", 0) > 0
            )
            else 0
        ),
    )
    neg_log_q = terms.cross_entropy

    target_probability = torch.exp(-neg_log_q)
    slot_positions = torch.arange(
        model.config.draft.spec_config.protocol.num_model_slots - block_size,
        model.config.draft.spec_config.protocol.num_model_slots,
        device=hidden.device,
    )
    loss_weights = dflash_loss_weights(
        options,
        target_probability,
        weight_mask,
        slot_positions=slot_positions,
        sequence_anchor_scale=sequence_anchor_scale,
    )
    if options.loss_type == "dflash":
        loss_den = loss_weights.sum()
    elif options.loss_type in DPACE_LOSS_TYPES:
        valid_anchors = (weight_mask > 0).any(dim=-1)
        # Each valid anchor contributes 1 / A_b to the denominator, so
        # reducing all chunks yields the number of valid sequences. This
        # preserves D-PACE's total credit mass while balancing sequences
        # with different numbers of sampled anchors.
        loss_den = (
            valid_anchors.to(weight_mask.dtype) * sequence_anchor_scale.squeeze(-1)
        ).sum()
    else:  # defensive: __init__ validates the configured loss type.
        raise ValueError(f"unknown loss_type {options.loss_type!r}")

    ce_loss_num = (neg_log_q * loss_weights).sum()
    if options.lk_loss_type in {"lambda", "tv"}:
        tv_loss_num = ((1.0 - target_probability) * loss_weights).sum()
    else:
        tv_loss_num = ce_loss_num.new_zeros(())
    target_probability_num = (target_probability.detach() * weight_mask).sum()

    selector_terms = SelectorTerms.zeros(ce_loss_num)
    if (
        getattr(model, "candidate_selector", None) is not None
        and getattr(options, "selector_loss_alpha", 0) > 0
    ):
        # Gate on static configuration only: zero-scaled warmup must retain
        # the selector graph for DDP(find_unused_parameters=False).
        # Hidden states provide a direct backbone gradient; top-k values
        # provide the path through logits (and the fused loss backward).
        selector_hidden = (
            hidden.detach()
            if getattr(options, "selector_stop_gradient", False)
            else hidden
        )
        selector_values = (
            terms.topk_values.detach()
            if getattr(options, "selector_stop_gradient", False)
            else terms.topk_values
        )
        selector = selector_token_terms(
            candidate_selector=candidate_selector,
            hidden=selector_hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            topk_values=selector_values,
            topk_ids=terms.topk_ids,
        )
        # A candidate miss has no selector target and carries no selector CE.
        selector_weights = loss_weights * selector.target_is_candidate.float()
        selector_mask = weight_mask * selector.target_is_candidate.float()
        selector_terms = SelectorTerms(
            ce_num=(selector.cross_entropy * selector_weights).sum(),
            probability_num=(
                torch.exp(-selector.cross_entropy.detach()) * selector_mask
            ).sum(),
            correct_num=(
                (selector.selected_ids == target_ids).float() * selector_weights
            ).sum(),
            weight_den=selector_weights.sum(),
            covered_num=selector_mask.sum(),
        )

    with torch.no_grad():
        predicted_ids = terms.argmax_ids
        correct_num = (
            ((predicted_ids == target_ids) & (weight_mask > 0.5)).sum().float()
        )
        accuracy_den = weight_mask.sum()
    return DFlashObjectiveTerms(
        ce_loss_num=ce_loss_num,
        tv_loss_num=tv_loss_num,
        loss_den=loss_den,
        target_probability_num=target_probability_num,
        correct_num=correct_num,
        accuracy_den=accuracy_den,
        selector_ce_num=selector_terms.ce_num,
        selector_probability_num=selector_terms.probability_num,
        selector_correct_num=selector_terms.correct_num,
        selector_weight_den=selector_terms.weight_den,
        selector_covered_num=selector_terms.covered_num,
    )


def compute_lk_weight(
    options,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Return the LK-lambda CE weight, or ``None`` for other objectives."""

    if options.lk_loss_type != "lambda":
        return None
    acceptance = probability_num / probability_den.clamp_min(1.0)
    return options.kl_scale * torch.exp(-options.kl_decay * acceptance.detach())


def compose_token_objective(
    options,
    ce_num: torch.Tensor,
    tv_num: torch.Tensor,
    probability_num: torch.Tensor,
    probability_den: torch.Tensor,
) -> torch.Tensor:
    """Compose a hard-target CE/TV/LK numerator after chunk reduction."""

    if options.lk_loss_type is None or options.lk_loss_type == "alpha":
        # With a one-hot target distribution, LK-alpha is exactly NLL/CE.
        return ce_num
    if options.lk_loss_type == "tv":
        return tv_num
    if options.lk_loss_type == "lambda":
        kl_weight = compute_lk_weight(options, probability_num, probability_den)
        return kl_weight * ce_num + (1.0 - kl_weight) * tv_num
    raise ValueError(f"unknown lk_loss_type {options.lk_loss_type!r}")
