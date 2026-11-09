"""Losses: dspark."""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from specforge.algorithms.losses.blocks import logits_loss_terms, project_logits
from specforge.algorithms.metrics.blocks import _scatter_accepted_prefix


def dspark_loss_weight_mask(
    model,
    options,
    eval_mask: torch.Tensor,
) -> torch.Tensor:
    loss_weight_mask = eval_mask.to(torch.float32)
    if options.loss_decay_gamma is not None and options.loss_decay_gamma > 0:
        positions = torch.arange(
            model.config.draft.spec_config.protocol.num_model_slots,
            device=eval_mask.device,
        ).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(options.loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def dspark_objective_chunk_terms(
    model,
    options,
    teacher_head,
    hidden: torch.Tensor,
    prev_token_ids: torch.Tensor,
    target_ids: torch.Tensor,
    loss_weights: torch.Tensor,
    eval_mask: torch.Tensor,
    aligned_target_hidden: Optional[torch.Tensor],
    block_index: Optional[torch.Tensor] = None,
    *,
    total_blocks: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Return additive loss and telemetry numerators for one block slice."""

    draft_logits = project_logits(model, hidden)
    if model.markov_head is not None:
        draft_logits = model.markov_head.apply_block_logits(
            draft_logits, token_ids=prev_token_ids, hidden_states=hidden
        )
    target_logits = None
    if aligned_target_hidden is not None:
        with torch.no_grad():
            target_logits = teacher_head(aligned_target_hidden)
    terms = logits_loss_terms(
        options, draft_logits, target_ids, teacher_logits=target_logits
    )
    cross_entropy, l1_per_token = terms.cross_entropy, terms.teacher_l1
    predicted_ids, draft_top1 = terms.argmax_ids, terms.top1_probability
    teacher_ids, teacher_top1 = (
        terms.teacher_argmax_ids,
        terms.teacher_top1_probability,
    )
    ce_num = (cross_entropy * loss_weights).sum()
    zero = ce_num.new_zeros(())
    l1_num = confidence_num = confidence_error_num = zero
    teacher_agreement_num = teacher_top1_num = draft_top1_num = zero
    tau_num = tau_den = zero
    accept_probability = None
    if l1_per_token is not None:
        accept_probability = (1.0 - 0.5 * l1_per_token).clamp(0.0, 1.0)
        if options.l1_loss_alpha > 0:
            l1_num = (l1_per_token * loss_weights).sum()

    confidence_pred = None
    if model.confidence_head is not None:
        features = hidden
        if model.config.draft.spec_config.confidence.with_markov:
            previous = model.markov_head.get_prev_embeddings(prev_token_ids)
            features = torch.cat((hidden, previous.to(hidden.dtype)), dim=-1)
        confidence_pred = model.confidence_head(features).float()
    if confidence_pred is not None and options.confidence_loss_alpha > 0:
        if accept_probability is None:
            raise ValueError(
                "DSpark confidence loss requires target_last_hidden_states"
            )
        confidence_per_token = F.binary_cross_entropy_with_logits(
            confidence_pred.float(),
            accept_probability.detach(),
            reduction="none",
        )
        confidence_num = (confidence_per_token * loss_weights).sum()
        confidence_error_num = (
            (confidence_pred.float().sigmoid() - accept_probability).abs()
            * loss_weights
        ).sum()

    with torch.no_grad():
        correct = ((predicted_ids == target_ids) & eval_mask).float()
        correct_num = correct.sum()
        eval_den = eval_mask.float().sum()
        ce_position_num = (cross_entropy.detach() * eval_mask).sum(dim=(0, 1))
        correct_position_num = correct.sum(dim=(0, 1))
        position_den = eval_mask.float().sum(dim=(0, 1))
        walk_accepted = _scatter_accepted_prefix(
            predicted_ids, target_ids, eval_mask, block_index, total_blocks
        )
        if aligned_target_hidden is not None:
            teacher_agreement_num = (
                (predicted_ids == teacher_ids).float() * eval_mask
            ).sum()
            teacher_top1_num = (teacher_top1 * eval_mask).sum()
            draft_top1_num = (draft_top1 * eval_mask).sum()
            valid_blocks = eval_mask.any(dim=-1).float()
            accepted_expectation = (accept_probability.detach() * eval_mask).cumprod(
                dim=-1
            ).sum(dim=-1) + 1.0
            tau_num = (accepted_expectation * valid_blocks).sum()
            tau_den = valid_blocks.sum()

    return (
        ce_num,
        l1_num,
        confidence_num,
        confidence_error_num,
        correct_num,
        eval_den,
        ce_position_num,
        correct_position_num,
        position_den,
        teacher_agreement_num,
        teacher_top1_num,
        draft_top1_num,
        tau_num,
        tau_den,
        walk_accepted,
    )
