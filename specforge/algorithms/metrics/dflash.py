"""Metrics: dflash."""

from typing import Optional

import torch
import torch.nn as nn

from specforge.algorithms.losses.blocks import selector_token_terms
from specforge.algorithms.losses.dflash import dflash_logits, dflash_loss_weights
from specforge.algorithms.metrics.blocks import (
    _expected_accepted_length_num,
    _scatter_accepted_prefix,
)
from specforge.algorithms.metrics.dflash_terms import (
    DFlashMetricTerms,
    _DFlashGreedyDiagnostics,
    _DFlashSelectorDiagnostics,
    _DFlashTeacherTerms,
    _DFlashUnaryDiagnostics,
)
from specforge.algorithms.metrics.prefix import hard_label_prefix_counts


@torch.no_grad()
def metric_top_k(
    model,
    options,
) -> int:
    """Candidate-set width: the selector's top-k, else ``metric_top_k``.

    Never wider than the vocabulary, so tiny test vocabularies stay valid.
    """

    candidate_selector = getattr(model, "candidate_selector", None)
    top_k = (
        int(candidate_selector.top_k)
        if candidate_selector is not None
        else options.metric_top_k
    )
    vocab_size = getattr(model.embed_tokens, "num_embeddings", None)
    return top_k if vocab_size is None else min(top_k, int(vocab_size))


@torch.no_grad()
def dflash_metric_chunk_terms(
    model,
    options,
    teacher_head,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    predecessor_ids: torch.Tensor,
    aligned_target_hidden: Optional[torch.Tensor] = None,
    sequence_anchor_scale: Optional[torch.Tensor] = None,
    block_index: Optional[torch.Tensor] = None,
    *,
    total_blocks: int = 0,
) -> DFlashMetricTerms:
    """Return additive unary, selector, and teacher diagnostics for one chunk.

    The unary, objective-weight, and teacher families apply to every
    DFlash-family draft; the selector and greedy-path families are only
    computed when the draft carries a DFlash2 candidate selector.
    """

    candidate_selector = getattr(model, "candidate_selector", None)
    unary = dflash_unary_diagnostics(
        model,
        hidden,
        target_ids,
        top_k=metric_top_k(model, options),
    )
    loss_weights = dflash_loss_weights(
        options,
        unary.hard_label_probability,
        weight_mask,
        slot_positions=torch.arange(weight_mask.shape[-1], device=hidden.device),
        sequence_anchor_scale=sequence_anchor_scale,
    )
    selector = None
    if candidate_selector is not None:
        selector = dflash2_selector_diagnostics(
            candidate_selector=candidate_selector,
            unary=unary,
            hidden=hidden,
            target_ids=target_ids,
            predecessor_ids=predecessor_ids,
            loss_weights=loss_weights,
            weight_mask=weight_mask,
        )
    position_den = weight_mask.sum(dim=(0, 1))
    greedy = dflash_greedy_diagnostics(
        candidate_selector=candidate_selector,
        unary=unary,
        hidden=hidden,
        target_ids=target_ids,
        weight_mask=weight_mask,
        position_den=position_den,
    )
    teacher = dflash_teacher_terms(
        teacher_head,
        unary=unary,
        greedy_ids=greedy.selected_ids,
        aligned_target_hidden=aligned_target_hidden,
        weight_mask=weight_mask,
        position_den=position_den,
    )
    return reduce_dflash_metric_terms(
        unary=unary,
        selector=selector,
        greedy=greedy,
        teacher=teacher,
        target_ids=target_ids,
        weight_mask=weight_mask,
        loss_weights=loss_weights,
        position_den=position_den,
        block_index=block_index,
        total_blocks=total_blocks,
    )


def dflash_unary_diagnostics(
    model,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    top_k: int,
) -> _DFlashUnaryDiagnostics:
    """Project draft states once and derive the strict unary top-k view."""

    objective_logits = dflash_logits(model, hidden).float()
    probabilities = torch.softmax(objective_logits.float(), dim=-1)
    hard_label_probability = probabilities.gather(-1, target_ids.unsqueeze(-1)).squeeze(
        -1
    )
    topk_logits, candidate_ids = objective_logits.topk(
        min(top_k, objective_logits.shape[-1]),
        dim=-1,
    )
    target_matches = candidate_ids.eq(target_ids.unsqueeze(-1))
    return _DFlashUnaryDiagnostics(
        probabilities=probabilities,
        hard_label_probability=hard_label_probability,
        predicted_ids=objective_logits.argmax(dim=-1),
        topk_logits=topk_logits,
        candidate_ids=candidate_ids,
        target_is_candidate=target_matches.any(dim=-1),
    )


def dflash2_selector_diagnostics(
    *,
    candidate_selector: nn.Module,
    unary: _DFlashUnaryDiagnostics,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    loss_weights: torch.Tensor,
    weight_mask: torch.Tensor,
) -> _DFlashSelectorDiagnostics:
    """Evaluate the selector with ground-truth predecessor tokens."""

    selector = selector_token_terms(
        candidate_selector=candidate_selector,
        hidden=hidden,
        target_ids=target_ids,
        predecessor_ids=predecessor_ids,
        topk_values=unary.topk_logits,
        topk_ids=unary.candidate_ids,
    )
    target_is_candidate = selector.target_is_candidate.float()
    return _DFlashSelectorDiagnostics(
        cross_entropy=selector.cross_entropy,
        loss_weights=loss_weights * target_is_candidate,
        selected_ids=selector.selected_ids,
        covered_mask=weight_mask * target_is_candidate,
    )


def dflash_greedy_diagnostics(
    *,
    candidate_selector: Optional[nn.Module],
    unary: _DFlashUnaryDiagnostics,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    position_den: torch.Tensor,
) -> _DFlashGreedyDiagnostics:
    """Reduce per-block acceptance chains; walk the selector path if any."""

    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    zero = position_den.new_zeros(())
    if block_size <= 1:
        return _DFlashGreedyDiagnostics(
            selected_ids=None,
            correct_num=position_den.new_zeros(position_den.shape),
            block_den=zero,
            oracle_accepted_length_num=zero.clone(),
            accepted_length_num=zero.clone(),
            expected_accepted_length_num=zero.clone(),
        )

    # A block accepts its anchor, then only its leading run of supervised
    # slots: covered (oracle), hit by the greedy path (realized), or, for
    # the smooth D-PACE surrogate, weighted by the gold-token probability.
    supervised = weight_mask[:, :, 1:] > 0.5
    block_valid = supervised.any(dim=-1).float()
    oracle_accepted_length_num = _expected_accepted_length_num(
        unary.target_is_candidate[:, :, 1:], supervised, block_valid
    )
    expected_accepted_length_num = _expected_accepted_length_num(
        unary.hard_label_probability[:, :, 1:], supervised, block_valid
    )
    if candidate_selector is None:
        return _DFlashGreedyDiagnostics(
            selected_ids=None,
            correct_num=position_den.new_zeros(position_den.shape),
            block_den=block_valid.sum(),
            oracle_accepted_length_num=oracle_accepted_length_num,
            accepted_length_num=zero,
            expected_accepted_length_num=expected_accepted_length_num,
        )

    greedy_ids = candidate_selector.greedy_path(
        candidate_ids=unary.candidate_ids[:, :, 1:].reshape(
            batch_size * num_blocks,
            block_size - 1,
            candidate_selector.top_k,
        ),
        unary_logits=unary.topk_logits[:, :, 1:].reshape(
            batch_size * num_blocks,
            block_size - 1,
            candidate_selector.top_k,
        ),
        hidden_states=hidden[:, :, 1:].reshape(
            batch_size * num_blocks,
            block_size - 1,
            hidden_size,
        ),
        anchor_token_ids=target_ids[:, :, 0].reshape(-1),
    ).reshape(batch_size, num_blocks, block_size - 1)
    greedy_hit = greedy_ids == target_ids[:, :, 1:]
    return _DFlashGreedyDiagnostics(
        selected_ids=greedy_ids,
        correct_num=torch.cat(
            (
                position_den.new_zeros(1),
                (greedy_hit.float() * weight_mask[:, :, 1:]).sum(dim=(0, 1)),
            )
        ),
        block_den=block_valid.sum(),
        oracle_accepted_length_num=oracle_accepted_length_num,
        accepted_length_num=_expected_accepted_length_num(
            greedy_hit, supervised, block_valid
        ),
        expected_accepted_length_num=expected_accepted_length_num,
    )


def dflash_teacher_terms(
    teacher_head,
    *,
    unary: _DFlashUnaryDiagnostics,
    greedy_ids: Optional[torch.Tensor],
    aligned_target_hidden: Optional[torch.Tensor],
    weight_mask: torch.Tensor,
    position_den: torch.Tensor,
) -> _DFlashTeacherTerms:
    """Compare unary and greedy predictions with the frozen target head."""

    if aligned_target_hidden is None:
        return _DFlashTeacherTerms.zeros(position_den)

    block_size = aligned_target_hidden.shape[2]
    teacher_logits = teacher_head(aligned_target_hidden)
    teacher_probabilities = torch.softmax(teacher_logits.float(), dim=-1)
    teacher_ids = teacher_logits.argmax(dim=-1)
    expected_acceptance = (
        1.0 - 0.5 * (unary.probabilities - teacher_probabilities).abs().sum(dim=-1)
    ).clamp(0.0, 1.0)

    selector_greedy_agreement_num = position_den.new_zeros(position_den.shape)
    expected_accepted_length_num = position_den.new_zeros(())
    if block_size > 1:
        if greedy_ids is not None:
            selector_greedy_agreement_num = torch.cat(
                (
                    position_den.new_zeros(1),
                    (
                        (greedy_ids == teacher_ids[:, :, 1:]).float()
                        * weight_mask[:, :, 1:]
                    ).sum(dim=(0, 1)),
                )
            )
        supervised = weight_mask[:, :, 1:] > 0.5
        expected_accepted_length_num = _expected_accepted_length_num(
            expected_acceptance[:, :, 1:],
            supervised,
            supervised.any(dim=-1).float(),
        )
    return _DFlashTeacherTerms(
        expected_acceptance_num=(expected_acceptance * weight_mask).sum(dim=(0, 1)),
        unary_top1_agreement_num=(
            (unary.predicted_ids == teacher_ids).float() * weight_mask
        ).sum(dim=(0, 1)),
        unary_topk_mass_num=(
            teacher_probabilities.gather(-1, unary.candidate_ids).sum(dim=-1)
            * weight_mask
        ).sum(dim=(0, 1)),
        selector_greedy_agreement_num=selector_greedy_agreement_num,
        expected_accepted_length_num=expected_accepted_length_num,
    )


def reduce_dflash_metric_terms(
    *,
    unary: _DFlashUnaryDiagnostics,
    selector: Optional[_DFlashSelectorDiagnostics],
    greedy: _DFlashGreedyDiagnostics,
    teacher: _DFlashTeacherTerms,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    loss_weights: torch.Tensor,
    position_den: torch.Tensor,
    block_index: Optional[torch.Tensor],
    total_blocks: int,
) -> DFlashMetricTerms:
    """Reduce per-token diagnostics into the flat chunk-reducer contract."""

    covered_mask = weight_mask * unary.target_is_candidate.float()
    position_zeros = position_den.new_zeros(position_den.shape)
    if selector is None:
        selector_ce_num = position_zeros
        selector_weight_den = position_zeros
        selector_conditional_correct_num = position_zeros
    else:
        selector_ce_num = (selector.cross_entropy * selector.loss_weights).sum(
            dim=(0, 1)
        )
        selector_weight_den = selector.loss_weights.sum(dim=(0, 1))
        selector_conditional_correct_num = (
            (selector.selected_ids == target_ids).float() * selector.covered_mask
        ).sum(dim=(0, 1))
    prefix = hard_label_prefix_counts(
        unary_hit=unary.predicted_ids[:, :, 1:] == target_ids[:, :, 1:],
        covered=unary.target_is_candidate[:, :, 1:],
        selector_hit=(
            None
            if greedy.selected_ids is None
            else greedy.selected_ids == target_ids[:, :, 1:]
        ),
        supervised=weight_mask[:, :, 1:] > 0.5,
    )
    return DFlashMetricTerms(
        hard_label_probability_num=(unary.hard_label_probability * weight_mask).sum(
            dim=(0, 1)
        ),
        unary_correct_num=(
            (unary.predicted_ids == target_ids).float() * weight_mask
        ).sum(dim=(0, 1)),
        position_den=position_den,
        unary_topk_recall_num=covered_mask.sum(dim=(0, 1)),
        unary_topk_mass_num=(
            unary.probabilities.gather(-1, unary.candidate_ids).sum(dim=-1)
            * weight_mask
        ).sum(dim=(0, 1)),
        selector_ce_num=selector_ce_num,
        selector_weight_den=selector_weight_den,
        selector_conditional_correct_num=selector_conditional_correct_num,
        selector_covered_den=covered_mask.sum(dim=(0, 1)),
        selector_greedy_correct_num=greedy.correct_num,
        loss_weight_num=loss_weights.sum(dim=(0, 1)),
        teacher_expected_acceptance_num=teacher.expected_acceptance_num,
        teacher_unary_top1_agreement_num=teacher.unary_top1_agreement_num,
        teacher_unary_topk_mass_num=teacher.unary_topk_mass_num,
        teacher_selector_greedy_agreement_num=teacher.selector_greedy_agreement_num,
        block_den=greedy.block_den,
        oracle_accepted_length_num=greedy.oracle_accepted_length_num,
        greedy_accepted_length_num=greedy.accepted_length_num,
        expected_accepted_length_num=greedy.expected_accepted_length_num,
        teacher_expected_accepted_length_num=teacher.expected_accepted_length_num,
        prefix_reached_num=prefix.reached,
        prefix_accepted_num=prefix.accepted,
        selector_prefix_covered_num=prefix.selector_covered,
        selector_prefix_unary_correct_num=prefix.selector_unary_correct,
        prefix_eligible_num=prefix.eligible,
        unary_block_accepted=_scatter_accepted_prefix(
            unary.predicted_ids[..., 1:],
            target_ids[..., 1:],
            weight_mask[..., 1:] > 0.5,
            block_index,
            total_blocks,
        ),
        selector_block_accepted=_scatter_accepted_prefix(
            greedy.selected_ids,
            target_ids[..., 1:],
            weight_mask[..., 1:] > 0.5,
            block_index,
            total_blocks,
        ),
    )
