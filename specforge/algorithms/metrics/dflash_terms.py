"""Metrics: dflash terms."""

from typing import NamedTuple, Optional

import torch


class DFlashMetricTerms(NamedTuple):
    """Additive DFlash2 diagnostics for one metric chunk.

    Per-position fields hold one entry per block position; the accepted-length
    fields and ``block_den`` are per-block scalars. The final two tensors
    reconstruct accepted prefixes in sampled-anchor order across chunks.
    """

    hard_label_probability_num: torch.Tensor
    unary_correct_num: torch.Tensor
    position_den: torch.Tensor
    unary_topk_recall_num: torch.Tensor
    unary_topk_mass_num: torch.Tensor
    selector_ce_num: torch.Tensor
    selector_weight_den: torch.Tensor
    selector_weighted_correct_num: torch.Tensor
    selector_probability_num: torch.Tensor
    selector_conditional_correct_num: torch.Tensor
    selector_covered_den: torch.Tensor
    selector_greedy_correct_num: torch.Tensor
    loss_weight_num: torch.Tensor
    teacher_expected_acceptance_num: torch.Tensor
    teacher_unary_top1_agreement_num: torch.Tensor
    teacher_unary_topk_mass_num: torch.Tensor
    teacher_selector_greedy_agreement_num: torch.Tensor
    block_den: torch.Tensor
    oracle_accepted_length_num: torch.Tensor
    greedy_accepted_length_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor
    teacher_expected_accepted_length_num: torch.Tensor
    prefix_reached_num: torch.Tensor
    prefix_accepted_num: torch.Tensor
    selector_prefix_covered_num: torch.Tensor
    selector_prefix_unary_correct_num: torch.Tensor
    prefix_eligible_num: torch.Tensor
    unary_block_accepted: torch.Tensor
    selector_block_accepted: torch.Tensor

    @classmethod
    def zeros(
        cls,
        block_size: int,
        *,
        anchors_shape: tuple[int, ...],
        has_selector: bool,
        device: torch.device,
    ) -> "DFlashMetricTerms":
        pos = torch.zeros(block_size, device=device, dtype=torch.float32)
        scalar = torch.zeros((), device=device, dtype=torch.float32)
        prefix = torch.zeros((3, block_size), device=device, dtype=torch.float32)
        anchors = torch.zeros(anchors_shape, device=device, dtype=torch.float32)
        return cls(
            hard_label_probability_num=pos,
            unary_correct_num=pos,
            position_den=pos,
            unary_topk_recall_num=pos,
            unary_topk_mass_num=pos,
            selector_ce_num=pos,
            selector_weight_den=pos,
            selector_weighted_correct_num=pos,
            selector_probability_num=pos,
            selector_conditional_correct_num=pos,
            selector_covered_den=pos,
            selector_greedy_correct_num=pos,
            loss_weight_num=pos,
            teacher_expected_acceptance_num=pos,
            teacher_unary_top1_agreement_num=pos,
            teacher_unary_topk_mass_num=pos,
            teacher_selector_greedy_agreement_num=pos,
            block_den=scalar,
            oracle_accepted_length_num=scalar,
            greedy_accepted_length_num=scalar,
            expected_accepted_length_num=scalar,
            teacher_expected_accepted_length_num=scalar,
            prefix_reached_num=prefix,
            prefix_accepted_num=prefix,
            selector_prefix_covered_num=pos,
            selector_prefix_unary_correct_num=pos,
            prefix_eligible_num=pos,
            unary_block_accepted=anchors,
            selector_block_accepted=anchors if has_selector else scalar,
        )


class _DFlashUnaryDiagnostics(NamedTuple):
    """Full-vocabulary unary outputs reused by all DFlash2 diagnostics."""

    probabilities: torch.Tensor
    hard_label_probability: torch.Tensor
    predicted_ids: torch.Tensor
    topk_logits: torch.Tensor
    candidate_ids: torch.Tensor
    target_is_candidate: torch.Tensor


class _DFlashSelectorDiagnostics(NamedTuple):
    """Teacher-forced selector outputs and their effective metric masks."""

    cross_entropy: torch.Tensor
    loss_weights: torch.Tensor
    selected_ids: torch.Tensor
    covered_mask: torch.Tensor


class _DFlashGreedyDiagnostics(NamedTuple):
    """Per-block acceptance chains plus the greedy selector path when present."""

    selected_ids: Optional[torch.Tensor]
    correct_num: torch.Tensor
    block_den: torch.Tensor
    oracle_accepted_length_num: torch.Tensor
    accepted_length_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor


class _DFlashTeacherTerms(NamedTuple):
    """Additive target-distribution diagnostics for one metric chunk."""

    expected_acceptance_num: torch.Tensor
    unary_top1_agreement_num: torch.Tensor
    unary_topk_mass_num: torch.Tensor
    selector_greedy_agreement_num: torch.Tensor
    expected_accepted_length_num: torch.Tensor

    @classmethod
    def zeros(cls, position_den: torch.Tensor) -> "_DFlashTeacherTerms":
        return cls(
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(position_den.shape),
            position_den.new_zeros(()),
        )
