"""Render additive block diagnostics using the historical metric names."""

import torch

from .blocks import (
    add_position_ratios,
    add_prefix_metrics,
    add_ratio_family,
    compute_walk_accepted_length_terms,
)


def format_block_metrics(
    terms,
    weight_mask,
    anchor_positions,
    top_k,
    *,
    has_selector,
    has_teacher,
    selector_alpha=None,
    kl_weight=None,
):
    """Return ratio pairs and raw counts; normalization belongs to the trainer.

    selector_alpha is the effective selector coefficient while that objective
    trains. kl_weight is the (kl_scale, kl_decay) of an LK-lambda token loss on
    hard labels. Each adds its own keys and is None otherwise.
    """
    ratio_metrics, sum_metrics = {}, {}
    block_valid = (weight_mask[..., 1:] > 0.5).any(dim=-1)
    ratio_metrics["dflash/hard_label/walk_accepted_length"] = (
        compute_walk_accepted_length_terms(
            terms.unary_block_accepted, anchor_positions, block_valid
        )
    )
    if has_selector:
        ratio_metrics["dflash2/selector/walk_accepted_length"] = (
            compute_walk_accepted_length_terms(
                terms.selector_block_accepted, anchor_positions, block_valid
            )
        )
    for name, numerators in (
        ("dflash/hard_label/unary_top1_accuracy", terms.unary_correct_num),
        ("dflash/hard_label/unary_probability", terms.hard_label_probability_num),
        (f"dflash/hard_label/unary_top{top_k}_recall", terms.unary_topk_recall_num),
        (f"dflash/hard_label/unary_top{top_k}_mass", terms.unary_topk_mass_num),
    ):
        add_ratio_family(ratio_metrics, name, numerators, terms.position_den)
    ratio_metrics["target_probability"] = ratio_metrics[
        "dflash/hard_label/unary_probability"
    ]
    if kl_weight is not None:
        kl_scale, kl_decay = kl_weight
        accepted, count = ratio_metrics["target_probability"]
        # The CE share LK-lambda gave this prediction set, weighted by its size.
        share = kl_scale * torch.exp(-kl_decay * accepted / count.clamp_min(1.0))
        ratio_metrics["objective/lk_kl_weight"] = (share * count, count)
    add_position_ratios(
        ratio_metrics,
        "dflash/objective/loss_weight_share",
        terms.loss_weight_num,
        terms.loss_weight_num.sum().expand_as(terms.loss_weight_num),
    )
    ratio_metrics[f"dflash/hard_label/unary_top{top_k}_oracle_accepted_length"] = (
        terms.oracle_accepted_length_num.detach(),
        terms.block_den.detach(),
    )
    ratio_metrics["dflash/hard_label/unary_gold_probability_chain_length"] = (
        terms.expected_accepted_length_num.detach(),
        terms.block_den.detach(),
    )
    add_prefix_metrics(
        ratio_metrics, sum_metrics, terms, top_k, has_selector=has_selector
    )
    if has_teacher:
        ratio_metrics["dflash/teacher/unary_overlap_chain_length"] = (
            terms.teacher_expected_accepted_length_num.detach(),
            terms.block_den.detach(),
        )
        for name, numerators in (
            (
                "dflash/teacher/unary_distribution_overlap",
                terms.teacher_expected_acceptance_num,
            ),
            (
                "dflash/teacher/unary_top1_agreement",
                terms.teacher_unary_top1_agreement_num,
            ),
            (
                f"dflash/teacher/unary_top{top_k}_mass",
                terms.teacher_unary_topk_mass_num,
            ),
        ):
            add_ratio_family(ratio_metrics, name, numerators, terms.position_den)
    if has_selector:
        add_ratio_family(
            ratio_metrics,
            "dflash2/selector/self_conditioned_marginal_accuracy",
            terms.selector_greedy_correct_num,
            terms.position_den,
        )
        add_ratio_family(
            ratio_metrics,
            "dflash2/selector/teacher_forced_covered_accuracy",
            terms.selector_conditional_correct_num,
            terms.selector_covered_den,
        )
        ratio_metrics["dflash2/selector/greedy_accepted_length"] = (
            terms.greedy_accepted_length_num.detach(),
            terms.block_den.detach(),
        )
        if has_teacher:
            add_ratio_family(
                ratio_metrics,
                "dflash2/selector/self_conditioned_teacher_argmax_agreement",
                terms.teacher_selector_greedy_agreement_num,
                terms.position_den,
            )
    if selector_alpha is not None:
        add_ratio_family(
            ratio_metrics,
            "dflash2/selector/loss",
            terms.selector_ce_num,
            terms.selector_weight_den,
        )
        weight = terms.selector_weight_den.sum().detach()
        ratio_metrics["dflash2/objective/weighted_covered_accuracy"] = (
            terms.selector_weighted_correct_num.sum().detach(),
            weight,
        )
        ratio_metrics["dflash2/selector/teacher_forced_covered_gold_probability"] = (
            terms.selector_probability_num.sum().detach(),
            terms.selector_covered_den.sum().detach(),
        )
        # A ratio over one, so accumulation and ranks average it, not sum it.
        ratio_metrics["selector_loss_alpha"] = (
            selector_alpha * weight.new_ones(()),
            weight.new_ones(()),
        )
    return ratio_metrics, sum_metrics
