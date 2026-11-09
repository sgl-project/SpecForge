"""Losses: dflash reduction."""

from functools import partial
from typing import Dict

import torch

from specforge.algorithms.losses.blocks import anchor_scales, gather_teacher_hidden
from specforge.algorithms.losses.dflash import (
    compose_token_objective,
    compute_lk_weight,
    dflash_objective_chunk_terms,
)
from specforge.algorithms.metrics.blocks import (
    add_position_ratios,
    add_prefix_metrics,
    add_ratio_family,
    compute_walk_accepted_length_terms,
)
from specforge.algorithms.metrics.dflash import dflash_metric_chunk_terms, metric_top_k
from specforge.algorithms.metrics.dflash_terms import DFlashMetricTerms
from specforge.algorithms.result import ObjectiveResult
from specforge.core.chunking import checkpointed_chunk_reduce
from specforge.core.position_weights import DPACE_LOSS_TYPES


def reduce_dflash(
    model,
    options,
    teacher_head,
    output_hidden,
    blocks,
    *,
    target_last_hidden_states=None,
    selector_loss_alpha=None,
    collect_detailed_metrics=True,
):
    """Reduce prepared DFlash blocks without materializing all vocabulary logits."""
    target_ids, weight_mask = blocks.labels, blocks.weights
    predecessor_ids = torch.cat((target_ids[..., :1], target_ids[..., :-1]), -1)
    anchor_positions = blocks.anchors
    safe_label_indices = blocks.label_positions
    bsz, device = target_ids.shape[0], target_ids.device
    hidden_4d = output_hidden.reshape(
        bsz,
        anchor_positions.shape[1],
        model.config.draft.spec_config.protocol.num_model_slots,
        -1,
    )
    aligned_target_hidden = (
        gather_teacher_hidden(
            target_last_hidden_states,
            safe_label_indices,
        )
        if (
            target_last_hidden_states is not None
            and collect_detailed_metrics
            and options.teacher_metrics
        )
        else None
    )
    sequence_anchor_scale = None
    if options.loss_type in DPACE_LOSS_TYPES:
        sequence_anchor_scale = anchor_scales(weight_mask)
    metric_terms = None
    if collect_detailed_metrics:
        # Reduce the detached diagnostics first: their full-vocabulary
        # temporaries are released before the objective keeps state for
        # backward (the fused logits loss keeps its BF16 logits).
        metric_terms = DFlashMetricTerms(
            *checkpointed_chunk_reduce(
                partial(
                    dflash_metric_chunk_terms,
                    model,
                    options,
                    teacher_head,
                    total_blocks=anchor_positions.shape[1],
                ),
                hidden_4d.detach(),
                target_ids,
                weight_mask,
                predecessor_ids,
                aligned_target_hidden,
                sequence_anchor_scale,
                torch.arange(anchor_positions.shape[1], device=device).unsqueeze(0),
                chunk_size=options.objective_chunk_blocks,
                dim=1,
            )
        )
    objective_inputs = (
        hidden_4d,
        target_ids,
        weight_mask,
        predecessor_ids,
        sequence_anchor_scale,
    )
    if model.config.draft.spec_config.protocol.num_model_slots > 1:
        # Slot 0 is the clean, zero-weight anchor in DFlash objectives.
        # Skip its projection for every backend; diagnostics above retain it.
        # Keep the single-slot case nonempty so a zero loss retains its graph.
        objective_inputs = (
            hidden_4d[:, :, 1:],
            target_ids[:, :, 1:],
            weight_mask[:, :, 1:],
            predecessor_ids[:, :, 1:],
            sequence_anchor_scale,
        )
    (
        ce_loss_num,
        tv_loss_num,
        loss_den,
        target_probability_num,
        correct_num,
        accuracy_denom,
        selector_ce_num,
        selector_probability_num,
        selector_correct_num,
        selector_weight_den,
        selector_covered_num,
    ) = checkpointed_chunk_reduce(
        partial(dflash_objective_chunk_terms, model, options),
        *objective_inputs,
        chunk_size=options.objective_chunk_blocks,
        dim=1,
        checkpoint=options.objective_checkpointing,
    )
    token_loss_num = compose_token_objective(
        options,
        ce_loss_num,
        tv_loss_num,
        target_probability_num,
        accuracy_denom,
    )
    loss_num = token_loss_num
    effective_selector_alpha = (
        getattr(options, "selector_loss_alpha", 0.0)
        if selector_loss_alpha is None
        else float(selector_loss_alpha)
    )
    if effective_selector_alpha < 0:
        raise ValueError("selector_loss_alpha must be >= 0")
    selector_loss_num = loss_num.new_zeros(())
    has_selector_objective = (
        getattr(model, "candidate_selector", None) is not None
        and getattr(options, "selector_loss_alpha", 0) > 0
    )
    if has_selector_objective:
        # The selector is a categorical distribution over the serving
        # top-k. Keep its proper, calibrated CE independent of the base
        # model's optional LK/TV composition.
        selector_loss_num = selector_ce_num
        loss_num = loss_num + effective_selector_alpha * selector_loss_num

    loss_denominator = loss_den
    ratio_metrics = {
        "acc": (correct_num.detach(), accuracy_denom.detach()),
        "target_probability": (
            target_probability_num.detach(),
            accuracy_denom.detach(),
        ),
        "lk_loss" if options.lk_loss_type is not None else "ce_loss": (
            token_loss_num.detach(),
            loss_denominator.detach(),
        ),
    }
    lk_kl_weight = compute_lk_weight(options, target_probability_num, accuracy_denom)
    if lk_kl_weight is not None:
        ratio_metrics["objective/lk_kl_weight"] = (
            (lk_kl_weight * accuracy_denom).detach(),
            accuracy_denom.detach(),
        )
    candidate_selector = getattr(model, "candidate_selector", None)
    sum_metrics = {}
    if collect_detailed_metrics:
        terms = metric_terms
        block_valid = (weight_mask[..., 1:] > 0.5).any(dim=-1)
        ratio_metrics["dflash/hard_label/walk_accepted_length"] = (
            compute_walk_accepted_length_terms(
                terms.unary_block_accepted, anchor_positions, block_valid
            )
        )
        if candidate_selector is not None:
            ratio_metrics["dflash2/selector/walk_accepted_length"] = (
                compute_walk_accepted_length_terms(
                    terms.selector_block_accepted, anchor_positions, block_valid
                )
            )
        top_k = metric_top_k(model, options)
        for name, numerators in (
            (
                "dflash/hard_label/unary_top1_accuracy",
                terms.unary_correct_num,
            ),
            (
                "dflash/hard_label/unary_probability",
                terms.hard_label_probability_num,
            ),
            (
                f"dflash/hard_label/unary_top{top_k}_recall",
                terms.unary_topk_recall_num,
            ),
            (
                f"dflash/hard_label/unary_top{top_k}_mass",
                terms.unary_topk_mass_num,
            ),
        ):
            add_ratio_family(
                ratio_metrics,
                name,
                numerators,
                terms.position_den,
            )
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
            ratio_metrics,
            sum_metrics,
            terms,
            top_k,
            has_selector=candidate_selector is not None,
        )
        if aligned_target_hidden is not None:
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
                add_ratio_family(
                    ratio_metrics,
                    name,
                    numerators,
                    terms.position_den,
                )
        if candidate_selector is not None:
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
            if aligned_target_hidden is not None:
                add_ratio_family(
                    ratio_metrics,
                    "dflash2/selector/self_conditioned_teacher_argmax_agreement",
                    terms.teacher_selector_greedy_agreement_num,
                    terms.position_den,
                )
    if has_selector_objective:
        ratio_metrics.update(
            {
                "dflash2/objective/weighted_covered_accuracy": (
                    selector_correct_num.detach(),
                    selector_weight_den.detach(),
                ),
                "dflash2/selector/teacher_forced_covered_gold_probability": (
                    selector_probability_num.detach(),
                    selector_covered_num.detach(),
                ),
                "dflash2/selector/loss": (
                    selector_loss_num.detach(),
                    selector_weight_den.detach(),
                ),
            }
        )
        if collect_detailed_metrics:
            add_position_ratios(
                ratio_metrics,
                "dflash2/selector/loss",
                terms.selector_ce_num,
                terms.selector_weight_den,
            )
    metrics: Dict[str, object] = {
        "accuracy_denom": accuracy_denom.detach(),
        "ratio_metrics": ratio_metrics,
        "sum_metrics": sum_metrics,
    }
    if has_selector_objective:
        metrics["selector_loss_alpha"] = effective_selector_alpha
    # Reduce all chunks before flooring the denominator so the result does
    # not depend on how many chunks happen to contain no effective weight.
    denominator_floor = torch.finfo(loss_denominator.dtype).tiny
    loss = loss_num / loss_denominator.clamp_min(denominator_floor)
    metrics["loss_terms"] = (loss_num, loss_denominator.detach())
    accuracy = correct_num / accuracy_denom.clamp_min(1)
    numerator, denominator = metrics.pop("loss_terms")
    ratios = metrics.pop("ratio_metrics", {})
    ratios.setdefault("accuracy", (correct_num.detach(), accuracy_denom.detach()))
    sums = metrics.pop("sum_metrics", {})
    return ObjectiveResult(numerator, denominator, ratios, metrics, sums)
