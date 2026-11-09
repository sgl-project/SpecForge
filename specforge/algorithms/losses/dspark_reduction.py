"""Losses: dspark reduction."""

from functools import partial
from typing import Optional

import torch

from specforge.algorithms.losses.blocks import gather_teacher_hidden
from specforge.algorithms.losses.dspark import (
    dspark_loss_weight_mask,
    dspark_objective_chunk_terms,
)
from specforge.algorithms.metrics.blocks import compute_walk_accepted_length_terms
from specforge.algorithms.result import ObjectiveResult
from specforge.core.chunking import checkpointed_chunk_reduce


def reduce_dspark(
    model,
    options,
    teacher_head,
    *,
    output_hidden: torch.Tensor,
    target_ids: torch.Tensor,
    eval_mask: torch.Tensor,
    prev_token_ids: torch.Tensor,
    safe_label_indices: torch.Tensor,
    target_last_hidden_states: Optional[torch.Tensor],
    anchor_positions: torch.Tensor,
    collect_detailed_metrics: bool,
) -> ObjectiveResult:
    """Token-pooled DSpark objective under the configured chunk/checkpoint policy."""

    batch_size, num_blocks, block_size = target_ids.shape
    hidden_4d = output_hidden.reshape(
        batch_size,
        num_blocks,
        block_size,
        -1,
    )
    loss_weights = dspark_loss_weight_mask(model, options, eval_mask)
    local_loss_den = loss_weights.sum()
    need_target = options.l1_loss_alpha > 0 or (
        options.confidence_loss_alpha > 0
        and getattr(model, "confidence_head", None) is not None
    )
    aligned_target_hidden = None
    if need_target:
        if target_last_hidden_states is None:
            raise ValueError(
                "DSpark L1/confidence loss requires target_last_hidden_states"
            )
        aligned_target_hidden = gather_teacher_hidden(
            target_last_hidden_states,
            safe_label_indices,
        )

    totals = checkpointed_chunk_reduce(
        partial(
            dspark_objective_chunk_terms,
            model,
            options,
            teacher_head,
            total_blocks=num_blocks,
        ),
        hidden_4d,
        prev_token_ids,
        target_ids,
        loss_weights,
        eval_mask,
        aligned_target_hidden,
        (
            torch.arange(num_blocks, device=target_ids.device).unsqueeze(0)
            if collect_detailed_metrics
            else None
        ),
        chunk_size=options.objective_chunk_blocks,
        dim=1,
        checkpoint=options.objective_checkpointing,
    )

    (
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
    ) = totals

    numerator = (
        options.ce_loss_alpha * ce_num
        + options.l1_loss_alpha * l1_num
        + options.confidence_loss_alpha * confidence_num
    )

    ratio_metrics = {
        "acc": (correct_num, eval_den),
        "ce_loss": (ce_num.detach(), local_loss_den.detach()),
        "l1_loss": (l1_num.detach(), local_loss_den.detach()),
        "confidence_loss": (
            confidence_num.detach(),
            local_loss_den.detach(),
        ),
        "confidence_abs_error": (
            confidence_error_num.detach(),
            local_loss_den.detach(),
        ),
        "ce_position": (ce_position_num, position_den),
        "accuracy_position": (correct_position_num, position_den),
    }
    if aligned_target_hidden is not None:
        ratio_metrics.update(
            {
                "teacher_agreement": (teacher_agreement_num, eval_den),
                "teacher_top1_prob": (teacher_top1_num, eval_den),
                "draft_top1_prob": (draft_top1_num, eval_den),
                "tau_probabilistic": (tau_num, tau_den),
            }
        )
    if collect_detailed_metrics:
        ratio_metrics["dspark/hard_label/walk_accepted_length"] = (
            compute_walk_accepted_length_terms(
                walk_accepted, anchor_positions, eval_mask.any(dim=-1)
            )
        )
    return ObjectiveResult(
        numerator,
        local_loss_den.detach(),
        {
            name: (num.detach(), den.detach())
            for name, (num, den) in ratio_metrics.items()
        },
    )
