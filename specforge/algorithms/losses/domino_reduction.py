"""Losses: domino reduction."""

from functools import partial

import torch

from specforge.algorithms.losses.domino import domino_objective_chunk_terms
from specforge.algorithms.metrics.blocks import compute_walk_accepted_length_terms
from specforge.algorithms.result import ObjectiveResult
from specforge.core.chunking import checkpointed_chunk_reduce


def reduce_domino(
    model,
    options,
    output_hidden,
    blocks,
    *,
    lambda_base=0.0,
    collect_detailed_metrics=True,
):
    """Reduce GRU-corrected and base losses under one chunking policy."""
    target_ids, weight_mask = blocks.labels, blocks.weights
    anchor_positions = blocks.anchors
    bsz, n, bs = target_ids.shape
    device = target_ids.device
    hidden4d = output_hidden.reshape(bsz, n, bs, -1)
    eval_weight_mask = weight_mask
    if options.loss_decay_gamma is not None and options.loss_decay_gamma > 0:
        k = torch.arange(
            model.config.draft.spec_config.protocol.num_model_slots, device=device
        ).view(1, 1, -1)
        offset = 0 if model.config.draft.spec_config.block.shift_label else 1
        decay_weights = torch.exp(
            -(k - offset).clamp(min=0).float() / options.loss_decay_gamma
        )
        weight_mask = weight_mask * decay_weights

    (
        final_num,
        base_num,
        loss_den,
        correct_num,
        base_correct_num,
        accuracy_denom,
        accept_num,
        base_accept_num,
        accept_den,
        walk_accepted,
        base_walk_accepted,
    ) = checkpointed_chunk_reduce(
        partial(domino_objective_chunk_terms, model, options, total_blocks=n),
        hidden4d,
        blocks.predecessors,
        target_ids,
        weight_mask,
        eval_weight_mask,
        (
            torch.arange(n, device=device).unsqueeze(0)
            if collect_detailed_metrics
            else None
        ),
        chunk_size=options.objective_chunk_blocks,
        dim=1,
        checkpoint=options.objective_checkpointing,
    )

    valid_token_count = loss_den + 1e-6
    final_loss = final_num / valid_token_count
    base_loss = base_num / valid_token_count
    loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
    accuracy = correct_num / (accuracy_denom + 1e-6)
    metrics = {
        "final_loss": final_loss.detach(),
        "base_loss": base_loss.detach(),
        "base_accuracy": (base_correct_num / (accuracy_denom + 1e-6)).detach(),
        "accept_len": (accept_num / (accept_den + 1e-6)).detach(),
        "base_accept_len": (base_accept_num / (accept_den + 1e-6)).detach(),
        # Telemetry does not participate in the objective. Keeping this as
        # a host scalar avoids a tiny H2D copy that otherwise synchronizes
        # the whole forward stream once per micro-step.
        "lambda_base": float(lambda_base),
        "accuracy_denom": accuracy_denom.detach(),
    }
    # Hand the trainer raw numerator/denominator so gradients are
    # normalized by the globally reduced token count instead of a
    # mean of per-rank ratios.
    metrics["loss_terms"] = (
        (1.0 - lambda_base) * final_num + lambda_base * base_num,
        loss_den.detach(),
    )
    if collect_detailed_metrics:
        acceptance_slice = (
            slice(None, -1)
            if model.config.draft.spec_config.block.shift_label
            else slice(1, None)
        )
        block_valid = (eval_weight_mask[..., acceptance_slice] > 0).any(dim=-1)
        metrics["ratio_metrics"] = {
            f"domino/{head}/walk_accepted_length": compute_walk_accepted_length_terms(
                accepted, anchor_positions, block_valid
            )
            for head, accepted in (
                ("final", walk_accepted),
                ("base", base_walk_accepted),
            )
        }

    numerator, denominator = metrics.pop("loss_terms")
    ratios = metrics.pop("ratio_metrics", {})
    ratios.setdefault("accuracy", (correct_num.detach(), accuracy_denom.detach()))
    sums = metrics.pop("sum_metrics", {})
    return ObjectiveResult(numerator, denominator, ratios, metrics, sums)
