"""Losses: domino."""

from typing import Optional, Tuple

import torch

from specforge.algorithms.losses.blocks import logits_loss_backend, project_logits
from specforge.algorithms.metrics.blocks import (
    _scatter_accepted_prefix,
    compute_accept_len,
)


def domino_objective_chunk_terms(
    model,
    options,
    hidden: torch.Tensor,
    prev_ids: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    eval_weight_mask: torch.Tensor,
    block_index: Optional[torch.Tensor] = None,
    *,
    total_blocks: int = 0,
) -> Tuple[torch.Tensor, ...]:
    """Return additive Domino loss and telemetry terms for one block slice."""
    from specforge.core.domino_loss import domino_weighted_cross_entropy

    block_size = hidden.shape[2]
    base_logits = project_logits(model, hidden)
    head_token_embeddings = model.embed_tokens(prev_ids)
    spec = model.config.draft.spec_config
    prefix = (
        head_token_embeddings
        if spec.block.shift_label
        else head_token_embeddings[..., 1:, :]
    )
    correction_logits = model.gru_head(hidden[..., spec.suffix_start :, :], prefix)
    final_num, base_num, predicted_ids, base_predicted_ids = (
        domino_weighted_cross_entropy(
            base_logits.reshape(-1, base_logits.shape[-1]),
            correction_logits.reshape(-1, correction_logits.shape[-1]),
            target_ids.reshape(-1),
            weight_mask.reshape(-1),
            block_size=block_size,
            suffix_start=model.config.draft.spec_config.suffix_start,
            use_fused=logits_loss_backend(options, base_logits) == "fused",
        )
    )
    loss_den = weight_mask.sum()

    with torch.no_grad():
        predicted_ids = predicted_ids.reshape_as(target_ids)
        base_predicted_ids = base_predicted_ids.reshape_as(target_ids)
        binary_accuracy_mask = eval_weight_mask > 0.5
        correct_num = (
            ((predicted_ids == target_ids) & binary_accuracy_mask).sum().float()
        )
        base_correct_num = (
            ((base_predicted_ids == target_ids) & binary_accuracy_mask).sum().float()
        )
        accuracy_den = eval_weight_mask.sum()

        valid_mask = eval_weight_mask > 0
        accepted = compute_accept_len(predicted_ids, target_ids, valid_mask)
        base_accepted = compute_accept_len(
            base_predicted_ids,
            target_ids,
            valid_mask,
        )
        valid_blocks = valid_mask.any(dim=-1).float()
        accept_num = ((accepted + 1.0) * valid_blocks).sum()
        base_accept_num = ((base_accepted + 1.0) * valid_blocks).sum()
        accept_den = valid_blocks.sum()
        acceptance_slice = (
            slice(None, -1)
            if model.config.draft.spec_config.block.shift_label
            else slice(1, None)
        )
        # Before the first rejection, teacher-forced and generated histories agree.
        walk_accepted, base_walk_accepted = (
            _scatter_accepted_prefix(
                ids[..., acceptance_slice],
                target_ids[..., acceptance_slice],
                valid_mask[..., acceptance_slice],
                block_index,
                total_blocks,
            )
            for ids in (predicted_ids, base_predicted_ids)
        )

    return (
        final_num,
        base_num,
        loss_den,
        correct_num,
        base_correct_num,
        accuracy_den,
        accept_num,
        base_accept_num,
        accept_den,
        walk_accepted,
        base_walk_accepted,
    )
