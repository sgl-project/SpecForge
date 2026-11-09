"""Losses: blocks."""

from typing import Literal, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from specforge.algorithms.losses.dflash_terms import SelectorTokenTerms
from specforge.core import fused_loss


def gather_teacher_hidden(
    target_last_hidden_states: torch.Tensor,
    safe_label_indices: torch.Tensor,
) -> torch.Tensor:
    """Gather the frozen target state that predicts each hard label."""

    target_pred_indices = (safe_label_indices - 1).clamp(
        0, target_last_hidden_states.shape[1] - 1
    )
    batch_size = target_last_hidden_states.shape[0]
    hidden_size = target_last_hidden_states.shape[-1]
    gather_indices = target_pred_indices.reshape(batch_size, -1, 1).expand(
        -1, -1, hidden_size
    )
    return torch.gather(
        target_last_hidden_states,
        1,
        gather_indices,
    ).reshape(*safe_label_indices.shape, hidden_size)


def selector_token_terms(
    candidate_selector: nn.Module,
    hidden: torch.Tensor,
    target_ids: torch.Tensor,
    predecessor_ids: torch.Tensor,
    topk_values: torch.Tensor,
    topk_ids: torch.Tensor,
) -> SelectorTokenTerms:
    """Return teacher-forced selector terms; callers control gradients and weights."""
    target_matches = topk_ids.eq(target_ids.unsqueeze(-1))
    target_is_candidate = target_matches.any(dim=-1)
    target_candidate_index = target_matches.long().argmax(dim=-1)
    selector_logits = candidate_selector.score_candidates(
        candidate_ids=topk_ids,
        unary_logits=topk_values,
        hidden_states=hidden,
        predecessor_ids=predecessor_ids,
    )
    selector_ce = F.cross_entropy(
        selector_logits.float().reshape(-1, selector_logits.shape[-1]),
        target_candidate_index.reshape(-1),
        reduction="none",
    ).reshape_as(target_ids)
    selected_ids = topk_ids.gather(
        -1, selector_logits.argmax(dim=-1, keepdim=True)
    ).squeeze(-1)
    return SelectorTokenTerms(
        cross_entropy=selector_ce,
        target_is_candidate=target_is_candidate,
        selected_ids=selected_ids,
    )


def anchor_scales(weight_mask: torch.Tensor) -> torch.Tensor:
    """Return 1 / valid-anchor-count for every anchor in each sequence."""

    valid_anchor_counts = (weight_mask > 0).any(dim=-1).sum(dim=1, keepdim=True)
    return (
        valid_anchor_counts.to(weight_mask.dtype)
        .clamp_min(1.0)
        .reciprocal()
        .unsqueeze(-1)
        .expand(-1, weight_mask.shape[1], -1)
    )


def logits_loss_backend(options, logits: torch.Tensor) -> Literal["eager", "fused"]:
    """Choose the loss implementation independently of projection and memory policy."""
    if options.fused_logits_loss and fused_loss.supports_fused_ce_l1_loss(logits):
        return "fused"
    return "eager"


def logits_loss_terms(
    options,
    logits: torch.Tensor,
    target_ids: torch.Tensor,
    *,
    teacher_logits: Optional[torch.Tensor] = None,
    top_k: int = 0,
) -> fused_loss.CEL1Loss:
    loss_fn = (
        fused_loss.fused_ce_l1_loss
        if logits_loss_backend(options, logits) == "fused"
        else fused_loss.ce_l1_loss
    )
    return loss_fn(logits, target_ids, teacher_logits=teacher_logits, top_k=top_k)


def project_logits(model, hidden: torch.Tensor) -> torch.Tensor:
    """Apply the LM head to block states without choosing loss or memory policy."""
    batch_size, num_blocks, block_size, hidden_size = hidden.shape
    return model.lm_head(
        hidden.reshape(batch_size, num_blocks * block_size, hidden_size)
    ).reshape(batch_size, num_blocks, block_size, -1)
