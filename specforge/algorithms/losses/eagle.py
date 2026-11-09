"""Fused soft cross entropy and optional LK acceptance objectives."""

from typing import Callable, Optional, Tuple

import torch

from specforge.core.lk_loss import (
    _reduce_acceptance_rows,
    compute_acceptance_rate,
    compute_lk_loss,
)


def eagle_loss(
    *,
    logits: torch.Tensor,
    target_p: torch.Tensor,
    target_p_on_draft: torch.Tensor,
    position_mask: torch.Tensor,
    lk_loss_type: Optional[str],
    kl_scale: float,
    kl_decay: float,
    reduce_metrics_fn: Optional[
        Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
    reduce_loss_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute step loss and acceptance rate for KL/LK objectives.

    Args:
        logits: Draft model logits for current step.
        target_p: Renormalized target distribution over draft-vocab tokens (for KL).
        target_p_on_draft: Original target probabilities restricted to draft-vocab tokens (for acceptance terms).
        position_mask: Mask indicating valid tokens for loss/metric aggregation.
        lk_loss_type: LK objective mode (`None`, `"alpha"`, or `"lambda"`).
        kl_scale: Scale factor for lambda LK mixing weight.
        kl_decay: Decay factor for lambda LK mixing weight.
        reduce_metrics_fn: Optional distributed reducer for metric numer/denom.
        reduce_loss_fn: Optional distributed reducer for KL loss.
    """
    acceptance_rows = None
    if logits.is_cuda:
        from specforge.core.loss import LogSoftmaxLoss

    if not logits.is_cuda:
        kl_loss = (
            -(target_p * logits.float().log_softmax(-1) * position_mask).sum(-1).mean()
        )
    elif lk_loss_type is None and target_p_on_draft.dtype == torch.float32:
        kl_loss, acceptance_rows = LogSoftmaxLoss.apply(
            logits, target_p, position_mask, target_p_on_draft
        )
    else:
        kl_loss = LogSoftmaxLoss.apply(logits, target_p, position_mask)
    if reduce_loss_fn is not None:
        kl_loss = reduce_loss_fn(kl_loss)

    with torch.set_grad_enabled(lk_loss_type is not None):
        if acceptance_rows is not None:
            acceptance_rate, log_acceptance_rate = _reduce_acceptance_rows(
                acceptance_rows, position_mask, reduce_fn=reduce_metrics_fn
            )
        else:
            acceptance_rate, log_acceptance_rate = compute_acceptance_rate(
                logits=logits,
                target_probs=target_p_on_draft,
                position_mask=position_mask,
                reduce_fn=reduce_metrics_fn,
            )

    if lk_loss_type is None:
        loss = kl_loss
    else:
        loss = compute_lk_loss(
            kl_loss=kl_loss,
            acceptance_rate=acceptance_rate,
            log_acceptance_rate=log_acceptance_rate,
            lk_loss_type=lk_loss_type,
            kl_scale=kl_scale,
            kl_decay=kl_decay,
        )
    return acceptance_rate.detach(), loss
