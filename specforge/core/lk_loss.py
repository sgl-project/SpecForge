from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F


def expected_acceptance_rate(
    target_probs: torch.Tensor,
    draft_probs: torch.Tensor,
) -> torch.Tensor:
    """Compute token-wise expected acceptance rates for speculative decoding."""
    if target_probs.shape != draft_probs.shape:
        raise ValueError(
            f"target_probs and draft_probs must have the same shape, "
            f"got {target_probs.shape} and {draft_probs.shape}"
        )
    return torch.minimum(target_probs, draft_probs).sum(dim=-1)


def compute_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float = 1e-8,
    reduce_fn: Optional[
        Callable[..., Tuple[torch.Tensor, torch.Tensor]]
    ] = None,
) -> torch.Tensor:
    """Compute expected acceptance rate from draft logits and target probabilities."""
    draft_p = F.softmax(logits.to(torch.float32), dim=-1).to(target_probs.dtype)
    acceptance_rate_per_token = expected_acceptance_rate(
        target_probs=target_probs,
        draft_probs=draft_p,
    )

    mask = position_mask.squeeze(-1)
    if mask.dtype == torch.bool:
        mask = mask.float()
    else:
        mask = mask.to(dtype=acceptance_rate_per_token.dtype)

    numerator = (acceptance_rate_per_token * mask).sum()
    denominator = mask.sum().clamp_min(eps)
    if reduce_fn is not None:
        numerator, denominator = reduce_fn(
            local_correct=numerator, local_denom=denominator
        )
        denominator = denominator.clamp_min(eps)
    return numerator / denominator


def compute_lk_loss(
    *,
    kl_loss: torch.Tensor,
    acceptance_rate: torch.Tensor,
    lk_loss_type: str,
    kl_scale: float,
    kl_decay: float,
) -> torch.Tensor:
    """Combine KL and LK objectives according to the selected LK loss type."""

    if lk_loss_type == "alpha":
        return -torch.log(acceptance_rate)
    if lk_loss_type == "lambda":
        lk_loss = 1.0 - acceptance_rate
        kl_weight = kl_scale * torch.exp(-kl_decay * acceptance_rate)
        return kl_weight * kl_loss + (1.0 - kl_weight) * lk_loss
    raise ValueError(f"Unknown lk loss type: {lk_loss_type}")
