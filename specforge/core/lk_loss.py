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


def _masked_mean(
    values_per_token: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]],
) -> torch.Tensor:
    """Compute a masked mean, with optional distributed reduction."""
    mask = position_mask.squeeze(-1)
    if mask.dtype == torch.bool:
        mask = mask.float()
    else:
        mask = mask.to(dtype=values_per_token.dtype)

    numerator = (values_per_token * mask).sum()
    denominator = mask.sum().clamp_min(eps)
    if reduce_fn is not None:
        numerator, denominator = reduce_fn(
            local_correct=numerator, local_denom=denominator
        )
        denominator = denominator.clamp_min(eps)
    return numerator / denominator


def _acceptance_rate_per_token_from_logits(
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    chunk_size: int = 256,
) -> torch.Tensor:
    """Return per-token expected acceptance from draft logits and target probs.

    The fp32 softmax over the draft vocab is computed in token chunks so it
    never materializes for the whole sequence at once; each chunk reduces to
    one scalar per token, keeping peak memory flat in sequence length.
    """
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target_probs.reshape(-1, target_probs.shape[-1])
    per_token = []
    for i in range(0, flat_logits.shape[0], chunk_size):
        draft_p = F.softmax(
            flat_logits[i : i + chunk_size].to(torch.float32), dim=-1
        ).to(flat_target.dtype)
        per_token.append(
            expected_acceptance_rate(
                target_probs=flat_target[i : i + chunk_size], draft_probs=draft_p
            )
        )
    return torch.cat(per_token).reshape(logits.shape[:-1])


def compute_acceptance_rate(
    *,
    logits: torch.Tensor,
    target_probs: torch.Tensor,
    position_mask: torch.Tensor,
    eps: float = 1e-8,
    reduce_fn: Optional[Callable[..., Tuple[torch.Tensor, torch.Tensor]]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return masked means of acceptance and log-acceptance over valid positions."""
    acceptance_rate_per_token = _acceptance_rate_per_token_from_logits(
        logits=logits,
        target_probs=target_probs,
    )
    acceptance_rate = _masked_mean(
        values_per_token=acceptance_rate_per_token,
        position_mask=position_mask,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    log_acceptance_rate_per_token = torch.where(
        acceptance_rate_per_token > 0, torch.log(acceptance_rate_per_token), 0
    )
    log_acceptance_rate = _masked_mean(
        values_per_token=log_acceptance_rate_per_token,
        position_mask=position_mask,
        eps=eps,
        reduce_fn=reduce_fn,
    )
    return acceptance_rate, log_acceptance_rate


def compute_lk_loss(
    *,
    kl_loss: torch.Tensor,
    acceptance_rate: torch.Tensor,
    log_acceptance_rate: torch.Tensor,
    lk_loss_type: str,
    kl_scale: float,
    kl_decay: float,
) -> torch.Tensor:
    """Compute LK loss from KL loss and acceptance rate."""
    if lk_loss_type == "alpha":
        return -log_acceptance_rate
    if lk_loss_type == "lambda":
        acc_det = acceptance_rate.detach()
        kl_weight = kl_scale * torch.exp(-kl_decay * acc_det)
        return kl_weight * kl_loss + (1 - kl_weight) * (1 - acceptance_rate)
    raise ValueError(f"Unknown lk loss type: {lk_loss_type}")
