"""Detached position weights, independent of draft architecture and loss backend.

Functions operate along the last (token-position) axis. Callers provide aligned
probabilities and masks, then apply the final mask, loss coefficients, and
method-specific normalization. These functions do not project hidden states or
choose the per-token loss, chunking, or checkpoint policy.
"""

import torch

DPACE_LOSS_TYPES = frozenset(
    {"dpace", "dpace-cumulative-confidence-only", "dpace-continuation-value-only"}
)


@torch.no_grad()
def dpace_weights(
    target_probability: torch.Tensor,
    mask: torch.Tensor,
    *,
    alpha: float,
    mode: str = "dpace",
) -> torch.Tensor:
    """Return D-PACE credit weights for matching ``[..., positions]`` inputs.

    ``target_probability`` is the draft probability of each hard target token;
    ``mask`` is a binary supervision mask. ``alpha`` in [0, 1] smooths the
    probabilities toward one. Masked slots leave prefix products unchanged and
    contribute nothing to suffix sums. Returned weights are detached and are
    not finally masked or normalized across anchors, sequences, or workers.
    """
    smooth = (1.0 - alpha) * target_probability + alpha
    smooth = torch.where(mask > 0, smooth, torch.ones_like(smooth))
    prefix = torch.cumprod(smooth, dim=-1)

    if mode == "dpace-cumulative-confidence-only":
        return prefix

    suffix = torch.flip(
        torch.cumsum(torch.flip(prefix * mask, dims=[-1]), dim=-1), dims=[-1]
    )
    if mode == "dpace":
        return suffix
    if mode == "dpace-continuation-value-only":
        return suffix / prefix.clamp_min(torch.finfo(prefix.dtype).tiny)
    raise ValueError(f"unknown D-PACE mode {mode!r}")
