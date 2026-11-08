"""Tensor-only initialization of DSpine's fixed codes and learned readouts."""

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from specforge.config.draft._validation import positive_int


@torch.no_grad()
def initialize_transfer_space(
    head_weight: Tensor,
    codes: Tensor,
    readouts: Sequence[nn.Linear],
    *,
    chunk_size: int = 4096,
) -> None:
    """Write whitened token codes and principal directions into existing tensors.

    The caller supplies target weights and owns readiness/resume decisions. This
    performs no file reading and preserves destination identity, device and dtype.
    Covariance accumulation and eigendecomposition use FP32, as in legacy DSpine.
    """
    positive_int("chunk_size", chunk_size)
    if head_weight.ndim != 2 or codes.ndim != 2:
        raise ValueError("head_weight and codes must be matrices")
    vocab, width = head_weight.shape
    rank = codes.shape[1]
    if codes.shape[0] != vocab or not 0 < rank <= min(width, vocab - 1):
        raise ValueError("transfer code dimensions must match vocabulary and rank")
    if not readouts or any(r.weight.shape != (rank, width) for r in readouts):
        raise ValueError("readouts must match transfer rank and head hidden width")
    if not head_weight.is_floating_point() or not codes.is_floating_point():
        raise ValueError("head_weight and codes must be floating point")
    covariance = torch.zeros(
        width, width, device=head_weight.device, dtype=torch.float32
    )
    total = covariance.new_zeros(width)
    for rows in head_weight.split(chunk_size):
        if not torch.isfinite(rows).all():
            raise ValueError("head_weight must be finite")
        normalized = F.normalize(rows.float(), dim=-1)
        total.add_(normalized.sum(dim=0))
        covariance.add_(normalized.T @ normalized)
    mean = total / vocab
    covariance.sub_(vocab * torch.outer(mean, mean))
    values, directions = torch.linalg.eigh(covariance)
    values, directions = values[-rank:], directions[:, -rank:]
    if not torch.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Target head has insufficient positive principal directions for transfer_rank"
        )
    for start in range(0, vocab, chunk_size):
        rows = head_weight[start : start + chunk_size]
        normalized = F.normalize(
            (F.normalize(rows.float(), dim=-1) - mean) @ directions / values.sqrt(),
            dim=-1,
        )
        codes[start : start + len(rows)].copy_(normalized)
    for readout in readouts:
        readout.weight.copy_(directions.T)
