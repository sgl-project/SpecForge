"""Vocabulary-chunked teacher projection without full vocabulary logits."""

from bisect import bisect_left
from typing import Optional, Tuple

import torch
import torch.nn.functional as F


@torch.no_grad()
def tiled_logsumexp_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    chunk_size: int = 32768,
    selected_token_ids: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
    """Return full-vocabulary log-normalizer, argmax and selected logits.

    hidden has shape [..., hidden_size]; weight is [vocabulary, hidden_size].
    Optional sorted token IDs select retained logits. All results are detached,
    with normalization computed in FP32 and the first token winning ties.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    vocab_size = weight.shape[0]
    lead_shape = hidden.shape[:-1]
    device = hidden.device

    neg_inf = float("-inf")
    running_max = torch.full(
        (*lead_shape, 1), neg_inf, dtype=torch.float32, device=device
    )
    running_sumexp = torch.zeros((*lead_shape, 1), dtype=torch.float32, device=device)
    running_argval = torch.full(lead_shape, neg_inf, dtype=torch.float32, device=device)
    running_argmax = torch.zeros(lead_shape, dtype=torch.long, device=device)

    selected_logits = None
    selected_ids_host = None
    selected_start = 0
    if selected_token_ids is not None:
        # Copy once so chunk boundaries do not synchronize the device inside the loop.
        selected_ids_host = selected_token_ids.tolist()
        selected_logits = torch.empty(
            (*lead_shape, len(selected_ids_host)), dtype=torch.float32, device=device
        )

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_logits = F.linear(hidden, weight[start:end]).float()

        if selected_ids_host is not None:
            selected_end = bisect_left(selected_ids_host, end, lo=selected_start)
            if selected_end > selected_start:
                chunk_indices = selected_token_ids[selected_start:selected_end] - start
                torch.index_select(
                    chunk_logits,
                    -1,
                    chunk_indices,
                    out=selected_logits[..., selected_start:selected_end],
                )
            selected_start = selected_end

        chunk_max = chunk_logits.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        running_sumexp = running_sumexp * torch.exp(running_max - new_max) + torch.exp(
            chunk_logits - new_max
        ).sum(dim=-1, keepdim=True)
        running_max = new_max

        # Ascending scan + strict-greater update keeps the lowest global index on ties.
        chunk_val, chunk_idx = chunk_logits.max(dim=-1)
        chunk_idx = chunk_idx + start
        take = chunk_val > running_argval
        running_argmax = torch.where(take, chunk_idx, running_argmax)
        running_argval = torch.where(take, chunk_val, running_argval)

    log_z = running_max + torch.log(running_sumexp)
    return log_z, running_argmax, selected_logits
