"""Autograd-capable selected-KV execution with bounded query chunks.

This PyTorch implementation avoids a full attention score matrix, but is not a
fused sparse kernel. Indexer scoring/visibility costs are accounted separately.
"""

import torch

from ...config import AttentionConfig
from ..base import AttentionContext
from ..selection import SparseSelection
from .reference import (
    ExplicitSparseInputs,
    expanded_kv,
    masked_probabilities,
    source_visibility,
)
from .sparse import SparseAttentionResult


def gathered_sparse_attention(
    inputs: ExplicitSparseInputs,
    selection: SparseSelection,
    context: AttentionContext,
    options: AttentionConfig,
    *,
    query_chunk_size: int = 32,
) -> SparseAttentionResult:
    if len(inputs.sources) != 1 or selection.windows:
        raise NotImplementedError(
            "Gathered execution needs one source and explicit indexed selections"
        )
    source = inputs.sources[0]
    b, h, q, _ = inputs.query.shape
    s = source.key.shape[2]
    pieces = []
    for entry in selection.indexed:
        if entry.source != source.name:
            raise ValueError("Unknown KV source")
        width = entry.indices.shape[-1]
        valid = (
            torch.arange(width, device=inputs.query.device) < entry.lengths[..., None]
        )
        if ((entry.lengths < 0) | (entry.lengths > width)).any():
            raise ValueError("selection lengths are out of range")
        if (
            valid
            & (
                (entry.indices < 0)
                | (entry.indices >= (s + entry.block_size - 1) // entry.block_size)
            )
        ).any():
            raise ValueError("valid selection indices are out of range")
        indices = entry.indices[..., None] * entry.block_size + torch.arange(
            entry.block_size, device=inputs.query.device
        )
        valid = valid[..., None] & (indices < s)
        pieces.append(indices.masked_fill(~valid, s).flatten(-2).expand(b, h, q, -1))
    indices = (
        torch.cat(pieces, -1)
        if pieces
        else torch.empty((b, h, q, 0), device=inputs.query.device, dtype=torch.long)
    )
    indices = indices.sort(-1).values
    unique = torch.ones_like(indices, dtype=torch.bool)
    unique[..., 1:] = indices[..., 1:] != indices[..., :-1]
    valid_indices = unique & (indices < s)
    allowed, bias = source_visibility(inputs, source, context, options)
    k, v = expanded_kv(inputs.query, source)
    dtype = torch.float64 if inputs.query.dtype == torch.float64 else torch.float32
    outputs, validity = [], []
    bi = torch.arange(b, device=k.device)[:, None, None, None]
    hi = torch.arange(h, device=k.device)[None, :, None, None]
    for start in range(0, q, query_chunk_size):
        end = min(q, start + query_chunk_size)
        idx = indices[:, :, start:end].clamp(0, max(s - 1, 0))
        if not s or idx.shape[-1] == 0:
            # Preserve zero gradients to all inputs for empty selections.
            zero = (
                inputs.query[:, :, start:end].sum(-1, keepdim=True) * 0
                + (k.sum() + v.sum()) * 0
            )
            outputs.append(zero.expand(b, h, end - start, v.shape[-1]))
            validity.append(
                torch.zeros(b, h, end - start, device=k.device, dtype=torch.bool)
            )
            continue
        selected_k, selected_v = k[bi, hi, idx], v[bi, hi, idx]
        mask = allowed[:, :, start:end].gather(-1, idx) & valid_indices[:, :, start:end]
        logits = (
            inputs.query[:, :, start:end, None, :].to(dtype) * selected_k.to(dtype)
        ).sum(-1) * inputs.scale
        if bias is not None:
            logits = logits + bias[:, :, start:end].gather(-1, idx)
        probs = masked_probabilities(logits, mask)
        outputs.append((probs[..., None] * selected_v.to(dtype)).sum(-2))
        validity.append(mask.any(-1))
    return SparseAttentionResult(
        torch.cat(outputs, 2).transpose(1, 2).to(inputs.query.dtype),
        torch.cat(validity, 2).transpose(1, 2),
    )
