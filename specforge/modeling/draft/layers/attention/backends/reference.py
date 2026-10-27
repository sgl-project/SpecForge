"""Small, differentiable sparse oracle. Explicitly allocates dense score matrices.

Never select this backend implicitly for production training or inference.
"""

from dataclasses import dataclass

import torch
from torch.nn.attention.flex_attention import BlockMask

from ...config import AttentionConfig
from ..base import AttentionContext
from ..selection import SparseSelection
from .sparse import SparseAttentionResult


@dataclass(frozen=True)
class SparseKVSource:
    name: str
    key: torch.Tensor  # [B, Hkv, S, Dk]
    value: torch.Tensor  # [B, Hkv, S, Dv]
    positions: torch.Tensor  # [B or 1, S], logical visibility positions
    mask: torch.Tensor | BlockMask | None = None  # source-local [B, H, Q, S]


@dataclass(frozen=True)
class ExplicitSparseInputs:
    query: torch.Tensor  # [B, H, Q, Dk]
    sources: tuple[SparseKVSource, ...]
    query_positions: torch.Tensor  # [B or 1, Q]
    scale: float


def visibility_mask(
    mask: torch.Tensor | BlockMask | None, query: torch.Tensor, key_length: int
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Materialize algorithm visibility only for explicit reference/indexer use."""
    b, h, q, _ = query.shape
    shape = (b, h, q, key_length)
    if isinstance(mask, BlockMask):
        # Evaluate logical positions, not padded BlockMask metadata indices.
        bi = torch.arange(b, device=query.device)[:, None, None, None]
        hi = torch.arange(h, device=query.device)[None, :, None, None]
        qi = torch.arange(q, device=query.device)[None, None, :, None]
        ki = torch.arange(key_length, device=query.device)[None, None, None, :]
        mask = mask.mask_mod(bi, hi, qi, ki)
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=query.device), None
    mask = torch.broadcast_to(mask, shape)
    if mask.dtype == torch.bool:
        return mask, None
    if not mask.is_floating_point():
        raise TypeError("attention mask must be boolean or floating point")
    if torch.isnan(mask).any() or torch.isposinf(mask).any():
        raise ValueError("attention bias cannot contain NaN or positive infinity")
    allowed = torch.isfinite(mask) & (mask != torch.finfo(mask.dtype).min)
    return allowed, mask.masked_fill(~allowed, 0)


def source_visibility(
    inputs: ExplicitSparseInputs,
    source: SparseKVSource,
    context: AttentionContext,
    options: AttentionConfig,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if context.attention_mask is not None and len(inputs.sources) != 1:
        raise ValueError("Multiple KV sources require source-local masks")
    allowed, bias = visibility_mask(source.mask, inputs.query, source.key.shape[2])
    if context.attention_mask is not None:
        extra, extra_bias = visibility_mask(
            context.attention_mask, inputs.query, source.key.shape[2]
        )
        allowed = allowed & extra
        if extra_bias is not None:
            bias = extra_bias if bias is None else bias + extra_bias
    distance = inputs.query_positions[:, :, None] - source.positions[:, None, :]
    if options.is_causal:
        allowed = allowed & (distance[:, None] >= 0)
    if options.sliding_window is not None:
        allowed = allowed & (distance[:, None].abs() < options.sliding_window)
    return allowed, bias


def selection_mask(
    inputs: ExplicitSparseInputs, source: SparseKVSource, selection: SparseSelection
) -> torch.Tensor:
    """Set union with checked indices. Invalid padding is never gathered."""
    b, h, q, _ = inputs.query.shape
    s = source.key.shape[2]
    selected = torch.zeros((b, h, q, s), dtype=torch.bool, device=inputs.query.device)
    for entry in selection.indexed:
        if entry.source != source.name:
            continue
        if entry.indices.shape[:3] not in ((b, 1, q), (b, h, q)):
            raise ValueError("selection batch/query/head dimensions do not match")
        width = entry.indices.shape[-1]
        valid = torch.arange(width, device=selected.device) < entry.lengths[..., None]
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
        if not s:
            continue
        offsets = torch.arange(entry.block_size, device=selected.device)
        indices = (entry.indices[..., None] * entry.block_size + offsets).flatten(-2)
        valid = (
            valid[..., None] & (indices.unflatten(-1, (width, entry.block_size)) < s)
        ).flatten(-2)
        # Scatter counts, not booleans: invalid slots must not overwrite valid ones.
        counts = torch.zeros_like(selected, dtype=torch.int32)
        counts.scatter_add_(
            -1,
            indices.clamp(0, s - 1).expand(b, h, q, -1),
            valid.int().expand(b, h, q, -1),
        )
        selected |= counts > 0
    for window in selection.windows:
        if window.source == source.name:
            distance = inputs.query_positions[:, :, None] - source.positions[:, None, :]
            selected |= distance[:, None].abs() < window.size
    return selected


def expanded_kv(
    query: torch.Tensor, source: SparseKVSource
) -> tuple[torch.Tensor, torch.Tensor]:
    heads = query.shape[1]
    if heads % source.key.shape[1]:
        raise ValueError("KV heads must divide query heads")
    groups = heads // source.key.shape[1]
    return (
        source.key.repeat_interleave(groups, 1),
        source.value.repeat_interleave(groups, 1),
    )


def masked_probabilities(scores: torch.Tensor, allowed: torch.Tensor) -> torch.Tensor:
    valid = allowed.any(-1, keepdim=True)
    scores = scores.masked_fill(~allowed, float("-inf"))
    return scores.masked_fill(~valid, 0).softmax(-1).masked_fill(~allowed, 0)


def reference_sparse_attention(
    inputs: ExplicitSparseInputs,
    selection: SparseSelection,
    context: AttentionContext,
    options: AttentionConfig,
) -> SparseAttentionResult:
    if not inputs.sources:
        raise ValueError("At least one KV source is required")
    names = [s.name for s in inputs.sources]
    if len(set(names)) != len(names):
        raise ValueError("KV source names must be unique")
    if any(e.source not in names for e in (*selection.indexed, *selection.windows)):
        raise ValueError("selection refers to an unknown KV source")
    dtype = torch.float64 if inputs.query.dtype == torch.float64 else torch.float32
    scores, values, masks = [], [], []
    for source in inputs.sources:
        k, v = expanded_kv(inputs.query, source)
        allowed, bias = source_visibility(inputs, source, context, options)
        allowed = allowed & selection_mask(inputs, source, selection)
        logits = inputs.query.to(dtype) @ k.to(dtype).transpose(-1, -2) * inputs.scale
        if bias is not None:
            logits = logits + bias
        scores.append(logits)
        masks.append(allowed)
        values.append(v.to(dtype))
    allowed = torch.cat(masks, -1)
    probabilities = masked_probabilities(torch.cat(scores, -1), allowed)
    output = probabilities @ torch.cat(values, -2)
    return SparseAttentionResult(
        output.transpose(1, 2).to(inputs.query.dtype), allowed.any(-1).transpose(1, 2)
    )
