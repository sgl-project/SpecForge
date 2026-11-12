"""Eager softmax attention with explicit mask, scaling, and dropout semantics."""

import torch
from torch import Tensor
from torch.nn import functional as F

from specforge.config.draft._validation import boolean, finite_number, positive_number


def _repeat_kv(hidden: Tensor, groups: int) -> Tensor:
    if groups == 1:
        return hidden
    batch, heads, length, width = hidden.shape
    return (
        hidden[:, :, None, :, :]
        .expand(batch, heads, groups, length, width)
        .reshape(batch, heads * groups, length, width)
    )


def eager_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    *,
    attention_mask: Tensor,
    scale: float,
    dropout_p: float,
    training: bool,
    return_lse: bool = False,
    empty_rows: Tensor | None = None,
) -> Tensor | tuple[Tensor, Tensor]:
    """Return [batch, query_heads, query_length, value_dim].

    Inputs use [batch, heads, sequence, dim]; grouped KV heads are supported.
    The caller supplies the complete visibility policy as a boolean mask:
    True means visible. Masks must be rank four and broadcast to [batch, heads, Q, K].
    Fully blocked rows return zero with zero gradients. No causal/window policy
    is inferred, and additive float biases are not supported. Prepared callers may
    supply empty_rows = ~attention_mask.any(-1, keepdim=True) to reuse the reduction.
    """
    positive_number("scale", scale)
    finite_number("dropout_p", dropout_p)
    if not 0 <= dropout_p < 1:
        raise ValueError("dropout_p must be in [0, 1)")
    boolean("training", training)
    if any(tensor.ndim != 4 for tensor in (query, key, value)):
        raise ValueError("query, key, and value must have rank four")
    batch, heads, queries, width = query.shape
    kv_heads, keys = key.shape[1:3]
    if (
        heads == 0
        or kv_heads == 0
        or width == 0
        or heads % kv_heads
        or key.shape[0] != batch
        or value.shape[:3] != key.shape[:3]
        or key.shape[-1] != width
    ):
        raise ValueError(
            "incompatible query/key/value dimensions or grouped head counts"
        )
    if any(
        tensor.device != query.device
        or tensor.dtype != query.dtype
        or not tensor.is_floating_point()
        for tensor in (query, key, value)
    ):
        raise ValueError("query/key/value must share a floating dtype and device")
    if not isinstance(attention_mask, Tensor):
        raise TypeError("eager attention requires a boolean tensor mask")
    target = (batch, heads, queries, keys)
    if attention_mask.ndim != 4 or any(
        size not in (1, expected)
        for size, expected in zip(attention_mask.shape, target)
    ):
        raise ValueError("attention_mask must broadcast to [batch, heads, Q, K]")
    if attention_mask.device != query.device or attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must be boolean on the query device")

    groups = heads // kv_heads
    scores = torch.matmul(query, _repeat_kv(key, groups).transpose(-1, -2)) * scale
    scores = scores.masked_fill(~attention_mask, -torch.inf)
    if empty_rows is None:
        empty_rows = ~attention_mask.any(dim=-1, keepdim=True)
    elif (
        empty_rows.shape != (*attention_mask.shape[:-1], 1)
        or empty_rows.dtype != torch.bool
        or empty_rows.device != query.device
    ):
        raise ValueError(
            "empty_rows must be boolean with shape [B, H, Q, 1] matching the mask"
        )
    # Avoid NaN softmax/gradients for fully blocked rows. Unmasked numerical
    # overflow remains visible rather than being treated as an empty row.
    scores = scores.masked_fill(empty_rows, 0)
    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(query.dtype)
    weights = weights.masked_fill(empty_rows, 0)
    weights = F.dropout(weights, p=dropout_p, training=training)
    output = torch.matmul(weights, _repeat_kv(value, groups))
    if return_lse:
        lse = (
            scores.float()
            .logsumexp(dim=-1)
            .masked_fill(empty_rows.squeeze(-1), -torch.inf)
        )
        return output, lse
    return output
