"""Target-context attention for LongSpark training.

Each sampled anchor attends to three sources:

* the global memory: learned queries that summarize the anchor's full target
  prefix into a fixed number of slots;
* the recent window: the last ``window`` rows of target K/V before the anchor;
* the draft block itself.

Captured target K/V is stored once per source sequence and shared by all of its
anchors. The three sources are combined with an exact softmax merge, so no
per-anchor K/V tensor is ever built.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from .longspark_kernels import recent_window_attention

# Temporary-memory bounds for reading the global memory over long prefixes.
# They only change how the computation is chunked, not its result.
_ANCHOR_CHUNK_SIZE = 32
_KEY_CHUNK_THRESHOLD = 2048
_KEY_CHUNK_SIZE = 512


@dataclass(frozen=True)
class SoftmaxState:
    """Unnormalized online-softmax state of one attention read."""

    max_logits: torch.Tensor
    normalizer: torch.Tensor
    weighted_values: torch.Tensor

    def output(self) -> torch.Tensor:
        return (
            self.weighted_values
            / self.normalizer.clamp_min(torch.finfo(self.weighted_values.dtype).tiny)[
                ..., None
            ]
        )

    def normalized(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(output, logsumexp)``; rows without visible keys get ``-inf``."""
        visible = self.normalizer > 0
        safe_normalizer = self.normalizer.clamp_min(
            torch.finfo(self.normalizer.dtype).tiny
        )
        lse = self.max_logits + torch.log(safe_normalizer)
        lse = torch.where(visible, lse, torch.full_like(lse, float("-inf")))
        return self.output(), lse


def _grouped_attention(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
    key_chunk_size: Optional[int] = None,
) -> SoftmaxState:
    """GQA softmax state over ``[batch, layers, heads, rows, dim]`` inputs.

    K/V are read in their original storage and dtype; only scores and
    accumulators are promoted. ``key_chunk_size`` splits the keys into chunks
    whose states are merged exactly.
    """
    if query.ndim != 5 or keys.ndim != 5 or values.shape != keys.shape:
        raise ValueError("query must be rank 5 and keys/values must be matching rank 5")
    batch, layers, query_heads, query_rows, head_dim = query.shape
    key_batch, key_layers, kv_heads, key_rows, key_head_dim = keys.shape
    if (key_batch, key_layers, key_head_dim) != (batch, layers, head_dim):
        raise ValueError("query and K/V batch/layer/head dimensions must align")
    if query_heads % kv_heads:
        raise ValueError(
            "query heads must be divisible by KV heads for grouped attention"
        )
    if attention_mask.shape != (batch, query_rows, key_rows):
        raise ValueError(
            f"attention_mask must have shape {(batch, query_rows, key_rows)}, "
            f"got {tuple(attention_mask.shape)}"
        )
    if attention_mask.dtype != torch.bool:
        raise ValueError("attention_mask must have dtype torch.bool")
    if key_chunk_size is not None and key_chunk_size < key_rows:
        merged: Optional[SoftmaxState] = None
        for start in range(0, key_rows, key_chunk_size):
            stop = min(start + key_chunk_size, key_rows)
            part = _grouped_attention(
                query=query,
                keys=keys[..., start:stop, :],
                values=values[..., start:stop, :],
                attention_mask=attention_mask[..., start:stop],
            )
            merged = part if merged is None else _merge_softmax_states((merged, part))
        return merged

    # Cast the query rather than K/V so that K/V are never copied.
    query_for_scores = (
        query if query.dtype == keys.dtype else query.to(dtype=keys.dtype)
    )
    groups = query_heads // kv_heads
    grouped_query = query_for_scores.reshape(
        batch, layers, kv_heads, groups, query_rows, head_dim
    )
    scores = torch.einsum("blhgrd,blhtd->blhgrt", grouped_query, keys).reshape(
        batch, layers, query_heads, query_rows, key_rows
    )
    accumulator_dtype = (
        torch.float64
        if keys.dtype == torch.float64 or query.dtype == torch.float64
        else torch.float32
    )
    scores = scores.to(dtype=accumulator_dtype) * (head_dim**-0.5)
    expanded_mask = attention_mask[:, None, None, :, :]
    has_visible_key = attention_mask.any(dim=-1)[:, None, None, :]
    masked_scores = scores.masked_fill(~expanded_mask, float("-inf"))
    max_logits = masked_scores.max(dim=-1).values
    safe_max = torch.where(has_visible_key, max_logits, torch.zeros_like(max_logits))
    # Mask before exp: an invisible score can otherwise overflow and leave
    # inf * 0 in backward even though its forward weight is later zeroed.
    weights = torch.exp(masked_scores - safe_max[..., None])
    normalizer = weights.sum(dim=-1)
    grouped_weights = weights.to(dtype=values.dtype).reshape(
        batch, layers, kv_heads, groups, query_rows, key_rows
    )
    weighted_values = torch.einsum(
        "blhgrt,blhtd->blhgrd",
        grouped_weights,
        values,
    ).reshape(batch, layers, query_heads, query_rows, head_dim)
    weighted_values = weighted_values.to(dtype=accumulator_dtype)
    return SoftmaxState(
        max_logits=torch.where(
            has_visible_key, max_logits, torch.zeros_like(max_logits)
        ),
        normalizer=normalizer,
        weighted_values=weighted_values,
    )


def _grouped_attention_single_layer(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    attention_mask: torch.Tensor,
) -> SoftmaxState:
    """:func:`_grouped_attention` for ``[batch, heads, rows, dim]`` inputs."""
    if query.ndim != 4 or keys.ndim != 4 or values.shape != keys.shape:
        raise ValueError("4-D query/K/V tensors are required")
    state = _grouped_attention(
        query=query[:, None],
        keys=keys[:, None],
        values=values[:, None],
        attention_mask=attention_mask,
    )
    return SoftmaxState(
        max_logits=state.max_logits[:, 0],
        normalizer=state.normalizer[:, 0],
        weighted_values=state.weighted_values[:, 0],
    )


def _merge_softmax_states(parts: Sequence[SoftmaxState]) -> SoftmaxState:
    max_logits = torch.zeros_like(parts[0].max_logits)
    has_visible_key = torch.zeros_like(parts[0].normalizer, dtype=torch.bool)
    for part in parts:
        visible = part.normalizer > 0
        first_visible = visible & ~has_visible_key
        max_logits = torch.where(first_visible, part.max_logits, max_logits)
        max_logits = torch.where(
            visible & has_visible_key,
            torch.maximum(max_logits, part.max_logits),
            max_logits,
        )
        has_visible_key = has_visible_key | visible
    normalizer = torch.zeros_like(parts[0].normalizer)
    weighted_values = torch.zeros_like(parts[0].weighted_values)
    for part in parts:
        visible = part.normalizer > 0
        # Empty chunks have a placeholder maximum of zero. Exclude them
        # before exp, especially when every visible score is very negative.
        scale = torch.exp(
            (part.max_logits - max_logits).masked_fill(~visible, float("-inf"))
        )
        normalizer = normalizer + scale * part.normalizer
        weighted_values = weighted_values + scale[..., None] * part.weighted_values
    return SoftmaxState(max_logits, normalizer, weighted_values)


def _merge_normalized(
    parts: Sequence[tuple[torch.Tensor, torch.Tensor]],
) -> torch.Tensor:
    """Merge normalized ``(output, logsumexp)`` reads into one attention output.

    Sources with ``-inf`` logsumexp (no visible keys) contribute zero weight.
    """
    first_output = parts[0][0]
    accumulator_dtype = (
        torch.float64
        if any(
            output.dtype == torch.float64 or lse.dtype == torch.float64
            for output, lse in parts
        )
        else torch.float32
    )
    lse_stack = torch.stack(
        [lse.to(dtype=accumulator_dtype) for _, lse in parts], dim=0
    )
    merged_lse = torch.logsumexp(lse_stack, dim=0)
    safe_lse = torch.where(
        torch.isfinite(merged_lse), merged_lse, torch.zeros_like(merged_lse)
    )
    merged_output = torch.zeros_like(first_output, dtype=accumulator_dtype)
    for output, lse in parts:
        contribution = torch.where(
            torch.isfinite(lse),
            torch.exp(lse.to(dtype=accumulator_dtype) - safe_lse),
            torch.zeros_like(safe_lse),
        )
        merged_output = merged_output + contribution[..., None] * output.to(
            dtype=accumulator_dtype
        )
    return merged_output.to(dtype=first_output.dtype)


class GlobalMemory(nn.Module):
    """Learned queries that summarize the full target prefix per draft layer."""

    def __init__(
        self, *, num_layers: int, num_heads: int, num_slots: int, head_dim: int
    ):
        super().__init__()
        self.queries = nn.Parameter(
            torch.empty(num_layers, num_heads, num_slots, head_dim)
        )
        nn.init.normal_(self.queries, mean=0.0, std=head_dim**-0.5)

    @property
    def num_slots(self) -> int:
        return self.queries.shape[2]

    def expand(
        self, batch_size: int, *, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return (
            self.queries[None]
            .expand(batch_size, -1, -1, -1, -1)
            .to(device=device, dtype=dtype)
        )

    def read(
        self,
        *,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        prefix_mask: torch.Tensor,
        key_chunk_size: Optional[int] = None,
    ) -> SoftmaxState:
        """Attend from ``query`` to the visible prefix of ``keys``/``values``."""
        mask = prefix_mask[:, None, :].expand(-1, self.num_slots, keys.shape[-2])
        return _grouped_attention(
            query=query,
            keys=keys,
            values=values,
            attention_mask=mask,
            key_chunk_size=key_chunk_size,
        )


class TargetContext:
    """Captured target K/V for one batch, shared by all draft layers."""

    def __init__(
        self,
        *,
        global_memory: GlobalMemory,
        keys: torch.Tensor,
        values: torch.Tensor,
        prefix_mask: torch.Tensor,
        source_indices: torch.Tensor,
        window: int,
    ) -> None:
        """
        Args:
            keys/values: ``[sources, layers, kv_heads, tokens, head_dim]``.
            prefix_mask: ``[anchors, tokens]``, the target tokens visible to
                each anchor.
            source_indices: ``[anchors]``, the source sequence of each anchor.
            window: number of most recent target rows read directly.
        """
        if keys.shape != values.shape or keys.ndim != 5:
            raise ValueError(
                "target K/V must be matching [sources, layers, heads, tokens, dim]"
            )
        anchors = int(source_indices.numel())
        if anchors <= 0 or source_indices.ndim != 1:
            raise ValueError("source_indices must be a non-empty rank-1 tensor")
        if prefix_mask.shape != (anchors, keys.shape[-2]):
            raise ValueError("prefix_mask must be [anchors, target_tokens]")
        if prefix_mask.dtype != torch.bool:
            raise ValueError("prefix_mask must use dtype torch.bool")
        if torch.any(source_indices < 0) or torch.any(source_indices >= keys.shape[0]):
            raise ValueError("source_indices contains an invalid source")
        if not prefix_mask.any(dim=-1).all():
            raise ValueError("every anchor needs a visible target prefix")

        self.global_memory = global_memory
        self.keys = keys
        self.values = values
        self.prefix_mask = prefix_mask
        self.window = window
        self.source_indices = source_indices.to(device=keys.device, dtype=torch.long)
        self.global_queries = global_memory.expand(
            anchors, device=keys.device, dtype=keys.dtype
        )
        self.memory = self._read_global_memory()

    @property
    def batch_size(self) -> int:
        return int(self.source_indices.numel())

    def _source_rows(self):
        for source_index in self.source_indices.unique(sorted=True).tolist():
            rows = torch.nonzero(
                self.source_indices == int(source_index), as_tuple=False
            ).flatten()
            yield int(source_index), rows

    def _source_kv(
        self,
        *,
        source_index: int,
        count: int,
        layer_index: Optional[int] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Expanded views of one source's K/V; nothing is copied.
        if layer_index is None:
            source_keys = self.keys[source_index : source_index + 1]
            source_values = self.values[source_index : source_index + 1]
        else:
            source_keys = self.keys[source_index : source_index + 1, layer_index]
            source_values = self.values[source_index : source_index + 1, layer_index]
        return (
            source_keys.expand(count, *source_keys.shape[1:]),
            source_values.expand(count, *source_values.shape[1:]),
        )

    def _read_global_memory(self) -> SoftmaxState:
        """Read the global memory of every anchor over its full prefix."""
        batch, layers, heads, slots, head_dim = self.global_queries.shape
        accumulator_dtype = (
            torch.float64
            if self.global_queries.dtype == torch.float64
            else torch.float32
        )
        state_max = self.global_queries.new_empty(
            (batch, layers, heads, slots), dtype=accumulator_dtype
        )
        state_normalizer = torch.empty_like(state_max)
        state_weighted_values = self.global_queries.new_empty(
            (batch, layers, heads, slots, head_dim), dtype=accumulator_dtype
        )
        for source_index, rows in self._source_rows():
            for row_chunk in rows.split(_ANCHOR_CHUNK_SIZE):
                group_keys, group_values = self._source_kv(
                    source_index=source_index, count=row_chunk.numel()
                )
                group_query = self.global_queries.index_select(0, row_chunk)
                group_prefix_mask = self.prefix_mask.index_select(0, row_chunk)
                key_chunk_size = (
                    _KEY_CHUNK_SIZE
                    if group_keys.shape[-2] > _KEY_CHUNK_THRESHOLD
                    else None
                )
                if key_chunk_size is None or not (
                    torch.is_grad_enabled() and group_query.requires_grad
                ):
                    group_state = self.global_memory.read(
                        query=group_query,
                        keys=group_keys,
                        values=group_values,
                        prefix_mask=group_prefix_mask,
                        key_chunk_size=key_chunk_size,
                    )
                else:
                    # Recompute chunked reads in backward to bound activations.
                    def read_chunked(query, keys, values, prefix_mask):
                        state = self.global_memory.read(
                            query=query,
                            keys=keys,
                            values=values,
                            prefix_mask=prefix_mask,
                            key_chunk_size=key_chunk_size,
                        )
                        return (
                            state.max_logits,
                            state.normalizer,
                            state.weighted_values,
                        )

                    group_state = SoftmaxState(
                        *checkpoint(
                            read_chunked,
                            group_query,
                            group_keys,
                            group_values,
                            group_prefix_mask,
                            use_reentrant=False,
                        )
                    )
                state_max = state_max.index_copy(0, row_chunk, group_state.max_logits)
                state_normalizer = state_normalizer.index_copy(
                    0, row_chunk, group_state.normalizer
                )
                state_weighted_values = state_weighted_values.index_copy(
                    0, row_chunk, group_state.weighted_values
                )
        return SoftmaxState(state_max, state_normalizer, state_weighted_values)

    def attend(
        self,
        *,
        layer_index: int,
        query: torch.Tensor,
        block_keys: torch.Tensor,
        block_values: torch.Tensor,
        block_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Attend to the global memory, the recent window and the draft block."""
        if query.shape[0] != self.batch_size:
            raise ValueError("draft query does not match the number of anchors")
        if block_keys.shape != block_values.shape or block_keys.ndim != 4:
            raise ValueError("draft-block K/V must be matching rank-4 tensors")
        if block_attention_mask.shape != (
            self.batch_size,
            query.shape[-2],
            block_keys.shape[-2],
        ):
            raise ValueError("draft-block attention mask has an invalid shape")

        output = torch.empty_like(query)
        for source_index, rows in self._source_rows():
            group_query = query.index_select(0, rows)
            group_prefix_mask = self.prefix_mask.index_select(0, rows)
            lengths = group_prefix_mask.sum(dim=-1, dtype=torch.long)
            group_global_query = self.global_queries.index_select(0, rows)[
                :, layer_index
            ]
            group_global_values = self.memory.output().index_select(0, rows)[
                :, layer_index
            ]
            global_output, global_lse = _grouped_attention_single_layer(
                query=group_query,
                keys=group_global_query,
                values=group_global_values.to(dtype=group_global_query.dtype),
                attention_mask=torch.ones(
                    (
                        rows.numel(),
                        group_query.shape[-2],
                        self.global_memory.num_slots,
                    ),
                    dtype=torch.bool,
                    device=group_query.device,
                ),
            ).normalized()
            block_state = _grouped_attention_single_layer(
                query=group_query,
                keys=block_keys.index_select(0, rows),
                values=block_values.index_select(0, rows),
                attention_mask=block_attention_mask.index_select(0, rows),
            )
            if group_query.is_cuda:
                window_output, window_lse = recent_window_attention(
                    query=group_query,
                    keys=self.keys[:, layer_index],
                    values=self.values[:, layer_index],
                    source_indices=self.source_indices.index_select(0, rows),
                    prefix_lengths=lengths,
                    window=self.window,
                    scale=group_query.shape[-1] ** -0.5,
                )
            else:
                # Dense PyTorch fallback for CPU runs and tests.
                group_keys, group_values = self._source_kv(
                    source_index=source_index,
                    count=rows.numel(),
                    layer_index=layer_index,
                )
                token_positions = torch.arange(
                    group_prefix_mask.shape[-1], device=group_prefix_mask.device
                )
                window_mask = group_prefix_mask & token_positions[None, :].ge(
                    (lengths - self.window).clamp_min(0)[:, None]
                )
                window_output, window_lse = _grouped_attention_single_layer(
                    query=group_query,
                    keys=group_keys,
                    values=group_values,
                    attention_mask=window_mask[:, None, :].expand(
                        -1, group_query.shape[-2], -1
                    ),
                ).normalized()
            block_output, block_lse = block_state.normalized()
            merged_output = _merge_normalized(
                (
                    (global_output, global_lse),
                    (window_output, window_lse),
                    (block_output, block_lse),
                )
            )
            output = output.index_copy(0, rows, merged_output.to(dtype=query.dtype))
        return output
