"""Triton kernels for LongSpark's recent-window attention.

Each anchor attends to the last ``window`` rows of its own source sequence in
the captured target K/V cache. Rows are addressed through ``source_indices`` and
``prefix_lengths``, so no per-anchor copy of the window is materialized. Target
K/V is frozen, so the backward kernel only computes the query gradient.
"""

from __future__ import annotations

from typing import Optional

import torch

try:  # Triton is optional; CPU runs use the PyTorch path in longspark_context.
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in CPU-only installs.
    triton = None
    tl = None


_BLOCK_N = 32


if triton is not None:

    @triton.jit
    def _window_attention_fwd_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        source_ptr,
        length_ptr,
        output_ptr,
        lse_ptr,
        stride_qa,
        stride_qh,
        stride_qr,
        stride_qd,
        stride_ks,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vs,
        stride_vh,
        stride_vt,
        stride_vd,
        QUERY_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        QUERY_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        query_row = row_id % QUERY_ROWS
        anchor_head = row_id // QUERY_ROWS
        query_head = anchor_head % QUERY_HEADS
        anchor = anchor_head // QUERY_HEADS
        kv_head = query_head // (QUERY_HEADS // KV_HEADS)

        source = tl.load(source_ptr + anchor)
        prefix_length = tl.load(length_ptr + anchor)
        window_start = tl.maximum(prefix_length - WINDOW, 0)
        window_length = prefix_length - window_start

        offsets_d = tl.arange(0, BLOCK_D)
        dimension_mask = offsets_d < HEAD_DIM
        query_offsets = (
            anchor * stride_qa
            + query_head * stride_qh
            + query_row * stride_qr
            + offsets_d * stride_qd
        )
        query = tl.load(query_ptr + query_offsets, mask=dimension_mask, other=0.0)
        query = query.to(tl.float32)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)
        offsets_n = tl.arange(0, BLOCK_N)

        for token_begin in range(0, WINDOW, BLOCK_N):
            token_offsets = token_begin + offsets_n
            token_mask = token_offsets < window_length
            token_positions = window_start + token_offsets
            key_offsets = (
                source * stride_ks
                + kv_head * stride_kh
                + token_positions[:, None] * stride_kt
                + offsets_d[None, :] * stride_kd
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            scores = tl.where(token_mask, scores, -float("inf"))

            block_max = tl.max(scores, axis=0)
            next_max = tl.maximum(running_max, block_max)
            previous_scale = tl.exp(running_max - next_max)
            probabilities = tl.exp(scores - next_max)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            value_offsets = (
                source * stride_vs
                + kv_head * stride_vh
                + token_positions[:, None] * stride_vt
                + offsets_d[None, :] * stride_vd
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            accumulator = accumulator * previous_scale + tl.sum(
                probabilities[:, None] * values,
                axis=0,
            )
            running_sum = running_sum * previous_scale + tl.sum(
                probabilities,
                axis=0,
            )
            running_max = next_max

        output = accumulator / running_sum
        output_offsets = row_id * HEAD_DIM + offsets_d
        tl.store(output_ptr + output_offsets, output, mask=dimension_mask)
        tl.store(lse_ptr + row_id, running_max + tl.log(running_sum))

    @triton.jit
    def _window_attention_bwd_dq_kernel(
        query_ptr,
        key_ptr,
        value_ptr,
        source_ptr,
        length_ptr,
        output_ptr,
        lse_ptr,
        grad_output_ptr,
        grad_lse_ptr,
        grad_query_ptr,
        stride_qa,
        stride_qh,
        stride_qr,
        stride_qd,
        stride_ks,
        stride_kh,
        stride_kt,
        stride_kd,
        stride_vs,
        stride_vh,
        stride_vt,
        stride_vd,
        QUERY_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        QUERY_ROWS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        WINDOW: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_D: tl.constexpr,
        BLOCK_N: tl.constexpr,
    ):
        row_id = tl.program_id(0)
        query_row = row_id % QUERY_ROWS
        anchor_head = row_id // QUERY_ROWS
        query_head = anchor_head % QUERY_HEADS
        anchor = anchor_head // QUERY_HEADS
        kv_head = query_head // (QUERY_HEADS // KV_HEADS)

        source = tl.load(source_ptr + anchor)
        prefix_length = tl.load(length_ptr + anchor)
        window_start = tl.maximum(prefix_length - WINDOW, 0)
        window_length = prefix_length - window_start

        offsets_d = tl.arange(0, BLOCK_D)
        dimension_mask = offsets_d < HEAD_DIM
        query_offsets = (
            anchor * stride_qa
            + query_head * stride_qh
            + query_row * stride_qr
            + offsets_d * stride_qd
        )
        query = tl.load(query_ptr + query_offsets, mask=dimension_mask, other=0.0)
        query = query.to(tl.float32)
        output_offsets = row_id * HEAD_DIM + offsets_d
        output = tl.load(output_ptr + output_offsets, mask=dimension_mask, other=0.0)
        grad_output = tl.load(
            grad_output_ptr + output_offsets,
            mask=dimension_mask,
            other=0.0,
        )
        logsumexp = tl.load(lse_ptr + row_id)
        grad_logsumexp = tl.load(grad_lse_ptr + row_id)
        output_dot = tl.sum(grad_output * output, axis=0)

        grad_query = tl.zeros((BLOCK_D,), dtype=tl.float32)
        offsets_n = tl.arange(0, BLOCK_N)
        for token_begin in range(0, WINDOW, BLOCK_N):
            token_offsets = token_begin + offsets_n
            token_mask = token_offsets < window_length
            token_positions = window_start + token_offsets
            key_offsets = (
                source * stride_ks
                + kv_head * stride_kh
                + token_positions[:, None] * stride_kt
                + offsets_d[None, :] * stride_kd
            )
            keys = tl.load(
                key_ptr + key_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            scores = tl.sum(keys * query[None, :], axis=1) * SCALE
            probabilities = tl.exp(scores - logsumexp)
            probabilities = tl.where(token_mask, probabilities, 0.0)

            value_offsets = (
                source * stride_vs
                + kv_head * stride_vh
                + token_positions[:, None] * stride_vt
                + offsets_d[None, :] * stride_vd
            )
            values = tl.load(
                value_ptr + value_offsets,
                mask=token_mask[:, None] & dimension_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            value_dot = tl.sum(values * grad_output[None, :], axis=1)
            score_gradient = probabilities * (value_dot - output_dot + grad_logsumexp)
            grad_query += tl.sum(score_gradient[:, None] * keys, axis=0) * SCALE

        tl.store(grad_query_ptr + output_offsets, grad_query, mask=dimension_mask)


def _validate_inputs(
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    source_indices: torch.Tensor,
    prefix_lengths: torch.Tensor,
    window: int,
) -> None:
    if triton is None:
        raise RuntimeError("recent-window attention requires Triton")
    if not query.is_cuda:
        raise ValueError("recent-window attention requires CUDA tensors")
    if keys.device != query.device or values.device != query.device:
        raise ValueError("query and K/V must be on one CUDA device")
    if keys.shape != values.shape or keys.ndim != 4:
        raise ValueError("K/V must be matching [sources, heads, tokens, dim]")
    if query.ndim != 4:
        raise ValueError("query must be [anchors, heads, rows, dim]")
    anchors, query_heads, _, head_dim = query.shape
    if keys.shape[-1] != head_dim:
        raise ValueError("query and K/V head dimensions must match")
    if query_heads % keys.shape[1]:
        raise ValueError("query heads must be divisible by K/V heads")
    if query.dtype != keys.dtype or keys.dtype != values.dtype:
        raise ValueError("query and K/V must use one dtype")
    if query.dtype not in {torch.bfloat16, torch.float16, torch.float32}:
        raise ValueError("recent-window attention supports bf16, fp16 and fp32")
    if source_indices.shape != (anchors,) or prefix_lengths.shape != (anchors,):
        raise ValueError("source indices and lengths must have one row per anchor")
    if source_indices.device != query.device or prefix_lengths.device != query.device:
        raise ValueError("index metadata must be on the query device")
    if source_indices.dtype not in {torch.int32, torch.int64}:
        raise ValueError("source indices must be integral")
    if prefix_lengths.dtype not in {torch.int32, torch.int64}:
        raise ValueError("prefix lengths must be integral")
    if window <= 0:
        raise ValueError("window must be positive")
    if window % _BLOCK_N:
        raise ValueError(f"window must be a multiple of {_BLOCK_N}")
    # Keep validation on the CUDA stream.  Converting either reduction to a
    # Python bool would synchronize every selected draft layer with the host.
    torch._assert_async(
        torch.all((source_indices >= 0) & (source_indices < keys.shape[0])),
        "source index is out of range",
    )
    torch._assert_async(
        torch.all((prefix_lengths > 0) & (prefix_lengths <= keys.shape[-2])),
        "prefix length is out of range",
    )


class _RecentWindowAttention(torch.autograd.Function):
    """Autograd wrapper for the recent-window kernels.

    Target K/V are frozen by the caller, so backward returns only the query
    gradient. ``output`` and ``logsumexp`` stay in FP32 for the softmax merge.
    """

    @staticmethod
    def forward(
        ctx,
        query: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        source_indices: torch.Tensor,
        prefix_lengths: torch.Tensor,
        window: int,
        scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _validate_inputs(
            query,
            keys,
            values,
            source_indices,
            prefix_lengths,
            window,
        )
        anchors, query_heads, query_rows, head_dim = query.shape
        output = torch.empty(query.shape, device=query.device, dtype=torch.float32)
        logsumexp = torch.empty(
            (anchors, query_heads, query_rows),
            device=query.device,
            dtype=torch.float32,
        )
        block_d = triton.next_power_of_2(head_dim)
        grid = (anchors * query_heads * query_rows,)
        _window_attention_fwd_kernel[grid](
            query,
            keys,
            values,
            source_indices,
            prefix_lengths,
            output,
            logsumexp,
            *query.stride(),
            *keys.stride(),
            *values.stride(),
            QUERY_HEADS=query_heads,
            KV_HEADS=keys.shape[1],
            QUERY_ROWS=query_rows,
            HEAD_DIM=head_dim,
            WINDOW=window,
            SCALE=scale,
            BLOCK_D=block_d,
            BLOCK_N=_BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        ctx.save_for_backward(
            query,
            keys,
            values,
            source_indices,
            prefix_lengths,
            output,
            logsumexp,
        )
        ctx.window = int(window)
        ctx.scale = float(scale)
        return output, logsumexp

    @staticmethod
    def backward(
        ctx,
        grad_output: Optional[torch.Tensor],
        grad_logsumexp: Optional[torch.Tensor],
    ):
        (
            query,
            keys,
            values,
            source_indices,
            prefix_lengths,
            output,
            logsumexp,
        ) = ctx.saved_tensors
        if grad_output is None:
            grad_output = torch.zeros_like(output)
        else:
            grad_output = grad_output.contiguous()
        if grad_logsumexp is None:
            grad_logsumexp = torch.zeros_like(logsumexp)
        else:
            grad_logsumexp = grad_logsumexp.contiguous()
        grad_query = torch.empty(
            query.shape,
            device=query.device,
            dtype=torch.float32,
        )
        anchors, query_heads, query_rows, head_dim = query.shape
        block_d = triton.next_power_of_2(head_dim)
        grid = (anchors * query_heads * query_rows,)
        _window_attention_bwd_dq_kernel[grid](
            query,
            keys,
            values,
            source_indices,
            prefix_lengths,
            output,
            logsumexp,
            grad_output,
            grad_logsumexp,
            grad_query,
            *query.stride(),
            *keys.stride(),
            *values.stride(),
            QUERY_HEADS=query_heads,
            KV_HEADS=keys.shape[1],
            QUERY_ROWS=query_rows,
            HEAD_DIM=head_dim,
            WINDOW=ctx.window,
            SCALE=ctx.scale,
            BLOCK_D=block_d,
            BLOCK_N=_BLOCK_N,
            num_warps=4,
            num_stages=2,
        )
        return (
            grad_query.to(dtype=query.dtype),
            None,
            None,
            None,
            None,
            None,
            None,
        )


def recent_window_attention(
    *,
    query: torch.Tensor,
    keys: torch.Tensor,
    values: torch.Tensor,
    source_indices: torch.Tensor,
    prefix_lengths: torch.Tensor,
    window: int,
    scale: Optional[float] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Attend to the last ``window`` target rows before each anchor.

    Args:
        query: ``[anchors, query_heads, query_rows, head_dim]``.
        keys/values: ``[sources, kv_heads, tokens, head_dim]``.
        source_indices: source sequence of every anchor.
        prefix_lengths: number of visible target tokens for every anchor.

    Returns:
        A pair ``(output, natural_logsumexp)`` in FP32.
    """

    if scale is None:
        scale = query.shape[-1] ** -0.5
    return _RecentWindowAttention.apply(
        query,
        keys.detach(),
        values.detach(),
        source_indices,
        prefix_lengths,
        int(window),
        float(scale),
    )


__all__ = ["recent_window_attention"]
