"""Triton implementation of the DFlash2 grouped dynamic convolution.

For flattened token rows ``r`` (block position ``i = r % block_size``),
channels ``h`` and groups ``g = h // group_size``, the convolution is::

    out[r, h] = sum_{t <= i} (base[t, h] + delta[r, t, g]) * x[r - t, h]

The eager module materializes ``base + delta`` at full
``[rows, taps, hidden]`` size, and autograd keeps it alive until backward.
These kernels form each coefficient in registers instead, accumulate in FP32
and round once on store. Backward recomputes from the saved input and dynamic
kernel, and writes only the three input gradients:

    grad_x[r, h] = sum_{i + t < block_size} (base[t, h] + delta[r + t, t, g])
                   * grad_out[r + t, h]
    grad_delta[r, t, g] = sum_{h in g} grad_out[r, h] * x[r - t, h]
    grad_base[t, h] = sum_{r : i >= t} grad_out[r, h] * x[r - t, h]

``grad_base`` is reduced per row tile into an FP32 buffer and summed on the
host stream, which keeps the result deterministic.
"""

import torch
import triton
import triton.language as tl

__all__ = ["dflash2_grouped_conv_fused", "supports_group_size"]

# Tiles hold 4096 elements (64 FP32 values per thread at two warps); larger
# tiles spill registers in the backward kernel on H200. Channel widths are
# minimums: a wider group gets one group per program and fewer rows.
_TILE_ELEMENTS = 4096
_FORWARD_BLOCK_H = 128
_BACKWARD_BLOCK_H = 64
_NUM_WARPS = 2
_MAX_GROUP_SIZE = 1024


def supports_group_size(group_size: int) -> bool:
    """Return whether the kernels can tile ``group_size`` channels per group."""
    return 1 <= group_size <= _MAX_GROUP_SIZE and (group_size & (group_size - 1)) == 0


def dflash2_grouped_conv_fused(hidden_states, delta, base, block_size, group_size):
    """Apply the grouped dynamic convolution with the fused autograd kernels.

    Tensor layout:
        ``hidden_states``: ``[batch, seq_len, hidden]``
        ``delta``: ``[batch, seq_len, taps, hidden // group_size]``; any row
        stride is accepted as long as the group dimension is contiguous
        ``base``: ``[taps, hidden]``
    """
    return _DFlash2GroupedConv.apply(
        hidden_states,
        delta,
        base,
        block_size,
        group_size,
    )


class _DFlash2GroupedConv(torch.autograd.Function):
    """Convolve without saving the full per-token coefficient tensor.

    Saves only the input rows and the dynamic kernel (``hidden // group_size``
    values per tap and row) and recomputes every coefficient in backward.
    """

    @staticmethod
    def forward(ctx, hidden_states, delta, base, block_size, group_size):
        batch_size, sequence_length, hidden_size = hidden_states.shape
        taps = base.shape[0]
        num_rows = batch_size * sequence_length
        num_groups = hidden_size // group_size

        rows = hidden_states.reshape(num_rows, hidden_size).contiguous()
        dynamic = _as_row_view(delta, num_rows, taps, num_groups)
        base = base.contiguous()
        output = torch.empty_like(rows)

        block_m, block_h, num_warps = _calculate_grouped_conv_settings(
            group_size, _FORWARD_BLOCK_H
        )
        grid = (triton.cdiv(num_rows, block_m), triton.cdiv(hidden_size, block_h))
        _dflash2_grouped_conv_forward_kernel[grid](
            rows,
            dynamic,
            base,
            output,
            num_rows,
            hidden_size,
            num_groups,
            dynamic.stride(0),
            dynamic.stride(1),
            BLOCK_SIZE=block_size,
            TAPS=taps,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            GROUPS_PER_TILE=block_h // group_size,
            num_warps=num_warps,
        )

        ctx.block_size = block_size
        ctx.group_size = group_size
        ctx.delta_shape = delta.shape
        ctx.save_for_backward(rows, dynamic, base)
        return output.view(batch_size, sequence_length, hidden_size)

    @staticmethod
    def backward(ctx, grad_output):
        rows, dynamic, base = ctx.saved_tensors
        num_rows, hidden_size = rows.shape
        taps = base.shape[0]
        group_size = ctx.group_size
        num_groups = hidden_size // group_size

        grad_rows = grad_output.reshape(num_rows, hidden_size).contiguous()
        grad_hidden = torch.empty_like(rows)
        grad_delta = torch.empty(
            num_rows,
            taps,
            num_groups,
            device=rows.device,
            dtype=dynamic.dtype,
        )
        block_m, block_h, num_warps = _calculate_grouped_conv_settings(
            group_size, _BACKWARD_BLOCK_H
        )
        num_row_tiles = triton.cdiv(num_rows, block_m)
        grad_base_partials = torch.empty(
            num_row_tiles,
            taps,
            hidden_size,
            device=rows.device,
            dtype=torch.float32,
        )

        grid = (num_row_tiles, triton.cdiv(hidden_size, block_h))
        _dflash2_grouped_conv_backward_kernel[grid](
            rows,
            dynamic,
            base,
            grad_rows,
            grad_hidden,
            grad_delta,
            grad_base_partials,
            num_rows,
            hidden_size,
            num_groups,
            dynamic.stride(0),
            dynamic.stride(1),
            BLOCK_SIZE=ctx.block_size,
            TAPS=taps,
            GROUP_SIZE=group_size,
            BLOCK_M=block_m,
            GROUPS_PER_TILE=block_h // group_size,
            num_warps=num_warps,
        )

        grad_base = grad_base_partials.sum(dim=0).to(base.dtype)
        return (
            grad_hidden.view(grad_output.shape),
            grad_delta.view(ctx.delta_shape),
            grad_base,
            None,
            None,
        )


def _as_row_view(delta, num_rows, taps, num_groups):
    """View ``delta`` as ``[rows, taps, groups]`` with unit group stride."""
    dynamic = delta.reshape(num_rows, taps, num_groups)
    if dynamic.stride(2) != 1:
        dynamic = dynamic.contiguous()
    return dynamic


def _calculate_grouped_conv_settings(group_size, min_block_h):
    """Choose the row tile, channel tile and warp count for the GPU backend."""
    block_h = max(min_block_h, group_size)
    block_m = _TILE_ELEMENTS // block_h
    num_warps = _NUM_WARPS

    # Preserve the NVIDIA thread count on AMD targets with 64-lane wavefronts.
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        num_warps = num_warps * 32 // warp_size

    return block_m, block_h, max(num_warps, 1)


@triton.jit
def _dflash2_grouped_conv_forward_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    out_ptr,
    num_rows,
    hidden_size,
    num_groups,
    delta_row_stride,
    delta_tap_stride,
    BLOCK_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
):
    """Write one ``[BLOCK_M, GROUPS_PER_TILE, GROUP_SIZE]`` output tile.

    The tile is 3D so each dynamic coefficient is loaded once per row and
    group, then broadcast across the group's channels in registers.
    """
    rows = tl.program_id(0).to(tl.int64) * BLOCK_M + tl.arange(0, BLOCK_M)
    groups = tl.program_id(1) * GROUPS_PER_TILE + tl.arange(0, GROUPS_PER_TILE)
    channels = groups[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    row_mask = rows < num_rows
    group_mask = groups < num_groups
    positions = rows % BLOCK_SIZE

    output = tl.zeros((BLOCK_M, GROUPS_PER_TILE, GROUP_SIZE), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        # Tap ``t`` reads ``x[row - t]`` only from inside the same block.
        tap_rows = row_mask & (positions >= tap)
        dynamic = tl.load(
            delta_ptr
            + rows[:, None] * delta_row_stride
            + tap * delta_tap_stride
            + groups[None, :],
            mask=tap_rows[:, None] & group_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        base = tl.load(
            base_ptr + tap * hidden_size + channels,
            mask=group_mask[:, None],
            other=0.0,
        ).to(tl.float32)
        source = tl.load(
            x_ptr + (rows - tap)[:, None, None] * hidden_size + channels[None, :, :],
            mask=tap_rows[:, None, None] & group_mask[None, :, None],
            other=0.0,
        ).to(tl.float32)
        output += (base[None, :, :] + dynamic[:, :, None]) * source

    tl.store(
        out_ptr + rows[:, None, None] * hidden_size + channels[None, :, :],
        output.to(out_ptr.dtype.element_ty),
        mask=row_mask[:, None, None] & group_mask[None, :, None],
    )


@triton.jit
def _dflash2_grouped_conv_backward_kernel(
    x_ptr,
    delta_ptr,
    base_ptr,
    grad_out_ptr,
    grad_x_ptr,
    grad_delta_ptr,
    grad_base_partial_ptr,
    num_rows,
    hidden_size,
    num_groups,
    delta_row_stride,
    delta_tap_stride,
    BLOCK_SIZE: tl.constexpr,
    TAPS: tl.constexpr,
    GROUP_SIZE: tl.constexpr,
    BLOCK_M: tl.constexpr,
    GROUPS_PER_TILE: tl.constexpr,
):
    """Write one tile of ``grad_x`` and ``grad_delta`` plus a ``grad_base`` partial.

    Row tile ``m`` stores its per-tap column sums at
    ``grad_base_partial[m, tap, :]``. Rows masked out of a tap contribute zero
    to both coefficient gradients, matching the eager zero padding.
    """
    row_tile = tl.program_id(0).to(tl.int64)
    rows = row_tile * BLOCK_M + tl.arange(0, BLOCK_M)
    groups = tl.program_id(1) * GROUPS_PER_TILE + tl.arange(0, GROUPS_PER_TILE)
    channels = groups[:, None] * GROUP_SIZE + tl.arange(0, GROUP_SIZE)[None, :]
    row_mask = rows < num_rows
    group_mask = groups < num_groups
    positions = rows % BLOCK_SIZE
    tile_mask = row_mask[:, None, None] & group_mask[None, :, None]
    tile_offsets = rows[:, None, None] * hidden_size + channels[None, :, :]

    grad_out = tl.load(grad_out_ptr + tile_offsets, mask=tile_mask, other=0.0).to(
        tl.float32
    )
    grad_x = tl.zeros((BLOCK_M, GROUPS_PER_TILE, GROUP_SIZE), dtype=tl.float32)
    for tap in tl.static_range(TAPS):
        base = tl.load(
            base_ptr + tap * hidden_size + channels,
            mask=group_mask[:, None],
            other=0.0,
        ).to(tl.float32)

        # ``out[row + t]`` read ``x[row]`` through tap ``t``.
        target_rows = row_mask & (positions + tap < BLOCK_SIZE)
        target_dynamic = tl.load(
            delta_ptr
            + (rows + tap)[:, None] * delta_row_stride
            + tap * delta_tap_stride
            + groups[None, :],
            mask=target_rows[:, None] & group_mask[None, :],
            other=0.0,
        ).to(tl.float32)
        if tap == 0:
            target_grad = grad_out
        else:
            target_grad = tl.load(
                grad_out_ptr + tile_offsets + tap * hidden_size,
                mask=target_rows[:, None, None] & group_mask[None, :, None],
                other=0.0,
            ).to(tl.float32)
        grad_x += (base[None, :, :] + target_dynamic[:, :, None]) * target_grad

        # ``out[row]`` read ``x[row - t]`` through tap ``t``.
        source_rows = row_mask & (positions >= tap)
        source = tl.load(
            x_ptr + tile_offsets - tap * hidden_size,
            mask=source_rows[:, None, None] & group_mask[None, :, None],
            other=0.0,
        ).to(tl.float32)
        product = grad_out * source
        tl.store(
            grad_delta_ptr
            + (rows[:, None] * TAPS + tap) * num_groups
            + groups[None, :],
            tl.sum(product, axis=2).to(grad_delta_ptr.dtype.element_ty),
            mask=row_mask[:, None] & group_mask[None, :],
        )
        tl.store(
            grad_base_partial_ptr + (row_tile * TAPS + tap) * hidden_size + channels,
            tl.sum(product, axis=0),
            mask=group_mask[:, None],
        )

    tl.store(
        grad_x_ptr + tile_offsets,
        grad_x.to(grad_x_ptr.dtype.element_ty),
        mask=tile_mask,
    )
