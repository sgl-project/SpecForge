"""Fused Domino cross-entropy without vocabulary-sized intermediates.

The CUDA path processes base and compact-correction logits in vocabulary tiles.
It never writes full corrected logits, padded corrections, base/final softmax or
log-softmax tensors, or a separate corrected-logit gradient. Backward rebuilds
each probability tile from the saved row maximum and shifted exponential sum,
then writes only the required base and compact-correction gradients.

The unfused reference path materializes corrected logits for readability and
non-CUDA execution.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


def domino_weighted_cross_entropy(
    base_logits, correction_logits, targets, weights,
    block_size, suffix_start, *, use_fused: bool,
):
    """Return weighted final/base loss sums and predictions for Domino logits.

    ``use_fused=True`` requires CUDA; otherwise the reference path materializes
    final logits. ``suffix_start`` is the first corrected position in each
    block. Only the two logit tensors are differentiable; targets, weights, and
    layout arguments are treated as fixed inputs. Incompatible inputs raise
    ``ValueError``.

    Tensor layout:
        ``num_rows = num_blocks * block_size``
        ``base_logits``: ``[num_rows, vocab_size]``
        ``correction_logits``: ``[num_blocks * suffix_size, vocab_size]``
        ``targets`` and ``weights``: ``[num_rows]``

    Returns:
        ``final_loss_sum``: weighted corrected-logit losses summed over rows.
        ``base_loss_sum``: weighted base-logit losses summed over rows.
        ``final_pred``: ``[num_rows]`` predictions from corrected logits.
        ``base_pred``: ``[num_rows]`` predictions from base logits.
    """
    if base_logits.dim() != 2 or correction_logits.dim() != 2:
        raise ValueError("Domino cross entropy expects 2D logits")
    if block_size < 1 or not 0 <= suffix_start < block_size:
        raise ValueError("suffix_start must select a non-empty Domino suffix")
    num_rows, vocab_size = base_logits.shape
    if num_rows % block_size:
        raise ValueError("base-logit rows must be divisible by block_size")
    num_blocks = num_rows // block_size
    suffix_size = block_size - suffix_start
    expected_correction_rows = num_blocks * suffix_size
    if correction_logits.shape != (expected_correction_rows, vocab_size):
        raise ValueError("correction logits do not match the configured suffix width")
    if targets.shape != (num_rows,) or weights.shape != (num_rows,):
        raise ValueError("targets and weights must have one value per logit row")
    if base_logits.device != correction_logits.device or any(
        tensor.device != base_logits.device for tensor in (targets, weights)
    ):
        raise ValueError("Domino cross-entropy inputs must be on the same device")
    if base_logits.dtype != correction_logits.dtype:
        raise ValueError("base and correction logits must have the same dtype")

    if not use_fused:
        return _domino_weighted_cross_entropy_reference(
            base_logits, correction_logits, targets, weights,
            block_size, suffix_start,
        )
    if not base_logits.is_cuda:
        raise ValueError("Fused Domino cross entropy requires CUDA logits")

    return _DominoCrossEntropy.apply(
        base_logits, correction_logits, targets, weights,
        block_size, suffix_start,
    )


def _domino_weighted_cross_entropy_reference(
    base_logits, correction_logits, targets, weights,
    block_size, suffix_start,
):
    """Reference implementation that materializes the corrected logits.

    Tensor shapes:
        base_logits: ``[num_blocks * block_size, vocab_size]``
        correction_logits:
            ``[num_blocks * (block_size - suffix_start), vocab_size]``
        targets: ``[num_blocks * block_size]``
        weights: ``[num_blocks * block_size]``
    """
    num_blocks = base_logits.shape[0] // block_size
    suffix_size = block_size - suffix_start
    base_logits_3d = base_logits.reshape(num_blocks, block_size, -1)
    correction_logits_3d = correction_logits.reshape(
        num_blocks, suffix_size, -1
    )
    final_logits = torch.cat(
        [
            base_logits_3d[:, :suffix_start],
            base_logits_3d[:, suffix_start:] + correction_logits_3d,
        ],
        dim=1,
    ).reshape_as(base_logits)
    final_losses = F.cross_entropy(final_logits, targets, reduction="none")
    base_losses = F.cross_entropy(base_logits, targets, reduction="none")
    final_loss_sum = (final_losses * weights).sum()
    base_loss_sum = (base_losses * weights).sum()
    return (
        final_loss_sum, base_loss_sum,
        final_logits.argmax(-1), base_logits.argmax(-1),
    )


class _DominoCrossEntropy(torch.autograd.Function):
    """Return weighted loss sums and predictions from the fused kernels.

    Saves each row's maximum logit and shifted exponential sum so backward can
    reconstruct both softmaxes without saving full probability tensors.
    """

    @staticmethod
    def forward(
        ctx, base_logits, correction_logits, targets, weights,
        block_size, suffix_start,
    ):
        num_rows, vocab_size = base_logits.shape
        base_logits = base_logits.contiguous()
        correction_logits = correction_logits.contiguous()
        targets = targets.contiguous()
        weights = weights.contiguous()

        device = base_logits.device

        # kernel outputs
        final_losses = torch.empty(num_rows, device=device, dtype=torch.float32)
        base_losses = torch.empty_like(final_losses)
        final_max = torch.empty_like(final_losses)
        final_exp_sum = torch.empty_like(final_losses)
        base_max = torch.empty_like(final_losses)
        base_exp_sum = torch.empty_like(final_losses)
        final_pred = torch.empty(num_rows, device=device, dtype=torch.long)
        base_pred = torch.empty_like(final_pred)

        vocab_block_size, num_warps = _calculate_domino_ce_settings(vocab_size)
        _domino_cross_entropy_forward_kernel[(num_rows,)](
            base_logits, base_logits.stride(0),
            correction_logits, correction_logits.stride(0),
            targets, weights,
            final_losses, base_losses,
            final_max, final_exp_sum, base_max, base_exp_sum,
            final_pred, base_pred, vocab_size,
            BLOCK_SIZE=block_size,
            SUFFIX_START=suffix_start,
            VOCAB_BLOCK_SIZE=vocab_block_size,
            num_warps=num_warps,
        )

        ctx.block_size = block_size
        ctx.suffix_start = suffix_start
        ctx.save_for_backward(
            base_logits, correction_logits, targets, weights,
            final_max, final_exp_sum, base_max, base_exp_sum,
        )
        ctx.mark_non_differentiable(final_pred, base_pred)
        return (
            final_losses.sum(), base_losses.sum(),
            final_pred, base_pred,
        )

    @staticmethod
    def backward(
        ctx, grad_final_loss, grad_base_loss, _grad_final_pred, _grad_base_pred,
    ):
        (
            base_logits, correction_logits, targets, weights,
            final_max, final_exp_sum, base_max, base_exp_sum,
        ) = ctx.saved_tensors
        num_rows, vocab_size = base_logits.shape
        grad_base_logits = torch.empty_like(base_logits)
        grad_correction_logits = torch.empty_like(correction_logits)
        vocab_block_size, num_warps = _calculate_domino_ce_settings(vocab_size)
        _domino_cross_entropy_backward_kernel[(num_rows,)](
            base_logits, base_logits.stride(0),
            correction_logits, correction_logits.stride(0),
            targets, weights,
            grad_final_loss, grad_base_loss,
            grad_base_logits, grad_base_logits.stride(0),
            grad_correction_logits, grad_correction_logits.stride(0),
            final_max, final_exp_sum, base_max, base_exp_sum, vocab_size,
            BLOCK_SIZE=ctx.block_size,
            SUFFIX_START=ctx.suffix_start,
            VOCAB_BLOCK_SIZE=vocab_block_size,
            num_warps=num_warps,
        )
        return (
            grad_base_logits, grad_correction_logits,
            None, None, None, None,
        )


def _calculate_domino_ce_settings(vocab_size):
    """Choose the vocabulary tile and warp count for the active GPU backend."""
    vocab_block_size = min(triton.next_power_of_2(vocab_size), 2048)
    num_warps = 8 if vocab_block_size >= 2048 else 4

    # Preserve the NVIDIA thread count on AMD targets with 64-lane wavefronts.
    # Note: this isn't tested, someone should tune this
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        num_warps = num_warps * 32 // warp_size

    return vocab_block_size, max(num_warps, 1)


@triton.jit
def _domino_cross_entropy_forward_kernel(
    base_logits_ptr, base_logits_stride,
    correction_logits_ptr, correction_logits_stride,
    targets_ptr, weights_ptr,
    final_losses_ptr, base_losses_ptr,
    final_max_ptr, final_exp_sum_ptr, base_max_ptr, base_exp_sum_ptr,
    final_pred_ptr, base_pred_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUFFIX_START: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr,
):
    """Write one set of outputs per flattened logit row.

    For each row, the kernel writes:

    - ``final_losses`` / ``base_losses``: weighted cross-entropy per row.
      ``_DominoCrossEntropy.forward`` sums them for its scalar loss outputs.
    - ``final_max`` / ``base_max``: largest logit in the row.
    - ``final_exp_sum`` / ``base_exp_sum``: sum of
      ``exp(logit - largest_logit)`` over the row.
    - ``final_pred`` / ``base_pred``: index of the largest logit.

    For corrected logits ``[2, 5, 4]`` with target index 2 and weight 1:

    - ``final_max[row] = 5``
    - ``final_exp_sum[row] = exp(2 - 5) + exp(5 - 5) + exp(4 - 5)``
    - ``final_losses[row] = 5 + log(final_exp_sum[row]) - 4``
    - ``final_pred[row] = 1``

    Backward reconstructs each probability as
    ``exp(logit - final_max) / final_exp_sum``. The base outputs use the same
    calculation with uncorrected base logits.
    """
    row = tl.program_id(0).to(tl.int64)
    has_correction, correction_row = _domino_correction_mapping(
        row, BLOCK_SIZE, SUFFIX_START,
    )

    base_logits_ptr += row * base_logits_stride
    correction_logits_ptr += correction_row * correction_logits_stride
    target = tl.load(targets_ptr + row)
    weight = tl.load(weights_ptr + row).to(tl.float32)
    base_target_raw = tl.load(base_logits_ptr + target)
    correction_target_raw = tl.load(
        correction_logits_ptr + target, mask=has_correction, other=0.0,
    )
    base_target = base_target_raw.to(tl.float32)
    final_target = (base_target_raw + correction_target_raw).to(tl.float32)

    final_max = float("-inf")
    final_exp_sum = 0.0
    final_argmax = 0
    base_max = float("-inf")
    base_exp_sum = 0.0
    base_argmax = 0

    # Iterates over the one row of base logits.
    # Each iteration:
    # - loads one chunk of base_logits
    # - loads the corresponding correction_logits chunk (or zeros if base-only row).
    # - updates the {base/final}{max, exp_sum, argmax} variables
    for i in range(0, vocab_size, VOCAB_BLOCK_SIZE):
        offsets = i + tl.arange(0, VOCAB_BLOCK_SIZE)
        mask = offsets < vocab_size
        base_block_raw = tl.load(
            base_logits_ptr + offsets, mask=mask, other=float("-inf"),
        )
        correction_block_raw = tl.load(
            correction_logits_ptr + offsets, mask=mask & has_correction, other=0.0,
        )
        base_block = base_block_raw.to(tl.float32)
        final_block = (base_block_raw + correction_block_raw).to(tl.float32)

        final_max, final_exp_sum, final_argmax = _update_online_logsumexp_and_argmax(
            final_block, mask, i, final_max, final_exp_sum, final_argmax,
        )
        base_max, base_exp_sum, base_argmax = _update_online_logsumexp_and_argmax(
            base_block, mask, i, base_max, base_exp_sum, base_argmax,
        )

    final_loss = weight * (tl.log(final_exp_sum) + final_max - final_target)
    base_loss = weight * (tl.log(base_exp_sum) + base_max - base_target)
    tl.store(final_losses_ptr + row, final_loss)
    tl.store(base_losses_ptr + row, base_loss)
    tl.store(final_max_ptr + row, final_max.to(tl.float32))
    tl.store(final_exp_sum_ptr + row, final_exp_sum.to(tl.float32))
    tl.store(base_max_ptr + row, base_max.to(tl.float32))
    tl.store(base_exp_sum_ptr + row, base_exp_sum.to(tl.float32))
    tl.store(final_pred_ptr + row, final_argmax)
    tl.store(base_pred_ptr + row, base_argmax)


@triton.jit
def _domino_cross_entropy_backward_kernel(
    base_logits_ptr, base_logits_stride,
    correction_logits_ptr, correction_logits_stride,
    targets_ptr, weights_ptr,
    grad_final_loss_ptr, grad_base_loss_ptr,
    grad_base_logits_ptr, grad_base_logits_stride,
    grad_correction_logits_ptr, grad_correction_logits_stride,
    final_max_ptr, final_exp_sum_ptr, base_max_ptr, base_exp_sum_ptr,
    vocab_size,
    BLOCK_SIZE: tl.constexpr,
    SUFFIX_START: tl.constexpr,
    VOCAB_BLOCK_SIZE: tl.constexpr,
):
    """Reconstruct both softmaxes and write one row of input gradients.

    For one ``x[vocab_size]`` logit row and fixed scalar target index ``y``::

        CE(x, y) = logsumexp(x) - x[y]
        d CE(x, y) / d x[j] = softmax(x)[j] - 1[j == y]

    ``logsumexp(x)`` reduces the vocabulary row to one scalar, as does selecting
    ``x[y]``. Their difference is the row's scalar cross-entropy; its gradient
    with respect to ``x`` has shape ``[vocab_size]``.

    The kernel evaluates the softmax term stably using the forward statistics:
    ``softmax(x)[j] = exp(x[j] - row_max) / row_exp_sum``.

    Including the row weight and upstream loss gradients gives::

        grad_from_final_loss = grad_final_loss * weight
                               * (softmax(final_logits) - one_hot(y))
        grad_base_logits = grad_from_final_loss
                           + grad_base_loss * weight
                             * (softmax(base_logits) - one_hot(y))
        grad_correction_logits = grad_from_final_loss

    Any normalization applied to the returned loss sums is already included in
    the incoming ``grad_final_loss`` and ``grad_base_loss`` values.

    Base logits feed both losses. Correction logits feed only the final loss
    and exist only for suffix rows; prefix rows have no correction to update.
    """
    row = tl.program_id(0).to(tl.int64)
    has_correction, correction_row = _domino_correction_mapping(
        row, BLOCK_SIZE, SUFFIX_START,
    )

    base_logits_ptr += row * base_logits_stride
    grad_base_logits_ptr += row * grad_base_logits_stride
    correction_logits_ptr += correction_row * correction_logits_stride
    grad_correction_logits_ptr += correction_row * grad_correction_logits_stride

    target = tl.load(targets_ptr + row)
    weight = tl.load(weights_ptr + row).to(tl.float32)
    final_scale = tl.load(grad_final_loss_ptr).to(tl.float32) * weight
    base_scale = tl.load(grad_base_loss_ptr).to(tl.float32) * weight
    final_max = tl.load(final_max_ptr + row).to(tl.float32)
    final_exp_sum = tl.load(final_exp_sum_ptr + row).to(tl.float32)
    base_max = tl.load(base_max_ptr + row).to(tl.float32)
    base_exp_sum = tl.load(base_exp_sum_ptr + row).to(tl.float32)

    # Iterates over one row of base logits.
    # Each iteration:
    # - loads base_logits and correction_logits chunks
    # - reconstructs the base and final softmax probabilities
    # - writes grad_base_logits with contributions from both losses
    # - writes grad_correction_logits from the final loss for suffix rows
    for i in range(0, vocab_size, VOCAB_BLOCK_SIZE):
        offsets = i + tl.arange(0, VOCAB_BLOCK_SIZE)
        mask = offsets < vocab_size
        base_block_raw = tl.load(
            base_logits_ptr + offsets, mask=mask, other=0.0,
        )
        correction_block_raw = tl.load(
            correction_logits_ptr + offsets, mask=mask & has_correction, other=0.0,
        )
        base_block = base_block_raw.to(tl.float32)
        final_block = (base_block_raw + correction_block_raw).to(tl.float32)
        target_grad = tl.where(offsets == target, 1.0, 0.0)
        final_grad = final_scale * (
            tl.exp(final_block - final_max) / final_exp_sum - target_grad
        )
        base_grad = final_grad + base_scale * (
            tl.exp(base_block - base_max) / base_exp_sum - target_grad
        )
        tl.store(grad_base_logits_ptr + offsets, base_grad, mask=mask)
        tl.store(
            grad_correction_logits_ptr + offsets,
            final_grad, mask=mask & has_correction,
        )


@triton.jit
def _domino_correction_mapping(
    row, BLOCK_SIZE: tl.constexpr, SUFFIX_START: tl.constexpr,
):
    """Map a flattened base row to Domino's compact correction layout.

    For ``BLOCK_SIZE=4`` and ``SUFFIX_START=1``::

        base row:        0  1  2  3 | 4  5  6  7
        correction row:  -  0  1  2 | -  3  4  5

    ``-`` marks a base-only row with no correction.
    """
    row_in_block = row % BLOCK_SIZE
    suffix_size: tl.constexpr = BLOCK_SIZE - SUFFIX_START
    has_correction = row_in_block >= SUFFIX_START
    correction_row = (row // BLOCK_SIZE) * suffix_size + (
        row_in_block - SUFFIX_START
    )
    correction_row = tl.where(has_correction, correction_row, 0)
    return has_correction, correction_row


@triton.jit
def _update_online_logsumexp_and_argmax(
    logits, mask, block_offset, previous_max, previous_exp_sum, previous_argmax,
):
    """Update the row maximum, shifted exponential sum, and leftmost argmax."""
    block_max, block_argmax = tl.max(
        tl.where(mask, logits, float("-inf")),
        axis=0, return_indices=True, return_indices_tie_break_left=True,
    )
    block_argmax += block_offset
    take_block = block_max > previous_max
    argmax = tl.where(take_block, block_argmax, previous_argmax)
    max_logit = tl.maximum(previous_max, block_max)
    exp_sum = previous_exp_sum * tl.exp(previous_max - max_logit) + tl.sum(
        tl.where(mask, tl.exp(logits - max_logit), 0.0)
    )
    return max_logit, exp_sum, argmax
