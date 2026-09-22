"""Triton implementation of the DFlash2 unary head objective.

The reference DFlash2 objective projects draft states through the frozen target
LM head, upcasts the BF16 logits to FP32, and derives cross-entropy, the strict
selector top-k, and the accuracy argmax from that FP32 copy. Under activation
checkpointing the whole projection then runs a second time in backward.

This path keeps a single BF16 logit buffer per objective chunk:

* forward runs the same cuBLAS projection, one Triton pass for the FP32 online
  log-sum-exp, target logit, and first-index argmax, and ``torch.topk`` on the
  BF16 logits (BF16 -> FP32 is exact, so the candidate values are identical);
* backward writes the FP32 logit gradient
  ``g * (softmax - one_hot(target)) + scatter(top-k grads)`` with one BF16
  rounding, the same rounding point as the reference, in place into the saved
  logits, then runs the input-gradient GEMM. The target head is frozen, so no
  weight gradient is formed.
"""

import torch
import torch.nn.functional as F
import triton
import triton.language as tl

__all__ = ["dflash_unary_head_fused"]


def dflash_unary_head_fused(hidden, weight, targets, top_k):
    """Return ``(neg_log_q, topk_values, topk_ids, argmax_ids)`` for ``hidden``.

    ``hidden`` is ``[..., hidden_size]`` BF16, ``weight`` the frozen
    ``[vocab_size, hidden_size]`` BF16 head, and ``targets`` has ``hidden``'s
    leading shape. ``neg_log_q`` is the FP32 per-row cross-entropy and
    ``topk_values`` the FP32 strict top-k logits; both are differentiable with
    respect to ``hidden``. ``top_k=0`` skips the top-k outputs' computation and
    returns empty candidate tensors.
    """
    leading_shape = hidden.shape[:-1]
    neg_log_q, topk_values, topk_ids, argmax_ids = _DFlash2UnaryHead.apply(
        hidden.reshape(-1, hidden.shape[-1]),
        weight,
        targets.reshape(-1),
        int(top_k),
    )
    return (
        neg_log_q.reshape(leading_shape),
        topk_values.reshape(*leading_shape, topk_values.shape[-1]),
        topk_ids.reshape(*leading_shape, topk_ids.shape[-1]),
        argmax_ids.reshape(leading_shape),
    )


class _DFlash2UnaryHead(torch.autograd.Function):
    """Frozen-head projection with fused FP32 cross-entropy statistics.

    Saves the BF16 logits, the FP32 row log-sum-exp, targets, and the top-k
    values and ids. Backward reuses the saved logits as its gradient buffer, so
    the graph must not be backpropagated twice.
    """

    @staticmethod
    def forward(ctx, hidden, weight, targets, top_k):
        logits = F.linear(hidden, weight)
        num_rows, vocab_size = logits.shape
        targets = targets.contiguous()
        device = logits.device

        lse = torch.empty(num_rows, device=device, dtype=torch.float32)
        target_logit = torch.empty_like(lse)
        argmax_ids = torch.empty(num_rows, device=device, dtype=torch.long)
        if num_rows > 0:
            _launch_stats(logits, targets, lse, target_logit, argmax_ids)
        neg_log_q = lse - target_logit

        if top_k > 0:
            topk_values, topk_ids = torch.topk(logits, top_k, dim=-1)
        else:
            topk_values = logits.new_empty(num_rows, 0)
            topk_ids = targets.new_empty(num_rows, 0)

        ctx.save_for_backward(weight, logits, lse, targets, topk_values, topk_ids)
        ctx.mark_non_differentiable(topk_ids, argmax_ids)
        return neg_log_q, topk_values.float(), topk_ids, argmax_ids

    @staticmethod
    def backward(ctx, grad_neg_log_q, grad_topk_values, _grad_topk_ids, _grad_argmax):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        weight, logits, lse, targets, topk_values, topk_ids = ctx.saved_tensors
        num_rows, vocab_size = logits.shape
        if grad_neg_log_q is None:
            grad_neg_log_q = lse.new_zeros(num_rows)
        grad_neg_log_q = grad_neg_log_q.float().contiguous()

        # ``logits`` is internal to this node and dead after this point; it
        # becomes the BF16 logit-gradient buffer.
        grad_logits = logits
        if num_rows > 0:
            vocab_block, num_warps = _grad_settings(vocab_size)
            _unary_head_grad_kernel[(num_rows, triton.cdiv(vocab_size, vocab_block))](
                grad_logits,
                grad_logits.stride(0),
                targets,
                lse,
                grad_neg_log_q,
                vocab_size,
                VOCAB_BLOCK=vocab_block,
                num_warps=num_warps,
            )
            top_k = topk_ids.shape[-1]
            if grad_topk_values is not None and top_k > 0:
                # Top-k ids are distinct within a row, so every candidate
                # rewrites one gradient entry with its complete FP32 value.
                _unary_head_topk_grad_kernel[(num_rows,)](
                    grad_logits,
                    grad_logits.stride(0),
                    targets,
                    lse,
                    grad_neg_log_q,
                    topk_values.contiguous(),
                    topk_ids.contiguous(),
                    grad_topk_values.float().contiguous(),
                    top_k,
                    BLOCK_K=triton.next_power_of_2(top_k),
                    num_warps=1,
                )
        grad_hidden = torch.matmul(grad_logits, weight)
        return grad_hidden, None, None, None


def _num_warps_for_backend(num_warps):
    """Preserve the NVIDIA thread count on AMD targets with 64-lane wavefronts."""
    # Note: this isn't tested, someone should tune this
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        num_warps = num_warps * 32 // warp_size
    return max(num_warps, 1)


def _stats_settings(num_rows, vocab_size):
    """Choose the vocabulary tile, row split, and warp count for statistics.

    Each row is split over several programs when there are too few rows to
    keep every SM streaming; a second tiny kernel merges the partial maxima,
    exponential sums, and argmaxes.
    """
    # Measured on H200 for 1792 x 248320 BF16 logits: 4096-wide tiles with
    # four warps and a 4-way row split stream at ~3 TB/s.
    vocab_block = min(triton.next_power_of_2(vocab_size), 4096)
    num_tiles = triton.cdiv(vocab_size, vocab_block)
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    num_splits = 1
    while num_rows * num_splits < 32 * num_sms and num_splits * 2 <= min(num_tiles, 64):
        num_splits *= 2
    return vocab_block, num_splits, _num_warps_for_backend(4)


def _grad_settings(vocab_size):
    """Choose the vocabulary tile and warp count for the gradient kernel."""
    vocab_block = min(triton.next_power_of_2(vocab_size), 4096)
    num_warps = 8 if vocab_block >= 2048 else 4
    return vocab_block, _num_warps_for_backend(num_warps)


def _launch_stats(logits, targets, lse, target_logit, argmax_ids):
    """Write each row's FP32 log-sum-exp, target logit, and first argmax."""
    num_rows, vocab_size = logits.shape
    vocab_block, num_splits, num_warps = _stats_settings(num_rows, vocab_size)
    split_size = triton.cdiv(triton.cdiv(vocab_size, num_splits), vocab_block)
    split_size *= vocab_block
    num_splits = triton.cdiv(vocab_size, split_size)
    partial_max = torch.empty(
        num_rows, num_splits, device=logits.device, dtype=torch.float32
    )
    partial_exp_sum = torch.empty_like(partial_max)
    partial_argmax = torch.empty(
        num_rows, num_splits, device=logits.device, dtype=torch.int32
    )
    _unary_head_partial_stats_kernel[(num_rows, num_splits)](
        logits,
        logits.stride(0),
        partial_max,
        partial_exp_sum,
        partial_argmax,
        vocab_size,
        split_size,
        NUM_SPLITS=num_splits,
        VOCAB_BLOCK=vocab_block,
        num_warps=num_warps,
    )
    _unary_head_combine_stats_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        targets,
        partial_max,
        partial_exp_sum,
        partial_argmax,
        lse,
        target_logit,
        argmax_ids,
        vocab_size,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        num_warps=1,
    )


@triton.jit
def _unary_head_partial_stats_kernel(
    logits_ptr,
    logits_stride,
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_argmax_ptr,
    vocab_size,
    split_size,
    NUM_SPLITS: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
):
    """Reduce one vocabulary split of one row to (max, shifted exp sum, argmax).

    ``argmax`` is the leftmost maximal index inside the split, so merging splits
    in index order reproduces ``torch.argmax``'s first-index tie break.
    """
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    logits_ptr += row * logits_stride
    start = split * split_size
    end = tl.minimum(start + split_size, vocab_size)

    row_max = float("-inf")
    exp_sum = 0.0
    argmax = 0
    for offset in range(start, end, VOCAB_BLOCK):
        offsets = offset + tl.arange(0, VOCAB_BLOCK)
        mask = offsets < end
        block = tl.load(
            logits_ptr + offsets,
            mask=mask,
            other=float("-inf"),
        ).to(tl.float32)
        block_max, block_argmax = tl.max(
            block,
            axis=0,
            return_indices=True,
            return_indices_tie_break_left=True,
        )
        argmax = tl.where(block_max > row_max, block_argmax + offset, argmax)
        new_max = tl.maximum(row_max, block_max)
        exp_sum = exp_sum * tl.exp(row_max - new_max) + tl.sum(tl.exp(block - new_max))
        row_max = new_max

    tl.store(partial_max_ptr + row * NUM_SPLITS + split, row_max)
    tl.store(partial_exp_sum_ptr + row * NUM_SPLITS + split, exp_sum)
    tl.store(partial_argmax_ptr + row * NUM_SPLITS + split, argmax)


@triton.jit
def _unary_head_combine_stats_kernel(
    logits_ptr,
    logits_stride,
    targets_ptr,
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_argmax_ptr,
    lse_ptr,
    target_logit_ptr,
    argmax_ptr,
    vocab_size,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
):
    """Merge a row's split statistics and read its FP32 target logit.

    ``lse = max + log(sum_s exp_sum_s * exp(max_s - max))``; the argmax is taken
    from the first split that attains the row maximum.
    """
    row = tl.program_id(0).to(tl.int64)
    splits = tl.arange(0, BLOCK_SPLITS)
    split_mask = splits < NUM_SPLITS
    partial_max = tl.load(
        partial_max_ptr + row * NUM_SPLITS + splits,
        mask=split_mask,
        other=float("-inf"),
    )
    partial_exp_sum = tl.load(
        partial_exp_sum_ptr + row * NUM_SPLITS + splits,
        mask=split_mask,
        other=0.0,
    )
    partial_argmax = tl.load(
        partial_argmax_ptr + row * NUM_SPLITS + splits,
        mask=split_mask,
        other=0,
    )
    row_max = tl.max(partial_max, axis=0)
    exp_sum = tl.sum(
        tl.where(split_mask, partial_exp_sum * tl.exp(partial_max - row_max), 0.0)
    )
    first_split = tl.min(
        tl.where(split_mask & (partial_max == row_max), splits, BLOCK_SPLITS),
        axis=0,
    )
    argmax = tl.sum(tl.where(splits == first_split, partial_argmax, 0))

    target = tl.load(targets_ptr + row)
    target_in_bounds = (target >= 0) & (target < vocab_size)
    target_logit = tl.load(
        logits_ptr + row * logits_stride + target,
        mask=target_in_bounds,
        other=float("nan"),
    ).to(tl.float32)
    tl.store(lse_ptr + row, row_max + tl.log(exp_sum))
    tl.store(target_logit_ptr + row, target_logit)
    tl.store(argmax_ptr + row, argmax.to(tl.int64))


@triton.jit
def _unary_head_grad_kernel(
    logits_ptr,
    logits_stride,
    targets_ptr,
    lse_ptr,
    grad_neg_log_q_ptr,
    vocab_size,
    VOCAB_BLOCK: tl.constexpr,
):
    """Overwrite one logit tile with ``g * (softmax - one_hot(target))``.

    The value is formed in FP32 and rounded to the logit dtype once. Rows with
    zero upstream gradient write exact zeros.
    """
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    offsets = tile * VOCAB_BLOCK + tl.arange(0, VOCAB_BLOCK)
    mask = offsets < vocab_size
    pointers = logits_ptr + row * logits_stride + offsets
    logits = tl.load(pointers, mask=mask, other=0.0).to(tl.float32)
    grad = tl.load(grad_neg_log_q_ptr + row)
    lse = tl.load(lse_ptr + row)
    target = tl.load(targets_ptr + row)
    logit_grad = grad * tl.exp(logits - lse)
    logit_grad = tl.where(offsets == target, logit_grad - grad, logit_grad)
    tl.store(pointers, logit_grad.to(logits_ptr.dtype.element_ty), mask=mask)


@triton.jit
def _unary_head_topk_grad_kernel(
    logits_ptr,
    logits_stride,
    targets_ptr,
    lse_ptr,
    grad_neg_log_q_ptr,
    topk_values_ptr,
    topk_ids_ptr,
    grad_topk_values_ptr,
    top_k,
    BLOCK_K: tl.constexpr,
):
    """Rewrite a row's top-k gradient entries with the selector contribution.

    Runs after ``_unary_head_grad_kernel`` overwrote the logits, so the saved
    top-k values (exact copies of those logits) supply ``x``. Each entry is
    recomputed as ``g * (softmax - one_hot) + g_topk`` in FP32 and rounded once.
    """
    row = tl.program_id(0).to(tl.int64)
    candidates = tl.arange(0, BLOCK_K)
    mask = candidates < top_k
    ids = tl.load(topk_ids_ptr + row * top_k + candidates, mask=mask, other=0)
    logits = tl.load(
        topk_values_ptr + row * top_k + candidates,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    grad_topk = tl.load(
        grad_topk_values_ptr + row * top_k + candidates,
        mask=mask,
        other=0.0,
    )
    grad = tl.load(grad_neg_log_q_ptr + row)
    lse = tl.load(lse_ptr + row)
    target = tl.load(targets_ptr + row)
    logit_grad = grad * tl.exp(logits - lse)
    logit_grad = tl.where(ids == target, logit_grad - grad, logit_grad) + grad_topk
    tl.store(
        logits_ptr + row * logits_stride + ids,
        logit_grad.to(logits_ptr.dtype.element_ty),
        mask=mask,
    )
