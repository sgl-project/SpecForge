"""Triton implementation of the DFlash2 unary head objective.

The reference DFlash2 objective projects draft states through the frozen target
LM head, upcasts the BF16 logits to FP32, and derives cross-entropy, the strict
selector top-k, and the accuracy argmax from that FP32 copy. Under activation
checkpointing the whole projection then runs a second time in backward.

This path keeps a single BF16 logit buffer per objective chunk:

* forward runs the same cuBLAS projection and one Triton pass for the FP32
  online log-sum-exp, target logit, first-index argmax, and strict top-k
  (BF16 -> FP32 is exact, so the candidate values are identical). The top-k
  selects ``torch.topk``'s candidate set, lowest index first on ties, and
  orders equal values by ascending index; wider candidate sets than
  ``MAX_FUSED_TOP_K`` fall back to ``torch.topk`` on the BF16 logits;
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

__all__ = ["MAX_FUSED_TOP_K", "dflash_unary_head_fused"]

# The in-kernel selection keeps one register list per row split and is tuned
# for the selector's small candidate sets; wider sets use ``torch.topk``.
MAX_FUSED_TOP_K = 64


def dflash_unary_head_fused(hidden, weight, targets, top_k):
    """Return ``(neg_log_q, topk_values, topk_ids, argmax_ids)`` for ``hidden``.

    ``hidden`` is ``[..., hidden_size]`` BF16, ``weight`` the frozen
    ``[vocab_size, hidden_size]`` BF16 head, and ``targets`` has ``hidden``'s
    leading shape. ``neg_log_q`` is the FP32 per-row cross-entropy and
    ``topk_values`` the FP32 strict top-k logits; both are differentiable with
    respect to ``hidden``. Candidates hold ``torch.topk``'s set in descending
    value order; up to ``MAX_FUSED_TOP_K`` of them, equal values are ordered
    by ascending id. ``top_k=0`` skips the top-k selection and returns empty
    candidate tensors.
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
    a second backward through a retained graph raises instead of recomputing.
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
        fused_top_k = top_k if top_k <= MAX_FUSED_TOP_K else 0
        topk_values = torch.empty(
            num_rows, fused_top_k, device=device, dtype=torch.float32
        )
        topk_ids = torch.empty(num_rows, fused_top_k, device=device, dtype=torch.long)
        if num_rows > 0:
            _launch_stats(
                logits,
                targets,
                lse,
                target_logit,
                argmax_ids,
                topk_values,
                topk_ids,
            )
        if top_k > fused_top_k:
            topk_values, topk_ids = torch.topk(logits, top_k, dim=-1)
            topk_values = topk_values.float()
        neg_log_q = lse - target_logit

        ctx.save_for_backward(weight, logits, lse, targets, topk_values, topk_ids)
        ctx.mark_non_differentiable(topk_ids, argmax_ids)
        return neg_log_q, topk_values, topk_ids, argmax_ids

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
                    topk_values,
                    topk_ids,
                    grad_topk_values.float().contiguous(),
                    top_k,
                    BLOCK_K=triton.next_power_of_2(top_k),
                    num_warps=1,
                )
            # Triton writes do not bump the version counter; do it so a
            # retained graph fails loudly instead of reusing the gradients.
            torch.autograd.graph.increment_version(grad_logits)
        grad_hidden = torch.matmul(grad_logits, weight)
        return grad_hidden, None, None, None


def _num_warps_for_backend(num_warps):
    """Preserve the NVIDIA thread count on AMD targets with 64-lane wavefronts."""
    # Note: this isn't tested, someone should tune this
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        warp_size = triton.runtime.driver.active.get_current_target().warp_size
        num_warps = num_warps * 32 // warp_size
    return max(num_warps, 1)


def _stats_settings(num_rows, vocab_size, top_k):
    """Choose the vocabulary tile, row split, and warp count for statistics.

    Each row is split over several programs when there are too few rows to
    keep every SM streaming; a second tiny kernel merges the partial maxima,
    exponential sums, argmaxes, and top-k lists.
    """
    # Measured on H200 for 1792 x 248320 BF16 logits. Without top-k, 4096-wide
    # tiles with four warps and a 4-way split stream at ~3 TB/s (0.31 ms).
    # The top-k selection's reductions are cheapest within one warp: 512-wide
    # single-warp tiles with a 2-way split take 0.71 ms for statistics plus
    # top-16, against 3.3 ms for ``torch.topk`` on the BF16 logits alone.
    if top_k > 0:
        vocab_block, num_warps, min_programs_per_sm = 512, 1, 16
    else:
        vocab_block, num_warps, min_programs_per_sm = 4096, 4, 32
    vocab_block = min(triton.next_power_of_2(vocab_size), vocab_block)
    num_tiles = triton.cdiv(vocab_size, vocab_block)
    num_sms = torch.cuda.get_device_properties(
        torch.cuda.current_device()
    ).multi_processor_count
    num_splits = 1
    while num_rows * num_splits < min_programs_per_sm * num_sms and (
        num_splits * 2 <= min(num_tiles, 64)
    ):
        num_splits *= 2
    return vocab_block, num_splits, _num_warps_for_backend(num_warps)


def _grad_settings(vocab_size):
    """Choose the vocabulary tile and warp count for the gradient kernel."""
    vocab_block = min(triton.next_power_of_2(vocab_size), 4096)
    num_warps = 8 if vocab_block >= 2048 else 4
    return vocab_block, _num_warps_for_backend(num_warps)


def _launch_stats(
    logits,
    targets,
    lse,
    target_logit,
    argmax_ids,
    topk_values,
    topk_ids,
):
    """Write each row's FP32 log-sum-exp, target logit, argmax, and top-k."""
    num_rows, vocab_size = logits.shape
    top_k = topk_ids.shape[-1]
    vocab_block, num_splits, num_warps = _stats_settings(num_rows, vocab_size, top_k)
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
    partial_topk_values = torch.empty(
        num_rows, num_splits, top_k, device=logits.device, dtype=torch.float32
    )
    partial_topk_ids = torch.empty(
        num_rows, num_splits, top_k, device=logits.device, dtype=torch.int32
    )
    _unary_head_partial_stats_kernel[(num_rows, num_splits)](
        logits,
        logits.stride(0),
        partial_max,
        partial_exp_sum,
        partial_argmax,
        partial_topk_values,
        partial_topk_ids,
        vocab_size,
        split_size,
        NUM_SPLITS=num_splits,
        VOCAB_BLOCK=vocab_block,
        TOP_K=top_k,
        BLOCK_K=triton.next_power_of_2(max(top_k, 1)),
        num_warps=num_warps,
    )
    _unary_head_combine_stats_kernel[(num_rows,)](
        logits,
        logits.stride(0),
        targets,
        partial_max,
        partial_exp_sum,
        partial_argmax,
        partial_topk_values,
        partial_topk_ids,
        lse,
        target_logit,
        argmax_ids,
        topk_values,
        topk_ids,
        vocab_size,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        TOP_K=top_k,
        BLOCK_CANDIDATES=triton.next_power_of_2(max(num_splits * top_k, 1)),
        num_warps=1,
    )


@triton.jit
def _unary_head_partial_stats_kernel(
    logits_ptr,
    logits_stride,
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_argmax_ptr,
    partial_topk_values_ptr,
    partial_topk_ids_ptr,
    vocab_size,
    split_size,
    NUM_SPLITS: tl.constexpr,
    VOCAB_BLOCK: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Reduce one vocabulary split of one row to its softmax and top-k state.

    Writes the split's max, shifted exponential sum, leftmost argmax, and its
    ``TOP_K`` largest logits (unordered). Ties keep the lower vocabulary index,
    so merging splits in index order reproduces ``torch.argmax`` and
    ``torch.topk``'s lowest-index tie break.

    The top-k list is updated only from logits strictly above its current
    smallest member; after the first tiles almost no tile has such a logit, so
    the selection adds little to the streaming reduction.
    """
    row = tl.program_id(0).to(tl.int64)
    split = tl.program_id(1)
    logits_ptr += row * logits_stride
    start = split * split_size
    end = tl.minimum(start + split_size, vocab_size)
    lanes = tl.arange(0, VOCAB_BLOCK)

    row_max = float("-inf")
    exp_sum = 0.0
    argmax = 0
    slots = tl.arange(0, BLOCK_K)
    # Padding slots hold +inf so they are never the smallest member.
    top_values = tl.where(slots < TOP_K, float("-inf"), float("inf"))
    top_ids = tl.full([BLOCK_K], 2147483647, tl.int32)
    for offset in range(start, end, VOCAB_BLOCK):
        offsets = offset + lanes
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

        if TOP_K > 0:
            threshold = tl.min(top_values, axis=0)
            # Later indices never displace an equal member: strict ``>``.
            candidate = block > threshold
            num_candidates = tl.sum(candidate.to(tl.int32), axis=0)
            for _ in range(0, tl.minimum(num_candidates, TOP_K)):
                value, index = tl.max(
                    tl.where(candidate, block, float("-inf")),
                    axis=0,
                    return_indices=True,
                    return_indices_tie_break_left=True,
                )
                # Replace the weakest member: the smallest value, and among
                # equal values the highest index.
                is_weakest = top_values == threshold
                weakest_id = tl.max(tl.where(is_weakest, top_ids, -1), axis=0)
                weakest_slot = tl.min(
                    tl.where(is_weakest & (top_ids == weakest_id), slots, BLOCK_K),
                    axis=0,
                )
                replace = (slots == weakest_slot) & (value > threshold)
                top_values = tl.where(replace, value, top_values)
                top_ids = tl.where(replace, index + offset, top_ids)
                candidate = candidate & (lanes != index)
                threshold = tl.min(top_values, axis=0)

    tl.store(partial_max_ptr + row * NUM_SPLITS + split, row_max)
    tl.store(partial_exp_sum_ptr + row * NUM_SPLITS + split, exp_sum)
    tl.store(partial_argmax_ptr + row * NUM_SPLITS + split, argmax)
    if TOP_K > 0:
        topk_offsets = (row * NUM_SPLITS + split) * TOP_K + slots
        tl.store(partial_topk_values_ptr + topk_offsets, top_values, mask=slots < TOP_K)
        tl.store(partial_topk_ids_ptr + topk_offsets, top_ids, mask=slots < TOP_K)


@triton.jit
def _unary_head_combine_stats_kernel(
    logits_ptr,
    logits_stride,
    targets_ptr,
    partial_max_ptr,
    partial_exp_sum_ptr,
    partial_argmax_ptr,
    partial_topk_values_ptr,
    partial_topk_ids_ptr,
    lse_ptr,
    target_logit_ptr,
    argmax_ptr,
    topk_values_ptr,
    topk_ids_ptr,
    vocab_size,
    NUM_SPLITS: tl.constexpr,
    BLOCK_SPLITS: tl.constexpr,
    TOP_K: tl.constexpr,
    BLOCK_CANDIDATES: tl.constexpr,
):
    """Merge a row's split statistics and read its FP32 target logit.

    ``lse = max + log(sum_s exp_sum_s * exp(max_s - max))``; the argmax is taken
    from the first split that attains the row maximum. The top-k is written in
    descending value order, equal values by ascending index; ``torch.topk``
    selects the same set but leaves the order of equal values unspecified.
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

    if TOP_K > 0:
        candidates = tl.arange(0, BLOCK_CANDIDATES)
        candidate_mask = candidates < NUM_SPLITS * TOP_K
        candidate_offsets = row * NUM_SPLITS * TOP_K + candidates
        values = tl.load(
            partial_topk_values_ptr + candidate_offsets,
            mask=candidate_mask,
            other=float("-inf"),
        )
        ids = tl.load(
            partial_topk_ids_ptr + candidate_offsets,
            mask=candidate_mask,
            other=2147483647,
        )
        for rank in range(TOP_K):
            value = tl.max(values, axis=0)
            best_id = tl.min(tl.where(values == value, ids, 2147483647), axis=0)
            tl.store(topk_values_ptr + row * TOP_K + rank, value)
            tl.store(topk_ids_ptr + row * TOP_K + rank, best_id.to(tl.int64))
            values = tl.where(ids == best_id, float("-inf"), values)


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
