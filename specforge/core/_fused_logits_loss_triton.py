"""Internal CUDA kernels for core.fused_loss.fused_ce_l1_loss.

The public loss contract, validation and eager reference live in fused_loss.py.
This module is imported lazily only when the fused CE/L1 backend is selected.
"""

import torch
import triton
import triton.language as tl

MAX_FUSED_TOP_K = 64


def _statistics(logits, targets, top_k=0):
    rows = logits.shape[0]
    lse = torch.empty(rows, device=logits.device, dtype=torch.float32)
    target = torch.empty_like(lse)
    top1 = torch.empty_like(lse)
    ids = torch.empty(rows, device=logits.device, dtype=torch.long)
    fused_k = top_k if top_k <= MAX_FUSED_TOP_K else 0
    topv = torch.empty(rows, fused_k, device=logits.device, dtype=torch.float32)
    topi = torch.empty(rows, fused_k, device=logits.device, dtype=torch.long)
    if rows:
        _launch_stats(logits, targets, lse, target, ids, topv, topi, top1)
    if fused_k != top_k:
        topv, topi = torch.topk(logits, top_k, dim=-1)
        topv = topv.float()
    return lse, target, ids, top1, topv, topi


class _FusedCEL1Loss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, teacher, top_k):
        lse, target, ids, top1, topv, topi = _statistics(logits, targets, top_k)
        teacher_lse = teacher_ids = teacher_top1 = l1_dot = l1 = None
        rows, vocab = logits.shape
        if teacher is not None:
            teacher_lse, _, teacher_ids, teacher_top1, _, _ = _statistics(
                teacher, targets
            )
            tiles = triton.cdiv(vocab, 4096)
            partial = torch.empty(
                2, rows, tiles, device=logits.device, dtype=torch.float32
            )
            if rows:
                _l1_stats_kernel[(rows, tiles)](
                    logits,
                    teacher,
                    lse,
                    teacher_lse,
                    partial,
                    vocab,
                    rows * tiles,
                    TILES=tiles,
                    BLOCK=4096,
                )
            l1, l1_dot = partial.sum(-1).unbind(0)
        ctx.save_for_backward(logits, targets, lse, teacher, teacher_lse, l1_dot, topi)
        ctx.set_materialize_grads(False)
        diagnostics = [topi, ids, top1]
        if teacher is not None:
            diagnostics.extend([teacher_ids, teacher_top1])
        ctx.mark_non_differentiable(*diagnostics)
        return lse - target, l1, topv, topi, ids, top1, teacher_ids, teacher_top1

    @staticmethod
    def backward(ctx, grad_ce, grad_l1, grad_topv, *_diagnostics):
        if not ctx.needs_input_grad[0]:
            return None, None, None, None
        logits, targets, lse, teacher, teacher_lse, l1_dot, topi = ctx.saved_tensors
        grad_ce = torch.zeros_like(lse) if grad_ce is None else grad_ce.contiguous()
        has_l1 = teacher is not None and grad_l1 is not None
        grad_l1 = None if grad_l1 is None else grad_l1.contiguous()
        grad_logits = torch.empty_like(logits)
        rows, vocab = logits.shape
        if rows:
            block, warps = _grad_settings(vocab)
            _logits_grad_kernel[(rows, triton.cdiv(vocab, block))](
                logits,
                grad_logits,
                targets,
                lse,
                grad_ce,
                teacher,
                teacher_lse,
                l1_dot,
                grad_l1,
                vocab,
                BLOCK=block,
                HAS_L1=has_l1,
                num_warps=warps,
            )
            if grad_topv is not None and topi.shape[-1]:
                _topk_grad_kernel[(rows,)](
                    logits,
                    grad_logits,
                    targets,
                    lse,
                    grad_ce,
                    teacher,
                    teacher_lse,
                    l1_dot,
                    grad_l1,
                    topi,
                    grad_topv.contiguous(),
                    vocab,
                    topi.shape[-1],
                    BLOCK=triton.next_power_of_2(topi.shape[-1]),
                    HAS_L1=has_l1,
                )
        return grad_logits, None, None, None


@triton.jit
def _l1_stats_kernel(
    logits,
    teacher,
    lse,
    teacher_lse,
    partial,
    vocab,
    plane,
    TILES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    p = tl.exp(
        tl.load(logits + row * vocab + cols, cols < vocab, other=float("-inf")).to(
            tl.float32
        )
        - tl.load(lse + row)
    )
    q = tl.exp(
        tl.load(teacher + row * vocab + cols, cols < vocab, other=float("-inf")).to(
            tl.float32
        )
        - tl.load(teacher_lse + row)
    )
    sign = (p > q).to(tl.float32) - (p < q).to(tl.float32)
    tl.store(partial + row * TILES + tile, tl.sum(tl.abs(p - q), 0))
    tl.store(partial + plane + row * TILES + tile, tl.sum(p * sign, 0))


@triton.jit
def _loss_gradient(
    logits,
    targets,
    lse,
    grad_ce,
    teacher,
    teacher_lse,
    l1_dot,
    grad_l1,
    row,
    cols,
    mask,
    vocab,
    HAS_L1: tl.constexpr,
):
    x = tl.load(logits + row * vocab + cols, mask, other=float("-inf")).to(tl.float32)
    p = tl.exp(x - tl.load(lse + row))
    grad = tl.load(grad_ce + row) * (
        p - (cols == tl.load(targets + row)).to(tl.float32)
    )
    if HAS_L1:
        y = tl.load(teacher + row * vocab + cols, mask, other=float("-inf")).to(
            tl.float32
        )
        q = tl.exp(y - tl.load(teacher_lse + row))
        sign = (p > q).to(tl.float32) - (p < q).to(tl.float32)
        # d sum|p-q| / dx_i = p_i * (sign_i - sum_j p_j * sign_j).
        grad += tl.load(grad_l1 + row) * p * (sign - tl.load(l1_dot + row))
    return grad


@triton.jit
def _logits_grad_kernel(
    logits,
    grad_logits,
    targets,
    lse,
    grad_ce,
    teacher,
    teacher_lse,
    l1_dot,
    grad_l1,
    vocab,
    BLOCK: tl.constexpr,
    HAS_L1: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    cols = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    grad = _loss_gradient(
        logits,
        targets,
        lse,
        grad_ce,
        teacher,
        teacher_lse,
        l1_dot,
        grad_l1,
        row,
        cols,
        cols < vocab,
        vocab,
        HAS_L1,
    )
    tl.store(grad_logits + row * vocab + cols, grad, cols < vocab)


@triton.jit
def _topk_grad_kernel(
    logits,
    grad_logits,
    targets,
    lse,
    grad_ce,
    teacher,
    teacher_lse,
    l1_dot,
    grad_l1,
    topi,
    grad_topv,
    vocab,
    top_k,
    BLOCK: tl.constexpr,
    HAS_L1: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    k = tl.arange(0, BLOCK)
    cols = tl.load(topi + row * top_k + k, k < top_k, other=0)
    grad = _loss_gradient(
        logits,
        targets,
        lse,
        grad_ce,
        teacher,
        teacher_lse,
        l1_dot,
        grad_l1,
        row,
        cols,
        k < top_k,
        vocab,
        HAS_L1,
    )
    grad += tl.load(grad_topv + row * top_k + k, k < top_k, other=0)
    tl.store(grad_logits + row * vocab + cols, grad, k < top_k)


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
    top1=None,
):
    """Write each row's FP32 log-sum-exp, target logit, argmax, and top-k."""
    num_rows, vocab_size = logits.shape
    if top1 is None:
        top1 = torch.empty_like(lse)
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
    _logits_partial_stats_kernel[(num_rows, num_splits)](
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
    _logits_combine_stats_kernel[(num_rows,)](
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
        top1,
        vocab_size,
        NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits),
        TOP_K=top_k,
        BLOCK_CANDIDATES=triton.next_power_of_2(max(num_splits * top_k, 1)),
        num_warps=1,
    )


@triton.jit
def _logits_partial_stats_kernel(
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
def _logits_combine_stats_kernel(
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
    top1_ptr,
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
    tl.store(top1_ptr + row, 1.0 / exp_sum)
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
