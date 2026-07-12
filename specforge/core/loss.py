"""
This file incorporates code from Unsloth licensed under the Apache License, Version 2.0.
See the original Unsloth repository at https://github.com/unslothai/unsloth.
The idea of in-place backward pass is from Liger-Kernel.
See the original Liger-Kernel repository at https://github.com/linkedin/Liger-Kernel.
"""

from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl
from torch.utils.checkpoint import checkpoint


# Reference implementation
@torch.compile(dynamic=None)
def _compute_loss(logits, target_p, position_mask):
    logits = logits.float()
    out_logp = nn.LogSoftmax(dim=2)(logits)
    plogp = target_p * out_logp
    loss = -torch.sum(position_mask * plogp, 2).mean()
    return loss


def compute_accept_len(
    pred_ids_4d: torch.Tensor,
    target_ids_4d: torch.Tensor,
    valid_mask_4d: torch.Tensor,
) -> torch.Tensor:
    """Compute the consecutive accepted token count for each block."""
    correct = (pred_ids_4d == target_ids_4d) | (~valid_mask_4d)
    accept_prefix = correct.long().cumprod(dim=2) * valid_mask_4d.long()
    return accept_prefix.sum(dim=2).float()


def _compute_domino_chunk_weighted_losses(
    output_hidden_flat: torch.Tensor,
    prefix_states_flat: torch.Tensor,
    flat_targets: torch.Tensor,
    flat_weights: torch.Tensor,
    valid_token_count: torch.Tensor,
    lambda_base: float,
    start: int,
    end: int,
    lm_head: nn.Module,
    embed_proj: nn.Module,
    block_size: int,
    suffix_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    idx = torch.arange(start, end, device=output_hidden_flat.device)
    pos_in_block = idx % block_size
    block_idx = idx // block_size

    chunk_hidden = output_hidden_flat[idx]
    chunk_targets = flat_targets[idx]
    chunk_weights = flat_weights[idx]
    base_logits = lm_head(chunk_hidden)

    base_loss_per_token = F.cross_entropy(base_logits, chunk_targets, reduction="none")
    base_numer = (base_loss_per_token * chunk_weights).sum()

    final_numer = torch.zeros_like(base_numer)
    prefix_mask = pos_in_block < suffix_start
    if prefix_mask.any():
        final_numer = (
            final_numer
            + (base_loss_per_token[prefix_mask] * chunk_weights[prefix_mask]).sum()
        )

    suffix_mask = ~prefix_mask
    if suffix_mask.any():
        suffix_idx = block_idx[suffix_mask] * (block_size - suffix_start) + (
            pos_in_block[suffix_mask] - suffix_start
        )
        concat_features = torch.cat(
            [chunk_hidden[suffix_mask], prefix_states_flat[suffix_idx]], dim=-1
        )
        final_logits = base_logits[suffix_mask] + embed_proj(concat_features)
        final_loss_per_token = F.cross_entropy(
            final_logits, chunk_targets[suffix_mask], reduction="none"
        )
        final_numer = (
            final_numer + (final_loss_per_token * chunk_weights[suffix_mask]).sum()
        )

    final_loss = final_numer / valid_token_count
    base_loss = base_numer / valid_token_count
    loss = (1.0 - lambda_base) * final_loss + lambda_base * base_loss
    return loss, final_loss, base_loss


def compute_domino_chunked_weighted_losses(
    output_hidden: torch.Tensor,
    prefix_states: torch.Tensor,
    target_ids: torch.Tensor,
    weight_mask: torch.Tensor,
    lambda_base: float,
    logit_chunk_size: int,
    lm_head: nn.Module,
    embed_proj: nn.Module,
    block_size: int,
    suffix_start: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Domino CE in checkpointed token chunks to bound logit memory."""
    if logit_chunk_size <= 0:
        raise ValueError("logit_chunk_size must be positive")

    output_hidden_flat = output_hidden.reshape(-1, output_hidden.shape[-1])
    prefix_states_flat = prefix_states.reshape(-1, prefix_states.shape[-1])
    flat_targets = target_ids.reshape(-1)
    flat_weights = weight_mask.reshape(-1)
    valid_token_count = flat_weights.sum() + 1e-6

    loss = torch.zeros((), device=output_hidden.device)
    final_loss = torch.zeros((), device=output_hidden.device)
    base_loss = torch.zeros((), device=output_hidden.device)
    total_tokens = flat_targets.numel()

    for start in range(0, total_tokens, logit_chunk_size):
        end = min(total_tokens, start + logit_chunk_size)

        def chunk_fn(hidden_flat, prefix_flat, start=start, end=end):
            return _compute_domino_chunk_weighted_losses(
                output_hidden_flat=hidden_flat,
                prefix_states_flat=prefix_flat,
                flat_targets=flat_targets,
                flat_weights=flat_weights,
                valid_token_count=valid_token_count,
                lambda_base=lambda_base,
                start=start,
                end=end,
                lm_head=lm_head,
                embed_proj=embed_proj,
                block_size=block_size,
                suffix_start=suffix_start,
            )

        chunk_loss, chunk_final_loss, chunk_base_loss = checkpoint(
            chunk_fn,
            output_hidden_flat,
            prefix_states_flat,
            use_reentrant=False,
        )
        loss = loss + chunk_loss
        final_loss = final_loss + chunk_final_loss.detach()
        base_loss = base_loss + chunk_base_loss.detach()

    return loss, final_loss, base_loss


@torch.no_grad()
def compute_domino_chunked_metrics(
    output_hidden: torch.Tensor,
    prefix_states: torch.Tensor,
    target_ids: torch.Tensor,
    eval_weight_mask: torch.Tensor,
    final_loss: torch.Tensor,
    base_loss: torch.Tensor,
    lambda_base: float,
    logit_chunk_size: int,
    lm_head: nn.Module,
    embed_proj: nn.Module,
    block_size: int,
    suffix_start: int,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute Domino predictions and metrics without materializing full logits."""
    if logit_chunk_size <= 0:
        raise ValueError("logit_chunk_size must be positive")

    output_hidden_flat = output_hidden.reshape(-1, output_hidden.shape[-1])
    prefix_states_flat = prefix_states.reshape(-1, prefix_states.shape[-1])
    flat_targets = target_ids.reshape(-1)
    binary_eval_mask = eval_weight_mask.reshape(-1)
    actual_token_count = binary_eval_mask.sum() + 1e-6
    total_tokens = flat_targets.numel()

    pred_ids = torch.empty_like(flat_targets)
    base_pred_ids = torch.empty_like(flat_targets)

    for start in range(0, total_tokens, logit_chunk_size):
        end = min(total_tokens, start + logit_chunk_size)
        idx = torch.arange(start, end, device=output_hidden.device)
        pos_in_block = idx % block_size
        block_idx = idx // block_size
        chunk_hidden = output_hidden_flat[idx]
        base_logits = lm_head(chunk_hidden)
        chunk_base_pred_ids = torch.argmax(base_logits, dim=-1)
        chunk_pred_ids = chunk_base_pred_ids.clone()

        suffix_mask = pos_in_block >= suffix_start
        if suffix_mask.any():
            suffix_idx = block_idx[suffix_mask] * (block_size - suffix_start) + (
                pos_in_block[suffix_mask] - suffix_start
            )
            concat_features = torch.cat(
                [chunk_hidden[suffix_mask], prefix_states_flat[suffix_idx]], dim=-1
            )
            final_logits = base_logits[suffix_mask] + embed_proj(concat_features)
            chunk_pred_ids[suffix_mask] = torch.argmax(final_logits, dim=-1)

        pred_ids[idx] = chunk_pred_ids
        base_pred_ids[idx] = chunk_base_pred_ids

    correct = (pred_ids == flat_targets) & (binary_eval_mask > 0.5)
    accuracy = correct.sum().float() / actual_token_count

    base_correct = (base_pred_ids == flat_targets) & (binary_eval_mask > 0.5)
    base_accuracy = base_correct.sum().float() / actual_token_count

    bsz, n, bs = target_ids.shape
    valid_mask_4d = (eval_weight_mask > 0).bool()
    pred_accept_len = compute_accept_len(
        pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
    )
    base_accept_len = compute_accept_len(
        base_pred_ids.view(bsz, n, bs), target_ids, valid_mask_4d
    )

    valid_block_mask = valid_mask_4d.any(dim=2)
    num_valid_blocks = valid_block_mask.sum().float() + 1e-6
    avg_accept_len = (
        (pred_accept_len + 1.0) * valid_block_mask.float()
    ).sum() / num_valid_blocks
    base_avg_accept_len = (
        (base_accept_len + 1.0) * valid_block_mask.float()
    ).sum() / num_valid_blocks

    metrics = {
        "final_loss": final_loss.detach(),
        "base_loss": base_loss.detach(),
        "base_accuracy": base_accuracy.detach(),
        "accept_len": avg_accept_len.detach(),
        "base_accept_len": base_avg_accept_len.detach(),
        "lambda_base": torch.tensor(lambda_base, device=final_loss.device),
    }
    return accuracy, metrics


def _calculate_settings(n):
    # reference: https://github.com/unslothai/unsloth/blob/fd753fed99ed5f10ef8a9b7139588d9de9ddecfb/unsloth/kernels/utils.py#L43

    MAX_FUSED_SIZE = 131072
    BLOCK_SIZE = triton.next_power_of_2(n)
    if BLOCK_SIZE > MAX_FUSED_SIZE:
        raise RuntimeError(
            f"Cannot launch Triton kernel since n = {n} exceeds the recommended Triton blocksize = {MAX_FUSED_SIZE}."
        )

    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8

    # AMD GPU (ROCm)
    if hasattr(torch.version, "hip") and torch.version.hip is not None:
        num_warps //= 2

    return BLOCK_SIZE, num_warps


@triton.jit
def log_softmax_forward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    position_mask_stride,
    loss_ptr,
    loss_stride,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id * position_mask_stride
    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        return

    m = float("-inf")
    d = 0.0

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(
            logits_ptr + offsets, mask=mask, other=float("-inf")
        ).cast(tl.float32)
        block_max = tl.max(tl.where(mask, logits_block, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(
            tl.where(mask, tl.exp(logits_block - m_new), 0.0)
        )
        m = m_new

    loss = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        # log-softmax: log(exp(x - max) / sum) = (x - max) - log(sum)
        normalized_logits = logits_block - m
        log_normalizer = tl.log(d)
        log_softmax_logits = normalized_logits - log_normalizer
        weighted_log_prob = target_block * log_softmax_logits
        loss += tl.sum(tl.where(mask, weighted_log_prob, 0.0))

    loss_ptr += program_id * loss_stride
    m_ptr += program_id
    d_ptr += program_id
    tl.store(loss_ptr, -loss)
    tl.store(m_ptr, m.to(tl.float32))
    tl.store(d_ptr, d.to(tl.float32))


@triton.jit
def log_softmax_backward_kernel(
    logits_ptr,
    logits_stride,
    target_ptr,
    target_stride,
    position_mask_ptr,
    grad_output_ptr,
    scaling_factor,
    m_ptr,
    d_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0).to(tl.int64)
    logits_ptr += program_id * logits_stride
    target_ptr += program_id * target_stride
    position_mask_ptr += program_id

    position_mask = tl.load(position_mask_ptr)
    if position_mask == 0:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(logits_ptr + offsets, 0.0, mask=mask)
        return

    m_ptr += program_id
    d_ptr += program_id
    m = tl.load(m_ptr).to(tl.float32)
    d = tl.load(d_ptr).to(tl.float32)
    grad_output = tl.load(grad_output_ptr).to(tl.float32)
    grad_output = grad_output * scaling_factor

    # First pass: compute sum of (target * grad_output)
    target_grad_sum = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_grad_sum += tl.sum(tl.where(mask, target_block * grad_output, 0.0))

    # Second pass: compute log-softmax gradients
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        logits_block = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        target_block = tl.load(target_ptr + offsets, mask=mask, other=0.0).cast(
            tl.float32
        )
        softmax_prob = tl.exp(logits_block - m) / d
        normalized_grad = softmax_prob * target_grad_sum
        grad_block = -(target_block * grad_output - normalized_grad)
        tl.store(logits_ptr + offsets, grad_block.to(tl.float32), mask=mask)


class LogSoftmaxLoss(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, target, position_mask):
        B, T, V = logits.shape
        loss = torch.zeros((B * T, 1), device=logits.device)
        logits_flat = logits.contiguous().view(B * T, V)
        target_flat = target.contiguous().view(B * T, V)
        position_mask_flat = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        m = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        d = torch.zeros((B * T,), device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_forward_kernel[grid](
            logits_flat,
            logits_flat.stride(0),
            target_flat,
            target_flat.stride(0),
            position_mask_flat,
            position_mask_flat.stride(0),
            loss,
            loss.stride(0),
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        ctx.save_for_backward(logits.detach(), target, position_mask, m, d)
        return loss.squeeze(1).mean()

    @staticmethod
    def backward(ctx, grad_output):
        logits, target, position_mask, m, d = ctx.saved_tensors
        B, T, V = logits.shape
        scaling_factor = 1.0 / (B * T)
        logits = logits.contiguous().view(B * T, V)
        target = target.contiguous().view(B * T, V)
        position_mask = position_mask.contiguous().view(B * T, 1).bool()
        grid = (B * T,)
        BLOCK_SIZE, num_warps = _calculate_settings(V)
        log_softmax_backward_kernel[grid](
            logits,
            logits.stride(0),
            target,
            target.stride(0),
            position_mask,
            grad_output,
            scaling_factor,
            m,
            d,
            V,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        logits = logits.view(B, T, V)
        return logits, None, None


if __name__ == "__main__":
    device = "cuda"
    B, T, V = 1, 1024, 16000
    logits = torch.randn(B, T, V, device=device, requires_grad=True)
    logits2 = logits.clone().detach().requires_grad_(True)
    target = torch.randn(B, T, V, device=device)
    position_mask = torch.randint(0, 2, (B, T, 1), dtype=torch.bool, device=device)
    position_mask = torch.ones((B, T, 1), dtype=torch.bool, device=device)
    output1 = LogSoftmaxLoss.apply(logits, target, position_mask)
    output2 = _compute_loss(logits2, target, position_mask)
    torch.testing.assert_close(output1, output2, rtol=1e-4, atol=1e-4)
    output1.backward()
    output2.backward()
    torch.testing.assert_close(logits.grad, logits2.grad, rtol=1e-4, atol=1e-4)
