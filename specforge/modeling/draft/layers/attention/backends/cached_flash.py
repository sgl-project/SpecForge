"""Tensor-only EAGLE branch-cache kernels, shared by FlashAttention and USP."""

from typing import Optional

import torch
import torch.distributed as dist

try:
    from flash_attn import flash_attn_varlen_func as _std_flash_attn_varlen_func
    from flash_attn.bert_padding import pad_input as _std_flash_pad_input
    from flash_attn.bert_padding import unpad_input as _std_flash_unpad_input
    from flash_attn.flash_attn_interface import (
        _flash_attn_varlen_backward as _std_flash_attn_varlen_backward,
    )
except ImportError as exc:
    _std_flash_attn_varlen_func = None
    _std_flash_pad_input = None
    _std_flash_unpad_input = None
    _std_flash_attn_varlen_backward = None
    _std_flash_attn_import_error = exc
else:
    _std_flash_attn_import_error = None


def _raise_standard_flash_attn_unavailable() -> None:
    raise RuntimeError(
        "LlamaFlashAttention requires the standard flash-attn interface "
        f"(import error: {_std_flash_attn_import_error!r})"
    )


def _standard_flash_attn_varlen_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    attention_mask: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(
        q, attention_mask
    )
    k_unpad, _, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(
        k, attention_mask
    )
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    out_unpad, lse_unpad, _ = _std_flash_attn_varlen_func(
        q_unpad,
        k_unpad,
        v_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p=0.0,
        softmax_scale=softmax_scale,
        causal=causal,
        return_attn_probs=True,
    )
    out = _std_flash_pad_input(out_unpad, indices_q, q.shape[0], q.shape[1])
    lse_padded = _std_flash_pad_input(
        lse_unpad.transpose(0, 1), indices_q, q.shape[0], q.shape[1]
    ).transpose(1, 2)
    return out, lse_padded, indices_q, cu_seqlens_q


def _standard_flash_attn_varlen_backward_call(
    dout: torch.Tensor,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    out: torch.Tensor,
    softmax_lse: torch.Tensor,
    attention_mask: torch.Tensor,
    dq: torch.Tensor,
    dk: torch.Tensor,
    dv: torch.Tensor,
    softmax_scale: float,
    causal: bool,
) -> None:
    q_unpad, indices_q, cu_seqlens_q, max_seqlen_q, _ = _std_flash_unpad_input(
        q, attention_mask
    )
    k_unpad, indices_k, cu_seqlens_k, max_seqlen_k, _ = _std_flash_unpad_input(
        k, attention_mask
    )
    v_unpad, _, _, _, _ = _std_flash_unpad_input(v, attention_mask)
    dout_unpad, _, _, _, _ = _std_flash_unpad_input(dout, attention_mask)
    out_unpad, _, _, _, _ = _std_flash_unpad_input(out, attention_mask)
    lse_unpad, _, _, _, _ = _std_flash_unpad_input(
        softmax_lse.transpose(1, 2), attention_mask
    )
    lse_unpad = lse_unpad.transpose(0, 1).contiguous()

    dq_unpad = torch.empty_like(q_unpad)
    dk_unpad = torch.empty_like(k_unpad)
    dv_unpad = torch.empty_like(v_unpad)
    _std_flash_attn_varlen_backward(
        dout_unpad,
        q_unpad,
        k_unpad,
        v_unpad,
        out_unpad,
        lse_unpad,
        dq_unpad,
        dk_unpad,
        dv_unpad,
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        0.0,
        softmax_scale,
        causal,
        -1,
        -1,
        0.0,
        None,
        False,
        None,
    )
    dq.copy_(_std_flash_pad_input(dq_unpad, indices_q, q.shape[0], q.shape[1]))
    dk.copy_(_std_flash_pad_input(dk_unpad, indices_k, k.shape[0], k.shape[1]))
    dv.copy_(_std_flash_pad_input(dv_unpad, indices_k, v.shape[0], v.shape[1]))


def _cached_merge_dims(q: torch.Tensor, cache_k: torch.Tensor):
    bsz, q_len, num_heads, head_dim = q.shape
    num_blocks = cache_k.shape[1]
    num_kv_heads = cache_k.shape[3]
    num_groups = num_heads // num_kv_heads
    return bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups


def _kernel_lse(lse: torch.Tensor, bsz: int, q_len: int, num_heads: int):
    return lse.reshape(bsz, q_len, num_heads).transpose(1, 2).contiguous()


def _current_valid_lengths(
    attention_mask: Optional[torch.Tensor],
    bsz: int,
    q_len: int,
    lck: int,
    device: torch.device,
) -> torch.Tensor:
    if attention_mask is None or attention_mask.dim() != 2:
        return torch.full((bsz,), q_len, dtype=torch.long, device=device)
    return attention_mask.sum(dim=-1, dtype=torch.long).sub(lck).clamp_(0, q_len)


class _FlashCachedMergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, cache_k, cache_v, valid_lengths, softmax_scale: float):
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _cached_merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long).clamp_(
            0, q_len
        )
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0)
            < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)

        k0 = cache_k[:, 0].contiguous()
        v0 = cache_v[:, 0].contiguous()
        out0, lse0_kernel, _, _ = _standard_flash_attn_varlen_forward(
            q.contiguous(),
            k0,
            v0,
            attention_mask,
            softmax_scale=softmax_scale,
            causal=True,
        )
        out0_expanded = out0.view(
            bsz, q_len, num_kv_heads, num_groups, head_dim
        ).float()
        neg_inf = float("-inf")
        lse0 = (
            lse0_kernel.transpose(1, 2)
            .view(bsz, q_len, num_kv_heads, num_groups)
            .float()
        )
        lse0 = torch.where(valid_mask, lse0, neg_inf)
        lse_terms = [lse0]
        attn_terms = [out0_expanded]
        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            lse_terms.append(torch.where(valid_mask, lse_i, neg_inf))
            attn_terms.append(vi.expand_as(out0_expanded))

        merged_lse = torch.logsumexp(torch.stack(lse_terms, dim=-1), dim=-1)
        out = sum(
            term * torch.exp(lse - merged_lse).unsqueeze(-1)
            for term, lse in zip(attn_terms, lse_terms)
        )
        out = torch.where(valid_mask.unsqueeze(-1), out, 0.0)
        merged_lse = torch.where(valid_mask, merged_lse, 0.0)
        ctx.save_for_backward(q, cache_k, cache_v, out, merged_lse, valid_lengths)
        ctx.softmax_scale = softmax_scale
        return out.to(q.dtype).reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out):
        q, cache_k, cache_v, out, merged_lse, valid_lengths = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _cached_merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale
        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)
        valid_lengths = valid_lengths.to(device=q.device, dtype=torch.long)
        valid_mask = (
            torch.arange(q_len, device=q.device).unsqueeze(0)
            < valid_lengths.unsqueeze(1)
        ).view(bsz, q_len, 1, 1)
        attention_mask = valid_mask.view(bsz, q_len)
        grad_out = torch.where(valid_mask, grad_out, 0.0)

        grad_out_f = grad_out.float().view(
            bsz, q_len, num_kv_heads, num_groups, head_dim
        )
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_f = out.float()
        out_expanded = out_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_q = out.to(q.dtype).reshape(bsz, q_len, num_heads, head_dim)

        dq = torch.zeros_like(q_f)
        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())

        dq0 = torch.zeros_like(q)
        dk0 = torch.zeros_like(cache_k[:, 0])
        dv0 = torch.zeros_like(cache_v[:, 0])
        merged_lse_kernel = _kernel_lse(merged_lse, bsz, q_len, num_heads)
        _standard_flash_attn_varlen_backward_call(
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out_q.contiguous(),
            merged_lse_kernel,
            attention_mask,
            dq0,
            dk0,
            dv0,
            softmax_scale=scale,
            causal=True,
        )
        dq += dq0.float()
        dcache_k[:, 0] += dk0.float()
        dcache_v[:, 0] += dv0.float()

        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.where(valid_mask, torch.exp(lse_i - merged_lse), 0.0)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (
                grad_out_f * (vi.expand_as(out_expanded) - out_expanded)
            ).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return (
            dq.to(q.dtype),
            dcache_k.to(cache_k.dtype),
            dcache_v.to(cache_v.dtype),
            None,
            None,
        )


def _update_ring_out_and_lse(
    out: torch.Tensor | None,
    lse: torch.Tensor | None,
    block_out: torch.Tensor,
    block_lse: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    block_out = block_out.float()
    block_lse = block_lse.float()
    if out is None or lse is None:
        return block_out, block_lse
    new_lse = torch.logaddexp(lse, block_lse)
    out = out * torch.exp(lse - new_lse).unsqueeze(-1) + block_out * torch.exp(
        block_lse - new_lse
    ).unsqueeze(-1)
    return out, new_lse


class _USPRingFlashCachedMergeFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, q, cache_k, cache_v, ring_group: dist.ProcessGroup, softmax_scale: float
    ):
        from specforge.layers.ring.ring_flash_attn import ring_flash_attn_forward

        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _cached_merge_dims(q, cache_k)
        )
        q_expanded = q.view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        out_ring, lse_ring = ring_flash_attn_forward(
            ring_group,
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            softmax_scale=softmax_scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        if lse_ring.dim() == 3 and lse_ring.shape[1] == num_heads:
            lse_ring = lse_ring.transpose(1, 2)
        acc_out = out_ring.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        acc_lse = lse_ring.reshape(bsz, q_len, num_kv_heads, num_groups)

        for i in range(1, num_blocks):
            ki = cache_k[:, i].unsqueeze(-2).float()
            vi = cache_v[:, i].unsqueeze(-2).float()
            lse_i = (q_expanded.float() * ki).sum(-1) * softmax_scale
            out_i = vi.expand(bsz, q_len, num_kv_heads, num_groups, head_dim)
            acc_out, acc_lse = _update_ring_out_and_lse(acc_out, acc_lse, out_i, lse_i)

        ctx.save_for_backward(q, cache_k, cache_v, acc_out, acc_lse)
        ctx.ring_group = ring_group
        ctx.softmax_scale = softmax_scale
        return acc_out.to(q.dtype).reshape_as(q)

    @staticmethod
    def backward(ctx, grad_out):
        from specforge.layers.ring.ring_flash_attn import ring_flash_attn_backward

        q, cache_k, cache_v, out, merged_lse = ctx.saved_tensors
        bsz, q_len, num_heads, head_dim, num_blocks, num_kv_heads, num_groups = (
            _cached_merge_dims(q, cache_k)
        )
        scale = ctx.softmax_scale

        if grad_out.ndim == 3:
            grad_out = grad_out.view(bsz, q_len, num_heads, head_dim)
        grad_out_f = grad_out.float().view(
            bsz, q_len, num_kv_heads, num_groups, head_dim
        )
        q_f = q.float()
        q_expanded = q_f.view(bsz, q_len, num_kv_heads, num_groups, head_dim)
        out_expanded = out.float().view(bsz, q_len, num_kv_heads, num_groups, head_dim)

        dcache_k = torch.zeros_like(cache_k.float())
        dcache_v = torch.zeros_like(cache_v.float())
        merged_lse_kernel = _kernel_lse(merged_lse, bsz, q_len, num_heads)
        dq, dk0, dv0 = ring_flash_attn_backward(
            ctx.ring_group,
            grad_out.contiguous(),
            q.contiguous(),
            cache_k[:, 0].contiguous(),
            cache_v[:, 0].contiguous(),
            out.to(q.dtype).reshape_as(q).contiguous(),
            merged_lse_kernel,
            softmax_scale=scale,
            dropout_p=0.0,
            causal=True,
            window_size=(-1, -1),
            alibi_slopes=None,
            deterministic=False,
        )
        dq = dq.float()
        dcache_k[:, 0] = dk0.float()
        dcache_v[:, 0] = dv0.float()

        for i in range(1, num_blocks):
            ki = cache_k[:, i].float().unsqueeze(-2)
            vi = cache_v[:, i].float().unsqueeze(-2)
            lse_i = (q_expanded * ki).sum(-1) * scale
            wi = torch.exp(lse_i - merged_lse)
            d_out_i = grad_out_f * wi.unsqueeze(-1)
            d_lse_i = wi * (
                grad_out_f * (vi.expand_as(out_expanded) - out_expanded)
            ).sum(-1)
            dq += (d_lse_i.unsqueeze(-1) * scale * ki).reshape_as(q)
            dcache_k[:, i] += (d_lse_i.unsqueeze(-1) * scale * q_expanded).sum(dim=3)
            dcache_v[:, i] += d_out_i.sum(dim=3)

        return (
            dq.to(q.dtype),
            dcache_k.to(cache_k.dtype),
            dcache_v.to(cache_v.dtype),
            None,
            None,
        )
