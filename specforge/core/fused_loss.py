"""
This file incorporates code from vllm-project/speculators licensed under
the Apache License, Version 2.0. See the original repository at
https://github.com/vllm-project/speculators (file: src/speculators/losses/fused.py
and losses/eager.py), derived in turn from specforge/core/loss.py
(Unsloth / Liger-Kernel lineage).
Changes for SpecForge: lazy Triton import so the module loads on hosts
without Triton; eager reference co-located in the same module.
The label-ids CE variant (``_OP_CE_LABEL``, ``fused_ce_label_loss`` /
``ce_label_loss``) is a SpecForge addition with no upstream counterpart.

Upstream kernel documentation (speculators losses/fused.py):

One kernel pair serves every fused loss: forward streams the selected
reduction from online-softmax statistics; backward recomputes probabilities
from five saved scalars per row and applies the closed-form gradient. No
``[T, V]`` intermediate is ever materialized or saved -- the memory win over
the eager losses. ``OP`` picks the reduction at Triton specialization time;
``nla`` and ``lk_hybrid`` are ``[1, T]`` torch compositions of the primitives
instead of kernel branches.

Per-row gradients (dp = softmax(logits), draft; tp = softmax(targets),
treated as constant), L the row loss:

    kl_div   dp - tp
    rkl      dp * (log dp - log tp - L)
    jsd      0.5 * dp * (log(dp / mix) - KL(dp || mix)),  mix = (dp + tp) / 2
    ce       dp - onehot(argmax tp)
    ce_label dp - onehot(label id); reads one int32 label per row instead of
             the [T, V] targets (SpecForge addition)
    tv       -dp * (1[dp <= tp] - s_s),  s_s = sum of dp where dp <= tp
"""

# Triton kernels idiomatically use uppercase constexpr/dim names (B, T, V,
# BLOCK_SIZE) and inline block-size tuning thresholds; exempt this file from the
# pep8-naming and magic-value lints rather than fight those conventions.
# ruff: noqa: N803, N806, PLR2004

from __future__ import annotations

import torch

__all__ = [
    # Fused Triton losses (require Triton plus a CUDA or Ascend NPU accelerator)
    "fused_kl_div_loss",
    "fused_reverse_kl_div_loss",
    "fused_js_div_loss",
    "fused_ce_loss",
    "fused_ce_label_loss",
    "fused_tv_loss",
    "fused_nla_loss",
    "fused_lk_hybrid_loss",
    # Eager reference/fallback implementations (pure PyTorch, CPU-safe)
    "kl_div_loss",
    "reverse_kl_div_loss",
    "js_div_loss",
    "ce_loss",
    "ce_label_loss",
    "tv_loss",
    "neg_log_acceptance_loss",
    "lk_hybrid_loss",
]

MAX_FUSED_SIZE = 131072
# Ascend NPU's Unified Buffer (~192 KB) cannot fit the double-row load these
# kernels perform (logits + targets per block) beyond 4096 elements per block;
# triton-ascend raises "ub overflow" at BLOCK_SIZE >= 8192. CE is single-row
# but shares the same cap so all _FusedLoss OPs use one path.
MAX_FUSED_SIZE_NPU = 4096

# tl.constexpr instances: Triton kernels may only read globals wrapped this way.
# Plain values live at module scope so importing never needs Triton;
# _require_triton() re-wraps them as tl.constexpr before the first launch.
_LOG2 = 0.6931471805599453

# Reduction selector, baked into the kernel at specialization time. Pass
# ``.value`` (a plain int) across the autograd boundary -- Dynamo cannot
# represent a constexpr object as an autograd.Function argument and would
# graph-break; the ``OP: tl.constexpr`` parameter re-wraps it in the kernel.
_OP_KL = 0
_OP_RKL = 1
_OP_JSD = 2
_OP_CE = 3
_OP_TV = 4
# SpecForge addition: CE against ground-truth label ids (one int32 per row)
# instead of a [T, V] target distribution -- the MTP/DFlash-style path where
# no teacher distribution exists.
_OP_CE_LABEL = 5

# stats [_N_STATS, n_rows]: [0]/[1] draft row max / sum-exp, [2]/[3] same for
# targets (unused by ce), [4] per-OP -- kl_div/rkl: row loss L | jsd:
# KL(dp||mix) | tv: s_s | ce: argmax as float32 (exact: vocab < 2**24) |
# ce_label: label id as float32, or the -1 sentinel for ignored rows.
_N_STATS = 5

_NLA_EPS = 1e-5

_TRITON_READY = False


def _next_power_of_2(n):
    """Pure-Python twin of ``triton.next_power_of_2``.

    ``_calculate_settings`` must stay Triton-free so the device-cap logic (and
    its unit tests) runs on hosts without Triton.
    """
    return 1 << (n - 1).bit_length() if n > 1 else 1


def _calculate_settings(n, device):
    max_size = MAX_FUSED_SIZE_NPU if device.type == "npu" else MAX_FUSED_SIZE
    BLOCK_SIZE = min(_next_power_of_2(n), max_size)
    # triton-ascend does not require extra num_warps tuning
    num_warps = 4
    if BLOCK_SIZE >= 32768:
        num_warps = 32
    elif BLOCK_SIZE >= 8192:
        num_warps = 16
    elif BLOCK_SIZE >= 2048:
        num_warps = 8
    if getattr(torch.version, "hip", None) is not None:  # AMD wavefronts are 64-wide
        num_warps //= 2
    return BLOCK_SIZE, num_warps


# The kernel bodies below are verbatim from speculators losses/fused.py. Their
# ``@triton.jit`` decorators are applied lazily by ``_require_triton()`` below
# (which also publishes ``tl`` and the constexpr-wrapped globals the bodies
# resolve at compile time), so importing this module never requires Triton.
# The ``tl.constexpr`` annotations stay unevaluated strings thanks to the
# ``__future__`` annotations import at the top of the file.
def _online_stats(row_ptr, n_cols, BLOCK_SIZE: tl.constexpr):
    """Online max (m) and sum-exp (d) over one row."""
    m = float("-inf")
    d = 0.0
    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(row_ptr + offsets, mask=mask, other=float("-inf")).cast(tl.float32)
        block_max = tl.max(tl.where(mask, x, float("-inf")))
        m_new = tl.maximum(m, block_max)
        d = d * tl.exp(m - m_new) + tl.sum(tl.where(mask, tl.exp(x - m_new), 0.0))
        m = m_new
    return m, d


def _log_mix(ldp, ltp):
    """log((exp(ldp) + exp(ltp)) / 2): the JSD mixture, stable in log space."""
    mx = tl.maximum(ldp, ltp)
    return mx + tl.log(tl.exp(ldp - mx) + tl.exp(ltp - mx)) - _LOG2


def loss_forward_kernel(  # noqa: C901 -- constexpr OP branches, pruned per instance
    logits_ptr,
    targets_ptr,
    loss_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """One row per program: online softmax stats, then the OP's reduction.

    Masked tail lanes can hold inf/nan before ``tl.where`` selects; they
    never enter an accumulator.
    """
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    # targets_ptr is NOT offset here: its layout depends on OP. The
    # distribution ops carry [n_rows, n_cols] target logits (advanced by
    # pid * n_cols inside their branches); ce_label carries one int32 label
    # per row (indexed by pid directly).

    m_d, z_d = _online_stats(logits_ptr, n_cols, BLOCK_SIZE)
    lse_d = m_d + tl.log(z_d)
    tl.store(stats_ptr + 0 * stats_row + pid, m_d)
    tl.store(stats_ptr + 1 * stats_row + pid, z_d)

    if OP == _OP_CE:
        # Only the target argmax is needed; no target softmax stats.
        targets_ptr += pid * n_cols
        best_val = float("-inf")
        best_idx = 0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            y = tl.load(targets_ptr + offsets, mask=mask, other=float("-inf")).cast(
                tl.float32
            )
            block_max = tl.max(y)
            block_arg = tl.argmax(y, axis=0)
            # strict > keeps the first occurrence, the torch.argmax convention
            update = block_max > best_val
            best_idx = tl.where(update, i + block_arg, best_idx)
            best_val = tl.where(update, block_max, best_val)
        logit_at_target = tl.load(logits_ptr + best_idx).cast(tl.float32)
        tl.store(loss_ptr + pid, lse_d - logit_at_target)
        # mypy reads traced Triton code as Python and types best_idx as int
        tl.store(stats_ptr + 4 * stats_row + pid, best_idx.to(tl.float32))  # type: ignore[attr-defined]
    elif OP == _OP_CE_LABEL:
        # Ground-truth label ids: one int32 per row, no [n_rows, n_cols]
        # targets to scan. Labels outside [0, n_cols) -- e.g. the -100 ignore
        # index -- give loss 0 and the -1 sentinel in stats[4], which the
        # backward kernel maps to an all-zero gradient row.
        label = tl.load(targets_ptr + pid).to(tl.int32)
        in_range = (label >= 0) & (label < n_cols)
        safe_label = tl.where(in_range, label, 0)
        logit_at_label = tl.load(logits_ptr + safe_label).cast(tl.float32)
        tl.store(loss_ptr + pid, tl.where(in_range, lse_d - logit_at_label, 0.0))
        tl.store(
            stats_ptr + 4 * stats_row + pid,
            tl.where(in_range, label, -1).to(tl.float32),
        )
    else:
        targets_ptr += pid * n_cols
        m_t, z_t = _online_stats(targets_ptr, n_cols, BLOCK_SIZE)
        lse_t = m_t + tl.log(z_t)
        tl.store(stats_ptr + 2 * stats_row + pid, m_t)
        tl.store(stats_ptr + 3 * stats_row + pid, z_t)

        acc = 0.0
        extra = 0.0
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ldp = x - lse_d
            ltp = y - lse_t
            dp = tl.exp(ldp)
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                acc += tl.sum(tl.where(mask, tp * (ltp - ldp), 0.0))
            elif OP == _OP_RKL:
                acc += tl.sum(tl.where(mask, dp * (ldp - ltp), 0.0))
            elif OP == _OP_JSD:
                lmp = _log_mix(ldp, ltp)
                acc += tl.sum(tl.where(mask, dp * (ldp - lmp), 0.0))
                extra += tl.sum(tl.where(mask, tp * (ltp - lmp), 0.0))
            elif OP == _OP_TV:
                acc += tl.sum(tl.where(mask, tl.minimum(dp, tp), 0.0))
                extra += tl.sum(tl.where(mask & (dp <= tp), dp, 0.0))

        # Lint exemptions: `in`/merged compares aren't constexpr-foldable here.
        if OP == _OP_KL or OP == _OP_RKL:  # noqa: PLR1714, SIM109
            tl.store(loss_ptr + pid, acc)
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_JSD:
            tl.store(loss_ptr + pid, 0.5 * (acc + extra))
            tl.store(stats_ptr + 4 * stats_row + pid, acc)
        elif OP == _OP_TV:
            tl.store(loss_ptr + pid, 1.0 - acc)
            tl.store(stats_ptr + 4 * stats_row + pid, extra)


def loss_backward_kernel(
    logits_ptr,
    targets_ptr,
    grad_in_ptr,
    grad_out_ptr,
    stats_ptr,
    stats_row,
    n_cols,
    OP: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Recompute probabilities from the saved stats and apply the OP gradient."""
    pid = tl.program_id(0).to(tl.int64)
    logits_ptr += pid * n_cols
    grad_in_ptr += pid * n_cols

    go = tl.load(grad_out_ptr + pid).cast(tl.float32)
    zero_row = go == 0.0
    if OP == _OP_CE_LABEL:
        # Ignored positions carry the -1 sentinel in stats[4]; their gradient
        # is the all-zero row (same as a zero upstream gradient).
        zero_row = zero_row | (tl.load(stats_ptr + 4 * stats_row + pid) < 0.0)
    if zero_row:
        for i in range(0, n_cols, BLOCK_SIZE):
            offsets = i + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            tl.store(grad_in_ptr + offsets, 0.0, mask=mask)
        return

    m_d = tl.load(stats_ptr + 0 * stats_row + pid)
    z_d = tl.load(stats_ptr + 1 * stats_row + pid)
    lse_d = m_d + tl.log(z_d)
    extra = tl.load(stats_ptr + 4 * stats_row + pid)
    if OP == _OP_CE or OP == _OP_CE_LABEL:  # noqa: PLR1714, SIM109
        # Both CE variants pass None for targets_ptr; the target index
        # (argmax for CE, label id for ce_label) travels via stats[4].
        # Ignored ce_label rows already took the early-out above.
        target_idx = extra.to(tl.int32)
    else:
        targets_ptr += pid * n_cols
        m_t = tl.load(stats_ptr + 2 * stats_row + pid)
        z_t = tl.load(stats_ptr + 3 * stats_row + pid)
        lse_t = m_t + tl.log(z_t)

    for i in range(0, n_cols, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_cols
        x = tl.load(logits_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
        ldp = x - lse_d
        dp = tl.exp(ldp)
        if OP == _OP_CE or OP == _OP_CE_LABEL:  # noqa: PLR1714, SIM109
            grad = dp - (offsets == target_idx).to(tl.float32)
        else:
            y = tl.load(targets_ptr + offsets, mask=mask, other=0.0).cast(tl.float32)
            ltp = y - lse_t
            tp = tl.exp(ltp)
            if OP == _OP_KL:
                grad = dp - tp
            elif OP == _OP_RKL:
                grad = dp * (ldp - ltp - extra)
            elif OP == _OP_JSD:
                grad = 0.5 * dp * (ldp - _log_mix(ldp, ltp) - extra)
            elif OP == _OP_TV:
                grad = -dp * ((dp <= tp).to(tl.float32) - extra)
        tl.store(grad_in_ptr + offsets, go * grad, mask=mask)


def _require_triton():
    """Import Triton and assemble the fused kernels on first fused-path use.

    Triton is imported here rather than at module scope so this file still
    imports on hosts without Triton (CPU-only boxes, or NPU images before
    triton-ascend is installed); only the fused losses then fail, with a clear
    error, while the eager references below remain usable.
    """
    global _TRITON_READY
    if _TRITON_READY:
        return
    try:
        import triton
        import triton.language as tl
    except ImportError as exc:
        raise RuntimeError(
            "The fused losses in specforge.core.fused_loss require Triton "
            "(CUDA) or triton-ascend (NPU), which is not importable on this "
            "host. Use the eager reference losses in this module instead."
        ) from exc
    module_globals = globals()
    # Publish what the kernel bodies resolve from module globals at compile
    # time, re-wrap the selectors/globals as tl.constexpr exactly as upstream
    # defines them, and apply the deferred @triton.jit decorators.
    module_globals["tl"] = tl
    module_globals["_LOG2"] = tl.constexpr(_LOG2)
    module_globals["_OP_KL"] = tl.constexpr(_OP_KL)
    module_globals["_OP_RKL"] = tl.constexpr(_OP_RKL)
    module_globals["_OP_JSD"] = tl.constexpr(_OP_JSD)
    module_globals["_OP_CE"] = tl.constexpr(_OP_CE)
    module_globals["_OP_TV"] = tl.constexpr(_OP_TV)
    module_globals["_OP_CE_LABEL"] = tl.constexpr(_OP_CE_LABEL)
    module_globals["_online_stats"] = triton.jit(_online_stats)
    module_globals["_log_mix"] = triton.jit(_log_mix)
    module_globals["loss_forward_kernel"] = triton.jit(loss_forward_kernel)
    module_globals["loss_backward_kernel"] = triton.jit(loss_backward_kernel)
    _TRITON_READY = True


class _FusedLoss(torch.autograd.Function):
    """Shared autograd wrapper; ``op`` selects the loss reduction.

    Targets intentionally receive no gradient; use the eager implementation when
    target gradients are required.
    """

    @staticmethod
    def forward(ctx, logits, targets, op):
        B, T, V = logits.shape
        logits_flat = logits.contiguous().view(B * T, V)
        if op == _OP_CE_LABEL.value:
            # Label ids are one int32 per row, not a [B * T, V] distribution.
            targets_flat = targets.contiguous().view(-1)
        else:
            targets_flat = targets.contiguous().view(B * T, V)
        loss = torch.empty(B * T, device=logits.device, dtype=torch.float32)
        stats = torch.empty(_N_STATS, B * T, device=logits.device, dtype=torch.float32)
        BLOCK_SIZE, num_warps = _calculate_settings(V, logits_flat.device)
        loss_forward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            loss,
            stats,
            stats.stride(0),
            V,
            OP=op,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        # CE backward only needs the target index cached in stats (argmax for
        # CE, label id for ce_label); both CE variants pass None for targets.
        ctx.save_for_backward(
            logits_flat,
            None if op in (_OP_CE.value, _OP_CE_LABEL.value) else targets_flat,
            stats,
        )
        ctx.op = op
        ctx.shape = (B, T, V)
        ctx.settings = (BLOCK_SIZE, num_warps)
        return loss.view(B, T)

    @staticmethod
    def backward(ctx, grad_output):
        logits_flat, targets_flat, stats = ctx.saved_tensors
        B, T, V = ctx.shape
        BLOCK_SIZE, num_warps = ctx.settings
        grad_in = torch.empty_like(logits_flat)
        loss_backward_kernel[(B * T,)](
            logits_flat,
            targets_flat,
            grad_in,
            grad_output.contiguous().view(-1),
            stats,
            stats.stride(0),
            V,
            OP=ctx.op,
            BLOCK_SIZE=BLOCK_SIZE,
            num_warps=num_warps,
        )
        return grad_in.view(B, T, V), None, None


def fused_kl_div_loss(logits, targets):
    """Per-position forward KL ``[1, T]``; fused twin of ``kl_div_loss``."""
    _require_triton()
    return _FusedLoss.apply(logits, targets, _OP_KL.value)


def fused_reverse_kl_div_loss(logits, targets):
    """Per-position reverse KL ``[1, T]``; fused twin of ``reverse_kl_div_loss``."""
    _require_triton()
    return _FusedLoss.apply(logits, targets, _OP_RKL.value)


def fused_js_div_loss(logits, targets):
    """Per-position JSD ``[1, T]``; fused twin of ``js_div_loss``."""
    _require_triton()
    return _FusedLoss.apply(logits, targets, _OP_JSD.value)


def fused_ce_loss(logits, targets):
    """Per-position CE vs argmax(targets) ``[1, T]``; fused twin of ``ce_loss``."""
    _require_triton()
    return _FusedLoss.apply(logits, targets, _OP_CE.value)


def fused_ce_label_loss(logits, labels):
    """Per-position CE vs ground-truth label ids; fused twin of ``ce_label_loss``.

    Args:
        logits: Draft model logits, shape ``[batch, seq_len, vocab]``.
        labels: Ground-truth token ids, shape ``[batch, seq_len]`` (cast to
            int32 internally). Follows ``F.cross_entropy`` ``ignore_index=-100``
            semantics: any label outside ``[0, vocab)`` -- e.g. -100 padding in
            MTP/DFlash targets -- is treated as an ignore position by the
            kernel (loss 0, all-zero gradient row) rather than validated.

    Returns:
        Per-position cross-entropy, shape ``[batch, seq_len]``, fp32.
    """
    if labels.shape != logits.shape[:2]:
        raise ValueError(
            f"labels shape {tuple(labels.shape)} must match logits.shape[:2] "
            f"{tuple(logits.shape[:2])}"
        )
    _require_triton()
    return _FusedLoss.apply(logits, labels.to(torch.int32), _OP_CE_LABEL.value)


def fused_tv_loss(logits, targets):
    """Per-position TV distance ``[1, T]`` from draft/target logits (fused Triton)."""
    _require_triton()
    return _FusedLoss.apply(logits, targets, _OP_TV.value)


def fused_nla_loss(logits, targets):
    """Per-position negative-log-acceptance ``[1, T] = -log(alpha)``; composes on TV."""
    return -torch.log((1.0 - fused_tv_loss(logits, targets)).clamp_min(_NLA_EPS))


def fused_lk_hybrid_loss(logits, targets, eta: float = 3.0):
    """Per-position hybrid LK ``[1, T]``; fused KL + TV composed as in
    ``lk_hybrid_loss`` (blend weight detached).
    """
    kl = fused_kl_div_loss(logits, targets)
    tv = fused_tv_loss(logits, targets)
    weight = torch.exp(-eta * (1.0 - tv).detach())
    return weight * kl + (1.0 - weight) * tv


# ---------------------------------------------------------------------------
# Eager reference implementations (verbatim from speculators losses/eager.py).
# They are the numerical-validation reference for the fused kernels and the
# fallback for hosts without Triton; note they materialize full [B, T, V]
# probabilities, so wide vocabularies can easily OOM with the eager path.
# ---------------------------------------------------------------------------


def kl_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position KL divergence from draft logits to target logits.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (softmax applied internally).

    Returns:
        Per-position KL divergence with shape [1, seq_len].
    """
    logits = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    elementwise_loss = torch.nn.functional.kl_div(
        logits, target_p, reduction="none", log_target=False
    ).sum(
        dim=-1
    )  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def reverse_kl_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position reverse KL divergence from draft logits to target logits.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (log-softmax applied internally).

    Returns:
        Per-position reverse KL divergence with shape [1, seq_len].
    """
    draft_logq = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_logp = torch.nn.functional.log_softmax(targets, dim=-1, dtype=torch.float32)
    elementwise_loss = torch.nn.functional.kl_div(
        target_logp, draft_logq, reduction="none", log_target=True
    ).sum(
        dim=-1
    )  # shape: [1, seq_len]

    return elementwise_loss  # noqa: RET504


def js_div_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position Jensen-Shannon divergence between draft and target.

    ``JSD(p, q) = 0.5 * KL(p || m) + 0.5 * KL(q || m)`` with ``m = (p + q) / 2``.
    Symmetric and bounded by ``log 2`` (Lin 1991, "Divergence measures based on
    the Shannon entropy"), it balances forward KL's mass-covering pull with
    reverse KL's mode-seeking pull and keeps gradients finite where either
    distribution assigns near-zero probability. Compared to plain KL, this
    avoids unbounded penalties on tokens the target barely supports; compared
    to TV, it provides smoother, better-conditioned gradients for draft
    training.

    Args:
        logits: Draft model logits (log-softmax applied internally).
        targets: Target model logits (log-softmax applied internally).

    Returns:
        Per-position JS divergence with shape [1, seq_len].
    """
    import math

    draft_logq = torch.nn.functional.log_softmax(logits, dim=-1, dtype=torch.float32)
    target_logp = torch.nn.functional.log_softmax(targets, dim=-1, dtype=torch.float32)
    # log m = log((p + q) / 2), computed in log space for stability
    log_m = torch.logaddexp(draft_logq, target_logp) - math.log(2.0)
    kl_target_to_mix = torch.nn.functional.kl_div(
        log_m, target_logp, reduction="none", log_target=True
    ).sum(dim=-1)
    kl_draft_to_mix = torch.nn.functional.kl_div(
        log_m, draft_logq, reduction="none", log_target=True
    ).sum(dim=-1)
    elementwise_loss = 0.5 * (kl_target_to_mix + kl_draft_to_mix)  # [1, seq_len]

    return elementwise_loss  # noqa: RET504


def ce_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position cross-entropy loss using argmax of target logits as labels.

    Args:
        logits: Draft model logits.
        targets: Target model logits (argmax taken to produce hard labels).

    Returns:
        Per-position cross-entropy loss with shape [1, seq_len].
    """
    batch_size, seq_len, draft_vocab_size = logits.shape
    target_ids = torch.argmax(targets, dim=-1)  # shape: [1, seq_len]

    elementwise_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, draft_vocab_size),
        target_ids.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    return elementwise_loss  # noqa: RET504


def ce_label_loss(
    logits: torch.Tensor,  # shape: [batch_size, seq_len, vocab_size]
    labels: torch.Tensor,  # shape: [batch_size, seq_len], ground-truth token ids
):
    """Compute per-position cross-entropy against ground-truth label ids.

    Unlike ``ce_loss`` -- which derives hard labels from the argmax of the
    target logits -- this takes integer labels directly: the MTP/DFlash-style
    ground-truth CE path, where no teacher distribution exists.

    Args:
        logits: Draft model logits.
        labels: Ground-truth token ids; positions labeled -100 are ignored
            (loss 0, no gradient), matching
            ``F.cross_entropy(ignore_index=-100)``.

    Returns:
        Per-position cross-entropy loss with shape [batch_size, seq_len].
    """
    batch_size, seq_len, vocab_size = logits.shape

    elementwise_loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        reduction="none",
        ignore_index=-100,
    ).reshape(batch_size, seq_len)

    return elementwise_loss  # noqa: RET504


def tv_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position total variation (TV) distance from draft to target.

    The rejection-sampling acceptance rate of speculative decoding equals the
    distributional overlap between target and draft,
    ``alpha = sum_v min(p_v, q_v) = 1 - d_TV(p, q)``. Minimizing this TV distance
    therefore directly optimizes the acceptance rate, whereas cross-entropy and
    KL only optimize it indirectly (KL is a loose upper bound on TV via Pinsker).

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position TV distance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # shape: [1, seq_len]
    elementwise_loss = 1.0 - overlap

    return elementwise_loss  # noqa: RET504


def neg_log_acceptance_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
):
    """Compute per-position negative log-acceptance (LK) loss.

    The speculative-decoding acceptance rate equals the draft/target distribution
    overlap, ``alpha = sum_v min(p_v, q_v)`` (the same quantity computed in
    ``tv_loss``). This loss is ``-log(alpha)``. Its gradient is
    ``(1 / alpha) * grad(TV)``: the ``1 / alpha`` factor amplifies the otherwise
    vanishing TV gradient when overlap is low (early training), giving TV's
    acceptance-optimal target a usable gradient from a cold start. When the target
    is a point mass, this loss reduces to cross-entropy.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).

    Returns:
        Per-position negative log-acceptance with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    elementwise_loss = -torch.log(overlap.clamp_min(_NLA_EPS))

    return elementwise_loss  # noqa: RET504


def lk_hybrid_loss(
    logits: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    targets: torch.Tensor,  # shape: [1, seq_len, draft_vocab_size]
    eta: float = 3.0,
):
    """Compute per-position hybrid LK loss (adaptive KL/TV blend).

    Blends KL divergence and total variation per position:
    ``L = lambda * KL(p||q) + (1 - lambda) * TV(p, q)`` with adaptive weight
    ``lambda = exp(-eta * sg[alpha])``, where ``alpha = sum_v min(p_v, q_v)`` is the
    acceptance rate (overlap) and ``sg`` is stop-gradient. When overlap is low
    (early training, misaligned draft) ``lambda -> 1`` and the loss leans on KL's
    strong gradient; as overlap grows ``lambda -> 0`` and it shifts to TV, which
    optimizes acceptance directly. This gives TV's acceptance-optimal target a
    usable gradient from a cold start.

    ``alpha`` in the weight is detached: it controls the blend but is not
    differentiated through; gradients flow only through the KL and TV terms.

    Source: Samarin et al., "LK Losses: Direct Acceptance Rate Optimization for
    Speculative Decoding" (arXiv 2602.23881), hybrid objective.

    Args:
        logits: Draft model logits (softmax applied internally to form q).
        targets: Target model logits (softmax applied internally to form p).
        eta: Blend temperature; larger shifts toward TV sooner. Default 3.0
            (the paper's best hybrid setting).

    Returns:
        Per-position hybrid loss with shape [1, seq_len].
    """
    draft_p = torch.nn.functional.softmax(logits, dim=-1, dtype=torch.float32)
    target_p = torch.nn.functional.softmax(targets, dim=-1, dtype=torch.float32)
    overlap = torch.minimum(draft_p, target_p).sum(dim=-1)  # alpha, shape: [1, seq_len]
    tv = 1.0 - overlap
    kl = kl_div_loss(logits, targets)  # reuse existing KL, shape: [1, seq_len]
    weight = torch.exp(-eta * overlap.detach())  # lambda = exp(-eta * sg[alpha])
    elementwise_loss = weight * kl + (1.0 - weight) * tv

    return elementwise_loss  # noqa: RET504
