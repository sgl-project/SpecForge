"""Per-position PyTorch losses for aligned draft and teacher vocabulary logits.

The eager reference functions are adapted from vllm-project/speculators
(losses/eager.py), licensed under the Apache License, Version 2.0.
"""

import torch

NLA_EPS = 1e-5


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
    elementwise_loss = -torch.log(overlap.clamp_min(NLA_EPS))

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


def renyi_half(logits, teacher):
    log_p = teacher.float().log_softmax(-1)
    log_q = logits.float().log_softmax(-1)
    return -2 * torch.logsumexp((log_p + log_q) * 0.5, dim=-1)


def neg_log_overlap(logits, teacher):
    # Stay in log space even when the two distributions barely overlap.
    log_p = teacher.float().log_softmax(-1)
    log_q = logits.float().log_softmax(-1)
    return -torch.logsumexp(torch.minimum(log_p, log_q), dim=-1)
