"""Teacher distributions and supervised-row selection for recurrent distillation."""

import torch
from torch import nn
from torch.nn import functional as F


def compute_target_p_padded(target, t2d, loss_mask, length):
    with torch.no_grad():
        (
            target_p,
            target_p_on_draft,
            target_token_ids,
            position_mask,
        ) = compute_target_p(
            target=target,
            t2d=t2d,
            loss_mask=loss_mask,
        )

        assert len(target_p.shape) == 3
        target_p_padded = F.pad(
            target_p,
            pad=(0, 0, 0, length),
            mode="constant",
            # For bitwise equality with previous code
            value=1 / target_p.shape[-1],
        )
        target_p_on_draft_padded = F.pad(
            target_p_on_draft,
            pad=(0, 0, 0, length),
            mode="constant",
            value=0.0,
        )
        target_token_ids_padded = F.pad(
            target_token_ids,
            pad=(0, length),
            mode="constant",
            value=0,
        )

        return (
            target_p_padded,
            target_p_on_draft_padded,
            target_token_ids_padded,
            position_mask,
        )


@torch.compile(dynamic=None)
def compute_target_p(target, t2d, loss_mask):
    target_head = target.float()
    target_token_ids = target_head.argmax(-1)
    target_mask = t2d[target_token_ids]
    target_mask = target_mask[..., None].int()
    position_mask = target_mask * loss_mask
    draft_target_head = target_head[..., t2d]
    target_p = nn.Softmax(dim=2)(draft_target_head)
    target_logsumexp = torch.logsumexp(target_head, dim=-1, keepdim=True)
    target_p_on_draft = torch.exp(draft_target_head - target_logsumexp)
    target_p = target_p.detach()
    target_p_on_draft = target_p_on_draft.detach()
    target_token_ids = target_token_ids.detach()
    return target_p, target_p_on_draft, target_token_ids, position_mask


def compute_target_p_eager(target, t2d, loss_mask, row_chunk=256):
    """Uncompiled variant of the teacher target_p computation.

    Kept uncompiled because the supervised-row count varies per batch, which would
    trigger repeated torch.compile recompilation. Mathematically identical to the
    compiled path; chunks over rows to bound the transient full-vocab fp32
    activation (which can reach several GB when the row count is large).
    """
    tps, tpds, toks, pms = [], [], [], []
    n = target.shape[1]
    for s in range(0, n, row_chunk):
        t = target[:, s : s + row_chunk].float()
        ids = t.argmax(-1)
        tm = t2d[ids][..., None].int()
        pms.append(tm * loss_mask[:, s : s + row_chunk])
        dth = t[..., t2d]
        tps.append(F.softmax(dth, dim=2).detach())
        lse = torch.logsumexp(t, dim=-1, keepdim=True)
        tpds.append(torch.exp(dth - lse).detach())
        toks.append(ids.detach())
    return (
        torch.cat(tps, 1),
        torch.cat(tpds, 1),
        torch.cat(toks, 1),
        torch.cat(pms, 1),
    )


def build_trim_pack(target, t2d, loss_mask, length, chunk_len=None):
    """A-level trim (--trim-loss-positions): keep only the rows that can carry loss.

    Derivation of the per-step row set. On the full-length path the loop applies
    ``padding(..., left=False)`` to ``position_mask`` / ``loss_mask`` once per TTT
    step, so at step j the mask seen at row p is ``mask[p + j]``; meanwhile
    ``step_view`` slices the padded teacher so row p is supervised by the teacher at
    absolute position ``p + j``. A row therefore contributes at step j iff
    ``p + j`` is supervised, i.e. ``p = s - j`` for some supervised position s.

    So the rows shift *down* by one per step while the teacher/mask positions stay
    pinned at the supervised set:

        step j:  rows = {s - j : s in sup, s >= j}   teacher/mask at those s

    That makes the teacher cheap: it only ever has to be evaluated at ``sup``
    (no sliding window), and each step just drops the entries whose row would fall
    off the front of the sequence.

    ``chunk_len`` is the number of rows the backbone actually produces per step.
    On single-rank backends it equals the full local length L (default). Under USP
    the backbone runs on this rank's own chunk (``usp_chunk_size = L -
    ttt_length``) while the local buffer keeps a ``ttt_length`` overlap tail:
    tail positions may act as teachers for own-chunk rows at deeper steps, but
    tail rows belong to the next rank and must never emit loss here — hence the
    additional ``s - j < chunk_len`` bound on the row set, and ``full_len`` (the
    loss denominator) becomes ``chunk_len`` to match the full path's
    mean-over-chunk semantics.

    A step whose row set comes out empty (all supervised positions out of reach
    at that depth) is padded with one dummy entry and reported with
    ``nrows_steps[j] == 0``; the caller zeroes its masks so the step contributes
    exactly zero loss while kernel launches and collective calls stay aligned
    across ranks.

    batch == 1 only (online training uses batch == 1 per rank); the caller falls
    back to the full-length path otherwise.

    Returns a dict with, per step j: ``rows_steps[j]`` (row indices into the draft
    hidden states), ``keep_steps[j]`` (which supervised entries survive),
    ``nrows_steps[j]`` (real row count, 0 for dead steps), plus the teacher
    tables evaluated once at ``sup`` and the mask values at ``sup``.
    """
    with torch.no_grad():
        B, L = loss_mask.shape[0], loss_mask.shape[1]
        assert B == 1, "trim path requires batch==1"
        if chunk_len is None:
            chunk_len = L
        sup = loss_mask.view(-1).nonzero(as_tuple=False).squeeze(-1)  # [n_sup]
        # Positions beyond chunk_len + length - 2 can never supervise any row at
        # any step (would need j >= length); dropping them keeps every later
        # index within the hidden/target sequence range even when loss_mask is
        # longer than the hidden states (offline pipelines pad it by one).
        sup = sup[sup < chunk_len + length - 1]
        if sup.numel() == 0:
            # Nothing reachable at any step; tell the caller to use the full path.
            return None
        # Teacher is only ever needed at the supervised positions themselves.
        target_sel = target[:, sup]  # [1, n_sup, V_target]
        lm_sel = loss_mask[:, sup]
        target_p_c, on_draft_c, token_ids_c, pm_sup = compute_target_p_eager(
            target_sel, t2d, lm_sel
        )
        lm_sup = loss_mask.view(-1)[sup].view(1, -1, 1)

        rows_steps, keep_steps, nrows_steps = [], [], []
        pad_idx = torch.zeros(1, dtype=sup.dtype, device=sup.device)
        for j in range(length):
            keep = (
                ((sup >= j) & (sup - j < chunk_len)).nonzero(as_tuple=False).squeeze(-1)
            )
            n = int(keep.numel())
            if n == 0:
                # Dead step: pad with one dummy entry; the caller zeroes its
                # masks so it contributes nothing.
                keep = pad_idx
                rows = pad_idx
            else:
                rows = sup[keep] - j
            rows_steps.append(rows)
            keep_steps.append(keep)
            nrows_steps.append(n)
        return dict(
            sup=sup,
            rows_steps=rows_steps,
            keep_steps=keep_steps,
            nrows_steps=nrows_steps,
            target_p_c=target_p_c,
            on_draft_c=on_draft_c,
            token_ids_c=token_ids_c,
            position_mask_sup=pm_sup,
            loss_mask_sup=lm_sup,
            full_len=chunk_len,
        )


def teacher_distribution(model, logits):
    vocabulary = model.vocabulary
    vocabulary.require_ready()
    keep = vocabulary.t2d[logits.argmax(-1)]
    return logits[..., vocabulary.t2d].float().softmax(-1), keep
