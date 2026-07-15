# coding=utf-8
# Licensed under the Apache License, Version 2.0 (the "License").
"""Compact (draft-vocabulary) teacher distribution for EAGLE-3 training.

Reproduces the teacher quantities of ``specforge.core.eagle3._compute_target_p`` from
the target's last hidden states and ``lm_head`` weight without materializing the full
``[B, S, vocab_size]`` fp32 logits: the draft-vocab logits come from a ``t2d``-sliced
head, and the full-vocab ``logsumexp``/``argmax`` from a streaming reduction over
vocabulary chunks.

Scope: offline training only. The teacher uses the full (unsharded, rank-replicated)
target ``lm_head`` weight, so the computation is a pure function of
``(hidden, weight, t2d)`` with no cross-rank/tensor-parallel sharding of the teacher
head; every rank computes identical teacher tensors from identical inputs. Online
compact training is not supported in this release.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

DEFAULT_VOCAB_CHUNK_SIZE = 32768


def validate_compact_teacher_config(
    *,
    draft_vocab_size: int,
    vocab_size: int,
    t2d: torch.Tensor,
) -> None:
    """Validate the draft/vocab sizes and the ``t2d`` mask, or raise ValueError."""
    if draft_vocab_size >= vocab_size:
        raise ValueError(
            "compact teacher requires draft_vocab_size < vocab_size; got "
            f"draft_vocab_size={draft_vocab_size}, vocab_size={vocab_size}."
        )
    if t2d is None:
        raise ValueError(
            "compact teacher requires a loaded t2d vocab mapping; got None."
        )
    if t2d.dtype != torch.bool:
        raise ValueError(f"t2d must be a bool tensor, got dtype {t2d.dtype}.")
    if t2d.dim() != 1 or t2d.shape[0] != vocab_size:
        raise ValueError(
            f"t2d must have shape [vocab_size]={[vocab_size]}, got {list(t2d.shape)}."
        )
    selected = int(t2d.sum().item())
    if selected != draft_vocab_size:
        raise ValueError(
            f"t2d selects {selected} tokens but draft_vocab_size={draft_vocab_size}."
        )


@torch.no_grad()
def tiled_logsumexp_argmax(
    hidden: torch.Tensor,
    weight: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Full-vocabulary fp32 ``logsumexp`` (``[..., 1]``) and ``argmax`` (``[...]``).

    Streams over vocabulary chunks without allocating ``[..., vocab_size]``. Chunk
    logits are upcast to fp32 to match ``target.float()``; ties resolve to the lowest
    index like ``torch.argmax``.
    """
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}.")

    vocab_size = weight.shape[0]
    lead_shape = hidden.shape[:-1]
    device = hidden.device

    neg_inf = float("-inf")
    running_max = torch.full(
        (*lead_shape, 1), neg_inf, dtype=torch.float32, device=device
    )
    running_sumexp = torch.zeros((*lead_shape, 1), dtype=torch.float32, device=device)
    running_argval = torch.full(lead_shape, neg_inf, dtype=torch.float32, device=device)
    running_argmax = torch.zeros(lead_shape, dtype=torch.long, device=device)

    for start in range(0, vocab_size, chunk_size):
        end = min(start + chunk_size, vocab_size)
        chunk_logits = F.linear(hidden, weight[start:end]).float()

        chunk_max = chunk_logits.max(dim=-1, keepdim=True).values
        new_max = torch.maximum(running_max, chunk_max)
        running_sumexp = running_sumexp * torch.exp(running_max - new_max) + torch.exp(
            chunk_logits - new_max
        ).sum(dim=-1, keepdim=True)
        running_max = new_max

        # Ascending scan + strict-greater update keeps the lowest global index on ties.
        chunk_val, chunk_idx = chunk_logits.max(dim=-1)
        chunk_idx = chunk_idx + start
        take = chunk_val > running_argval
        running_argmax = torch.where(take, chunk_idx, running_argmax)
        running_argval = torch.where(take, chunk_val, running_argval)

    log_z = running_max + torch.log(running_sumexp)
    return log_z, running_argmax


@torch.no_grad()
def compute_target_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact-teacher equivalent of ``_compute_target_p``, computed from hidden states.

    Returns detached ``(target_p, target_p_on_draft, target_token_ids, position_mask)``
    matching the full-vocab path within fp tolerance, without a ``[B, S, vocab]`` fp32
    tensor.
    """
    if t2d.dtype != torch.bool:
        raise ValueError(f"t2d must be a bool mask, got dtype {t2d.dtype}.")
    if t2d.shape[0] != lm_head_weight.shape[0]:
        raise ValueError(
            f"t2d length {t2d.shape[0]} must equal vocab_size {lm_head_weight.shape[0]}."
        )
    if hidden.shape[-1] != lm_head_weight.shape[1]:
        raise ValueError(
            f"hidden size {hidden.shape[-1]} must equal lm_head hidden size "
            f"{lm_head_weight.shape[1]}."
        )

    # weight[t2d] keeps the same row order as (full_logits)[..., t2d] column slicing.
    draft_logits = F.linear(hidden, lm_head_weight[t2d]).float()
    target_p = torch.softmax(draft_logits, dim=-1)

    log_z, target_token_ids = tiled_logsumexp_argmax(
        hidden, lm_head_weight, chunk_size=chunk_size
    )
    target_p_on_draft = torch.exp(draft_logits - log_z)

    target_mask = t2d[target_token_ids][..., None].int()
    position_mask = target_mask * loss_mask

    return (
        target_p.detach(),
        target_p_on_draft.detach(),
        target_token_ids.detach(),
        position_mask,
    )


@torch.no_grad()
def compute_target_p_padded_from_hidden(
    hidden: torch.Tensor,
    lm_head_weight: torch.Tensor,
    t2d: torch.Tensor,
    loss_mask: torch.Tensor,
    length: int,
    *,
    chunk_size: int = DEFAULT_VOCAB_CHUNK_SIZE,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compact-teacher equivalent of ``_compute_target_p_padded`` (same pad constants)."""
    (
        target_p,
        target_p_on_draft,
        target_token_ids,
        position_mask,
    ) = compute_target_from_hidden(
        hidden, lm_head_weight, t2d, loss_mask, chunk_size=chunk_size
    )

    assert target_p.dim() == 3
    target_p_padded = F.pad(
        target_p, pad=(0, 0, 0, length), mode="constant", value=1 / target_p.shape[-1]
    )
    target_p_on_draft_padded = F.pad(
        target_p_on_draft, pad=(0, 0, 0, length), mode="constant", value=0.0
    )
    target_token_ids_padded = F.pad(
        target_token_ids, pad=(0, length), mode="constant", value=0
    )

    return (
        target_p_padded,
        target_p_on_draft_padded,
        target_token_ids_padded,
        position_mask,
    )


def build_offline_teacher_inputs(
    *,
    compact: bool,
    target_model,
    target_hidden: torch.Tensor,
    chunk_size_arg: Optional[int],
):
    """Decide the offline teacher inputs for ``run_forward``.

    ``compact=False`` returns ``(target_model(target_hidden), {})`` (full logits, legacy).
    ``compact=True`` returns ``(None, {...})`` from ``target_model.fc.weight`` without
    calling ``target_model.forward``.
    """
    if not compact:
        return target_model(target_hidden), {}
    chunk_size = (
        chunk_size_arg if chunk_size_arg is not None else DEFAULT_VOCAB_CHUNK_SIZE
    )
    return None, {
        "target_hidden_for_compact": target_hidden,
        "target_head_weight": target_model.fc.weight,
        "compact_teacher_chunk_size": chunk_size,
    }


def validate_vocab_mapping_consistency(t2d: torch.Tensor, d2t: torch.Tensor) -> None:
    """Require the ascending ids selected by ``t2d`` to equal ``d2t + arange``."""
    selected = torch.nonzero(t2d.bool(), as_tuple=False).flatten()
    if selected.numel() != d2t.numel():
        raise ValueError(
            f"t2d selects {selected.numel()} tokens but d2t has {d2t.numel()} entries."
        )
    expected = d2t.to(device=selected.device, dtype=selected.dtype) + torch.arange(
        d2t.numel(), device=selected.device, dtype=selected.dtype
    )
    if not torch.equal(selected, expected):
        raise ValueError(
            "t2d/d2t mapping is inconsistent: nonzero(t2d) must equal "
            "d2t + arange(draft_vocab_size)."
        )


def validate_compact_teacher_enabled(
    *,
    is_online: bool,
    is_vlm: bool,
    draft_vocab_size: int,
    vocab_size: int,
    t2d: torch.Tensor,
    target_head_weight: torch.Tensor,
    chunk_size: Optional[int] = None,
) -> None:
    """Validate that the compact teacher path may be enabled, or raise ValueError.

    Hidden-size compatibility of ``target_head_weight`` is enforced at runtime by
    ``compute_target_from_hidden``.
    """
    if is_online:
        raise ValueError(
            "compact teacher supports offline training only; disable --compact-teacher "
            "or train offline."
        )
    if is_vlm:
        raise ValueError(
            "compact teacher does not support VLM training yet; disable --compact-teacher."
        )
    if target_head_weight is None:
        raise ValueError(
            "compact teacher (offline) requires a TargetHead lm_head weight (fc.weight)."
        )
    if target_head_weight.dim() != 2:
        raise ValueError(
            "target_head_weight must be 2-D [vocab_size, hidden_size], got shape "
            f"{list(target_head_weight.shape)}."
        )
    if target_head_weight.shape[0] != vocab_size:
        raise ValueError(
            f"target_head_weight has {target_head_weight.shape[0]} rows but vocab_size "
            f"is {vocab_size}."
        )
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(
            f"--compact-teacher-chunk-size must be a positive integer, got {chunk_size}."
        )
    validate_compact_teacher_config(
        draft_vocab_size=draft_vocab_size, vocab_size=vocab_size, t2d=t2d
    )
