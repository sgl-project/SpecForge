"""Anchor sampling for LongSpark training."""

from dataclasses import dataclass

import torch


@dataclass
class Anchors:
    source_indices: torch.Tensor
    positions: torch.Tensor
    current_tokens: torch.Tensor
    labels: torch.Tensor


def sample_anchors(input_ids, loss_mask, *, block_size, num_anchors, generator=None):
    """Sample up to ``num_anchors`` distinct anchors per row, without replacement.

    Anchor ``a`` is valid when all labels ``x[a + 1 : a + 1 + block_size]`` are
    supervised. Selected anchors are returned in increasing order per row.
    """
    if input_ids.ndim != 2 or input_ids.shape != loss_mask.shape:
        raise ValueError(
            "input_ids and loss_mask must have matching [batch, tokens] shapes"
        )
    if block_size <= 0 or num_anchors <= 0:
        raise ValueError("block size and number of anchors must be positive")
    batch, length = input_ids.shape
    if length < block_size + 2:
        raise ValueError("no complete supervised anchor block")
    positions = torch.arange(length, device=input_ids.device)
    offsets = torch.arange(1, block_size + 1, device=input_ids.device)
    labels_at = positions[:, None] + offsets[None, :]
    in_bounds = labels_at < length
    safe_labels = labels_at.clamp(max=length - 1)
    supervised = (
        torch.gather(
            loss_mask.bool()[:, None, :].expand(batch, length, length),
            2,
            safe_labels[None, :, :].expand(batch, -1, -1),
        )
        & in_bounds[None, :, :]
    )
    valid = supervised.all(dim=-1)
    valid[:, 0] = False
    counts = valid.sum(dim=1)
    count = min(num_anchors, int(counts.max().item()))
    if count == 0:
        raise ValueError("no complete supervised anchor block")
    # Rank valid positions by a uniform draw; invalid positions sort last.
    random = torch.rand(batch, length, generator=generator, device=input_ids.device)
    random = torch.where(valid, random, torch.full_like(random, 2.0))
    selected = random.sort(dim=1).indices[:, :count]
    selected_valid = valid.gather(1, selected)
    selected = (
        torch.where(selected_valid, selected, torch.full_like(selected, length))
        .sort(dim=1)
        .values
    )
    keep = torch.arange(count, device=input_ids.device)[None, :] < counts[
        :, None
    ].clamp(max=count)
    source, column = torch.nonzero(keep, as_tuple=True)
    anchor = selected[source, column]
    return Anchors(
        source_indices=source,
        positions=anchor,
        current_tokens=input_ids[source, anchor],
        labels=input_ids[source[:, None], anchor[:, None] + offsets[None, :]],
    )


def has_trainable_block(loss_mask, block_size):
    """Return whether ``loss_mask`` contains at least one valid anchor."""
    mask = torch.as_tensor(loss_mask).reshape(-1).bool()
    # The smallest valid anchor is 1, whose labels start at position 2.
    return mask.numel() >= block_size + 2 and bool(
        mask[2:].unfold(0, block_size, 1).all(dim=-1).any()
    )
