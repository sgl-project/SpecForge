"""Alignment and visibility only: learned fusion stays in the draft model."""

from dataclasses import dataclass

import torch
from torch import Tensor

from specforge.modeling.draft.layers.attention.masks import AttentionLayout
from specforge.modeling.draft.layers.inputs import LayerInputs


def gather(sequence: Tensor, positions: Tensor) -> Tensor:
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[
        batch.reshape(-1, *([1] * (positions.ndim - 1))),
        positions.clamp(0, sequence.shape[1] - 1),
    ]


def sequence_inputs(
    config, positions, visibility, *, active_queries=None, key_positions=None
):
    layout = AttentionLayout(
        query_positions=positions,
        key_positions=positions if key_positions is None else key_positions,
        visibility=visibility if callable(visibility) else visibility[:, None],
        active_queries=active_queries,
    )
    return tuple(
        LayerInputs(position_ids=positions, attention_layout=layout)
        for _ in config.layer_types
    )


def valid_tokens(batch):
    ids, mask = batch["input_ids"], batch["loss_mask"]
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] < 2:
        raise ValueError("input_ids and loss_mask must be [batch, sequence>=2]")
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    if padding.shape != ids.shape:
        raise ValueError("attention_mask must align with input_ids")
    return (mask > 0.5) & padding.bool()


@dataclass(frozen=True)
class Blocks:
    input_ids: Tensor
    labels: Tensor
    predecessors: Tensor
    weights: Tensor
    anchors: Tensor
    keep: Tensor
    label_positions: Tensor
    layer_inputs: tuple[LayerInputs, ...]


def prepare_blocks(
    config,
    batch,
    *,
    num_anchors=512,
    gamma=None,
    label_offset=0,
    input_offset=0,
    context_offset=0,
    prefix=False,
):
    """Sample adjacent supervised anchors and isolate each independent draft block.

    All positions are in the original, unshifted sequence. Offsets are explicit
    at algorithm call sites, including DMTP's external first candidate.
    """
    if num_anchors < 1 or (gamma is not None and gamma <= 0):
        raise ValueError("num_anchors and loss decay must be positive")
    ids = batch["input_ids"]
    supervised = valid_tokens(batch)
    if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
        raise ValueError("block objectives require unpacked sequences")
    valid = supervised[:, :-1] & supervised[:, 1:]
    anchors, keep = sample_anchors(valid, num_anchors, batch.get("max_valid_anchors"))
    spec, size = config.spec_config, config.spec_config.protocol.num_model_slots
    slots = torch.arange(size, device=ids.device)
    positions = anchors[..., None] + input_offset + slots
    label_positions = anchors[..., None] + label_offset + slots
    labels = gather(ids, label_positions)
    predecessors = gather(ids, label_positions - 1)
    predicted = spec.protocol.prediction_positions
    if predicted == tuple(range(predicted[0], predicted[-1] + 1)):
        prediction = (slots >= predicted[0]) & (slots <= predicted[-1])
    else:
        prediction = torch.zeros_like(slots, dtype=torch.bool)
        for position in predicted:
            prediction |= slots == position
    allowed = (
        keep[..., None]
        & (label_positions < ids.shape[1])
        & gather(supervised, label_positions)
    )
    if prefix:
        # Do not bridge an unsupervised gap inside a rollout.
        allowed = (allowed | ~prediction).long().cumprod(-1).bool() & allowed
    weights = (allowed & prediction).float()
    if gamma is not None:
        weights *= torch.exp(
            -(slots - spec.protocol.prediction_positions[0]).clamp_min(0).float()
            / gamma
        )
    noisy = torch.full_like(labels, spec.target_features.mask_token_id)
    noisy[..., 0] = gather(ids, anchors + input_offset).masked_fill(
        ~keep, spec.target_features.mask_token_id
    )
    query = positions.flatten(1)
    context = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
    keys = torch.cat((context, query), dim=1)
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))

    def visible(b, h, q, k):
        block = q // size
        context_visible = (
            (k < ids.shape[1])
            & (k < anchors[b, block] + input_offset + context_offset)
            & padding[b, k.clamp(max=ids.shape[1] - 1)].bool()
        )
        draft_visible = (k >= ids.shape[1]) & ((k - ids.shape[1]) // size == block)
        return keep[b, block] & (context_visible | draft_visible)

    layout = AttentionLayout(
        query_positions=query,
        key_positions=keys,
        visibility=visible,
        active_queries=keep[..., None].expand_as(positions).flatten(1),
    )
    entries = tuple(
        LayerInputs(position_ids=keys, attention_layout=layout)
        for _ in config.layer_types
    )
    return Blocks(
        noisy.flatten(1),
        labels,
        predecessors,
        weights,
        anchors,
        keep,
        label_positions,
        entries,
    )


def sample_anchors(valid, num_anchors, available=None):
    """Sample supervised predecessor positions, preserving sequence order."""
    if available is None:
        available = int(valid.sum(-1).max())
    count = min(num_anchors, max(1, available))
    scores = torch.rand(valid.shape, device=valid.device).masked_fill(~valid, 2)
    anchors = scores.argsort(-1)[:, :count]
    keep = valid.gather(1, anchors)
    anchors = anchors.masked_fill(~keep, valid.shape[1] + 1).sort(-1).values
    keep = anchors < valid.shape[1] + 1
    anchors = anchors.masked_fill(~keep, 0)
    return anchors, keep
