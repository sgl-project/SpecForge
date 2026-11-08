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


def sequence_inputs(config, positions, visibility, *, key_positions=None):
    layout = AttentionLayout(
        query_positions=positions,
        key_positions=positions if key_positions is None else key_positions,
        visibility=visibility[:, None],
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
    count = min(num_anchors, max(1, int(valid.sum(-1).max())))
    scores = torch.rand(valid.shape, device=ids.device).masked_fill(~valid, 2)
    anchors = scores.argsort(-1)[:, :count]
    keep = valid.gather(1, anchors)
    anchors = anchors.masked_fill(~keep, ids.shape[1]).sort(-1).values
    keep = anchors < ids.shape[1]
    anchors = anchors.masked_fill(~keep, 0)
    spec, size = config.spec_config, config.spec_config.protocol.num_model_slots
    slots = torch.arange(size, device=ids.device)
    positions = anchors[..., None] + input_offset + slots
    label_positions = anchors[..., None] + label_offset + slots
    labels = gather(ids, label_positions)
    predecessors = gather(ids, label_positions - 1)
    prediction = torch.zeros(size, dtype=torch.bool, device=ids.device)
    prediction[list(spec.protocol.prediction_positions)] = True
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
    context_visible = (
        context[:, None, None, :]
        < (anchors + input_offset + context_offset)[..., None, None]
    )
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    context_visible = context_visible & padding[:, None, None, :].bool()
    context_visible = context_visible.expand(-1, -1, size, -1).reshape(
        ids.shape[0], -1, ids.shape[1]
    )
    block_ids = torch.arange(count, device=ids.device).repeat_interleave(size)
    draft_visible = (block_ids[:, None] == block_ids[None, :]).expand(
        ids.shape[0], -1, -1
    )
    visible = torch.cat((context_visible, draft_visible), dim=-1)
    visible &= keep.repeat_interleave(size, dim=1)[..., None]
    layout = AttentionLayout(
        query_positions=query, key_positions=keys, visibility=visible[:, None]
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
