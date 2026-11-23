"""Alignment and visibility only: learned fusion stays in the draft model."""

import hashlib
from dataclasses import dataclass

import torch
from torch import Tensor

from specforge.modeling.draft.layers.attention.masks import (
    AttentionLayout,
    ContextBlocks,
)
from specforge.modeling.draft.layers.inputs import LayerInputs


def gather(sequence: Tensor, positions: Tensor) -> Tensor:
    batch = torch.arange(sequence.shape[0], device=sequence.device)
    return sequence[
        batch.reshape(-1, *([1] * (positions.ndim - 1))),
        positions.clamp(0, sequence.shape[1] - 1),
    ]


def sequence_inputs(
    config, positions, visibility, *, key_positions=None, rotary_positions=None
):
    layout = AttentionLayout(
        query_positions=positions,
        key_positions=positions if key_positions is None else key_positions,
        visibility=visibility[:, None],
        active_queries=visibility.any(-1),
    )
    return tuple(
        LayerInputs(
            position_ids=positions if rotary_positions is None else rotary_positions,
            attention_layout=layout,
        )
        for _ in config.layer_types
    )


def valid_tokens(batch):
    ids, mask = batch["input_ids"], batch["loss_mask"]
    if ids.ndim != 2 or mask.shape != ids.shape or ids.shape[1] < 2:
        raise ValueError("input_ids and loss_mask must be [batch, sequence>=2]")
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    if padding.shape != ids.shape:
        raise ValueError("attention_mask must align with input_ids")
    for name in (
        "hidden_states",
        "target_last_hidden_states",
        "mtp_hidden_states",
        "target_logits",
    ):
        if name in batch and (
            batch[name].ndim != 3 or batch[name].shape[:2] != ids.shape
        ):
            raise ValueError(
                f"{name} must align with input_ids [batch, sequence, width]"
            )
    if "position_ids" in batch:
        positions = batch["position_ids"]
        if positions.ndim not in (2, 3) or positions.shape[-2:] != ids.shape:
            raise ValueError(
                "position_ids must be [batch, sequence] or [axes, batch, sequence]"
            )
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


def prepare_anchor_capacity(batch, config):
    """Encode the host-side valid-anchor count in a tensor shape before H2D."""
    if batch["loss_mask"].device.type != "cpu":
        return {}
    valid = valid_tokens(batch)
    count = int((valid[:, :-1] & valid[:, 1:]).sum(-1).max())
    return {"anchor_capacity": torch.empty(count, dtype=torch.bool)}


def anchor_count(valid, num_anchors, available=None):
    """Anchors per row: ``num_anchors``, or fewer when no row has that many."""
    available = int(valid.sum(-1).max()) if available is None else available
    return min(num_anchors, max(1, available))


def select_anchors(valid, scores, count, *, stable=False):
    """The ``count`` valid positions with the lowest scores, sorted.

    A row with fewer valid positions is padded: its extra anchors are 0 and
    their ``keep`` is False. Uniform scores make every subset of a row's valid
    positions equally likely; where the scores come from is the caller's.
    ``stable`` breaks equal scores by position, so the selection does not
    depend on how far the row is padded.
    """
    scores = scores.masked_fill(~valid, 2)
    anchors = scores.argsort(dim=-1, stable=stable)[:, :count]
    keep = valid.gather(1, anchors)
    anchors = anchors.masked_fill(~keep, valid.shape[1]).sort(-1).values
    keep = anchors < valid.shape[1]
    return anchors.masked_fill(~keep, 0), keep


def sample_anchors(valid, num_anchors, available=None):
    """Sample sorted valid anchors on the batch's device, from the global generator."""
    count = anchor_count(valid, num_anchors, available)
    return select_anchors(valid, torch.rand(valid.shape, device=valid.device), count)


def sample_seed(seed, sample_id, epoch):
    """The 128-bit generator seed of one visit of one sample, stable across processes."""
    key = f"{seed}\0{sample_id}\0{epoch}".encode()
    return int.from_bytes(hashlib.blake2b(key, digest_size=16).digest(), "little")


def host_anchor_inputs(
    batch, *, num_anchors, seed, sample_ids, epoch, draws_per_anchor=0
):
    """Anchors drawn on the host, with one generator per visit of a sample.

    The selection is ``select_anchors``, as on the device; only the uniform
    scores differ in origin: float64 numbers from a PCG64 generator seeded with
    ``sample_seed`` (128 bits, so distinct visits do not share a stream), with
    ties broken by position. A sample's draws depend on the run seed, its id and
    the epoch, not on its batch, its rank, its padding or the global generator,
    so a replayed forward sees the same anchors and so does another data-parallel
    layout. Two things still couple a row to its batch: when no row of the batch
    has ``num_anchors`` valid anchor positions every row gets fewer slots, and a
    short row's unused slots are masked. Each generator first draws
    ``draws_per_anchor`` numbers per anchor, then one score per position.

    The loss mask may already be on the training device (a pooled feature
    read); the draw reads it on the host.

    Returns:
        ``anchors`` and ``anchor_keep`` ([batch, count]) and, when
        ``draws_per_anchor`` is positive, ``anchor_draws``
        ([batch, count, draws_per_anchor]) uniform on [0, 1).
    """
    import numpy as np

    batch = dict(batch)
    for name in ("input_ids", "loss_mask", "attention_mask"):
        if name in batch:
            batch[name] = batch[name].cpu()
    supervised = valid_tokens(batch)
    valid = supervised[:, :-1] & supervised[:, 1:]
    if len(sample_ids) != valid.shape[0]:
        raise ValueError(
            f"{len(sample_ids)} sample ids for a batch of {valid.shape[0]} rows"
        )
    count = anchor_count(valid, num_anchors)
    scores, draws = [], []
    for sample_id in sample_ids:
        generator = np.random.Generator(
            np.random.PCG64(sample_seed(seed, sample_id, epoch))
        )
        draws.append(generator.random(num_anchors * draws_per_anchor))
        scores.append(generator.random(valid.shape[1]))
    anchors, keep = select_anchors(
        valid, torch.from_numpy(np.stack(scores)), count, stable=True
    )
    inputs = {"anchors": anchors, "anchor_keep": keep}
    if draws_per_anchor:
        inputs["anchor_draws"] = (
            torch.from_numpy(np.stack(draws))
            .to(torch.float32)
            .view(len(sample_ids), num_anchors, draws_per_anchor)[:, :count]
        )
    return inputs


def prepare_blocks(
    config,
    batch,
    *,
    num_anchors=512,
    label_offset=0,
    input_offset=0,
    context_offset=0,
    prefix=False,
):
    """Sample adjacent supervised anchors and isolate each independent draft block.

    All positions are in the original, unshifted sequence. Offsets are explicit
    at algorithm call sites, including DMTP's external first candidate. Anchors
    the host drew (``host_anchor_inputs``) are used as given; otherwise they are
    sampled here, on the batch's device.
    """
    if num_anchors < 1:
        raise ValueError("num_anchors must be positive")
    ids = batch["input_ids"]
    supervised = valid_tokens(batch)
    if "lengths" in batch and batch["lengths"].numel() > ids.shape[0]:
        raise ValueError("block objectives require unpacked sequences")
    valid = supervised[:, :-1] & supervised[:, 1:]
    if "anchors" in batch:
        anchors, keep = batch["anchors"], batch["anchor_keep"]
        if anchors.shape != keep.shape or anchors.shape[0] != ids.shape[0]:
            raise ValueError("host anchors must be [batch, count] with their keep mask")
    else:
        capacity = batch.get("anchor_capacity")
        anchors, keep = sample_anchors(
            valid, num_anchors, None if capacity is None else capacity.numel()
        )
    spec, size = config.spec_config, config.spec_config.protocol.num_model_slots
    slots = torch.arange(size, device=ids.device)
    positions = anchors[..., None] + input_offset + slots
    label_positions = anchors[..., None] + label_offset + slots
    labels = gather(ids, label_positions)
    predecessors = gather(ids, label_positions - 1)
    prediction = torch.zeros(size, dtype=torch.bool, device=ids.device)
    for position in spec.protocol.prediction_positions:
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
    noisy = torch.full_like(labels, spec.target_features.mask_token_id)
    noisy[..., 0] = gather(ids, anchors + input_offset).masked_fill(
        ~keep, spec.target_features.mask_token_id
    )
    query = positions.flatten(1)
    context = torch.arange(ids.shape[1], device=ids.device).expand(ids.shape[0], -1)
    keys = torch.cat((context, query), dim=1)
    padding = batch.get("attention_mask", torch.ones_like(ids, dtype=torch.bool))
    layout = AttentionLayout(
        query_positions=query,
        key_positions=keys,
        context_blocks=ContextBlocks(
            block_size=size,
            context_lengths=(anchors + input_offset + context_offset).clamp(
                0, ids.shape[1]
            ),
            context_valid=padding.bool(),
            active_blocks=keep,
        ),
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
