"""
Shared, Ray-free data structures and utilities used by both
RolloutWorker and TrainWorker.

No Ray import here so that unit-tests can run without a Ray cluster.
"""

from dataclasses import dataclass
from typing import Optional

import torch


@dataclass
class RolloutBatch:
    """
    Data container produced by a RolloutWorker and consumed by a TrainWorker.

    All tensors are stored on **CPU** so that they can be serialised through
    the Ray object store without triggering CUDA IPC issues.

    Shapes (B = target_batch_size = tp_size * per_dp_batch_size):
        input_ids:      (B, seq_len)         – token IDs (padded)
        attention_mask: (B, seq_len)         – 1 for real tokens
        loss_mask:      (B, seq_len, 1)      – 1 for tokens that count in loss
        hidden_states:  (B, seq_len, 3*H)   – aux hidden states from target model
        target:         (B, seq_len, V)      – target-model logits (full vocab)
        pixel_values:   optional, VLM only
        image_grid_thw: optional, VLM only
        position_ids:   optional, VLM / USP only
    """

    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    loss_mask: torch.Tensor
    hidden_states: torch.Tensor
    target: torch.Tensor
    pixel_values: Optional[torch.Tensor] = None
    image_grid_thw: Optional[torch.Tensor] = None
    position_ids: Optional[torch.Tensor] = None


def batch_to_device(batch: RolloutBatch, device: torch.device) -> RolloutBatch:
    """Move every non-None tensor in *batch* to *device*."""

    def _to(t):
        return t.to(device) if t is not None else None

    return RolloutBatch(
        input_ids=_to(batch.input_ids),
        attention_mask=_to(batch.attention_mask),
        loss_mask=_to(batch.loss_mask),
        hidden_states=_to(batch.hidden_states),
        target=_to(batch.target),
        pixel_values=_to(batch.pixel_values),
        image_grid_thw=_to(batch.image_grid_thw),
        position_ids=_to(batch.position_ids),
    )


def batch_shard_by_tp(batch: RolloutBatch, tp_size: int, tp_rank: int) -> RolloutBatch:
    """
    Slice the **batch** dimension (dim=0) according to *tp_rank*.

    The full RolloutBatch carries target_batch_size = tp_size * per_dp_batch_size
    samples.  All workers belonging to the same TP group share the same
    dp_rank, so they received the same RolloutBatch.  Each worker takes its
    own slice of per_dp_batch_size samples via this function.

    Equivalent to the existing get_dp_data_shard_from_tp() in train_eagle3.py.
    SP (sequence-parallel) slicing is handled inside OnlineEagle3Model via
    the UspAdapter, NOT here.

    Args:
        batch:   Full RolloutBatch (target_batch_size samples).
        tp_size: Tensor-parallel world size.
        tp_rank: This worker's rank within its TP group.

    Returns:
        A new RolloutBatch containing only the samples for this tp_rank.
    """
    if tp_size == 1:
        return batch

    def _shard(t):
        if t is None:
            return None
        chunks = t.chunk(tp_size, dim=0)
        return chunks[tp_rank].contiguous()

    return RolloutBatch(
        input_ids=_shard(batch.input_ids),
        attention_mask=_shard(batch.attention_mask),
        loss_mask=_shard(batch.loss_mask),
        hidden_states=_shard(batch.hidden_states),
        target=_shard(batch.target),
        pixel_values=_shard(batch.pixel_values),
        image_grid_thw=_shard(batch.image_grid_thw),
        position_ids=_shard(batch.position_ids),
    )
