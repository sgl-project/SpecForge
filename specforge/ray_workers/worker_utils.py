"""
Shared, Ray-free data structures and utilities used by both
RolloutWorker and TrainWorker.

No Ray import here so that unit-tests can run without a Ray cluster.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F

# Dtype encoding for NCCL metadata transfer
_DTYPE_TO_INT = {
    torch.float32: 0,
    torch.float16: 1,
    torch.bfloat16: 2,
    torch.int64: 3,
    torch.int32: 4,
    torch.int16: 5,
    torch.bool: 6,
}
_INT_TO_DTYPE = {v: k for k, v in _DTYPE_TO_INT.items()}


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
    """Move every non-None tensor in *batch* to *device*.

    Uses pin_memory() + non_blocking transfer to overlap DMA copies
    for all tensors in parallel, reducing the CPU→GPU transfer wall time.
    """

    needs_sync = False

    def _to(t):
        nonlocal needs_sync
        if t is None:
            return None
        if t.is_cuda and device.type == "cuda":
            return t
        if device.type == "cuda" and not t.is_pinned():
            t = t.pin_memory()
        needs_sync = True
        return t.to(device, non_blocking=True)

    rb = RolloutBatch(
        input_ids=_to(batch.input_ids),
        attention_mask=_to(batch.attention_mask),
        loss_mask=_to(batch.loss_mask),
        hidden_states=_to(batch.hidden_states),
        target=_to(batch.target),
        pixel_values=_to(batch.pixel_values),
        image_grid_thw=_to(batch.image_grid_thw),
        position_ids=_to(batch.position_ids),
    )
    if needs_sync:
        torch.cuda.synchronize()
    return rb


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

    if tp_rank < 0 or tp_rank >= tp_size:
        raise ValueError(f"tp_rank ({tp_rank}) must be in [0, {tp_size})")

    B = batch.input_ids.shape[0]
    if B % tp_size != 0:
        raise ValueError(f"Batch size ({B}) must be divisible by tp_size ({tp_size})")
    shard_size = B // tp_size
    start = tp_rank * shard_size

    def _shard(t):
        if t is None:
            return None
        return t.narrow(0, start, shard_size).contiguous()

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


def batch_split(batch: RolloutBatch, n: int) -> List[RolloutBatch]:
    """Split a RolloutBatch into *n* equal chunks along dim=0.

    Used to split a large rollout result (accumulated from multiple
    train batches) back into individual per-step RolloutBatches.
    """
    if n == 1:
        return [batch]

    def _chunk(t):
        if t is None:
            return [None] * n
        return [c.contiguous() for c in t.chunk(n, dim=0)]

    chunks = {
        "input_ids": _chunk(batch.input_ids),
        "attention_mask": _chunk(batch.attention_mask),
        "loss_mask": _chunk(batch.loss_mask),
        "hidden_states": _chunk(batch.hidden_states),
        "target": _chunk(batch.target),
        "pixel_values": _chunk(batch.pixel_values),
        "image_grid_thw": _chunk(batch.image_grid_thw),
        "position_ids": _chunk(batch.position_ids),
    }
    return [RolloutBatch(**{k: v[i] for k, v in chunks.items()}) for i in range(n)]


def pad_and_concat_batches(
    batches: List[Dict[str, torch.Tensor]],
) -> Tuple[Dict[str, torch.Tensor], int]:
    """Pad variable-length data batches to the same seq_len and concatenate.

    Seq_len is assumed to be dim=1 for all tensors.

    Returns:
        (merged_dict, num_batches)
    """
    if len(batches) == 1:
        return batches[0], 1

    max_seq_len = max(b["input_ids"].shape[1] for b in batches)

    def _pad_and_cat(key):
        tensors = [b[key] for b in batches if b.get(key) is not None]
        if not tensors:
            return None
        padded = []
        for t in tensors:
            pad_len = max_seq_len - t.shape[1]
            if pad_len > 0:
                if t.ndim == 2:
                    t = F.pad(t, (0, pad_len), value=0)
                elif t.ndim == 3:
                    t = F.pad(t, (0, 0, 0, pad_len), value=0)
            padded.append(t)
        return torch.cat(padded, dim=0)

    merged = {
        k: _pad_and_cat(k) for k in batches[0].keys() if batches[0][k] is not None
    }
    return merged, len(batches)


# ─────────────────────────────────────────────────────────────────────────
# NCCL point-to-point transfer helpers
# ─────────────────────────────────────────────────────────────────────────

# (field_name, is_optional)
_ROLLOUT_BATCH_FIELDS = [
    ("input_ids", False),
    ("attention_mask", False),
    ("loss_mask", False),
    ("hidden_states", False),
    ("target", True),
]


def nccl_send_rollout_batch(
    batch: RolloutBatch,
    dst_rank: int,
    group: Optional[dist.ProcessGroup] = None,
) -> None:
    """Send a RolloutBatch to *dst_rank* via NCCL point-to-point ops.

    All tensors must be on CUDA.  Sends shape metadata first, then
    each tensor contiguously.  Optional fields that are None are skipped.

    Raises RuntimeError if any NCCL operation fails.
    """
    # Collect present fields
    present = []
    for name, _optional in _ROLLOUT_BATCH_FIELDS:
        t = getattr(batch, name)
        if t is not None:
            if not t.is_cuda:
                raise RuntimeError(
                    f"Tensor '{name}' must be on CUDA for NCCL send, got {t.device}"
                )
            present.append((name, t))

    device = present[0][1].device

    try:
        # Header: [num_present_fields, 0, 0, 0, 0, 0]
        header = torch.zeros(1, 6, dtype=torch.long, device=device)
        header[0, 0] = len(present)
        dist.send(header, dst=dst_rank, group=group)

        # Per-field metadata: [field_index, dtype_int, ndim, dim0, dim1, dim2]
        if present:
            field_name_to_idx = {
                name: i for i, (name, _) in enumerate(_ROLLOUT_BATCH_FIELDS)
            }
            meta = torch.zeros(len(present), 6, dtype=torch.long, device=device)
            for row, (name, t) in enumerate(present):
                meta[row, 0] = field_name_to_idx[name]
                meta[row, 1] = _DTYPE_TO_INT.get(t.dtype, 0)
                meta[row, 2] = t.ndim
                for d in range(t.ndim):
                    meta[row, 3 + d] = t.shape[d]
            dist.send(meta, dst=dst_rank, group=group)

        for _, t in present:
            dist.send(t.contiguous(), dst=dst_rank, group=group)
    except RuntimeError as e:
        raise RuntimeError(f"NCCL send to rank {dst_rank} failed: {e}") from e


def nccl_recv_rollout_batch(
    src_rank: int,
    device: torch.device,
    group: Optional[dist.ProcessGroup] = None,
) -> RolloutBatch:
    """Receive a RolloutBatch from *src_rank* via NCCL point-to-point ops.

    Returns a RolloutBatch with all tensors on *device* (CUDA).
    Optional fields not sent by the sender will be None.

    Raises RuntimeError if any NCCL operation fails.
    """
    assert device.type == "cuda", f"Device must be CUDA, got {device}"

    try:
        header = torch.zeros(1, 6, dtype=torch.long, device=device)
        dist.recv(header, src=src_rank, group=group)
        num_present = header[0, 0].item()

        if num_present > 0:
            meta = torch.zeros(num_present, 6, dtype=torch.long, device=device)
            dist.recv(meta, src=src_rank, group=group)
        else:
            meta = torch.zeros(0, 6, dtype=torch.long, device=device)

        idx_to_name = {i: name for i, (name, _) in enumerate(_ROLLOUT_BATCH_FIELDS)}
        tensors = {name: None for name, _ in _ROLLOUT_BATCH_FIELDS}

        for row in range(num_present):
            field_idx = meta[row, 0].item()
            dtype_int = meta[row, 1].item()
            ndim = meta[row, 2].item()
            shape = [meta[row, 3 + d].item() for d in range(ndim)]
            dtype = _INT_TO_DTYPE[dtype_int]
            buf = torch.empty(shape, dtype=dtype, device=device)
            dist.recv(buf, src=src_rank, group=group)
            tensors[idx_to_name[field_idx]] = buf

        return RolloutBatch(**tensors)
    except RuntimeError as e:
        raise RuntimeError(f"NCCL recv from rank {src_rank} failed: {e}") from e


def nccl_broadcast_rollout_batch(
    batch: Optional[RolloutBatch],
    src: int,
    group: dist.ProcessGroup,
    device: Optional[torch.device] = None,
) -> RolloutBatch:
    """Broadcast a RolloutBatch within a process group.

    The *src* rank (relative to *group*) must provide *batch*.
    All other ranks receive it.  Handles variable shapes by
    broadcasting metadata first.

    Args:
        batch:  RolloutBatch on the src rank, None on others.
        src:    Source rank within *group* (typically 0 = SP leader).
        group:  The process group to broadcast within.
        device: CUDA device (required on non-src ranks).

    Returns:
        RolloutBatch on all ranks (on CUDA).
    """
    my_rank = dist.get_rank(group)
    is_src = my_rank == src

    if is_src:
        assert batch is not None
        device = batch.input_ids.device

        # Collect present fields
        present = []
        for name, _optional in _ROLLOUT_BATCH_FIELDS:
            t = getattr(batch, name)
            if t is not None:
                present.append((name, t))

        # Broadcast header: [num_present]
        header = torch.tensor([len(present)], dtype=torch.long, device=device)
        dist.broadcast(header, src=src, group=group)

        # Broadcast metadata
        field_name_to_idx = {
            name: i for i, (name, _) in enumerate(_ROLLOUT_BATCH_FIELDS)
        }
        meta = torch.zeros(len(present), 6, dtype=torch.long, device=device)
        for row, (name, t) in enumerate(present):
            meta[row, 0] = field_name_to_idx[name]
            meta[row, 1] = _DTYPE_TO_INT.get(t.dtype, 0)
            meta[row, 2] = t.ndim
            for d in range(t.ndim):
                meta[row, 3 + d] = t.shape[d]
        dist.broadcast(meta, src=src, group=group)

        # Broadcast tensors
        for _, t in present:
            dist.broadcast(t.contiguous(), src=src, group=group)

        return batch
    else:
        assert device is not None

        # Receive header
        header = torch.zeros(1, dtype=torch.long, device=device)
        dist.broadcast(header, src=src, group=group)
        num_present = header[0].item()

        # Receive metadata
        meta = torch.zeros(num_present, 6, dtype=torch.long, device=device)
        dist.broadcast(meta, src=src, group=group)

        idx_to_name = {i: name for i, (name, _) in enumerate(_ROLLOUT_BATCH_FIELDS)}
        tensors = {name: None for name, _ in _ROLLOUT_BATCH_FIELDS}

        for row in range(num_present):
            field_idx = meta[row, 0].item()
            dtype_int = meta[row, 1].item()
            ndim = meta[row, 2].item()
            shape = [meta[row, 3 + d].item() for d in range(ndim)]
            dtype = _INT_TO_DTYPE[dtype_int]
            buf = torch.empty(shape, dtype=dtype, device=device)
            dist.broadcast(buf, src=src, group=group)
            tensors[idx_to_name[field_idx]] = buf

        return RolloutBatch(**tensors)
