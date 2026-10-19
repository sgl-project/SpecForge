"""Static shapes for the TPU trainer: every step reuses a compiled graph."""

from __future__ import annotations

import bisect
import os
from collections.abc import Callable, Mapping
from typing import Any

import torch

_DEFAULT_SEQ_BUCKETS = (512, 1024, 2048, 3072, 4096, 6144, 8192, 12288, 16384)


def seq_buckets() -> tuple[int, ...]:
    raw = os.environ.get("SPECFORGE_TPU_SEQ_BUCKETS")
    if not raw:
        return _DEFAULT_SEQ_BUCKETS
    return tuple(sorted(int(item) for item in raw.split(",")))


def bucket_for(length: int, buckets: tuple[int, ...]) -> int:
    index = bisect.bisect_left(buckets, length)
    if index == len(buckets):
        raise ValueError(
            f"sequence length {length} exceeds the largest TPU bucket {buckets[-1]}; "
            "extend SPECFORGE_TPU_SEQ_BUCKETS or lower data.max_length"
        )
    return buckets[index]


def pad_batch_to_bucket(
    batch: Mapping[str, torch.Tensor], buckets: tuple[int, ...]
) -> dict[str, torch.Tensor]:
    """Zero-pad axis 1 of every [B, S, ...] feature up to the S bucket."""
    length = int(batch["input_ids"].shape[1])
    target = bucket_for(length, buckets)
    padded = {}
    for key, tensor in batch.items():
        if not isinstance(tensor, torch.Tensor) or tensor.dim() < 2:
            padded[key] = tensor
            continue
        if int(tensor.shape[1]) != length:
            raise ValueError(f"feature {key!r} is not aligned with input_ids on axis 1")
        pad_shape = list(tensor.shape)
        pad_shape[1] = target - length
        padded[key] = torch.cat([tensor, tensor.new_zeros(pad_shape)], dim=1)
    return padded


def padded_collator(
    original: Callable[..., Callable[[Any], Any]], *args: Any, **kwargs: Any
) -> Callable[[Any], Any]:
    collate = original(*args, **kwargs)
    buckets = seq_buckets()

    def collate_to_bucket(features: Any) -> dict[str, torch.Tensor]:
        return pad_batch_to_bucket(collate(features), buckets)

    return collate_to_bucket


def shape_only_max_valid_anchors(
    original: Callable[[torch.Tensor], int | None], loss_mask: torch.Tensor
) -> int | None:
    # The anchor width must depend on the S bucket only; rows with fewer
    # valid anchors are already masked by the model's keep_mask.
    max_valid = original(loss_mask)
    if not max_valid:
        return max_valid
    return max(int(loss_mask.shape[1]) - 1, 0)
