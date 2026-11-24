"""Sequence buckets and anchor sizing for the TPU trainer."""

from __future__ import annotations

import bisect
import os


def seq_buckets() -> tuple[int, ...]:
    raw = os.environ.get("SPECFORGE_TPU_SEQ_BUCKETS")
    if not raw:
        return ()
    return tuple(sorted(int(item) for item in raw.split(",")))


def bucket_for(length: int, buckets: tuple[int, ...]) -> int:
    index = bisect.bisect_left(buckets, length)
    if index == len(buckets):
        raise ValueError(
            f"sequence length {length} exceeds the largest TPU bucket {buckets[-1]}; "
            "extend SPECFORGE_TPU_SEQ_BUCKETS or lower data.max_length"
        )
    return buckets[index]


def anchor_capacity(valid) -> int:
    """Reserve one slot per adjacent token pair; sampling masks invalid slots."""
    return valid.shape[1] - 1
