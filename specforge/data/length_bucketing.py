"""Metadata-only length grouping for a padded distributed sample plan.

The input and output use DistributedSampler's rank-strided global layout.
Grouping changes training batch composition, never sample membership: padding
duplicates and the incomplete microbatch dropped by each rank are retained.
"""

from __future__ import annotations

import hashlib
import json
import random
from collections.abc import Callable, Sequence
from typing import Optional, TypeVar

T = TypeVar("T")


def sample_length(ref, max_len: Optional[int] = None) -> Optional[int]:
    """Read a ref's token length without opening or materializing its features."""
    length = getattr(ref, "num_tokens", 0)
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        spec = getattr(ref, "feature_specs", {}).get("input_ids")
        shape = getattr(spec, "shape", ())
        if len(shape) == 1 or (len(shape) == 2 and shape[0] == 1):
            length = shape[-1]
        else:
            return None
    if not isinstance(length, int) or isinstance(length, bool) or length <= 0:
        return None
    return min(length, max_len) if max_len is not None else length


def length_bucket_fingerprint(refs, *, max_len: int) -> str:
    """Identify the ordered ids and effective lengths that determine grouping.

    This is a sampler-resume contract, not a tensor-content checksum. Streaming
    canonical JSON records keeps hashing metadata-only and bounds extra memory.
    """
    digest = hashlib.sha256()
    for ref in refs:
        record = json.dumps(
            [ref.sample_id, sample_length(ref, max_len=max_len)],
            ensure_ascii=True,
            separators=(",", ":"),
        )
        digest.update(record.encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def bucket_by_length(
    items: Sequence[T],
    *,
    length_fn: Callable[[T], Optional[int]],
    batch_size: int,
    dp_size: int = 1,
    length_bucket_size: int = 0,
    seed: int = 0,
    epoch: int = 0,
) -> list[T]:
    """Group similar lengths in bounded windows of global microbatches.

    ``length_bucket_size`` counts microbatches across *all* DP ranks, each of
    ``batch_size * dp_size`` samples; zero preserves the exact original order.
    Within a window, sorted samples form contiguous per-rank batches, then
    global microbatches are shuffled deterministically. Windows may span
    optimizer steps, so enabling grouping changes the stochastic training
    order. This helper does not change the global RNG state.

    The caller supplies an already shuffled and DP-padded plan. Its incomplete
    final global microbatch stays untouched, preserving the exact samples
    dropped by each rank's loader. A window containing an unknown/nonpositive
    length also stays untouched; no tensor reads are performed to guess it.
    """
    if length_bucket_size < 0:
        raise ValueError("length_bucket_size must be >= 0")
    if batch_size < 1 or dp_size < 1:
        raise ValueError("batch_size and dp_size must be positive")
    result = list(items)
    if length_bucket_size == 0 or not result:
        return result
    if len(result) % dp_size:
        raise ValueError("length bucketing requires a DP-padded global sample plan")

    quantum = batch_size * dp_size
    usable = len(result) // quantum * quantum
    window_size = quantum * length_bucket_size
    rng = random.Random(int(seed) + int(epoch))
    for start in range(0, usable, window_size):
        end = min(start + window_size, usable)
        window = result[start:end]
        lengths = [length_fn(item) for item in window]
        if any(
            not isinstance(length, int) or isinstance(length, bool) or length <= 0
            for length in lengths
        ):
            continue
        # Ties retain the upstream epoch shuffle, rather than sorting by id.
        order = sorted(range(len(window)), key=lengths.__getitem__, reverse=True)
        sorted_window = [window[index] for index in order]
        batches = []
        for offset in range(0, len(sorted_window), quantum):
            group = sorted_window[offset : offset + quantum]
            # Each rank receives a contiguous group of similar lengths after
            # the caller applies global_indices[rank::dp_size].
            batches.append(
                [
                    group[rank * batch_size + row]
                    for row in range(batch_size)
                    for rank in range(dp_size)
                ]
            )
        rng.shuffle(batches)
        result[start:end] = [item for batch in batches for item in batch]
    return result
