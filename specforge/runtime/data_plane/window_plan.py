# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Cost-aware scheduling: plan the micro-batch rounds of one optimizer window.

A round is one micro-batch per DP rank, synchronized by FSDP, so it costs as
much as its dearest micro-batch. ``plan_window`` keeps a window's sample set
and sorts it by a cost key into rounds of similar cost. Pure Python, shared by
the online distributor and the offline sampler.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
from typing import List, NamedTuple, Optional, Sequence, Tuple

#: ``auto`` resolves to the key the algorithm declares.
SCHEDULING_KEYS = ("auto", "length", "anchors")
#: ``SampleRef.metadata`` field written by producers that see the loss mask.
VALID_ANCHORS_METADATA_KEY = "valid_anchors"

Cost = Tuple[float, ...]
#: ``rounds[r][k]``: the sample indices rank ``k`` runs in round ``r``.
Rounds = List[List[List[int]]]


def count_valid_anchors(loss_mask) -> int:
    """Positions whose own and next token are supervised (DFlash anchor rule)."""
    if hasattr(loss_mask, "numel"):
        mask = loss_mask.reshape(-1)
        if mask.numel() < 2:
            return 0
        supervised = mask > 0.5
        return int((supervised[:-1] & supervised[1:]).sum().item())
    values = [float(value) > 0.5 for value in loss_mask]
    return sum(1 for a, b in zip(values[:-1], values[1:]) if a and b)


@dataclass(frozen=True)
class SchedulingPolicy:
    """Resolved ``training.cost_aware_*`` settings of one run."""

    key: str
    num_anchors: int = 512
    window_rounds: int = 0

    def __post_init__(self) -> None:
        if self.key == "auto" or self.key not in SCHEDULING_KEYS:
            raise ValueError(
                f"scheduling key must be one of {SCHEDULING_KEYS[1:]}, got {self.key!r}"
            )
        if (
            isinstance(self.num_anchors, bool)
            or not isinstance(self.num_anchors, int)
            or self.num_anchors <= 0
        ):
            raise ValueError(
                f"num_anchors must be a positive integer, got {self.num_anchors!r}"
            )
        if (
            isinstance(self.window_rounds, bool)
            or not isinstance(self.window_rounds, int)
            or self.window_rounds < 0
        ):
            raise ValueError("window_rounds must be a non-negative integer")

    @property
    def uses_anchors(self) -> bool:
        return self.key == "anchors"

    def with_key(self, key: str) -> "SchedulingPolicy":
        return replace(self, key=key)

    def contract(self) -> dict:
        """JSON-native record for checkpoints; a resume must reproduce it."""
        return {
            "key": self.key,
            "num_anchors": int(self.num_anchors),
            "window_rounds": int(self.window_rounds),
        }

    def sample_cost(
        self, length: Optional[int], valid_anchors: Optional[int]
    ) -> Optional[Cost]:
        """Cost of one sample, or None when a required quantity is unknown."""
        if length is None or length <= 0:
            return None
        if self.key == "length":
            return (float(length),)
        if valid_anchors is None:
            return None
        return (float(min(self.num_anchors, max(0, int(valid_anchors)))),)


class OfflineWindowPlan(NamedTuple):
    """Per-sample costs (aligned with the source refs) and the window shape."""

    costs: Sequence[Cost]
    batch_size: int
    window_rounds: int


def ref_length(ref) -> Optional[int]:
    length = int(getattr(ref, "num_tokens", 0) or 0)
    return length if length > 0 else None


def ref_valid_anchors(ref) -> Optional[int]:
    value = (getattr(ref, "metadata", None) or {}).get(VALID_ANCHORS_METADATA_KEY)
    return None if value is None else int(value)


def ref_cost(
    policy: SchedulingPolicy, ref, *, max_len: Optional[int] = None
) -> Optional[Cost]:
    """Cost key of a ``SampleRef`` from its metadata (None when unknown)."""
    length = ref_length(ref)
    if length is not None and max_len is not None:
        length = min(length, int(max_len))
    return policy.sample_cost(length, ref_valid_anchors(ref))


def plan_window(
    costs: Sequence[Cost],
    dp_size: int,
    batch_size: int,
    *,
    tiebreak: Optional[Sequence] = None,
    rng: Optional[random.Random] = None,
) -> Rounds:
    """Sort a window of whole rounds by decreasing cost into ``rounds[r][k]``.

    Ties follow ``tiebreak[i]`` (default: the index); rank ``k`` takes the
    ``k``-th contiguous ``batch_size`` slice of each round; ``rng`` shuffles
    the round order.
    """
    n = len(costs)
    round_size = dp_size * batch_size
    if dp_size < 1 or batch_size < 1:
        raise ValueError("dp_size and batch_size must be >= 1")
    if n % round_size:
        raise ValueError(f"{n} samples do not form whole rounds of {round_size}")
    if tiebreak is not None and len(tiebreak) != n:
        raise ValueError("tiebreak must have one entry per sample")
    order = (
        list(range(n))
        if tiebreak is None
        else sorted(range(n), key=lambda i: tiebreak[i])
    )
    order.sort(key=lambda i: costs[i], reverse=True)  # stable: ties keep tiebreak order
    rounds: Rounds = [
        [
            order[start + k * batch_size : start + (k + 1) * batch_size]
            for k in range(dp_size)
        ]
        for start in range(0, n, round_size)
    ]
    if rng is not None:
        rng.shuffle(rounds)
    return rounds


def round_robin_rounds(n: int, dp_size: int, batch_size: int) -> Rounds:
    """The unscheduled dispatch order (``index % dp_size``) in ``Rounds`` shape."""
    round_size = dp_size * batch_size
    if n % round_size:
        raise ValueError(f"{n} samples do not form whole rounds of {round_size}")
    return [
        [list(range(start + k, start + round_size, dp_size)) for k in range(dp_size)]
        for start in range(0, n, round_size)
    ]


def interleave_rounds(rounds: Rounds, dp_size: int) -> List[int]:
    """Flatten rounds so ``indices[k::dp_size]`` hands rank ``k`` its slices in order."""
    out: List[int] = []
    for rnd in rounds:
        if len(rnd) != dp_size or any(len(s) != len(rnd[0]) for s in rnd):
            raise ValueError("every round needs one equal-sized slice per rank")
        for j in range(len(rnd[0])):
            out.extend(rnd[k][j] for k in range(dp_size))
    return out


__all__ = [
    "SCHEDULING_KEYS",
    "VALID_ANCHORS_METADATA_KEY",
    "OfflineWindowPlan",
    "SchedulingPolicy",
    "count_valid_anchors",
    "interleave_rounds",
    "plan_window",
    "ref_cost",
    "ref_length",
    "ref_valid_anchors",
    "round_robin_rounds",
]
