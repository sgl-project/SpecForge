# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Deterministic online prompt planning shared by every online topology.

An online run walks its prepared prompts ``num_epochs`` times. Every epoch is
the same ``random.Random(seed + epoch)`` permutation of the prompt list, and
every pass after the first mints new task ids because the online feature store
is consume-once and commits dedup by sample id.

The disaggregated producer streams each whole permutation. A colocated
target-DP island takes a strided shard of the same permutation, truncated so
that every island receives the same, target-batch-aligned number of prompts.
Both topologies therefore see identical per-epoch sample identities, which keeps
colocated and disaggregated runs over the same prompts and seed comparable at
matched samples.
"""

from __future__ import annotations

import random
from typing import Dict, Iterator, List, Sequence


def online_prompt_seed(cfg) -> int:
    """Seed for online prompt ordering: ``training.prompt_seed`` or the run seed."""
    configured = cfg.training.prompt_seed
    return cfg.training.seed if configured is None else configured


def epoch_prompt_indices(prompts: Sequence, epoch: int, *, seed: int = 0) -> List[int]:
    """Return the deterministic prompt order of one epoch as source indices."""
    indices = list(range(len(prompts)))
    random.Random(int(seed) + int(epoch)).shuffle(indices)
    return indices


def epoch_online_prompt(
    prompt: Dict, index: int, epoch: int, prompt_epochs: int
) -> Dict:
    """Give *prompt* its epoch identity while keeping the single-epoch shape."""
    if prompt_epochs == 1:
        return prompt

    item = dict(prompt)
    metadata = dict(prompt.get("metadata") or {})
    if "task_id" in prompt:
        metadata.setdefault("base_task_id", str(prompt["task_id"]))
    metadata["prompt_index"] = index
    metadata["epoch"] = epoch
    metadata["prompt_epochs"] = prompt_epochs
    item["metadata"] = metadata
    item["task_id"] = f"epoch{epoch:04d}-prompt{index:012d}"
    return item


def epoch_online_prompts(
    prompts: Sequence[Dict],
    epoch: int,
    prompt_epochs: int,
    *,
    seed: int = 0,
) -> List[Dict]:
    """Materialize one whole epoch of the online prompt plan."""
    return [
        epoch_online_prompt(prompts[index], index, epoch, prompt_epochs)
        for index in epoch_prompt_indices(prompts, epoch, seed=seed)
    ]


def iter_epoch_online_prompt_batches(
    prompts: Sequence[Dict],
    epoch: int,
    prompt_epochs: int,
    *,
    seed: int = 0,
    batch_size: int = 4096,
) -> Iterator[List[Dict]]:
    """Yield one epoch in batches, bounding expanded token-list residency."""
    indices = epoch_prompt_indices(prompts, epoch, seed=seed)
    for start in range(0, len(indices), batch_size):
        yield [
            epoch_online_prompt(prompts[index], index, epoch, prompt_epochs)
            for index in indices[start : start + batch_size]
        ]


def shard_epoch_size(num_prompts: int, *, shard_count: int, batch_multiple: int) -> int:
    """Prompts per shard per epoch: equal across shards, a multiple of the batch.

    Shards are strided slices of the epoch permutation, so they differ by at
    most one prompt before truncation. Truncating every shard to the shortest
    batch-aligned length keeps all islands in lockstep for the whole epoch.
    """
    if shard_count < 1 or batch_multiple < 1:
        raise ValueError("shard_count and batch_multiple must be positive")
    return num_prompts // shard_count // batch_multiple * batch_multiple


def epoch_prompt_shard(
    prompts: Sequence,
    epoch: int,
    *,
    seed: int,
    shard_rank: int,
    shard_count: int,
    batch_multiple: int,
) -> List[int]:
    """Return one shard of an epoch's source indices (see ``shard_epoch_size``)."""
    if not 0 <= shard_rank < shard_count:
        raise ValueError(f"shard_rank {shard_rank} is outside [0, {shard_count})")
    per_shard = shard_epoch_size(
        len(prompts), shard_count=shard_count, batch_multiple=batch_multiple
    )
    indices = epoch_prompt_indices(prompts, epoch, seed=seed)
    return indices[shard_rank::shard_count][:per_shard]


def iter_sharded_online_prompts(
    prompts: Sequence[Dict],
    *,
    num_epochs: int,
    seed: int,
    shard_rank: int,
    shard_count: int,
    batch_multiple: int,
) -> Iterator[Dict]:
    """Lazily yield one shard's prompts across all epochs, in training order."""
    if num_epochs < 1:
        raise ValueError("num_epochs must be positive")
    for epoch in range(num_epochs):
        shard = epoch_prompt_shard(
            prompts,
            epoch,
            seed=seed,
            shard_rank=shard_rank,
            shard_count=shard_count,
            batch_multiple=batch_multiple,
        )
        for index in shard:
            yield epoch_online_prompt(prompts[index], index, epoch, num_epochs)


__all__ = [
    "epoch_online_prompt",
    "epoch_online_prompts",
    "epoch_prompt_indices",
    "epoch_prompt_shard",
    "iter_epoch_online_prompt_batches",
    "iter_sharded_online_prompts",
    "online_prompt_seed",
    "shard_epoch_size",
]
