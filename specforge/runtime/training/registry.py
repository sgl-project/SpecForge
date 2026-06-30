# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""StrategySpec registry: the per-draft-model knobs the launch wiring needs.

The DataFlow runtime's components (FeatureDataLoader / FSDPTrainingBackend /
TrainerCore / TrainerController / FeatureStore) are model-agnostic. The only
model-specific facts the launch layer needs to assemble a run are captured here,
once per strategy, instead of being hardcoded into every ``build_*`` function:

  * ``make_strategy``        — build the per-step strategy over the FSDP-wrapped
                               model (the seam ``TrainerCore`` consumes),
  * ``required_features``    — the feature contract (drives
                               ``CaptureConfig.from_strategy`` + loader validation),
  * the offline data path    — reader + per-sample transform + collate + target_repr,
  * ``make_online_collate``  — the online/streamed collate,
  * ``make_adapter``         — the online capture adapter (None => SGLangAdapter),
  * ``supports_online``      — whether an online capture path exists yet.

Registering a new model (dflash, domino, ...) is a ``StrategySpec`` entry next
to its model code — NOT a new family of ``build_*_runtime`` functions. This is
what stops ``launch.py`` from multiplying as (topologies x models): the topology
stays a named builder, the model becomes the ``strategy=`` parameter resolved
through here. Builders go from ``topologies x models`` to
``topologies + one spec per model``.

Import-light: this module references only the (light) strategy classes; the
model/dataset-heavy imports (``OfflineEagle3Dataset``, ``DataCollatorWithPadding``,
``OfflineManifestReader``) stay lazy inside the factory callables so importing
the registry never drags in a GPU/model environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Optional

import torch

from specforge.runtime.training.strategy import DraftTrainStrategy, Eagle3TrainStrategy


def concat_collate(feats: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """Concatenate pre-formed per-sample features along the batch dim.

    Online/streamed features arrive already in train form (input_ids /
    hidden_state(s) / target / loss_mask), so a plain cat is correct for
    equal-length / batch_size=1 batches (the rollout groups equal-length
    prompts). Variable-length padded batching is a follow-up; pass an explicit
    ``collate_fn`` to override. (This is the canonical online/streamed collate the
    builders default to; tests may still define an inline ``_cat_collate``.)
    """
    return {k: torch.cat([f[k] for f in feats], dim=0) for k in feats[0]}


@dataclass(frozen=True)
class StrategySpec:
    """Everything the launch layer needs to assemble a run for one draft model.

    A ``None`` factory means that data path is not wired for this strategy yet;
    the corresponding builder raises an actionable ``NotImplementedError`` rather
    than silently assembling the wrong (e.g. EAGLE3-shaped) features.
    """

    name: str
    required_features: FrozenSet[str]
    # (wrapped_model, *, target_head) -> DraftTrainStrategy. Strategies that own
    # their head (e.g. DFlash) accept and ignore target_head.
    make_strategy: Callable[..., DraftTrainStrategy]
    uses_target_head: bool = True
    # Offline data path.
    make_offline_reader: Optional[Callable[..., Any]] = None
    make_offline_transform: Optional[Callable[[int], Callable]] = None
    make_offline_collate: Optional[Callable[[], Callable]] = None
    # Online data path.
    make_online_collate: Optional[Callable[[], Callable]] = None
    # (target_model, *, device, t2d) -> capture adapter. None => default SGLangAdapter.
    make_adapter: Optional[Callable[..., Any]] = None
    supports_online: bool = False


_REGISTRY: Dict[str, StrategySpec] = {}


def register_strategy(spec: StrategySpec) -> None:
    """Register (or replace) a strategy spec by name."""
    _REGISTRY[spec.name] = spec


def resolve_strategy(name: str) -> StrategySpec:
    """Look up a registered spec; raises with the registered names on miss."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown training strategy {name!r}; registered: {available_strategies()}"
        ) from None


def available_strategies() -> List[str]:
    return sorted(_REGISTRY)


# --- EAGLE3 -----------------------------------------------------------------
# Fully wired (offline + online). These factories reproduce exactly what the
# launch builders used to hardcode, so the eagle3 path is byte-identical.


def _eagle3_offline_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        target_repr="hidden_state",
    )


def _eagle3_offline_transform(max_len):
    from specforge.data.preprocessing import OfflineEagle3Dataset

    return lambda raw: OfflineEagle3Dataset.process_data(raw, max_len)


def _eagle3_offline_collate():
    from specforge.data.utils import DataCollatorWithPadding

    return DataCollatorWithPadding()


register_strategy(
    StrategySpec(
        name="eagle3",
        required_features=frozenset(Eagle3TrainStrategy.required_features),
        make_strategy=lambda wrapped, *, target_head=None: Eagle3TrainStrategy(
            wrapped, target_head=target_head
        ),
        uses_target_head=True,
        make_offline_reader=_eagle3_offline_reader,
        make_offline_transform=_eagle3_offline_transform,
        make_offline_collate=_eagle3_offline_collate,
        make_online_collate=lambda: concat_collate,
        make_adapter=None,  # default SGLangAdapter
        supports_online=True,
    )
)


__all__ = [
    "StrategySpec",
    "concat_collate",
    "register_strategy",
    "resolve_strategy",
    "available_strategies",
]
