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
                               ``FeatureContract.from_strategy`` + loader validation),
  * the offline data path    — reader + per-sample transform + collate + target_repr,
  * ``make_online_collate``  — the online/streamed collate,
  * ``feature_schema``       — the store-ready dict shape the shared
                               ``PolicyFeatureAdapter`` emits online,
  * ``make_adapter``         — escape hatch for a bespoke online capture adapter
                               (None => PolicyFeatureAdapter over feature_schema),
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

from specforge.inference.adapters.policy import (
    DFLASH_FEATURE_SCHEMA,
    EAGLE3_FEATURE_SCHEMA,
    FeatureSchema,
)
from specforge.training.strategies.base import DraftTrainStrategy, Eagle3TrainStrategy


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
    # The store-ready dict shape the shared PolicyFeatureAdapter emits online.
    # None with supports_online=True falls back to the EAGLE3 schema.
    feature_schema: Optional[FeatureSchema] = None
    # Escape hatch: (target_model, *, device, t2d) -> bespoke capture adapter.
    # None => PolicyFeatureAdapter over feature_schema.
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
        feature_schema=EAGLE3_FEATURE_SCHEMA,
        supports_online=True,
    )
)


# --- DFlash -----------------------------------------------------------------
# DFlash uses its own feature schema ('hidden_states' = the concatenated target
# capture layers, NO eagle3 aux/target swap, NO target distribution / vocab map).
# Offline: the reader reuses OfflineManifestReader with dflash feature_keys; the
# transform slices to max_len (no swap); the collate pads + emits {input_ids,
# hidden_states, loss_mask}. Online: DFLASH_FEATURE_SCHEMA drives the shared
# PolicyFeatureAdapter to emit the same schema. The DFlashTrainStrategy already
# drops into the unchanged TrainerCore/Backend/Loader.

from specforge.training.strategies.base import DFlashTrainStrategy


def _dflash_offline_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    # OfflineManifestReader is schema-agnostic; only its EAGLE3 defaults
    # (feature_keys/strategy/target_repr) must be overridden. DFlash has no target
    # distribution, so target_repr=None (only stored in ref metadata).
    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        strategy="dflash",
        feature_keys=("input_ids", "loss_mask", "hidden_states"),
        target_repr=None,
    )


def _dflash_process_data(raw, max_len):
    """Offline DFlash per-sample normalization.

    Slices to ``max_len`` and restores a leading batch dim. NO eagle3-style
    aux->input / hidden->target swap and NO last-position loss zeroing (the online
    DFlash path does neither), so offline matches online training.
    """
    input_ids = raw["input_ids"][:max_len]
    loss_mask = raw["loss_mask"][:max_len]
    hidden_states = raw["hidden_states"]
    if hidden_states.dim() == 3:  # stored as [1, seq, W]; drop the saved batch dim
        hidden_states = hidden_states.squeeze(0)
    hidden_states = hidden_states[:max_len]
    return {
        "input_ids": input_ids[None, :],
        "loss_mask": loss_mask[None, :],
        "hidden_states": hidden_states[None, :, :],
    }


def _dflash_offline_transform(max_len):
    return lambda raw: _dflash_process_data(raw, max_len)


def _dflash_offline_collate():
    """Right-pad ragged samples to the batch max length and concat along batch.

    DataCollatorWithPadding is eagle3-specific (hardwires attention_mask + the
    hidden_state/target keys), so DFlash uses this minimal collate. loss_mask is
    zero-padded, so padded positions contribute no loss. (Single-rank; no SP
    sharding multiple — sequence-parallel offline DFlash is a follow-up.)
    """

    def collate(feats):
        maxlen = max(f["input_ids"].shape[-1] for f in feats)

        def pad2d(t):  # [1, n] -> [1, maxlen]
            n = t.shape[-1]
            if n == maxlen:
                return t
            return torch.cat([t, t.new_zeros(t.shape[0], maxlen - n)], dim=-1)

        def pad3d(t):  # [1, n, W] -> [1, maxlen, W]
            n = t.shape[1]
            if n == maxlen:
                return t
            return torch.cat(
                [t, t.new_zeros(t.shape[0], maxlen - n, t.shape[2])], dim=1
            )

        return {
            "input_ids": torch.cat([pad2d(f["input_ids"]) for f in feats], dim=0),
            "loss_mask": torch.cat([pad2d(f["loss_mask"]) for f in feats], dim=0),
            "hidden_states": torch.cat(
                [pad3d(f["hidden_states"]) for f in feats], dim=0
            ),
        }

    return collate


register_strategy(
    StrategySpec(
        name="dflash",
        required_features=frozenset(DFlashTrainStrategy.required_features),
        make_strategy=lambda wrapped, *, target_head=None: DFlashTrainStrategy(wrapped),
        uses_target_head=False,
        make_offline_reader=_dflash_offline_reader,
        make_offline_transform=_dflash_offline_transform,
        make_offline_collate=_dflash_offline_collate,
        make_online_collate=lambda: concat_collate,
        feature_schema=DFLASH_FEATURE_SCHEMA,
        supports_online=True,
    )
)


# --- Domino -----------------------------------------------------------------
# Domino reuses DFlash's draft model (projector_type="domino" head), feature
# schema, offline transform/collate, and capture path (same DFLASH_FEATURE_SCHEMA
# -> hidden_states). The ONE difference is the loss: it blends a base loss with a
# step-decayed weight, so DominoTrainStrategy reads the StepContext
# (forward_loss(batch, ctx)). That is the whole reason a new algorithm needs
# anything beyond a spec entry here.

from specforge.training.strategies.base import DominoTrainStrategy


def _domino_offline_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        strategy="domino",
        feature_keys=("input_ids", "loss_mask", "hidden_states"),
        target_repr=None,
    )


register_strategy(
    StrategySpec(
        name="domino",
        required_features=frozenset(DominoTrainStrategy.required_features),
        make_strategy=lambda wrapped, *, target_head=None: DominoTrainStrategy(wrapped),
        uses_target_head=False,
        make_offline_reader=_domino_offline_reader,
        make_offline_transform=_dflash_offline_transform,  # same schema as DFlash
        make_offline_collate=_dflash_offline_collate,
        make_online_collate=lambda: concat_collate,
        feature_schema=DFLASH_FEATURE_SCHEMA,  # same capture path as DFlash
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
