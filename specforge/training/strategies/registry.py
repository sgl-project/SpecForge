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
  * ``required_features``    — the capture config (drives
                               ``CaptureConfig.from_strategy`` + loader validation),
  * the offline data path    — reader + per-sample transform + collate + target_repr,
  * ``make_online_collate``  — the online/streamed collate,
  * ``feature_schema``       — the store-ready dict shape the shared
                               ``PolicyFeatureAdapter`` emits online,
  * ``assembly``             — method-owned draft/model/capture/prompt hooks,
  * ``online_capture``       — explicit shared-policy, server-only, or unsupported
                               capture semantics,
  * capability fields       — the modes/backends/modalities accepted by config.

Registering a new model (dflash, domino, ...) is a ``StrategySpec`` entry next
to its model code — NOT a new family of ``build_*_runtime`` functions. This is
what stops ``launch.py`` from multiplying as (topologies x models): the topology
stays a named builder, the model becomes the ``strategy=`` parameter resolved
through here. Builders go from ``topologies x models`` to
``topologies + one spec per model``.

Import-light: this module references only the (light) strategy classes; the
model/dataset-heavy imports (``process_offline_eagle3_sample``,
``DataCollatorWithPadding``, ``OfflineManifestReader``) stay lazy inside the
factory callables so importing the registry never drags in a GPU/model
environment.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional

import torch

from specforge.inference.adapters.policy import (
    DFLASH_FEATURE_SCHEMA,
    EAGLE3_FEATURE_SCHEMA,
    EAGLE3_VLM_FEATURE_SCHEMA,
    FeatureSchema,
)
from specforge.training.strategies.assembly import (
    DraftConfigSpec,
    OnlineCaptureMode,
    StrategyAssemblySpec,
    apply_dflash_overrides,
    build_dflash_model,
    build_domino_model,
    build_dspark_model,
    build_eagle3_draft,
    build_eagle3_model,
    build_peagle_draft,
    build_peagle_model,
    build_registered_draft,
    dflash_min_loss_tokens,
    dflash_needs_input_tools,
    domino_strategy_kwargs,
    eagle3_strategy_kwargs,
    peagle_resume_contract,
    populate_dflash_generated_config,
    resolve_dflash_capture_layers,
    resolve_eagle_capture_layers,
)
from specforge.training.strategies.base import (
    DraftTrainStrategy,
    Eagle3TrainStrategy,
    PEagleTrainStrategy,
)


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
    assembly: StrategyAssemblySpec
    uses_target_head: bool = True
    # Offline data path.
    make_offline_reader: Optional[Callable[..., Any]] = None
    make_offline_transform: Optional[Callable[[int], Callable]] = None
    make_offline_collate: Optional[Callable[[], Callable]] = None
    # Online data path.
    make_online_collate: Optional[Callable[[], Callable]] = None
    # The store-ready dict shape the shared PolicyFeatureAdapter emits online.
    feature_schema: Optional[FeatureSchema] = None
    modality_feature_schemas: Optional[Mapping[str, FeatureSchema]] = None
    # POLICY selects the shared PolicyFeatureAdapter. SERVER_ONLY requires a
    # caller-provided server capture source. A bespoke colocated-adapter hook is
    # intentionally not part of the contract until a strategy needs one.
    online_capture: OnlineCaptureMode = OnlineCaptureMode.UNSUPPORTED
    supported_modes: FrozenSet[str] = frozenset({"online"})
    supported_deployments: FrozenSet[str] = frozenset({"local_colocated"})
    supported_attention_backends: FrozenSet[str] = frozenset(
        {"eager", "sdpa", "flex_attention", "fa", "usp"}
    )
    supported_target_backends: FrozenSet[str] = frozenset({"sglang", "hf"})
    supported_modalities: FrozenSet[str] = frozenset({"text"})
    required_batch_size: Optional[int] = None
    allows_aux_layer_override: bool = False
    supports_compact_teacher: bool = False
    requires_disaggregated_vocab_mapping: bool = False
    supports_sharded_target_output: bool = False

    @property
    def supports_online(self) -> bool:
        return self.online_capture is not OnlineCaptureMode.UNSUPPORTED

    def feature_schema_for(self, modality: str) -> Optional[FeatureSchema]:
        if self.modality_feature_schemas and modality in self.modality_feature_schemas:
            return self.modality_feature_schemas[modality]
        return self.feature_schema


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


def _eagle3_offline_transform(
    max_len,
    *,
    ttt_length=1,
    use_usp_preprocess=False,
):
    from functools import partial

    if not use_usp_preprocess:
        from specforge.data.preprocessing import process_offline_eagle3_sample

        return partial(process_offline_eagle3_sample, max_len=max_len)

    import torch.distributed as dist

    from specforge.data.preprocessing import OfflineEagle3Dataset
    from specforge.distributed import get_draft_sp_group, get_sp_ring_group

    if not dist.is_available() or not dist.is_initialized():
        raise RuntimeError("USP preprocessing requires initialized process groups")
    sp_group = get_draft_sp_group()
    ring_group = get_sp_ring_group()
    return partial(
        OfflineEagle3Dataset.process_data_usp,
        max_len=max_len,
        ttt_length=ttt_length,
        sp_rank=dist.get_rank(sp_group),
        sp_size=dist.get_world_size(sp_group),
        ring_rank=dist.get_rank(ring_group),
        sp_ring_size=dist.get_world_size(ring_group),
    )


def _eagle3_offline_collate():
    from specforge.data.utils import DataCollatorWithPadding

    return DataCollatorWithPadding()


def _make_eagle3_strategy(wrapped, *, target_head=None, **kwargs):
    return Eagle3TrainStrategy(wrapped, target_head=target_head, **kwargs)


register_strategy(
    StrategySpec(
        name="eagle3",
        required_features=frozenset(Eagle3TrainStrategy.required_features),
        make_strategy=_make_eagle3_strategy,
        assembly=StrategyAssemblySpec(
            draft_config=DraftConfigSpec(
                architecture="LlamaForCausalLMEagle3",
                auto_model_type="llama",
                auto_num_hidden_layers=1,
                auto_draft_vocab_size=32000,
                supports_num_hidden_layers_override=True,
                required_num_hidden_layers=1,
            ),
            make_draft_model=build_eagle3_draft,
            make_model=build_eagle3_model,
            make_strategy_kwargs=eagle3_strategy_kwargs,
            resolve_capture_layers=resolve_eagle_capture_layers,
            vocab_mapping_modes=frozenset({"online", "offline"}),
            default_dataloader_num_workers=4,
            allow_missing_warm_start_embedding=True,
            colocated_target_repr="logits",
            server_target_repr="hidden_state",
        ),
        uses_target_head=True,
        make_offline_reader=_eagle3_offline_reader,
        make_offline_transform=_eagle3_offline_transform,
        make_offline_collate=_eagle3_offline_collate,
        make_online_collate=lambda: concat_collate,
        feature_schema=EAGLE3_FEATURE_SCHEMA,
        modality_feature_schemas={"qwen2_5_vl": EAGLE3_VLM_FEATURE_SCHEMA},
        online_capture=OnlineCaptureMode.POLICY,
        supported_modes=frozenset({"online", "offline"}),
        supported_deployments=frozenset({"local_colocated", "disaggregated"}),
        supported_attention_backends=frozenset({"sdpa", "flex_attention", "fa", "usp"}),
        supported_target_backends=frozenset({"sglang", "hf", "custom"}),
        supported_modalities=frozenset({"text", "qwen2_5_vl"}),
        allows_aux_layer_override=True,
        supports_compact_teacher=True,
        requires_disaggregated_vocab_mapping=True,
        supports_sharded_target_output=True,
    )
)


# --- P-EAGLE ----------------------------------------------------------------
# P-EAGLE consumes the EAGLE3 online capture schema and runs the COD objective.


register_strategy(
    StrategySpec(
        name="peagle",
        required_features=frozenset(PEagleTrainStrategy.required_features),
        make_strategy=lambda wrapped, *, target_head=None: PEagleTrainStrategy(
            wrapped, target_head=target_head
        ),
        assembly=StrategyAssemblySpec(
            draft_config=DraftConfigSpec(
                architecture="PEagleDraftModel",
                auto_model_type="llama",
                auto_num_hidden_layers=4,
                auto_draft_vocab_size=32000,
                supports_num_hidden_layers_override=True,
            ),
            make_draft_model=build_peagle_draft,
            make_model=build_peagle_model,
            resume_contract=peagle_resume_contract,
            resolve_capture_layers=resolve_eagle_capture_layers,
            vocab_mapping_modes=frozenset({"online"}),
            default_dataloader_num_workers=4,
            allow_missing_warm_start_embedding=True,
            capture_engine_strategy="eagle3",
            colocated_target_repr="logits",
        ),
        uses_target_head=True,
        make_online_collate=lambda: concat_collate,
        feature_schema=EAGLE3_FEATURE_SCHEMA,
        online_capture=OnlineCaptureMode.POLICY,
        supported_attention_backends=frozenset({"flex_attention"}),
        supported_target_backends=frozenset({"sglang", "hf", "custom"}),
        required_batch_size=1,
        allows_aux_layer_override=True,
    )
)


# --- DFlash -----------------------------------------------------------------
# DFlash uses its own feature schema ('hidden_states' = the concatenated target
# capture layers, NO eagle3 aux/target swap, NO target distribution / vocab map).
# Offline and online use that same contract, so both paths share one transform
# and collate implementation.  The shared PolicyFeatureAdapter emits the schema
# from the policy-driven target engine for online runs.

from specforge.training.strategies.base import DFlashTrainStrategy


def _dflash_offline_reader(hidden_states_path, *, run_id, ttt_length, max_len):
    from specforge.runtime.data_plane.offline_reader import OfflineManifestReader

    return OfflineManifestReader(
        hidden_states_path,
        run_id=run_id,
        ttt_length=ttt_length,
        max_len=max_len,
        strategy="dflash",
        feature_keys=("input_ids", "loss_mask", "hidden_states"),
        target_repr=None,
    )


def _dflash_offline_transform(max_len, **_topology):
    from functools import partial

    from specforge.data.preprocessing import process_offline_dflash_sample

    return partial(process_offline_dflash_sample, max_len=max_len)


def _dflash_collate():
    """Right-pad ragged samples to the batch max length and concat along batch.

    DataCollatorWithPadding is eagle3-specific (hardwires attention_mask + the
    hidden_state/target keys), so DFlash uses this minimal collate. loss_mask is
    zero-padded, so padded positions contribute no loss.
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
        assembly=StrategyAssemblySpec(
            draft_config=DraftConfigSpec(
                architecture="DFlashDraftModel",
                expected_auto_map_model="dflash.DFlashDraftModel",
                auto_model_type="qwen3",
                auto_num_hidden_layers=1,
                populate_generated=populate_dflash_generated_config,
                apply_overrides=apply_dflash_overrides,
                supports_num_hidden_layers_override=True,
                supports_block_size_override=True,
            ),
            make_draft_model=build_registered_draft,
            make_model=build_dflash_model,
            min_loss_tokens=dflash_min_loss_tokens,
            needs_input_tools=dflash_needs_input_tools,
            resolve_capture_layers=resolve_dflash_capture_layers,
            default_dataloader_num_workers=8,
        ),
        uses_target_head=False,
        make_offline_reader=_dflash_offline_reader,
        make_offline_transform=_dflash_offline_transform,
        make_offline_collate=_dflash_collate,
        make_online_collate=_dflash_collate,
        feature_schema=DFLASH_FEATURE_SCHEMA,
        online_capture=OnlineCaptureMode.POLICY,
        supported_modes=frozenset({"online", "offline"}),
        supported_deployments=frozenset({"local_colocated", "disaggregated"}),
        supported_attention_backends=frozenset({"eager", "sdpa", "flex_attention"}),
    )
)


# --- DSpark -----------------------------------------------------------------
# DSpark is intentionally wired only for the server-capture disaggregated online
# path. It reuses DFlash's captured layer feature (hidden_states), but also
# requires the target model's last hidden state so the consumer can compute the
# target distribution through its frozen lm_head.

from specforge.training.strategies.base import DSparkTrainStrategy


def _dspark_online_collate():
    def collate(feats):
        maxlen = max(f["input_ids"].shape[-1] for f in feats)

        def pad2d(t):
            n = t.shape[-1]
            if n == maxlen:
                return t
            return torch.cat([t, t.new_zeros(t.shape[0], maxlen - n)], dim=-1)

        def pad3d(t):
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
            "target_last_hidden_states": torch.cat(
                [pad3d(f["target_last_hidden_states"]) for f in feats], dim=0
            ),
        }

    return collate


register_strategy(
    StrategySpec(
        name="dspark",
        required_features=frozenset(DSparkTrainStrategy.required_features),
        make_strategy=lambda wrapped, *, target_head=None: DSparkTrainStrategy(wrapped),
        assembly=StrategyAssemblySpec(
            draft_config=DraftConfigSpec(
                architecture="DSparkDraftModel",
                expected_auto_map_model="dspark.DSparkDraftModel",
            ),
            make_draft_model=build_registered_draft,
            make_model=build_dspark_model,
            min_loss_tokens=dflash_min_loss_tokens,
            needs_input_tools=dflash_needs_input_tools,
            resolve_capture_layers=resolve_dflash_capture_layers,
            default_dataloader_num_workers=8,
            server_target_repr="hidden_state",
        ),
        uses_target_head=False,
        make_online_collate=_dspark_online_collate,
        online_capture=OnlineCaptureMode.SERVER_ONLY,
        supported_deployments=frozenset({"disaggregated"}),
        supported_attention_backends=frozenset({"eager", "sdpa", "flex_attention"}),
    )
)


# --- Domino -----------------------------------------------------------------
# Domino uses a DFlash-family draft model and reuses the DFlash offline/online
# feature schema, transform and collate. The loss blends a base loss with a
# step-decayed weight, so DominoTrainStrategy reads the StepContext
# (forward_loss(batch, ctx)).

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


def _make_domino_strategy(
    wrapped, *, target_head=None, lambda_start=1.0, decay_ratio=0.5
):
    return DominoTrainStrategy(
        wrapped, lambda_start=lambda_start, decay_ratio=decay_ratio
    )


register_strategy(
    StrategySpec(
        name="domino",
        required_features=frozenset(DominoTrainStrategy.required_features),
        make_strategy=_make_domino_strategy,
        assembly=StrategyAssemblySpec(
            draft_config=DraftConfigSpec(
                architecture="DominoDraftModel",
                expected_auto_map_model="domino.DominoDraftModel",
            ),
            make_draft_model=build_registered_draft,
            make_model=build_domino_model,
            make_strategy_kwargs=domino_strategy_kwargs,
            min_loss_tokens=dflash_min_loss_tokens,
            needs_input_tools=dflash_needs_input_tools,
            resolve_capture_layers=resolve_dflash_capture_layers,
            default_dataloader_num_workers=8,
        ),
        uses_target_head=False,
        make_offline_reader=_domino_offline_reader,
        make_offline_transform=_dflash_offline_transform,
        make_offline_collate=_dflash_collate,
        make_online_collate=_dflash_collate,
        feature_schema=DFLASH_FEATURE_SCHEMA,  # same capture path as DFlash
        online_capture=OnlineCaptureMode.POLICY,
        supported_modes=frozenset({"online", "offline"}),
        supported_deployments=frozenset({"local_colocated", "disaggregated"}),
        supported_attention_backends=frozenset({"eager", "sdpa", "flex_attention"}),
    )
)


__all__ = [
    "StrategySpec",
    "OnlineCaptureMode",
    "concat_collate",
    "register_strategy",
    "resolve_strategy",
    "available_strategies",
]
