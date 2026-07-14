# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Method-owned policies used by the package-level training assembler.

The public assembler owns topology and lifecycle.  This module owns the facts
that vary by draft method: how to create its config and model, which target
features it captures, and which dataset policies it needs.  All heavy imports
remain inside callables so resolving the strategy registry stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Callable, Dict, FrozenSet, List, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.config import Config


class OnlineCaptureMode(str, Enum):
    """How a strategy obtains online target features."""

    UNSUPPORTED = "unsupported"
    POLICY = "policy"
    SERVER_ONLY = "server_only"


@dataclass(frozen=True)
class DraftConfigSpec:
    """Single strategy-owned source of draft architecture/config policy."""

    architecture: str
    expected_auto_map_model: Optional[str] = None
    auto_model_type: Optional[str] = None
    auto_num_hidden_layers: Optional[int] = None
    auto_draft_vocab_size: Optional[int] = None
    populate_generated: Optional[Callable[[Dict[str, Any], Any, Any], None]] = None
    apply_overrides: Optional[Callable[[Any, Any], None]] = None
    supports_num_hidden_layers_override: bool = False
    supports_block_size_override: bool = False
    required_num_hidden_layers: Optional[int] = None

    @property
    def supports_auto_generation(self) -> bool:
        return (
            self.auto_model_type is not None and self.auto_num_hidden_layers is not None
        )


@dataclass
class StrategyModelParts:
    """Method-specific pieces returned to the topology-owned assembler."""

    model: Any
    target_head: Any = None
    capture_layers: Optional[List[int]] = None


def _empty_strategy_kwargs(_cfg: Config) -> Dict[str, Any]:
    return {}


def _one_loss_token(_cfg: Config) -> int:
    return 1


def _online_needs_input_tools(cfg: Config, _draft_model: Any) -> bool:
    return cfg.mode == "online"


def dflash_needs_input_tools(cfg: Config, draft_model: Any) -> bool:
    method_config = getattr(draft_model.config, "dflash_config", None) or {}
    needs_mask_fallback = (
        cfg.model.mask_token_id is None and method_config.get("mask_token_id") is None
    )
    return cfg.mode == "online" or needs_mask_fallback


def _no_capture_layers(_cfg: Config, _draft_config: Any, _target_config: Any):
    return None


@dataclass(frozen=True)
class StrategyAssemblySpec:
    """Composable method hooks consumed by ``training.assembly``."""

    draft_config: DraftConfigSpec
    make_draft_model: Callable[[Any, Any], Any]
    make_model: Callable[[Any, Any, Any, Any, Any], StrategyModelParts]
    make_strategy_kwargs: Callable[[Any], Dict[str, Any]] = _empty_strategy_kwargs
    min_loss_tokens: Callable[[Any], int] = _one_loss_token
    needs_input_tools: Callable[[Any, Any], bool] = _online_needs_input_tools
    resolve_capture_layers: Callable[[Any, Any, Any], Optional[List[int]]] = (
        _no_capture_layers
    )
    vocab_mapping_modes: FrozenSet[str] = frozenset()
    default_dataloader_num_workers: int = 8
    allow_missing_warm_start_embedding: bool = False
    capture_engine_strategy: Optional[str] = None
    colocated_target_repr: Optional[str] = None
    server_target_repr: Optional[str] = None


def _torch_dtype(cfg: Config):
    import torch

    return getattr(torch, cfg.model.torch_dtype)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _warm_start(cfg: Config, draft_model: Any, draft_config: Any) -> None:
    if not cfg.model.draft_checkpoint_path:
        return
    from specforge.training.model_loading import warm_start_draft_model

    warm_start_draft_model(
        draft_model,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy=cfg.training.strategy,
        cache_dir=cfg.model.cache_dir,
        trust_remote_code=cfg.model.trust_remote_code,
    )


def _load_vocab_mapping(cfg: Config, draft_model: Any) -> None:
    if cfg.model.vocab_mapping_path:
        draft_model.load_vocab_mapping(cfg.model.vocab_mapping_path)


def build_eagle3_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_model = AutoDraftModel.from_config(
        draft_config,
        attention_backend=cfg.training.attention_backend,
        torch_dtype=_torch_dtype(cfg),
    )
    _warm_start(cfg, draft_model, draft_config)
    _load_vocab_mapping(cfg, draft_model)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
        )
    draft_model.freeze_embedding()
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_peagle_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.draft.peagle import PEagleDraftModel

    draft_model = PEagleDraftModel(
        draft_config,
        norm_before_residual=cfg.training.norm_before_residual,
    )
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
        )
    _warm_start(cfg, draft_model, draft_config)
    _load_vocab_mapping(cfg, draft_model)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_registered_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
    )
    _warm_start(cfg, draft_model, draft_config)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def resolve_eagle_capture_layers(
    cfg: Config, draft_config: Any, target_config: Any
) -> List[int]:
    """Resolve EAGLE capture layers from run override, draft config, or target."""

    layers = cfg.model.aux_hidden_state_layer_ids
    if layers is None:
        eagle_config = (
            draft_config.get("eagle_config", {})
            if isinstance(draft_config, dict)
            else getattr(draft_config, "eagle_config", {})
        ) or {}
        layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        target_config = getattr(target_config, "text_config", target_config)
        num_layers = int(target_config.num_hidden_layers)
        layers = [1, num_layers // 2 - 1, num_layers - 4]
    layers = list(layers)
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError(
            "resolved EAGLE capture layers must contain exactly three "
            f"non-negative integers, got {layers!r}"
        )
    return layers


def resolve_dflash_capture_layers(
    _cfg: Config, draft_config: Any, _target_config: Any
) -> List[int]:
    method_config = (
        draft_config.get("dflash_config", {})
        if isinstance(draft_config, dict)
        else getattr(draft_config, "dflash_config", {})
    ) or {}
    layers = list(method_config.get("target_layer_ids", []))
    if not layers:
        raise ValueError("draft config does not define target capture layer ids")
    return layers


def _resolve_mask_token_id(cfg: Config, draft_model: Any, tokenizer: Any) -> int:
    from specforge.training.model_utils import resolve_mask_token_id

    return resolve_mask_token_id(
        explicit=cfg.model.mask_token_id,
        tokenizer=tokenizer,
        draft_model=draft_model,
        embedding_vocab_size=int(draft_model.config.vocab_size),
    )


def build_eagle3_model(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    target_config: Any,
    _tokenizer: Any,
) -> StrategyModelParts:
    from specforge.core.eagle3 import OnlineEagle3Model

    model = OnlineEagle3Model(
        draft_model=draft_model,
        length=cfg.training.ttt_length,
        attention_backend=cfg.training.attention_backend,
        lk_loss_type=cfg.training.lk_loss_type,
        kl_scale=cfg.training.kl_scale,
        kl_decay=cfg.training.kl_decay,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    needs_target_head = cfg.mode == "offline" or (
        cfg.training.deployment_mode == "disaggregated"
        and cfg.training.role == "consumer"
    )
    target_head = None
    if needs_target_head:
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    return StrategyModelParts(
        model=model,
        target_head=target_head,
    )


def build_peagle_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> StrategyModelParts:
    from specforge.core.peagle import OnlinePEagleModel

    mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
    model = OnlinePEagleModel(
        draft_model=draft_model,
        mask_token_id=mask_token_id,
        num_depths=cfg.training.num_depths,
        down_sample_ratio=cfg.training.down_sample_ratio,
        down_sample_ratio_min=cfg.training.down_sample_ratio_min,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    return StrategyModelParts(model=model)


def _build_dflash_family_model(
    cfg: Config,
    draft_model: Any,
    tokenizer: Any,
    model_factory: Callable[[Dict[str, Any]], Any],
) -> StrategyModelParts:
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    mask_token_id = _resolve_mask_token_id(cfg, draft_model, tokenizer)
    draft_model.mask_token_id = mask_token_id
    method_config = getattr(draft_model.config, "dflash_config", None)
    if method_config is None:
        draft_model.config.dflash_config = {}
        method_config = draft_model.config.dflash_config
    method_config["mask_token_id"] = mask_token_id
    method_config["target_layer_ids"] = list(draft_model.target_layer_ids)

    target_parts = TargetEmbeddingsAndHead.from_pretrained(
        cfg.model.target_model_path,
        embed_key=cfg.model.embedding_key,
        lm_head_key=cfg.model.lm_head_key,
        cache_dir=cfg.model.cache_dir,
        device=_device().type,
        dtype=_torch_dtype(cfg),
        trust_remote_code=cfg.model.trust_remote_code,
    )
    common = {
        "draft_model": draft_model,
        "target_lm_head": target_parts.lm_head,
        "target_embed_tokens": target_parts.embed_tokens,
        "mask_token_id": mask_token_id,
        "block_size": int(draft_model.block_size),
        "attention_backend": cfg.training.attention_backend,
        "num_anchors": cfg.training.num_anchors,
        "loss_decay_gamma": cfg.training.loss_decay_gamma,
    }
    model = model_factory(common).to(device=_device(), dtype=_torch_dtype(cfg))
    return StrategyModelParts(
        model=model,
        capture_layers=list(draft_model.target_layer_ids),
    )


def build_dflash_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> StrategyModelParts:
    from specforge.core.dflash import OnlineDFlashModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
        ),
    )


def build_domino_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> StrategyModelParts:
    from specforge.core.dflash import OnlineDominoModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDominoModel(
            **common,
            shift_label=bool(getattr(draft_model, "shift_label", False)),
        ),
    )


def build_dspark_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> StrategyModelParts:
    from specforge.core.dflash import OnlineDSparkModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDSparkModel(
            **common,
            dspark_ce_loss_alpha=cfg.training.dspark_ce_loss_alpha,
            dspark_l1_loss_alpha=cfg.training.dspark_l1_loss_alpha,
            dspark_confidence_head_alpha=(cfg.training.dspark_confidence_head_alpha),
        ),
    )


def eagle3_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "compact_teacher": cfg.training.compact_teacher,
        "compact_teacher_chunk_size": cfg.training.compact_teacher_chunk_size,
    }


def domino_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "lambda_start": cfg.training.lambda_base_start,
        "decay_ratio": cfg.training.lambda_base_decay_ratio,
    }


def dflash_min_loss_tokens(cfg: Config) -> int:
    from specforge.training.model_loading import resolve_draft_config

    block_size = getattr(resolve_draft_config(cfg), "block_size", None)
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 1
    ):
        raise ValueError(
            "DFlash-family draft config must define a positive integer block_size"
        )
    return 2 * block_size


def populate_dflash_generated_config(
    payload: Dict[str, Any], target_config: Any, _cfg: Config
) -> None:
    from specforge.modeling.draft.dflash import build_target_layer_ids

    target_layers = getattr(target_config, "num_hidden_layers", None)
    if not isinstance(target_layers, int) or target_layers < 1:
        raise ValueError(
            "DFlash auto-generation requires target num_hidden_layers, got "
            f"{target_layers!r}"
        )
    payload["num_target_layers"] = target_layers
    payload["block_size"] = 16
    payload["dflash_config"] = {
        "target_layer_ids": build_target_layer_ids(target_layers, 1)
    }


def apply_dflash_overrides(cfg: Config, draft_config: Any) -> None:
    if cfg.model.draft_num_hidden_layers is None:
        return
    from specforge.modeling.draft.dflash import build_target_layer_ids

    target_layers = int(draft_config.num_target_layers)
    method_config = dict(getattr(draft_config, "dflash_config", None) or {})
    method_config["target_layer_ids"] = build_target_layer_ids(
        target_layers, cfg.model.draft_num_hidden_layers
    )
    draft_config.dflash_config = method_config


__all__ = [
    "DraftConfigSpec",
    "OnlineCaptureMode",
    "StrategyAssemblySpec",
    "StrategyModelParts",
    "apply_dflash_overrides",
    "build_dflash_model",
    "build_domino_model",
    "build_dspark_model",
    "build_eagle3_draft",
    "build_eagle3_model",
    "build_peagle_draft",
    "build_peagle_model",
    "build_registered_draft",
    "dflash_min_loss_tokens",
    "dflash_needs_input_tools",
    "domino_strategy_kwargs",
    "eagle3_strategy_kwargs",
    "populate_dflash_generated_config",
    "resolve_dflash_capture_layers",
    "resolve_eagle_capture_layers",
]
