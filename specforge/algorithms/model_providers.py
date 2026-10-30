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
"""Algorithm-owned model and configuration providers.

The public assembler owns topology and lifecycle.  This module owns the facts
that vary by draft method: how to create its config and model, which target
features it captures, and which dataset policies it needs.  All heavy imports
remain inside callables so resolving the algorithm registry stays import-light.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional

if TYPE_CHECKING:
    from transformers import PretrainedConfig

    from specforge.config import Config
    from specforge.modeling.draft.dflash_kernels import DFlashKernels


@dataclass
class AlgorithmModelParts:
    """Algorithm-specific pieces returned to the application assembler."""

    model: Any
    target_head: Any = None
    capture_layers: Optional[List[int]] = None


def dflash_needs_input_tools(cfg: Config, draft_model: Any) -> bool:
    return cfg.mode == "online"


def _torch_dtype(cfg: Config):
    import torch

    return getattr(torch, cfg.model.torch_dtype)


def _device():
    from specforge.utils import get_local_device

    return get_local_device()


def _warm_start(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    *,
    allow_missing_embedding: bool = False,
) -> None:
    if not cfg.model.draft_checkpoint_path:
        return
    from specforge.training.model_loading import warm_start_draft_model

    warm_start_draft_model(
        draft_model,
        cfg.model.draft_checkpoint_path,
        draft_config=draft_config,
        strategy=cfg.training.strategy,
        allow_missing_embedding=allow_missing_embedding,
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
    _warm_start(
        cfg,
        draft_model,
        draft_config,
        allow_missing_embedding=True,
    )
    _load_vocab_mapping(cfg, draft_model)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
            cache_dir=cfg.model.cache_dir,
        )
    draft_model.freeze_embedding()
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_peagle_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.draft.peagle import PEagleDraftModel

    draft_model = PEagleDraftModel(draft_config)
    if cfg.model.load_target_embedding:
        draft_model.load_embedding(
            cfg.model.target_model_path,
            embedding_key=cfg.model.embedding_key,
            cache_dir=cfg.model.cache_dir,
        )
    _warm_start(
        cfg,
        draft_model,
        draft_config,
        allow_missing_embedding=True,
    )
    _load_vocab_mapping(cfg, draft_model)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def peagle_resume_contract(
    _cfg: Config, draft_model: Any, model: Any
) -> Dict[str, Any]:
    """Persist the resolved P-EAGLE model and objective semantics."""

    return {
        "peagle_num_draft_layers": len(draft_model.layers),
        "peagle_norm_before_residual": bool(draft_model.norm_before_residual),
        "peagle_num_depths": int(model.num_depths),
        "peagle_down_sample_ratio": float(model.down_sample_ratio),
        "peagle_down_sample_ratio_min": float(model.down_sample_ratio_min),
        "peagle_mask_token_id": int(model.mask_token_id),
    }


def _finish_registered_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    draft_model: Any,
):
    _warm_start(cfg, draft_model, draft_config)
    return draft_model.to(device=_device(), dtype=_torch_dtype(cfg))


def build_registered_draft(cfg: Config, draft_config: PretrainedConfig):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
    )
    return _finish_registered_draft(cfg, draft_config, draft_model)


def build_dflash_draft(
    cfg: Config,
    draft_config: PretrainedConfig,
    kernels: Optional[DFlashKernels],
):
    from specforge.modeling.auto import AutoDraftModel

    draft_config._attn_implementation = cfg.training.attention_backend
    draft_model = AutoDraftModel.from_config(
        draft_config,
        torch_dtype=_torch_dtype(cfg),
        dflash_kernels=kernels,
    )
    return _finish_registered_draft(cfg, draft_config, draft_model)


def resolve_eagle_capture_layers(
    cfg: Config, draft_config: Any, target_config: Any
) -> List[int]:
    """Read capture layers from the resolved inline draft settings."""

    eagle_config = (
        draft_config.get("eagle_config", {})
        if isinstance(draft_config, dict)
        else getattr(draft_config, "eagle_config", {})
    ) or {}
    layers = eagle_config.get("eagle_aux_hidden_state_layer_ids")
    if layers is None:
        raise ValueError(
            "inline draft eagle_config.eagle_aux_hidden_state_layer_ids is required"
        )
    layers = list(layers)
    if len(layers) != 3 or any(not isinstance(i, int) or i < 0 for i in layers):
        raise ValueError(
            "resolved EAGLE capture layers must contain exactly three "
            f"non-negative integers, got {layers!r}"
        )
    return layers


def _validate_dflash_block_size(draft_config: Any) -> None:
    if isinstance(draft_config, dict):
        method_config = draft_config.get("dflash_config", {}) or {}
        block_size = draft_config.get("block_size", method_config.get("block_size"))
    else:
        method_config = getattr(draft_config, "dflash_config", {}) or {}
        block_size = getattr(draft_config, "block_size", None)
        if block_size is None:
            block_size = method_config.get("block_size")
    if (
        not isinstance(block_size, int)
        or isinstance(block_size, bool)
        or block_size < 2
    ):
        raise ValueError(
            "DFlash-family draft config must define an integer block_size >= 2"
        )


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


def build_eagle3_model(
    cfg: Config,
    draft_model: Any,
    draft_config: Any,
    target_config: Any,
    _tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.eagle3.model import OnlineEagle3Model

    model = OnlineEagle3Model(
        draft_model=draft_model,
        length=cfg.training.ttt_length,
        attention_backend=cfg.training.attention_backend,
        lk_loss_type=cfg.training.lk_loss_type,
        kl_scale=cfg.training.kl_scale,
        kl_decay=cfg.training.kl_decay,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    needs_target_head = cfg.mode == "offline" or (
        cfg.deployment.mode == "disaggregated" and cfg.training.role == "consumer"
    )
    target_head = None
    if needs_target_head:
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            embedding_key=cfg.model.embedding_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    return AlgorithmModelParts(
        model=model,
        target_head=target_head,
    )


def build_peagle_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.peagle.model import OnlinePEagleModel

    mask_token_id = draft_model.config.mask_token_id
    model = OnlinePEagleModel(
        draft_model=draft_model,
        mask_token_id=mask_token_id,
        num_depths=cfg.training.num_depths,
        down_sample_ratio=cfg.training.down_sample_ratio,
        down_sample_ratio_min=cfg.training.down_sample_ratio_min,
    ).to(device=_device(), dtype=_torch_dtype(cfg))
    # Server capture transports the final target hidden state instead of full
    # vocabulary logits.  The consumer resolves that representation through the
    # same frozen target head used by offline EAGLE3; no target model is loaded
    # in the trainer process.
    target_head = None
    if (
        cfg.mode == "online"
        and cfg.deployment.mode == "disaggregated"
        and cfg.training.role == "consumer"
    ):
        from specforge.modeling.target.target_head import TargetHead

        target_head = TargetHead.from_pretrained(
            cfg.model.target_model_path,
            lm_head_key=cfg.model.lm_head_key,
            embedding_key=cfg.model.embedding_key,
            cache_dir=cfg.model.cache_dir,
            trust_remote_code=cfg.model.trust_remote_code,
        )
    return AlgorithmModelParts(model=model, target_head=target_head)


def _build_dflash_family_model(
    cfg: Config,
    draft_model: Any,
    tokenizer: Any,
    model_factory: Callable[[Dict[str, Any]], Any],
) -> AlgorithmModelParts:
    from specforge.modeling.target.target_utils import TargetEmbeddingsAndHead

    _validate_dflash_block_size(draft_model)
    mask_token_id = draft_model.mask_token_id
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
        "objective_chunk_blocks": cfg.training.objective_chunk_blocks,
        "objective_checkpointing": cfg.training.objective_checkpointing,
    }
    model = model_factory(common).to(device=_device(), dtype=_torch_dtype(cfg))
    return AlgorithmModelParts(
        model=model,
        capture_layers=list(draft_model.target_layer_ids),
    )


def build_dflash_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDFlashModel(
            **common,
            loss_type=cfg.training.loss_type,
            dpace_alpha=cfg.training.dpace_alpha,
            selector_loss_alpha=cfg.training.dflash2_selector_loss_alpha,
            selector_warmup_ratio=cfg.training.dflash2_selector_warmup_ratio,
            selector_ramp_ratio=cfg.training.dflash2_selector_ramp_ratio,
            selector_stop_gradient=cfg.training.dflash2_selector_stop_gradient,
            lk_loss_type=cfg.training.lk_loss_type,
            kl_scale=cfg.training.kl_scale,
            kl_decay=cfg.training.kl_decay,
            teacher_metrics=cfg.training.dflash_teacher_metrics,
            fused_logits_loss=cfg.training.fused_logits_loss,
        ),
    )


def build_domino_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDominoModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDominoModel(
            **common,
            shift_label=bool(getattr(draft_model, "shift_label", False)),
            fused_logits_loss=cfg.training.fused_logits_loss,
        ),
    )


def build_dspark_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDSparkModel(
            **common,
            dspark_ce_loss_alpha=cfg.training.dspark_ce_loss_alpha,
            dspark_l1_loss_alpha=cfg.training.dspark_l1_loss_alpha,
            fused_logits_loss=cfg.training.fused_logits_loss,
            dspark_confidence_head_alpha=(cfg.training.dspark_confidence_head_alpha),
        ),
    )


def build_dspine_model(
    cfg: Config,
    draft_model: Any,
    _draft_config: Any,
    _target_config: Any,
    tokenizer: Any,
) -> AlgorithmModelParts:
    from specforge.algorithms.dspine.model import OnlineDSpineModel

    return _build_dflash_family_model(
        cfg,
        draft_model,
        tokenizer,
        lambda common: OnlineDSpineModel(
            **common, fused_logits_loss=cfg.training.fused_logits_loss
        ),
    )


def eagle3_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "trim_loss_positions": cfg.training.trim_loss_positions,
        "compact_teacher": cfg.training.compact_teacher,
        "compact_teacher_chunk_size": cfg.training.compact_teacher_chunk_size,
    }


def domino_strategy_kwargs(cfg: Config) -> Dict[str, Any]:
    return {
        "lambda_start": cfg.training.lambda_base_start,
        "decay_ratio": cfg.training.lambda_base_decay_ratio,
    }


def dflash_min_loss_tokens(_cfg: Config, draft_config: Any) -> int:
    _validate_dflash_block_size(draft_config)
    return 2


__all__ = [
    "AlgorithmModelParts",
    "build_dflash_model",
    "build_domino_model",
    "build_dspark_model",
    "build_dspine_model",
    "build_eagle3_draft",
    "build_eagle3_model",
    "build_peagle_draft",
    "peagle_resume_contract",
    "build_peagle_model",
    "build_registered_draft",
    "dflash_min_loss_tokens",
    "dflash_needs_input_tools",
    "domino_strategy_kwargs",
    "eagle3_strategy_kwargs",
    "resolve_dflash_capture_layers",
    "resolve_eagle_capture_layers",
]
