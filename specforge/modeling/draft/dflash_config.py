"""Resolved DFlash-family architecture settings, without model allocation.

HF config objects remain the serialization format. These immutable views own
normalization and static validation; unknown checkpoint fields are retained.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, fields
from typing import Any, Optional, Self

from transformers import Qwen3Config

from .config_base import DraftConfig
from .config_base import boolean as _boolean
from .config_base import integer as _integer
from .config_base import validate_dimensions, validate_gqa

FULL_ATTENTION = "full_attention"
SLIDING_ATTENTION = "sliding_attention"
_VALID_DFLASH_LAYER_TYPES = {FULL_ATTENTION, SLIDING_ATTENTION}


def resolve_dflash_causal_block(config) -> Optional[bool]:
    """Read the direction override; None preserves the legacy per-layout defaults."""

    value = getattr(config, "is_causal", None)
    if value is not None and not isinstance(value, bool):
        raise ValueError("DFlash config.is_causal must be a boolean or null")
    return value


def resolve_dflash_attention_layout(
    config: Qwen3Config,
) -> tuple[tuple[str, ...], Optional[int]]:
    """Validate and return the configured per-layer DFlash attention layout."""

    num_hidden_layers = config.num_hidden_layers
    layer_types = config.layer_types
    if not isinstance(layer_types, (list, tuple)) or any(
        not isinstance(layer, str) for layer in layer_types
    ):
        raise ValueError("DFlash config.layer_types must be a list of attention types")
    layer_types = tuple(layer_types)

    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "DFlash config.layer_types must contain exactly "
            f"num_hidden_layers={num_hidden_layers} entries, got "
            f"{len(layer_types)}"
        )
    invalid = set(layer_types) - _VALID_DFLASH_LAYER_TYPES
    if invalid:
        raise ValueError(
            "DFlash config.layer_types supports only full_attention and "
            f"sliding_attention, got {sorted(invalid)}"
        )

    if SLIDING_ATTENTION not in layer_types:
        return layer_types, None

    sliding_window = config.sliding_window
    _integer(sliding_window, "config.sliding_window")
    return layer_types, sliding_window


def resolve_dflash_attention_mode(config: Qwen3Config) -> str:
    """Validate and return the configured draft attention mode.

    ``gqa`` and ``mha`` share :class:`Qwen3DFlashAttention`; ``mla`` swaps in
    the latent parameterization while retaining the family decoder,
    target-context injection, masks, and objectives.
    """

    dflash_config = getattr(config, "dflash_config", None) or {}
    attention_mode = str(dflash_config.get("attention_mode", "gqa")).lower()
    if attention_mode not in ("gqa", "mha", "mla"):
        raise ValueError(
            "DFlash dflash_config.attention_mode must be one of "
            f"gqa, mha, mla, got {attention_mode!r}"
        )
    return attention_mode


def _resolve_mla_rope_interleaved(config: Qwen3Config) -> bool:
    """Rotation convention from the standard MLA ``rope_interleave`` field."""

    return _boolean(
        getattr(config, "rope_interleave", True),
        "config.rope_interleave",
    )


def validate_dflash_mla_config(config: Qwen3Config) -> None:
    """Validate the standard MLA dimension fields carried by a draft config."""

    required = (
        "kv_lora_rank",
        "qk_nope_head_dim",
        "qk_rope_head_dim",
        "v_head_dim",
    )
    missing = [name for name in required if getattr(config, name, None) is None]
    if missing:
        raise ValueError(f"MLA draft config is missing required fields: {missing}")

    q_lora_rank = getattr(config, "q_lora_rank", None)
    if q_lora_rank is not None:
        _integer(q_lora_rank, "q_lora_rank")

    for name in ("kv_lora_rank", "qk_rope_head_dim", "v_head_dim"):
        _integer(getattr(config, name), name)

    _integer(config.qk_nope_head_dim, "qk_nope_head_dim", 0)
    qk_rope_head_dim = config.qk_rope_head_dim
    if qk_rope_head_dim % 2:
        raise ValueError(f"qk_rope_head_dim must be even, got {qk_rope_head_dim}")
    _resolve_mla_rope_interleaved(config)


def validate_dflash_attention_config(config: Qwen3Config) -> str:
    """Validate the selected attention parameterization and return its mode."""

    attention_mode = resolve_dflash_attention_mode(config)
    heads = _integer(config.num_attention_heads, "num_attention_heads")
    if attention_mode == "mla":
        validate_dflash_mla_config(config)
        return attention_mode

    head_dim = validate_gqa(config)
    kv_heads = config.num_key_value_heads
    if head_dim % 2:
        raise ValueError("head_dim must be even for DFlash rotary embeddings")
    if attention_mode == "mha" and kv_heads != heads:
        raise ValueError(
            "attention_mode 'mha' requires num_key_value_heads == "
            f"num_attention_heads, got {config.num_key_value_heads} and "
            f"{config.num_attention_heads}"
        )
    return attention_mode


def build_target_layer_ids(num_target_layers: int, num_draft_layers: int):
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    target_layer_ids = [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]
    return target_layer_ids


@dataclass(frozen=True, kw_only=True)
class DFlashConfig(DraftConfig):
    block_size: int
    layer_types: tuple[str, ...]
    sliding_window: int | None
    is_causal: bool | None
    attention_mode: str
    target_layer_ids: tuple[int, ...]
    mask_token_id: int | None
    projector_type: str | None
    pure_draft_prefix_len: int
    shift_label: bool

    @classmethod
    def from_model_config(cls, config: Qwen3Config) -> Self:
        method = getattr(config, "dflash_config", None)
        if method is None:
            method = {}
        if not isinstance(method, dict):
            raise ValueError("dflash_config must be an object")
        validate_dimensions(config)
        layer_types, window = resolve_dflash_attention_layout(config)
        mode = validate_dflash_attention_config(config)
        block_size = getattr(config, "block_size", None)
        if block_size is None:
            block_size = method.get("block_size")
        block_size = _integer(block_size, "block_size", 2)
        target_layers = method.get("target_layer_ids")
        if target_layers is None:
            target_count = _integer(
                getattr(config, "num_target_layers", None), "num_target_layers"
            )
            target_layers = build_target_layer_ids(
                target_count, config.num_hidden_layers
            )
        if not isinstance(target_layers, (list, tuple)) or not target_layers:
            raise ValueError(
                "target_layer_ids must be a non-empty list of layer indices"
            )
        for layer in target_layers:
            _integer(layer, "target_layer_ids entry", 0)
        mask_token_id = method.get("mask_token_id")
        if mask_token_id is not None:
            _integer(mask_token_id, "mask_token_id", 0)
        return cls(
            block_size=block_size,
            layer_types=layer_types,
            sliding_window=window,
            is_causal=resolve_dflash_causal_block(config),
            attention_mode=mode,
            target_layer_ids=tuple(target_layers),
            mask_token_id=mask_token_id,
            pure_draft_prefix_len=_integer(
                method.get("pure_draft_prefix_len", 0), "pure_draft_prefix_len", 0
            ),
            shift_label=_boolean(method.get("shift_label", False), "shift_label"),
            **cls._variant_fields(config, method, mode, block_size),
        )

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        projector = method.get("projector_type")
        if projector is not None and not isinstance(projector, str):
            raise ValueError("projector_type must be a string or null")
        return {"projector_type": projector}

    def to_model_config(self, config: Qwen3Config) -> Qwen3Config:
        """Return a normalized copy in the existing checkpoint JSON layout."""
        result = copy.deepcopy(config)
        method = dict(getattr(result, "dflash_config", None) or {})
        top_level = {"layer_types", "sliding_window", "is_causal", "block_size"}
        for field in fields(self):
            if field.name not in top_level:
                value = getattr(self, field.name)
                method[field.name] = list(value) if isinstance(value, tuple) else value
        # Top-level block_size retains its historical precedence; canonicalize
        # the alias as well so downstream readers cannot disagree.
        result.block_size = method["block_size"] = self.block_size
        result.layer_types = list(self.layer_types)
        result.sliding_window = self.sliding_window
        result.use_sliding_window = self.sliding_window is not None
        result.is_causal = self.is_causal
        result.dflash_config = method
        return result


def _projector(method: dict, expected: str) -> str:
    value = method.get("projector_type")
    if value is not None and value != expected:
        raise ValueError(f"projector_type must be {expected!r}, got {value!r}")
    return expected


@dataclass(frozen=True, kw_only=True)
class DFlash2Config(DFlashConfig):
    conv_kernel_size: int
    conv_group_size: int
    selector_rank: int
    selector_top_k: int
    output_multiplier: float
    final_logit_softcapping: float | None

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        values = super()._variant_fields(config, method, mode, block_size)
        for name in (
            "conv_kernel_size",
            "conv_group_size",
            "selector_rank",
            "selector_top_k",
        ):
            values[name] = _integer(method.get(name), f"dflash_config.{name}")
        if values["conv_kernel_size"] > block_size:
            raise ValueError("conv_kernel_size must not exceed block_size")
        if config.hidden_size % values["conv_group_size"]:
            raise ValueError("conv_group_size must divide hidden_size")
        if values["selector_top_k"] > config.vocab_size:
            raise ValueError("selector_top_k must be in [1, vocab_size]")
        multiplier = float(method.get("output_multiplier", 1.0))
        softcap = method.get("final_logit_softcapping")
        if not math.isfinite(multiplier):
            raise ValueError("output_multiplier must be finite")
        if softcap is not None:
            softcap = float(softcap)
            if not math.isfinite(softcap) or softcap <= 0:
                raise ValueError("final_logit_softcapping must be finite and > 0")
        return {
            **values,
            "output_multiplier": multiplier,
            "final_logit_softcapping": softcap,
        }


@dataclass(frozen=True, kw_only=True)
class DominoConfig(DFlashConfig):
    emb_dim: int
    gru_hidden_dim: int

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        return {
            "projector_type": _projector(method, "domino"),
            "emb_dim": _integer(method.get("emb_dim"), "emb_dim"),
            "gru_hidden_dim": _integer(method.get("gru_hidden_dim"), "gru_hidden_dim"),
        }


@dataclass(frozen=True, kw_only=True)
class DSparkConfig(DFlashConfig):
    markov_rank: int
    markov_head_type: str
    enable_confidence_head: bool
    confidence_head_with_markov: bool

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], mode: str, block_size: int
    ) -> dict[str, Any]:
        if mode == "gqa" and config.num_key_value_heads == config.num_attention_heads:
            raise ValueError(
                "DSpark defaults to GQA; set attention_mode='mha' for equal query/KV heads"
            )
        # Legacy configs used null to disable the optional Markov head.
        rank = method.get("markov_rank")
        rank = _integer(0 if rank is None else rank, "markov_rank", 0)
        head_type = str(method.get("markov_head_type", "vanilla")).lower()
        if rank and head_type not in ("vanilla", "gated", "rnn"):
            raise ValueError(f"Unsupported markov_head_type: {head_type!r}")
        enabled = method.get("enable_confidence_head")
        if "enable_confidence_head" not in method:
            enabled = float(method.get("confidence_head_alpha", 0.0) or 0.0) > 0
        with_markov = _boolean(
            method.get("confidence_head_with_markov", False),
            "confidence_head_with_markov",
        )
        if with_markov and rank == 0:
            raise ValueError(
                "confidence_head_with_markov=True requires markov_rank > 0"
            )
        return {
            "projector_type": _projector(method, "dspark"),
            "markov_rank": rank,
            "markov_head_type": head_type,
            "enable_confidence_head": _boolean(enabled, "enable_confidence_head"),
            "confidence_head_with_markov": with_markov,
        }
