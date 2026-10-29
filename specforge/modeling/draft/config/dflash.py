"""Resolved DFlash-family architecture settings, without model allocation.

HF config objects remain the serialization format. These immutable views own
normalization and static validation; unknown checkpoint fields are retained.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, fields
from typing import Any, Optional, Self

from transformers import Qwen3Config

from .base import DraftConfig
from .base import boolean as _boolean
from .base import integer as _integer
from .base import validate_dimensions, validate_gqa
from .components import validate_components

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
        validate_components(config, attention_types=("gqa", "mha", "mla"))
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
