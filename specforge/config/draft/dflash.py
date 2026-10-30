"""Resolved DFlash-family architecture settings, without model allocation.

Algorithm settings own target-context injection and prediction behavior.
Layer structure is resolved separately before these settings are validated.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any, Self

from transformers import Qwen3Config

from .base import DraftConfig
from .base import boolean as _boolean
from .base import integer as _integer
from .base import validate_dimensions


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
    attention_contract = "context_then_draft"
    block_size: int
    target_layer_ids: tuple[int, ...]
    mask_token_id: int | None
    projector_type: str | None
    pure_draft_prefix_len: int
    shift_label: bool

    @classmethod
    def validate_training_config(cls, config) -> None:
        mask = config.dflash_config.get("mask_token_id")
        if type(mask) is not int or not 0 <= mask < config.vocab_size:
            raise ValueError(
                "inline dflash_config.mask_token_id must be within vocab_size"
            )

    @classmethod
    def from_model_config(cls, config: Qwen3Config) -> Self:
        method = getattr(config, "dflash_config", None)
        if method is None:
            method = {}
        if not isinstance(method, Mapping):
            raise ValueError("dflash_config must be an object")
        validate_dimensions(config)
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
            target_layer_ids=tuple(target_layers),
            mask_token_id=mask_token_id,
            pure_draft_prefix_len=_integer(
                method.get("pure_draft_prefix_len", 0), "pure_draft_prefix_len", 0
            ),
            shift_label=_boolean(method.get("shift_label", False), "shift_label"),
            **cls._variant_fields(config, method, block_size),
        )

    @classmethod
    def _variant_fields(
        cls, config: Qwen3Config, method: dict[str, Any], block_size: int
    ) -> dict[str, Any]:
        projector = method.get("projector_type")
        if projector is not None and not isinstance(projector, str):
            raise ValueError("projector_type must be a string or null")
        return {"projector_type": projector}

    def to_model_config(self, config: Qwen3Config) -> Qwen3Config:
        """Return a normalized copy in the existing checkpoint JSON layout."""
        result = copy.deepcopy(config)
        method = dict(getattr(result, "dflash_config", None) or {})
        top_level = {"block_size"}
        for field in fields(self):
            if field.name not in top_level:
                value = getattr(self, field.name)
                method[field.name] = list(value) if isinstance(value, tuple) else value
        # Accept the legacy alias on input, but keep one runtime value.
        method.pop("block_size", None)
        result.block_size = self.block_size
        result.dflash_config = method
        return result


def _projector(method: dict, expected: str) -> str:
    value = method.get("projector_type")
    if value is not None and value != expected:
        raise ValueError(f"projector_type must be {expected!r}, got {value!r}")
    return expected
