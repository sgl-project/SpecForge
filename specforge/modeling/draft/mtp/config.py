"""Configuration of the native single-layer Qwen3.5 MTP draft."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from transformers import Qwen3Config

from ..config_base import (
    DraftConfig,
    boolean,
    method_config,
    validate_dimensions,
    validate_gqa,
)


@dataclass(frozen=True, kw_only=True)
class Qwen3_5MTPConfig(DraftConfig):
    _nested_fields = frozenset({"share_lm_head"})
    attn_output_gate: bool
    partial_rotary_factor: float
    share_lm_head: bool

    @classmethod
    def from_model_config(cls, config: Qwen3Config) -> Self:
        validate_dimensions(config)
        head_dim = validate_gqa(config)
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen3.5 MTP requires num_hidden_layers=1")
        factor = getattr(config, "partial_rotary_factor", 1.0)
        if (
            type(factor) not in (float, int)
            or not math.isfinite(factor)
            or not 0 < factor <= 1
        ):
            raise ValueError("partial_rotary_factor must be a finite number in (0, 1]")
        rotary_dim = int(head_dim * factor)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError(
                "partial_rotary_factor must produce a positive even rotary dimension"
            )
        return cls(
            attn_output_gate=boolean(
                getattr(config, "attn_output_gate", False), "attn_output_gate"
            ),
            partial_rotary_factor=float(factor),
            share_lm_head=boolean(
                method_config(config, "mtp_config").get("share_lm_head", True),
                "mtp_config.share_lm_head",
            ),
        )

    def to_model_config(self, config: Qwen3Config) -> Qwen3Config:
        result = super().to_model_config(config)
        result.mtp_config = {
            **method_config(result, "mtp_config"),
            "share_lm_head": self.share_lm_head,
        }
        return result
