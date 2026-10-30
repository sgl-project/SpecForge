"""Configuration of the native single-layer Qwen3.5 MTP draft."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Self

from transformers import Qwen3Config

from .base import DraftConfig, boolean, method_config, validate_dimensions, validate_gqa
from .rope import resolve_rope_type, validate_native_rope_type


@dataclass(frozen=True, kw_only=True)
class Qwen3_5MTPConfig(DraftConfig):
    attention_contract = "qwen_mtp"
    requires_causal = True
    _nested_fields = frozenset({"share_lm_head"})
    attn_output_gate: bool
    partial_rotary_factor: float
    rope_theta: float
    share_lm_head: bool

    @classmethod
    def validate_native_config(cls, config, target_config) -> None:
        """Check the Qwen native MTP contract without importing target settings.

        Target decoder depth/model_type differ from the extracted single-layer
        draft. Only fields used by the native MTP computation are compared.
        """
        if hasattr(config, "layers"):
            from .model import LayerConfig

            config = LayerConfig(config, config.layers[0])
        target = getattr(target_config, "text_config", target_config)
        validate_native_rope_type(config, target, supported_types={"default"})
        rope = getattr(target, "rope_parameters", None) or {}
        defaults = {
            "attn_output_gate": True,
            "partial_rotary_factor": 1.0,
            "attention_bias": False,
            "rope_theta": 10000.0,
        }
        differences = []
        for name in (
            "vocab_size",
            "hidden_size",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
            "hidden_act",
            "rms_norm_eps",
            "attention_bias",
            "attn_output_gate",
            "partial_rotary_factor",
            "rope_theta",
        ):
            expected = getattr(target, name, rope.get(name, defaults.get(name)))
            actual = getattr(config, name, defaults.get(name))
            if expected is None or actual != expected:
                differences.append(f"{name}: inline={actual!r}, native={expected!r}")
        if differences:
            raise ValueError(
                "inline draft settings do not match native Qwen MTP: "
                + "; ".join(differences)
            )

    @classmethod
    def from_model_config(cls, config: Qwen3Config) -> Self:
        validate_dimensions(config)
        head_dim = (
            config.layers[0].attention.head_dim
            if hasattr(config, "layers")
            else validate_gqa(config)
        )
        if config.num_hidden_layers != 1:
            raise ValueError("Qwen3.5 MTP requires num_hidden_layers=1")
        rope = getattr(config, "rope_parameters", None) or {}
        factor = getattr(
            config, "partial_rotary_factor", rope.get("partial_rotary_factor", 1.0)
        )
        theta = getattr(config, "rope_theta", rope.get("rope_theta", 10000.0))
        if type(theta) not in (float, int) or not math.isfinite(theta) or theta <= 0:
            raise ValueError("rope_theta must be a finite positive number")
        if resolve_rope_type(config, source="inline draft") != "default":
            raise ValueError("Qwen3.5 MTP supports only default RoPE")
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
            rope_theta=float(theta),
            share_lm_head=boolean(
                method_config(config, "mtp_config").get("share_lm_head", True),
                "mtp_config.share_lm_head",
            ),
        )

    def to_model_config(self, config: Qwen3Config) -> Qwen3Config:
        result = super().to_model_config(config)
        # New HF versions put these under rope_parameters, while the native
        # MTP rotary module consumes the resolved scalar attributes.
        result.rope_parameters = {
            **(getattr(result, "rope_parameters", None) or {}),
            "rope_theta": self.rope_theta,
            "partial_rotary_factor": self.partial_rotary_factor,
        }
        result.mtp_config = {
            **method_config(result, "mtp_config"),
            "share_lm_head": self.share_lm_head,
        }
        return result
