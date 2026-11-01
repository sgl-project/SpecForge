"""Component choices and validation, independent of tensor implementations.

Only implemented combinations are accepted. Reserved names identify extension
points, not runnable placeholders or fallbacks to another structure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from transformers import PretrainedConfig


@dataclass(frozen=True)
class ComponentConfig:
    type: str
    variant: str | None = None


def component_config(
    config: PretrainedConfig, name: str, default: str
) -> ComponentConfig:
    value = getattr(config, name, None)
    if value is None:
        value = {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    kind = value.get("type", default)
    if not isinstance(kind, str) or not kind:
        raise ValueError(f"{name}.type must be a non-empty string")
    variant = value.get("variant")
    if variant is not None and (not isinstance(variant, str) or not variant):
        raise ValueError(f"{name}.variant must be a non-empty string")
    return ComponentConfig(kind, variant)


def validate_components(
    config: PretrainedConfig,
    *,
    attention_types: tuple[str, ...],
    causal_only: bool = False,
) -> None:
    """Reject unsupported structures before allocating any model parameters."""
    for name, default, supported, reserved in (
        (
            "attention_config",
            "gqa",
            attention_types,
            ("gqa", "mha", "mla", "dsa", "qsa", "kda"),
        ),
        ("residual_config", "standard", ("standard",), ("standard", "gr", "mhc")),
        ("ffn_config", "dense", ("dense",), ("dense", "moe")),
    ):
        choice = component_config(config, name, default)
        if choice.type not in reserved:
            raise ValueError(f"Unknown {name}.type: {choice.type!r}")
        if choice.type not in supported:
            raise ValueError(
                f"{name}.type={choice.type!r} is not implemented for this architecture"
            )
        if choice.variant is not None:
            raise ValueError(
                f"{name}.variant={choice.variant!r} is not supported for {choice.type!r}"
            )
        allowed = {"type", "variant"}
        if name == "attention_config":
            allowed.add("is_causal")
            attention = getattr(config, name, None) or {}
            if "is_causal" in attention:
                causal = attention["is_causal"]
                if causal is not None and type(causal) is not bool:
                    raise ValueError(
                        "attention_config.is_causal must be a boolean or null"
                    )
                if causal_only and causal is not True:
                    raise ValueError(
                        "This architecture requires attention_config.is_causal=true"
                    )
            if (
                choice.type == "mha"
                and config.num_attention_heads != config.num_key_value_heads
            ):
                raise ValueError(
                    "attention_config.type='mha' requires num_attention_heads == num_key_value_heads"
                )
            if choice.type == "mla":
                allowed.update(MLA_FIELDS)
        extra = set(getattr(config, name, None) or {}) - allowed
        if extra:
            raise ValueError(f"Unknown {name} fields: {sorted(extra)}")


MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rope_interleave",
)
