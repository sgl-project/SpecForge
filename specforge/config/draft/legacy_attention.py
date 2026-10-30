"""Legacy attention fields; only decoding and checkpoint compatibility use these."""

from typing import Optional

from transformers import Qwen3Config

from .base import integer as _integer

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


MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rope_interleave",
)
