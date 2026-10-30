"""Legacy attention fields; only decoding and checkpoint compatibility use these."""

from typing import Optional

from transformers import Qwen3Config

from .base import boolean as _boolean
from .base import integer as _integer
from .base import validate_gqa

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


MLA_FIELDS = (
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rope_interleave",
)
