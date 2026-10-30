"""Versioned public draft format, shared by embedded YAML and saved JSON."""

from __future__ import annotations

import copy
from collections.abc import Mapping

SCHEMA_VERSION = 1
_LEGACY_STRUCTURE = {
    "layer_types",
    "sliding_window",
    "is_causal",
    "use_sliding_window",
    "max_window_layers",
    "attention_config",
    "ffn_config",
    "residual_config",
    "q_lora_rank",
    "kv_lora_rank",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "v_head_dim",
    "rope_interleave",
}


def decode_format(payload: Mapping) -> dict:
    """Unversioned inputs are legacy; versioned inputs have one structural form."""
    values = copy.deepcopy(dict(payload))
    if "schema_version" not in values:
        return values
    version = values.pop("schema_version")
    if type(version) is not int or version != SCHEMA_VERSION:
        raise ValueError(f"unsupported draft schema_version: {version!r}")
    if "layers" not in values:
        raise ValueError("draft schema_version=1 requires layers")
    if not isinstance(values["layers"], (Mapping, list, tuple)):
        raise ValueError("layers must be an object or a per-layer list")
    extra = set(values) & _LEGACY_STRUCTURE
    method_extra = set(values.get("dflash_config", {}) or {}) & {
        "attention_mode",
        "conv_kernel_size",
        "conv_group_size",
    }
    if extra or method_extra:
        raise ValueError(
            f"legacy structural fields are not allowed in version 1: {sorted(extra | method_extra)}; use layers"
        )
    return values


def encode_format(config) -> dict:
    """Write resolved layer data; no HF conversion or model mutation is involved."""
    from .auto import AutoDraftConfig

    values = AutoDraftConfig.resolve(config).to_dict()
    values.pop("_attn_implementation", None)
    return {"schema_version": SCHEMA_VERSION, **values}
