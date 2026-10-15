"""Geometry and positional transforms for captured target-attention K/V."""

import math
from types import SimpleNamespace

import torch


def target_kv_config(config):
    """Resolve the opt-in KV path without changing hidden-state defaults."""
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    options = getattr(config, "dflash_config", None) or {}
    source = conditioning_source(config)
    if source == "hidden":
        if any(name.startswith("target_kv_") for name in options):
            raise ValueError("target_kv options require conditioning_source='kv'")
        return None
    layers = options.get("target_layer_ids")
    if not isinstance(layers, (list, tuple)) or not layers:
        raise ValueError("target_kv requires explicit target_layer_ids")
    if any(type(i) is not int or i < 0 for i in layers) or len(set(layers)) != len(
        layers
    ):
        raise ValueError("target_layer_ids must be unique non-negative integers")
    for field in ("target_kv_heads", "target_kv_head_dim"):
        if type(options.get(field)) is not int or options[field] <= 0:
            raise ValueError(f"{field} must be a positive integer")
    mode = options.get("target_kv_position_mode", "cached")
    if mode not in ("cached", "derope_reproject"):
        raise ValueError("target_kv_position_mode must be cached or derope_reproject")
    norm = options.get("target_kv_context_key_norm", False)
    if type(norm) is not bool:
        raise ValueError("target_kv_context_key_norm must be a boolean")
    if mode == "cached" and norm:
        raise ValueError("context key normalization requires derope_reproject")
    resolved = {
        "layer_ids": tuple(layers),
        "heads": options["target_kv_heads"],
        "head_dim": options["target_kv_head_dim"],
        "position_mode": mode,
        "context_key_norm": norm,
    }
    if mode == "derope_reproject":
        theta = options.get("target_kv_rope_theta")
        dim = options.get("target_kv_rotary_dim")
        if type(theta) not in (float, int) or not math.isfinite(theta) or theta <= 1:
            raise ValueError("target_kv_rope_theta must be finite and > 1")
        if type(dim) is not int or dim <= 0 or dim % 2 or dim > resolved["head_dim"]:
            raise ValueError(
                "target_kv_rotary_dim must be positive, even and <= head_dim"
            )
        if options.get("target_kv_rope_layout") != "neox":
            raise ValueError("target_kv_rope_layout must explicitly be 'neox'")
        resolved.update(rope_theta=float(theta), rotary_dim=dim, rope_layout="neox")
    return resolved


def conditioning_source(config):
    """Read the saved setting, accepting the original DSpark config spelling."""
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    options = getattr(config, "dflash_config", None) or {}
    legacy = options.get("conditioning_source")
    source = getattr(config, "conditioning_source", None)

    def normalize(value):
        return "kv" if value == "target_kv" else value

    if (
        source is not None
        and legacy is not None
        and normalize(source) != normalize(legacy)
    ):
        raise ValueError("conflicting draft conditioning_source settings")
    source = normalize(source if source is not None else legacy or "hidden")
    if source not in ("hidden", "kv"):
        raise ValueError("conditioning_source must be 'hidden' or 'kv'")
    return source


def resolve_target_kv_config(config, target_config):
    """Derive target geometry once and persist it for standalone draft loading.

    Layer selection, positional transform and context-key normalization remain
    draft choices. Saved/explicit geometry is checked, never silently replaced.
    """
    conditioning_source(config)
    target = getattr(target_config, "text_config", target_config)
    if getattr(target, "model_type", None) not in ("qwen3", "qwen3_5_text"):
        raise ValueError("target_kv supports Qwen3 and Qwen3.5 text attention only")
    options = dict(getattr(config, "dflash_config", None) or {})
    rope = (
        getattr(target, "rope_parameters", None)
        or getattr(target, "rope_scaling", None)
        or {}
    )
    if rope.get("rope_type", "default") != "default":
        raise ValueError("target_kv capture supports only default target RoPE")
    head_dim = _target_head_dim(target)
    factor = rope.get(
        "partial_rotary_factor", getattr(target, "partial_rotary_factor", 1.0)
    )
    geometry = {
        "target_kv_heads": target.num_key_value_heads,
        "target_kv_head_dim": head_dim,
        "target_kv_rope_theta": rope.get(
            "rope_theta", getattr(target, "rope_theta", None)
        ),
        "target_kv_rotary_dim": int(head_dim * factor),
        "target_kv_rope_layout": "neox",
    }
    for name, value in geometry.items():
        if name in options and (
            isinstance(options[name], bool) or options[name] != value
        ):
            raise ValueError(f"saved/explicit {name} disagrees with the target model")
        options[name] = value
    options.pop("conditioning_source", None)
    candidate = SimpleNamespace(
        conditioning_source="kv",
        dflash_config=options,
        hidden_size=config.hidden_size,
        vocab_size=config.vocab_size,
    )
    validate_target_kv_config(candidate, target_config)
    config.conditioning_source = "kv"
    config.dflash_config = options
    return config


def _target_head_dim(target):
    return (
        getattr(target, "head_dim", None)
        or target.hidden_size // target.num_attention_heads
    )


def validate_target_kv_config(config, target_config):
    """Fail closed on incompatible capture geometry or unsupported target RoPE."""
    if isinstance(config, dict):
        config = SimpleNamespace(**config)
    kv = target_kv_config(config)
    if kv is None:
        raise ValueError("target KV validation requires conditioning_source='kv'")
    target = getattr(target_config, "text_config", target_config)
    if getattr(target, "model_type", None) not in ("qwen3", "qwen3_5_text"):
        raise ValueError("target_kv supports Qwen3 and Qwen3.5 text attention only")
    expected = (target.num_key_value_heads, _target_head_dim(target))
    if (kv["heads"], kv["head_dim"]) != expected:
        raise ValueError("target_kv head geometry disagrees with the target model")
    layer_types = (
        getattr(target, "layer_types", None)
        or ["full_attention"] * target.num_hidden_layers
    )
    if any(
        i >= len(layer_types) or layer_types[i] != "full_attention"
        for i in kv["layer_ids"]
    ):
        raise ValueError("target_kv layer IDs must select target full-attention layers")
    if (
        config.hidden_size != target.hidden_size
        or config.vocab_size != target.vocab_size
    ):
        raise ValueError(
            "draft hidden_size/vocab_size must match the frozen target head"
        )
    if kv["position_mode"] == "derope_reproject":
        rope = (
            getattr(target, "rope_parameters", None)
            or getattr(target, "rope_scaling", None)
            or {}
        )
        if rope.get("rope_type", "default") != "default":
            raise ValueError("target_kv re-encoding supports only default target RoPE")
        theta = rope.get("rope_theta", getattr(target, "rope_theta", None))
        factor = rope.get(
            "partial_rotary_factor", getattr(target, "partial_rotary_factor", 1.0)
        )
        if (
            theta != kv["rope_theta"]
            or int(_target_head_dim(target) * factor) != kv["rotary_dim"]
        ):
            raise ValueError(
                "target_kv rotary geometry disagrees with the target model"
            )
    return kv


def capture_geometry(config, target_config):
    """Capture semantics independent of the draft's ablation settings."""
    kv = validate_target_kv_config(config, target_config)
    target = getattr(target_config, "text_config", target_config)
    rope = (
        getattr(target, "rope_parameters", None)
        or getattr(target, "rope_scaling", None)
        or {}
    )
    if rope.get("rope_type", "default") != "default":
        raise ValueError("offline KV capture currently requires default target RoPE")
    theta = rope.get("rope_theta", getattr(target, "rope_theta", None))
    factor = rope.get(
        "partial_rotary_factor", getattr(target, "partial_rotary_factor", 1.0)
    )
    return {
        "layer_ids": list(kv["layer_ids"]),
        "heads": kv["heads"],
        "head_dim": kv["head_dim"],
        "rope_theta": theta,
        "rotary_dim": int(_target_head_dim(target) * factor),
        "rope_layout": "neox",
    }


def inverse_target_kv_rope(target_kv, position_ids, *, rope_theta, rotary_dim):
    """Undo partial NeoX RoPE; preserve values and non-rotary key channels.

    Input is [batch, sequence, selected layer, K/V, head, channel]. Frequencies
    are recomputed in FP32 so FSDP/BF16 buffer casts cannot round them first.
    This removes RoPE only, not target QK normalization or BF16 rounding.
    """
    if (
        target_kv.ndim != 6
        or target_kv.shape[3] != 2
        or not target_kv.is_floating_point()
    ):
        raise ValueError("target_kv must be floating [B,S,L,2,H,D]")
    if position_ids.shape != target_kv.shape[:2] or position_ids.dtype not in (
        torch.int32,
        torch.int64,
    ):
        raise ValueError("position_ids must be integer [B,S] matching target_kv")
    if (
        type(rotary_dim) is not int
        or rotary_dim <= 0
        or rotary_dim % 2
        or rotary_dim > target_kv.shape[-1]
    ):
        raise ValueError("rotary_dim must be positive, even and <= key head dimension")
    if (
        type(rope_theta) not in (int, float)
        or not math.isfinite(rope_theta)
        or rope_theta <= 1
    ):
        raise ValueError("rope_theta must be finite and > 1")
    dtype = torch.float64 if target_kv.dtype == torch.float64 else torch.float32
    with torch.autocast(device_type=target_kv.device.type, enabled=False):
        exponents = (
            torch.arange(0, rotary_dim, 2, device=target_kv.device, dtype=dtype)
            / rotary_dim
        )
        angles = position_ids.to(dtype)[..., None] * (1.0 / rope_theta**exponents)
        angles = torch.cat((angles, angles), dim=-1)[:, :, None, None, :]
        keys = target_kv[:, :, :, 0]
        rotated = keys[..., :rotary_dim].to(dtype)
        half = rotary_dim // 2
        rotated_half = torch.cat((-rotated[..., half:], rotated[..., :half]), dim=-1)
        unrotated = (rotated * angles.cos() - rotated_half * angles.sin()).to(
            target_kv.dtype
        )
    keys = torch.cat((unrotated, keys[..., rotary_dim:]), dim=-1)
    return torch.stack((keys, target_kv[:, :, :, 1]), dim=3)
