"""Geometry and positional transforms for captured target-attention K/V."""

import math

import torch


def target_kv_config(config):
    """Resolve the opt-in KV path without changing hidden-state defaults."""
    options = getattr(config, "dflash_config", None) or {}
    source = options.get("conditioning_source", "hidden")
    if source not in ("hidden", "target_kv"):
        raise ValueError("conditioning_source must be 'hidden' or 'target_kv'")
    if source == "hidden":
        if any(name.startswith("target_kv_") for name in options):
            raise ValueError(
                "target_kv options require conditioning_source='target_kv'"
            )
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


def validate_target_kv_config(config, target_config):
    """Fail closed on incompatible capture geometry or unsupported target RoPE."""
    kv = target_kv_config(config)
    if kv is None:
        raise ValueError("dspark_kv requires conditioning_source='target_kv'")
    target = getattr(target_config, "text_config", target_config)
    if getattr(target, "model_type", None) != "qwen3_5_text":
        raise ValueError("target_kv currently supports Qwen3.5 text attention only")
    expected = (target.num_key_value_heads, target.head_dim)
    if (kv["heads"], kv["head_dim"]) != expected:
        raise ValueError("target_kv head geometry disagrees with the target model")
    layer_types = target.layer_types
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
        rope = getattr(target, "rope_parameters", None) or {}
        if rope.get("rope_type", "default") != "default":
            raise ValueError("target_kv re-encoding supports only default target RoPE")
        theta = rope.get("rope_theta", getattr(target, "rope_theta", None))
        factor = rope.get(
            "partial_rotary_factor", getattr(target, "partial_rotary_factor", 1.0)
        )
        if (
            theta != kv["rope_theta"]
            or int(target.head_dim * factor) != kv["rotary_dim"]
        ):
            raise ValueError(
                "target_kv rotary geometry disagrees with the target model"
            )
    return kv


def capture_geometry(config, target_config):
    """Capture semantics independent of the draft's ablation settings."""
    kv = validate_target_kv_config(config, target_config)
    target = getattr(target_config, "text_config", target_config)
    rope = getattr(target, "rope_parameters", None) or {}
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
        "rotary_dim": int(target.head_dim * factor),
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
