"""Decode component data and isolate legacy flat-field compatibility."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, fields

from .base import integer
from .component_registry import component_registration, require_contract
from .layers import LayerSpec
from .legacy_attention import MLA_FIELDS

_LAYER_FIELDS = {"attention", "ffn", "residual", "conv"}
_ATTENTION_FIELDS = (
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    *MLA_FIELDS,
)
_LEGACY_LAYER_FIELDS = (
    "layer_types",
    "sliding_window",
    "is_causal",
    "use_sliding_window",
    "max_window_layers",
    "attention_config",
    "ffn_config",
    "residual_config",
)


def _component(kind, data, defaults):
    if not isinstance(data, Mapping):
        raise ValueError(f"{kind} must be an object")
    values = {**defaults, **data}
    name = values.get("type")
    registration = component_registration(kind, name)
    allowed = {field.name for field in fields(registration.spec_class)}
    extra = set(data) - allowed
    if extra:
        raise ValueError(f"Unknown {kind} fields: {sorted(extra)}")
    values = {key: value for key, value in values.items() if key in allowed}
    try:
        return registration.spec_class(**values)
    except TypeError as exc:
        raise ValueError(f"Invalid {kind} {name!r}: {exc}") from exc


def decode_layers(config, options_class):
    """Expand a shared layer declaration once; runtime always stores a tuple."""
    count = integer(config.num_hidden_layers, "num_hidden_layers")
    method = getattr(config, "dflash_config", None) or {}
    raw = getattr(config, "layers", None)
    explicit = raw is not None
    if isinstance(raw, Mapping):
        raw = [raw] * count
    if raw is not None and (not isinstance(raw, (list, tuple)) or len(raw) != count):
        raise ValueError("layers must contain exactly num_hidden_layers entries")
    requires_causal = options_class.requires_causal if options_class else False
    requires_conv = options_class.requires_conv if options_class else False
    contract = options_class.attention_contract if options_class else "eagle_cache"
    legacy_types = getattr(config, "layer_types", None) or ["full_attention"] * count
    if not explicit and len(legacy_types) != count:
        raise ValueError("layer_types must contain exactly num_hidden_layers entries")
    if (
        not explicit
        and getattr(config, "architectures", None) == ["DSparkDraftModel"]
        and method.get("attention_mode", "gqa") == "gqa"
        and config.num_attention_heads == config.num_key_value_heads
    ):
        raise ValueError(
            "DSpark defaults to GQA; set attention_mode='mha' for equal query/KV heads"
        )
    result = []
    for index in range(count):
        item = raw[index] if explicit else {}
        if isinstance(item, LayerSpec):
            layer = item
        else:
            if not isinstance(item, Mapping) or set(item) - _LAYER_FIELDS:
                raise ValueError(f"Unknown layer fields at layers[{index}]")
            layer_type = "full_attention" if explicit else legacy_types[index]
            if layer_type not in ("full_attention", "sliding_attention"):
                raise ValueError(
                    "layer_types supports only full_attention and sliding_attention"
                )
            causal = getattr(config, "is_causal", None)
            if causal is None:
                causal = requires_causal or layer_type == "sliding_attention"
            window = (
                getattr(config, "sliding_window", None)
                if layer_type == "sliding_attention"
                else None
            )
            if layer_type == "sliding_attention":
                integer(window, "sliding_window")
            attention_defaults = {
                "type": method.get("attention_mode", "gqa"),
                "is_causal": causal,
                "sliding_window": window,
                **{
                    key: getattr(config, key)
                    for key in _ATTENTION_FIELDS
                    if hasattr(config, key)
                },
            }
            attention_data = (
                item.get("attention", {})
                if explicit
                else (getattr(config, "attention_config", None) or {})
            )
            # Legacy null direction means the layout's original default.
            if not explicit and attention_data.get("is_causal", True) is None:
                attention_data = {**attention_data, "is_causal": causal}
            attention = _component("attention", attention_data, attention_defaults)
            ffn = _component(
                "ffn",
                (
                    item.get("ffn", {})
                    if explicit
                    else (getattr(config, "ffn_config", None) or {})
                ),
                {"type": "dense", "intermediate_size": config.intermediate_size},
            )
            residual = _component(
                "residual",
                (
                    item.get("residual", {})
                    if explicit
                    else (getattr(config, "residual_config", None) or {})
                ),
                {"type": "standard"},
            )
            conv_data = (
                item.get("conv")
                if explicit
                else (
                    {
                        "kernel_size": method.get("conv_kernel_size"),
                        "group_size": method.get("conv_group_size"),
                    }
                    if requires_conv
                    else None
                )
            )
            conv = (
                _component("conv", conv_data, {"type": "grouped"})
                if conv_data is not None
                else None
            )
            layer = LayerSpec(
                attention=attention, ffn=ffn, residual=residual, conv=conv
            )
        require_contract("attention", layer.attention, contract)
        if (
            contract != "context_then_draft"
            and layer.attention.sliding_window is not None
        ):
            raise ValueError(
                f"{contract} does not yet implement sliding-window visibility"
            )
        require_contract("ffn", layer.ffn, "tokenwise")
        require_contract("residual", layer.residual, "single_stream")
        if requires_causal and not layer.attention.is_causal:
            raise ValueError(
                f"{options_class.__name__.replace("DraftConfig", "").replace("Config", "")} requires causal attention in every layer"
            )
        if requires_conv != (layer.conv is not None):
            raise ValueError(
                "this architecture requires conv in every layer"
                if requires_conv
                else "this architecture does not consume conv"
            )
        if layer.conv:
            require_contract("conv", layer.conv, "draft_block")
            block_size = getattr(config, "block_size", None) or method.get("block_size")
            if layer.conv.kernel_size > integer(block_size, "block_size", 2):
                raise ValueError("conv_kernel_size must not exceed block_size")
            if config.hidden_size % layer.conv.group_size:
                raise ValueError("conv_group_size must divide hidden_size")
        result.append(layer)
    return tuple(result)


def strip_legacy_layer_fields(values):
    for name in _LEGACY_LAYER_FIELDS:
        values.pop(name, None)
    for name in MLA_FIELDS:
        values.pop(name, None)
    method = values.get("dflash_config")
    if method is not None:
        for name in ("attention_mode", "conv_kernel_size", "conv_group_size"):
            method.pop(name, None)
    return values


def legacy_layer_fields(layers, *, check=True):
    """Translate only layouts that the legacy serving configuration can express.

    Mixed full/window attention remains representable with one shared window.
    Projection dimensions, FFN, residual and convolution must otherwise agree.
    """
    first = layers[0]
    projections = lambda a: {
        key: value
        for key, value in asdict(a).items()
        if key not in ("is_causal", "sliding_window")
    }
    if check and any(
        projections(layer.attention) != projections(first.attention)
        or layer.ffn != first.ffn
        or layer.residual != first.residual
        or layer.conv != first.conv
        for layer in layers
    ):
        raise ValueError(
            "per-layer component layout cannot be represented by the legacy serving config"
        )
    if check and (
        first.attention.type not in ("gqa", "mha", "mla")
        or first.ffn.type != "dense"
        or first.residual.type != "standard"
    ):
        raise ValueError("component layout has no legacy serving representation")
    windows = {layer.attention.sliding_window for layer in layers} - {None}
    directions = {layer.attention.is_causal for layer in layers}
    if len(directions) == 1:
        causal = first.attention.is_causal
    elif all(
        layer.attention.is_causal == (layer.attention.sliding_window is not None)
        for layer in layers
    ):
        causal = None
    elif check:
        raise ValueError(
            "per-layer causal directions cannot be represented by the legacy serving config"
        )
    else:
        causal = first.attention.is_causal
    if check and len(windows) > 1:
        raise ValueError(
            "per-layer sliding windows cannot be represented by the legacy serving config"
        )
    attention = projections(first.attention)
    mode = attention.pop("type")
    result = {
        **attention,
        "layer_types": [layer.attention.layer_type for layer in layers],
        "is_causal": causal,
        "sliding_window": (
            first.attention.sliding_window
            if len(windows) > 1
            else next(iter(windows), None)
        ),
        "use_sliding_window": bool(windows),
        "intermediate_size": first.ffn.intermediate_size,
    }
    method = {"attention_mode": mode}
    if first.conv:
        method.update(
            conv_kernel_size=first.conv.kernel_size,
            conv_group_size=first.conv.group_size,
        )
    return result, method
