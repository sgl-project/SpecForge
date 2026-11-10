"""Legacy aliases and Qwen3-style attention/RoPE defaults, resolved once."""

from collections.abc import Mapping

from ..._validation import boolean, positive_int
from ..components import decode_component
from .yarn import expand_yarn

_MISSING = object()


def resolve_alias(name, *sources, default=_MISSING, ignore_none=False):
    values = [source[name] for source in sources if name in source]
    if ignore_none:
        values = [value for value in values if value is not None]
    if not values:
        if default is _MISSING:
            raise ValueError(f"legacy configuration is missing {name}")
        return default
    first = values[0]
    if any(
        value != first or (type(value) is bool) != (type(first) is bool)
        for value in values[1:]
    ):
        raise ValueError(f"conflicting legacy aliases for {name}")
    return first


def resolve_rope(payload: Mapping):
    """Normalize old/new RoPE spellings; unsupported kinds fail explicitly."""
    modern = payload.get("rope_parameters")
    scaling = payload.get("rope_scaling")
    for name, value in (("rope_parameters", modern), ("rope_scaling", scaling)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"legacy {name} must be an object or null")

    def normalize(source):
        values = dict(source)
        alias = {"rope_type": values.pop("type")} if "type" in values else {}
        values["rope_type"] = resolve_alias(
            "rope_type", values, alias, default="default"
        )
        # Registration does not imply compatibility with the old Qwen3 path.
        kind = values["rope_type"]
        if type(kind) is not str or kind not in {"default", "linear", "llama3", "yarn"}:
            raise ValueError(
                f"unsupported legacy Qwen3 rope_type {values['rope_type']!r}"
            )
        values["rope_theta"] = resolve_alias(
            "rope_theta", values, payload, default=10000.0
        )
        values["partial_rotary_factor"] = resolve_alias(
            "partial_rotary_factor", values, payload, default=1.0
        )
        if values["rope_type"] == "yarn":
            values = expand_yarn(values, payload.get("max_position_embeddings", 32768))
        return decode_component("rope", values)

    result = normalize(modern if modern is not None else (scaling or {}))
    if modern is not None and scaling is not None and result != normalize(scaling):
        raise ValueError(
            "conflicting legacy aliases for rope_parameters and rope_scaling"
        )
    return result


def resolve_attention_layout(payload: Mapping, depth: int):
    """Return layer kinds and each kind's explicit (causal, window) settings."""
    positive_int("num_hidden_layers", depth)
    use_window = payload.get("use_sliding_window", False)
    boolean("use_sliding_window", use_window)
    window = payload.get("sliding_window", 4096) if use_window else None
    if window is not None:
        positive_int("sliding_window", window)
    layers = payload.get("layer_types")
    if layers is None:
        threshold = payload.get("max_window_layers", 28)
        if type(threshold) is not int or threshold < 0:
            raise ValueError("max_window_layers must be a nonnegative integer")
        layers = tuple(
            (
                "sliding_attention"
                if window is not None and i >= threshold
                else "full_attention"
            )
            for i in range(depth)
        )
    if not isinstance(layers, (list, tuple)) or len(layers) != depth:
        raise ValueError("layer_types must contain num_hidden_layers entries")
    if any(
        type(name) is not str or name not in {"full_attention", "sliding_attention"}
        for name in layers
    ):
        raise ValueError(
            "legacy layer_types supports only full_attention and sliding_attention"
        )
    if "sliding_attention" in layers and window is None:
        raise ValueError(
            "sliding_attention requires use_sliding_window and sliding_window"
        )
    causal = payload.get("is_causal")
    if causal is not None:
        boolean("is_causal", causal)
    settings = {
        name: (
            name == "sliding_attention" if causal is None else causal,
            window if name == "sliding_attention" else None,
        )
        for name in dict.fromkeys(layers)
    }
    return tuple(layers), settings


def default_target_layer_ids(
    num_target_layers: int, num_draft_layers: int
) -> tuple[int, ...]:
    """Historical DFlash feature selection, resolved before node construction."""
    positive_int("num_target_layers", num_target_layers)
    positive_int("num_draft_layers", num_draft_layers)
    if num_draft_layers == 1:
        return (num_target_layers // 2,)
    layers = tuple(
        int(round(1 + index * (num_target_layers - 4) / (num_draft_layers - 1)))
        for index in range(num_draft_layers)
    )
    if any(layer < 0 or layer >= num_target_layers for layer in layers):
        raise ValueError(
            "default target_layer_ids are out of range; supply explicit layers"
        )
    if len(set(layers)) != len(layers):
        raise ValueError(
            "default target_layer_ids contain duplicates; supply explicit distinct layers"
        )
    return layers


def reject_ignored_settings(payload, **fixed):
    """Reject structural values that the historical model would silently ignore."""
    for name, expected in fixed.items():
        value = payload.get(name)
        if value is not None and value != expected:
            raise ValueError(f"legacy {name} must be {expected!r}")
