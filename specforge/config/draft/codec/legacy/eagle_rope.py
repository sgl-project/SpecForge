"""EAGLE's historical rotary rules, distinct from the HF/Qwen3 reader."""

import math
from collections.abc import Mapping

from ..._validation import finite_number, positive_number
from ..components import decode_component
from .context import _check_fields
from .rules import resolve_alias

_PARAMETERS = {
    "default": set(),
    "linear": {"factor"},
    "llama3": {
        "factor",
        "low_freq_factor",
        "high_freq_factor",
        "original_max_position_embeddings",
    },
    "mrope": {"mrope_section", "factor", "mrope_interleaved"},
    "yarn": {
        "factor",
        "original_max_position_embeddings",
        "beta_fast",
        "beta_slow",
        "mscale",
        "mscale_all_dim",
        "attention_factor",
        "truncate",
    },
}


def resolve_eagle_rope(payload: Mapping):
    modern, scaling = payload.get("rope_parameters"), payload.get("rope_scaling")
    for name, value in (("rope_parameters", modern), ("rope_scaling", scaling)):
        if value is not None and not isinstance(value, Mapping):
            raise ValueError(f"{name} must be an object or null")

    def normalize(source):
        values = dict(source)
        alias = {"rope_type": values.pop("type")} if "type" in values else {}
        kind = resolve_alias("rope_type", values, alias, default="default")
        if type(kind) is not str or kind not in _PARAMETERS:
            raise ValueError(f"unsupported EAGLE rope_type {kind!r}")
        _check_fields(
            values,
            _PARAMETERS[kind] | {"rope_type", "rope_theta", "partial_rotary_factor"},
            "EAGLE RoPE",
        )
        result = {
            "rope_type": kind,
            "rope_theta": resolve_alias("rope_theta", values, payload, default=10000.0),
            # The old EAGLE implementation always rotates the entire head.
            "partial_rotary_factor": 1.0,
        }
        if kind in {"llama3", "yarn"}:
            # LlamaConfig fills this before constructing the historical model.
            values.setdefault(
                "original_max_position_embeddings",
                payload.get("max_position_embeddings", 2048),
            )
        if kind == "linear":
            result["factor"] = values.get("factor")
        elif kind == "mrope":
            result["mrope_section"] = values.get("mrope_section")
            # Historical EAGLE always used contiguous sections.
            result["mrope_interleaved"] = False
        elif kind == "llama3":
            frequency_fields = (
                "low_freq_factor",
                "high_freq_factor",
                "original_max_position_embeddings",
            )
            if any(values.get(name) is None for name in frequency_fields):
                # LlamaRotaryEmbedding applies this scaling only when all
                # frequency parameters are present.
                result["rope_type"] = "default"
            else:
                result.update({name: values[name] for name in frequency_fields})
                result["factor"] = (
                    1.0 if values.get("factor") is None else values["factor"]
                )
        elif kind == "yarn":
            factor = values.get("factor")
            positive_number("factor", factor)

            def scale(name):
                if factor <= 1:
                    return 1.0
                value = values.get(name)
                finite_number(name, value)
                if value < 0:
                    raise ValueError(f"{name} must be nonnegative")
                return 0.1 * value * math.log(factor) + 1.0

            result.update(
                factor=factor,
                original_max_position_embeddings=values.get(
                    "original_max_position_embeddings"
                ),
                beta_fast=values.get("beta_fast"),
                beta_slow=values.get("beta_slow"),
                attention_factor=scale("mscale") / scale("mscale_all_dim"),
                truncate=True,
                # In EAGLE's GQA, mscale_all_dim affects only the ratio above;
                # it never multiplies full QK logits as it does in MLA.
                mscale_all_dim=0.0,
            )
        return decode_component("rope", result)

    result = normalize(modern if modern is not None else (scaling or {}))
    if modern is not None and scaling is not None and result != normalize(scaling):
        raise ValueError(
            "conflicting legacy aliases for rope_parameters and rope_scaling"
        )
    return result
