"""EAGLE's historical rotary rules, distinct from the HF/Qwen3 reader."""

import math
from collections.abc import Mapping

from ..._validation import finite_number, positive_number
from ..components import decode_component
from .auxiliary import _DEFAULTS
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
        legacy_type = values.pop("type", None)
        # Match the old model: rope_type takes precedence over the HF alias.
        kind = values.get("rope_type", legacy_type)
        if kind is None:
            kind = "default"
        if kind != "mrope":
            values.pop("mrope_section", None)
            values.pop("mrope_interleaved", None)
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
                payload.get(
                    "max_position_embeddings", _DEFAULTS["max_position_embeddings"]
                ),
            )
        if kind == "linear":
            result["factor"] = values.get("factor")
        elif kind == "mrope":
            if values.get("mrope_interleaved", False) or "factor" in values:
                raise ValueError(
                    "legacy EAGLE mrope does not support interleaving or factor"
                )
            result["mrope_section"] = values.get("mrope_section")
            # Historical EAGLE always used contiguous sections.
            result["mrope_interleaved"] = False
        elif kind == "llama3":
            for name in _PARAMETERS["llama3"]:
                if values.get(name) is None:
                    raise ValueError(f"legacy llama3 RoPE requires {name}")
                result[name] = values[name]
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
