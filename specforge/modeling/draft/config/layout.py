"""Translate the public JSON layout at the boundary with existing HF models.

The runtime keeps the established HF attributes used by training providers.
Serialization has a single canonical location for each interpreted field.
Legacy-only metadata is preserved, and conflicting new/old fields are rejected.
"""

from __future__ import annotations

import copy
from dataclasses import fields
from typing import Any

from .base import DraftConfig
from .components import MLA_FIELDS


def option_class(payload: dict[str, Any]) -> type[DraftConfig] | None:
    from ..registry import DRAFT_REGISTRY

    architectures = payload.get("architectures") or []
    if len(architectures) != 1:
        return None
    model = DRAFT_REGISTRY.get(architectures[0])
    return getattr(model, "draft_options_class", None)


def spec_locations(options: type[DraftConfig]) -> dict[str, tuple[str | None, str]]:
    """Read the serialization policy owned by the concrete algorithm config."""
    excluded = options.non_spec_fields
    result = {}
    for field in fields(options):
        if field.name not in excluded:
            section = (
                None
                if field.name in options.top_level_fields
                else options.legacy_section
            )
            result[field.name] = (section, field.name)
    result.update(options.spec_aliases)
    for alias in options.spec_aliases:
        old = options.spec_aliases[alias][1]
        if old != alias:
            result.pop(old, None)
    return result


def _object(payload: dict[str, Any], name: str) -> dict[str, Any]:
    value = payload.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be an object")
    return value


def _put(
    payload: dict[str, Any], section: str | None, name: str, value: Any, source: str
) -> None:
    target = payload if section is None else payload.setdefault(section, {})
    if not isinstance(target, dict):
        raise ValueError(f"{section} must be an object")
    if name in target and target[name] != value:
        raise ValueError(
            f"Conflicting {source} and {section + '.' if section else ''}{name}"
        )
    target[name] = copy.deepcopy(value)


def decode_layout(
    payload: dict[str, Any], options: type[DraftConfig] | None = None
) -> dict[str, Any]:
    """Convert a public or legacy payload to runtime HF attributes, without mutation."""
    payload = copy.deepcopy(payload)
    options = options or option_class(payload)
    if options is None:
        if "spec_config" in payload:
            raise ValueError(
                "spec_config requires a registered draft configuration class"
            )
        return payload
    if "spec_config" in payload:
        spec = _object(payload, "spec_config")
        if type(spec.get("schema_version")) is not int or spec["schema_version"] != 1:
            raise ValueError("spec_config.schema_version must be 1")
        if spec.get("algorithm") != options.algorithm:
            raise ValueError(
                f"spec_config.algorithm must be {options.algorithm!r} for this architecture"
            )
        locations = spec_locations(options)
        unknown = set(spec) - {"schema_version", "algorithm"} - locations.keys()
        if unknown:
            raise ValueError(
                f"Unknown spec_config fields for {options.algorithm}: {sorted(unknown)}"
            )
        for name, value in spec.items():
            if name in locations:
                section, key = locations[name]
                _put(payload, section, key, value, f"spec_config.{name}")
                if name == "block_size" and "block_size" in _object(
                    payload, "dflash_config"
                ):
                    _put(
                        payload,
                        "dflash_config",
                        "block_size",
                        value,
                        "spec_config.block_size",
                    )
        payload.pop("spec_config")
    attention = _object(payload, "attention_config")
    if options.legacy_section == "dflash_config":
        if "type" in attention:
            _put(
                payload,
                "dflash_config",
                "attention_mode",
                attention["type"],
                "attention_config.type",
            )
        if "is_causal" in attention:
            _put(
                payload,
                None,
                "is_causal",
                attention["is_causal"],
                "attention_config.is_causal",
            )
        for name in MLA_FIELDS:
            if name in attention:
                _put(payload, None, name, attention[name], f"attention_config.{name}")
    elif "is_causal" in attention and attention["is_causal"] is not True:
        raise ValueError("This architecture requires attention_config.is_causal=true")
    # Legacy inputs acquire the same explicit component defaults as saved configs.
    default_attention = {"type": "gqa", "is_causal": True}
    if options.legacy_section == "dflash_config":
        default_attention = {
            "type": str(
                _object(payload, "dflash_config").get("attention_mode", "gqa")
            ).lower(),
            "is_causal": payload.get("is_causal"),
        }
    payload.setdefault("attention_config", default_attention)
    payload.setdefault("residual_config", {"type": "standard"})
    payload.setdefault("ffn_config", {"type": "dense"})
    return payload


def encode_layout(
    payload: dict[str, Any], options: type[DraftConfig] | None = None
) -> dict[str, Any]:
    """Emit canonical JSON fields, leaving unrecognized legacy metadata intact."""
    payload = copy.deepcopy(payload)
    options = options or option_class(payload)
    if options is None:
        return payload
    spec = {"schema_version": 1, "algorithm": options.algorithm}
    for public, (section, name) in spec_locations(options).items():
        source = payload if section is None else payload.get(section, {})
        if name in source:
            spec[public] = source.pop(name)
    method = payload.get("dflash_config", {})
    if "enable_confidence_head" in spec:
        # This historical loss coefficient only selected whether to allocate
        # the head. The resolved structural flag now records that decision.
        method.pop("confidence_head_alpha", None)
    attention = dict(payload.get("attention_config", {}))
    if options.legacy_section == "dflash_config":
        attention["type"] = method.pop("attention_mode", attention.get("type", "gqa"))
        if "block_size" in method:
            spec.setdefault("block_size", method.pop("block_size"))
        attention["is_causal"] = payload.pop("is_causal", attention.get("is_causal"))
        if attention["type"] == "mla":
            for name in MLA_FIELDS:
                if name in payload:
                    attention[name] = payload.pop(name)
    else:
        attention.setdefault("type", "gqa")
        attention["is_causal"] = True
    payload["attention_config"] = attention
    payload.setdefault("residual_config", {"type": "standard"})
    payload.setdefault("ffn_config", {"type": "dense"})
    payload["spec_config"] = spec
    for section in ("dflash_config", "eagle_config", "mtp_config"):
        if section in payload and not payload[section]:
            payload.pop(section)
    return payload
