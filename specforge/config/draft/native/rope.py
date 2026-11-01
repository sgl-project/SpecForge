"""RoPE type checks shared by native draft configuration adapters."""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any


def resolve_rope_type(config: Any, *, source: str) -> str:
    """Read modern/legacy RoPE declarations without changing the config.

    An absent declaration means default RoPE. Conflicting aliases and
    per-layer layouts must not silently fall back to that default.
    """
    declarations = {}
    top_level = getattr(config, "rope_type", None)
    if top_level is not None:
        declarations["rope_type"] = top_level
    for field in ("rope_parameters", "rope_scaling"):
        parameters = getattr(config, field, None)
        if parameters is None:
            continue
        if not isinstance(parameters, Mapping):
            raise ValueError(f"{source} {field} must be an object or null")
        if any(isinstance(value, Mapping) for value in parameters.values()):
            raise ValueError(
                f"{source} {field} uses a per-layer/nested RoPE layout; "
                "the native draft adapter must select its RoPE parameters explicitly"
            )
        for key in ("rope_type", "type"):
            if key in parameters:
                declarations[f"{field}.{key}"] = parameters[key]
        if (
            parameters
            and field == "rope_scaling"
            and not any(key in parameters for key in ("rope_type", "type"))
        ):
            raise ValueError(f"{source} rope_scaling must declare rope_type or type")
    for field, value in declarations.items():
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{source} {field} must be a non-empty string")
    if len(set(declarations.values())) > 1:
        raise ValueError(
            f"{source} has conflicting RoPE type declarations: {declarations}"
        )
    return next(iter(declarations.values()), "default")


def validate_native_rope_type(
    config: Any,
    native_config: Any,
    *,
    supported_types: Collection[str],
) -> None:
    """Require matching RoPE types supported by this native draft adapter.

    The caller selects the native text/draft configuration and additionally
    checks the numerical parameters used by its RoPE implementation.
    """
    inline_type = resolve_rope_type(config, source="inline draft")
    native_type = resolve_rope_type(native_config, source="native draft")
    if inline_type != native_type:
        raise ValueError(
            "inline draft rope_type does not match native draft: "
            f"inline={inline_type!r}, native={native_type!r}; "
            f"supported types: {sorted(supported_types)}"
        )
    if native_type not in supported_types:
        raise ValueError(
            f"unsupported native draft rope_type={native_type!r}; "
            f"supported types: {sorted(supported_types)}"
        )
