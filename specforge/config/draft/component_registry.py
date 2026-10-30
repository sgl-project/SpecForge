"""Open component catalog with explicit execution contracts and lazy builders."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module
from types import MappingProxyType
from typing import Callable, Mapping

from .layers import ConvSpec, FFNSpec, GQASpec, MLASpec, ResidualSpec


@dataclass(frozen=True)
class ComponentRegistration:
    spec_class: type
    builders: Mapping[str, str | Callable]

    def __post_init__(self):
        object.__setattr__(self, "builders", MappingProxyType(dict(self.builders)))


_COMPONENTS: dict[tuple[str, str], ComponentRegistration] = {}


def register_component(kind: str, name: str, spec_class: type, *, builders: Mapping):
    """Register a spec and its supported caller contracts without loading torch."""
    from dataclasses import is_dataclass

    if not is_dataclass(spec_class) or not spec_class.__dataclass_params__.frozen:
        raise TypeError("component specs must be frozen dataclasses")
    if not builders:
        raise ValueError("a component must implement at least one execution contract")
    key = (kind, name)
    registration = ComponentRegistration(spec_class, builders)
    if key in _COMPONENTS and _COMPONENTS[key] != registration:
        raise ValueError(f"component {key!r} is already registered")
    _COMPONENTS[key] = registration


def component_registration(kind: str, name: str) -> ComponentRegistration:
    try:
        return _COMPONENTS[kind, name]
    except KeyError:
        raise ValueError(f"Unknown {kind} type: {name!r}; not implemented") from None


def require_contract(kind: str, spec, contract: str):
    registration = component_registration(kind, spec.type)
    if not isinstance(spec, registration.spec_class):
        raise TypeError(
            f"{kind} {spec.type!r} requires {registration.spec_class.__name__}"
        )
    if contract not in registration.builders:
        raise ValueError(f"{kind} {spec.type!r} does not implement {contract!r}")
    return registration


def build_component(kind: str, spec, contract: str, **context):
    builder = require_contract(kind, spec, contract).builders[contract]
    if isinstance(builder, str):
        module, name = builder.split(":")
        builder = getattr(import_module(module), name)
    return builder(spec=spec, **context)


_PREFIX = "specforge.modeling.draft.models.layers"
for _type in ("gqa", "mha"):
    register_component(
        "attention",
        _type,
        GQASpec,
        builders={
            "context_then_draft": f"{_PREFIX}.attention:build_gqa",
            "eagle_cache": f"{_PREFIX}.attention:build_eagle",
            "peagle": f"{_PREFIX}.attention:build_peagle",
            "qwen_mtp": f"{_PREFIX}.attention:build_mtp",
        },
    )
register_component(
    "attention",
    "mla",
    MLASpec,
    builders={
        "context_then_draft": f"{_PREFIX}.attention:build_mla",
    },
)
register_component(
    "ffn",
    "dense",
    FFNSpec,
    builders={
        "tokenwise": f"{_PREFIX}.ffn:build_dense",
    },
)
register_component(
    "residual",
    "standard",
    ResidualSpec,
    builders={
        "single_stream": f"{_PREFIX}.residual:build_standard",
    },
)
register_component(
    "conv",
    "grouped",
    ConvSpec,
    builders={
        "draft_block": f"{_PREFIX}.convolution:build_grouped",
    },
)
del _type
