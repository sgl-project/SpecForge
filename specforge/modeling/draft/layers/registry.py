"""Computational builders, separate from configuration schema registration."""

from collections.abc import Callable
from dataclasses import fields

from torch import nn

from specforge.config.draft.registry import component_schema, component_type

from .layout import ComponentLayout

_BUILDERS: dict[tuple[str, str], Callable[..., nn.Module]] = {}
_LAYOUTS: dict[tuple[str, str], ComponentLayout] = {}


def register_component_builder(
    kind: str, schema: type, builder: Callable[..., nn.Module]
) -> None:
    """Associate a registered node schema with a constructor, never replace one."""
    if not isinstance(schema, type):
        raise TypeError("component builder schema must be a type")
    name = component_type(kind, schema)
    if component_schema(kind, name) is not schema:
        raise ValueError(f"{kind}/{name}: builder must use the registered node schema")
    if not callable(builder):
        raise TypeError("component builder must be callable")
    key = (kind, name)
    existing = _BUILDERS.get(key)
    if existing is not None and existing is not builder:
        raise ValueError(f"component builder {kind}/{name} is already registered")
    _BUILDERS[key] = builder


def component_builder(kind: str, name: str) -> Callable[..., nn.Module]:
    """Resolve a constructor without allocating parameters or importing families."""
    component_schema(kind, name)
    try:
        return _BUILDERS[kind, name]
    except KeyError:
        raise ValueError(f"no modeling builder registered for {kind}/{name}") from None


def register_component_layout(kind: str, schema: type, layout: ComponentLayout) -> None:
    """Declare a built component's fused-parameter layout; `count` must be a node field."""
    name = component_type(kind, schema)
    if component_schema(kind, name) is not schema:
        raise ValueError(f"{kind}/{name}: layout must use the registered node schema")
    component_builder(kind, name)
    if not isinstance(layout, ComponentLayout):
        raise TypeError("component layout must be a ComponentLayout")
    if layout.count not in {field.name for field in fields(schema)}:
        raise ValueError(f"{kind}/{name}: {layout.count!r} is not a node field")
    key = (kind, name)
    existing = _LAYOUTS.get(key)
    if existing is not None and existing != layout:
        raise ValueError(f"component layout {kind}/{name} is already registered")
    _LAYOUTS[key] = layout


def component_layout(kind: str, name: str) -> ComponentLayout | None:
    """The declared layout, or None for components stored tensor for tensor."""
    component_schema(kind, name)
    return _LAYOUTS.get((kind, name))


def build_component(kind: str, config, **dimensions) -> nn.Module:
    """Build from one node and explicit dimensions/runtime device settings."""
    name = component_type(kind, config)
    if type(config) is not component_schema(kind, name):
        raise TypeError(f"{kind}/{name}: expected its registered configuration node")
    module = component_builder(kind, name)(config, **dimensions)
    if not isinstance(module, nn.Module):
        raise TypeError(f"{kind}/{name}: builder must return nn.Module")
    if getattr(module, "config", None) is not config:
        raise ValueError(f"{kind}/{name}: module must retain its configuration node")
    return module
