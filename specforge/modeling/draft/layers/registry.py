"""Computational builders, separate from configuration schema registration."""

from collections.abc import Callable
from inspect import signature

from torch import nn

from specforge.config.draft.registry import component_schema, component_type

# Runtime keywords supplied by each component's callers; **kwargs is allowed.
_BUILDER_KWARGS = {
    "norm": ("hidden_size", "device", "dtype"),
    "mlp": ("hidden_size", "device", "dtype"),
    "residual": (),
    "conv": ("hidden_size", "block_size", "device", "dtype"),
    "rope": ("head_dim",),
    "logits": (),
    "attention": (
        "input_size",
        "hidden_size",
        "norm",
        "rope",
        "backend",
        "training",
        "device",
        "dtype",
    ),
}

_BUILDERS: dict[tuple[str, str], Callable[..., nn.Module]] = {}


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
    try:
        signature(builder).bind(object(), **dict.fromkeys(_BUILDER_KWARGS[kind]))
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"{kind}/{name}: incompatible builder signature: {error}"
        ) from error
    _BUILDERS[key] = builder


def component_builder(kind: str, name: str) -> Callable[..., nn.Module]:
    """Resolve a constructor without allocating parameters or importing families."""
    component_schema(kind, name)
    try:
        return _BUILDERS[kind, name]
    except KeyError:
        raise ValueError(f"no modeling builder registered for {kind}/{name}") from None


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
