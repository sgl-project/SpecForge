"""Encode and decode registered component schemas and layer references.

Legacy aliases and defaults belong in the legacy codec, not in these readers.
"""

from collections.abc import Mapping
from dataclasses import asdict, fields

from ..layer import LayerSpec
from ..registry import (
    component_kind,
    component_schema,
    component_type,
    component_type_field,
)
from ._schema import _construct
from .nodes import decode_node, encode_node


def decode_component(kind: str, payload: Mapping, *, path: str | None = None):
    """Decode a typed component, without changing or filling source fields."""
    tag = component_type_field(kind)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path or kind}: expected a configuration object")
    if tag not in payload:
        raise ValueError(f"{path or kind}: missing field {tag!r}")
    try:
        schema = component_schema(kind, payload[tag])
    except ValueError as error:
        raise ValueError(f"{path or kind}.{tag}: {error}") from None
    values = {name: value for name, value in payload.items() if name != tag}
    return decode_node(schema, values, path or f"{kind}/{payload[tag]}")


def encode_component(kind: str, node) -> dict:
    """Write the registered type and all node fields, including explicit nulls."""
    tag = component_type_field(kind)
    if component_kind(type(node)) != kind:
        raise ValueError(f"{kind}: unsupported node {type(node).__name__}")
    name = component_type(kind, node)
    schema = component_schema(kind, name)
    if type(node) is not schema:
        raise ValueError(
            f"{kind}/{name}: expected {schema.__name__}, got {type(node).__name__}"
        )
    hash(node)
    return {
        tag: name,
        **{
            field.name: encode_node(getattr(node, field.name)) for field in fields(node)
        },
    }


def decode_layer(payload: Mapping) -> LayerSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("layer: expected a configuration object")
    return _construct(LayerSpec, payload, "layer")


def encode_layer(node: LayerSpec) -> dict:
    if type(node) is not LayerSpec:
        raise ValueError("layer: expected LayerSpec")
    return asdict(node)
