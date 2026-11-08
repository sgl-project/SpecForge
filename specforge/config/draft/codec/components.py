"""Encode and decode component schemas and layer references.

Legacy aliases and defaults belong in the legacy codec, not in these readers.
"""

from collections.abc import Mapping
from dataclasses import asdict

from ..layer import LayerSpec
from ..registry import component_schema
from ._schema import _construct
from .settings import decode_rope, encode_rope


def decode_component(kind: str, payload: Mapping):
    """Decode a typed component, without changing or filling source fields."""
    if not isinstance(payload, Mapping):
        raise ValueError(f"{kind}: expected a configuration object")
    if "type" not in payload:
        raise ValueError(f"{kind}: missing field 'type'")
    schema = component_schema(kind, payload["type"])
    values = {name: value for name, value in payload.items() if name != "type"}
    if kind == "attention" and values.get("rope_parameters") is not None:
        values["rope_parameters"] = decode_rope(values["rope_parameters"])
    return _construct(schema, values, f"{kind}/{payload['type']}")


def encode_component(kind: str, node) -> dict:
    """Write the registered type and all node fields, including explicit nulls."""
    name = getattr(node, "type", None)
    schema = component_schema(kind, name)
    if type(node) is not schema:
        raise ValueError(
            f"{kind}/{name}: expected {schema.__name__}, got {type(node).__name__}"
        )
    hash(node)
    payload = {"type": name, **asdict(node)}
    if kind == "attention" and node.rope_parameters is not None:
        payload["rope_parameters"] = encode_rope(node.rope_parameters)
    return payload


def decode_layer(payload: Mapping) -> LayerSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("layer: expected a configuration object")
    return _construct(LayerSpec, payload, "layer")


def encode_layer(node: LayerSpec) -> dict:
    if type(node) is not LayerSpec:
        raise ValueError("layer: expected LayerSpec")
    return asdict(node)
