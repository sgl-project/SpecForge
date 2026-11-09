"""Strict codecs for model-level RoPE and resolved target-logit settings."""

from collections.abc import Mapping
from dataclasses import asdict

from ..logits import MuPLogitTransformSpec, NoLogitTransformSpec, TargetLogitSpec
from ..registry import component_schema, component_type
from ..rope import RopeSpec
from ._schema import _construct
from .nodes import decode_node, encode_node

_LOGITS = {
    schema.type: schema for schema in (NoLogitTransformSpec, MuPLogitTransformSpec)
}


def _decode(payload, schemas, tag, path):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a configuration object")
    name = payload.get(tag)
    if not isinstance(name, str) or name not in schemas:
        raise ValueError(f"{path}: unknown or missing {tag} {name!r}")
    return _construct(
        schemas[name],
        {key: value for key, value in payload.items() if key != tag},
        path,
    )


def _encode(node, schemas, tag, path):
    if type(node) not in schemas.values():
        raise ValueError(f"{path}: unsupported node {type(node).__name__}")
    return {tag: getattr(node, tag), **asdict(node)}


def decode_rope(payload: Mapping) -> RopeSpec:
    if not isinstance(payload, Mapping):
        raise ValueError("rope_parameters: expected a configuration object")
    try:
        schema = component_schema("rope", payload.get("rope_type"))
    except ValueError as error:
        raise ValueError(f"rope_parameters.rope_type: {error}") from None
    return decode_node(
        schema,
        {key: value for key, value in payload.items() if key != "rope_type"},
        "rope_parameters",
    )


def encode_rope(node: RopeSpec) -> dict:
    if not isinstance(node, RopeSpec):
        raise ValueError(f"rope_parameters: unsupported node {type(node).__name__}")
    name = component_type("rope", node)
    if type(node) is not component_schema("rope", name):
        raise ValueError(f"rope/{name}: expected its registered configuration node")
    return {"rope_type": name, **encode_node(node)}


def decode_target_logits(payload: Mapping) -> TargetLogitSpec:
    return _decode(payload, _LOGITS, "type", "target_logit_transform")


def encode_target_logits(node: TargetLogitSpec) -> dict:
    return _encode(node, _LOGITS, "type", "target_logit_transform")
