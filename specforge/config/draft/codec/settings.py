"""Strict codecs for model-level RoPE and resolved target-logit settings."""

from collections.abc import Mapping

from ..logits import TargetLogitSpec
from ..registry import component_schema, component_type
from ..rope import RopeSpec
from .nodes import decode_node, encode_node


def _decode(kind, payload, tag, path):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a configuration object")
    try:
        schema = component_schema(kind, payload.get(tag))
    except ValueError as error:
        raise ValueError(f"{path}.{tag}: {error}") from None
    return decode_node(
        schema,
        {key: value for key, value in payload.items() if key != tag},
        path,
    )


def _encode(kind, node, base, tag, path):
    if not isinstance(node, base):
        raise ValueError(f"{path}: unsupported node {type(node).__name__}")
    name = component_type(kind, node)
    if type(node) is not component_schema(kind, name):
        raise ValueError(f"{path}: expected its registered configuration node")
    return {tag: name, **encode_node(node)}


def decode_rope(payload: Mapping) -> RopeSpec:
    return _decode("rope", payload, "rope_type", "rope_parameters")


def encode_rope(node: RopeSpec) -> dict:
    return _encode("rope", node, RopeSpec, "rope_type", "rope_parameters")


def decode_target_logits(payload: Mapping) -> TargetLogitSpec:
    return _decode("logits", payload, "type", "target_logit_transform")


def encode_target_logits(node: TargetLogitSpec) -> dict:
    return _encode("logits", node, TargetLogitSpec, "type", "target_logit_transform")
