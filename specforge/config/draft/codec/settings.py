"""Strict codecs for model-level RoPE and resolved target-logit settings."""

from collections.abc import Mapping
from dataclasses import asdict

from ..logits import MuPLogitTransformSpec, NoLogitTransformSpec, TargetLogitSpec
from ..rope import DefaultRoPESpec, LinearRoPESpec, NoRoPESpec, RopeSpec, YaRNSpec
from ._schema import _construct

_ROPE = {
    schema.rope_type: schema
    for schema in (DefaultRoPESpec, LinearRoPESpec, NoRoPESpec, YaRNSpec)
}
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
    return _decode(payload, _ROPE, "rope_type", "rope_parameters")


def encode_rope(node: RopeSpec) -> dict:
    return _encode(node, _ROPE, "rope_type", "rope_parameters")


def decode_target_logits(payload: Mapping) -> TargetLogitSpec:
    return _decode(payload, _LOGITS, "type", "target_logit_transform")


def encode_target_logits(node: TargetLogitSpec) -> dict:
    return _encode(node, _LOGITS, "type", "target_logit_transform")
