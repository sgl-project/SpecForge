"""Strict conversion of typed, nested architecture nodes to JSON values."""

from collections.abc import Mapping
from dataclasses import fields, is_dataclass
from types import UnionType
from typing import Union, get_args, get_origin, get_type_hints

from .._validation import finite_number
from .components import _construct


def _decode_value(annotation, value, path):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, UnionType):
        if value is None and type(None) in args:
            return None
        choices = [item for item in args if item is not type(None)]
        if len(choices) != 1:
            raise ValueError(f"{path}: union needs an explicit type discriminator")
        return _decode_value(choices[0], value, path)
    if isinstance(annotation, type) and is_dataclass(annotation):
        return decode_node(annotation, value, path)
    if origin is tuple:
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{path}: expected an array")
        if len(args) != 2 or args[1] is not Ellipsis:
            raise ValueError(f"{path}: expected a homogeneous tuple schema")
        return tuple(
            _decode_value(args[0], item, f"{path}[{index}]")
            for index, item in enumerate(value)
        )
    if annotation is float:
        finite_number(path, value)
        return value
    if annotation in (int, bool, str) and type(value) is annotation:
        return value
    raise ValueError(f"{path}: expected {annotation}, got {type(value).__name__}")


def decode_node(schema: type, payload: Mapping, path: str):
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a configuration object")
    annotations = get_type_hints(schema)
    names = {field.name for field in fields(schema) if field.init}
    values = {
        name: (
            _decode_value(annotations[name], value, f"{path}.{name}")
            if name in names
            else value
        )
        for name, value in payload.items()
    }
    return _construct(schema, values, path)


def encode_node(value):
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: encode_node(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, tuple):
        return [encode_node(item) for item in value]
    if type(value) is float:
        finite_number("node value", value)
    if value is None or type(value) in (bool, int, float, str):
        return value
    raise ValueError(f"unsupported node value {type(value).__name__}")
