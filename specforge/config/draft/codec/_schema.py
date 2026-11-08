"""Strict dataclass construction shared by the component and settings codecs."""

from collections.abc import Mapping
from dataclasses import MISSING, fields


def _construct(schema: type, values: Mapping, path: str):
    expected = {field.name: field for field in fields(schema) if field.init}
    unknown = values.keys() - expected.keys()
    if unknown:
        raise ValueError(f"{path}: unknown fields {sorted(unknown, key=str)}")
    missing = [
        name
        for name, field in expected.items()
        if name not in values
        and field.default is MISSING
        and field.default_factory is MISSING
    ]
    if missing:
        raise ValueError(f"{path}: missing fields {missing}")
    try:
        node = schema(**values)
        hash(node)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {exc}") from exc
    return node
