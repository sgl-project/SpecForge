"""Configuration schemas only; computational builders register in modeling."""

from dataclasses import MISSING, fields

from .attention import AttentionSpec, GQASpec, MHASpec
from .conv import ConvSpec, GroupedCausalConvSpec
from .mlp import DenseFFNSpec, FFNSpec
from .norm import NormSpec, RMSNormSpec
from .residual import ResidualSpec, StandardResidualSpec

_BASES = {
    "attention": AttentionSpec,
    "mlp": FFNSpec,
    "conv": ConvSpec,
    "norm": NormSpec,
    "residual": ResidualSpec,
}
_SCHEMAS: dict[tuple[str, str], type] = {}


def register_component_schema(kind: str, schema: type) -> None:
    """Register a frozen node under its kind and class-level type name.

    Repeating the same registration is harmless; replacing a schema is an
    error. LayerSpec is a fixed reference schema and is not registered here.
    """
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}; expected {sorted(_BASES)}")
    if not isinstance(schema, type) or not issubclass(schema, _BASES[kind]):
        raise TypeError(f"{kind} schema must inherit {_BASES[kind].__name__}")
    if (
        "__dataclass_fields__" not in schema.__dict__
        or not schema.__dataclass_params__.frozen
    ):
        raise TypeError(f"{kind} schema must be a frozen dataclass")
    for field in fields(schema):
        if field.name == "type":
            raise TypeError(
                "component type must be a ClassVar, not a configuration field"
            )
        if field.init and (
            field.default is not MISSING or field.default_factory is not MISSING
        ):
            raise TypeError(
                f"{kind}.{field.name}: structural fields must have no defaults"
            )
    name = getattr(schema, "type", None)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} schema must declare a nonempty type name")
    key = (kind, name)
    existing = _SCHEMAS.get(key)
    if existing is not None and existing is not schema:
        raise ValueError(
            f"component {kind}/{name} already registered to {existing.__name__}"
        )
    _SCHEMAS[key] = schema


def component_schema(kind: str, name: str) -> type:
    """Resolve a schema without importing a model, backend, or training code."""
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}")
    if not isinstance(name, str) or (kind, name) not in _SCHEMAS:
        available = sorted(key[1] for key in _SCHEMAS if key[0] == kind)
        raise ValueError(f"unknown {kind} type {name!r}; available: {available}")
    return _SCHEMAS[kind, name]


for _kind, _schema in (
    ("attention", GQASpec),
    ("attention", MHASpec),
    ("mlp", DenseFFNSpec),
    ("conv", GroupedCausalConvSpec),
    ("norm", RMSNormSpec),
    ("residual", StandardResidualSpec),
):
    register_component_schema(_kind, _schema)
