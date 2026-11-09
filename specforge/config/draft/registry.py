"""Configuration schemas and pure validation; computational builders live in modeling."""

from dataclasses import MISSING, fields
from typing import Callable

from .attention import AttentionSpec, GQASpec, MHASpec, MLASpec
from .conv import ConvSpec, GroupedCausalConvSpec
from .conv.constraints import validate_grouped_causal
from .logits import MuPLogitTransformSpec, NoLogitTransformSpec, TargetLogitSpec
from .mlp import DenseFFNSpec, FFNSpec
from .norm import NormSpec, RMSNormSpec
from .residual import ResidualSpec, StandardResidualSpec
from .rope import DefaultRoPESpec, LinearRoPESpec, NoRoPESpec, RopeSpec

_BASES = {
    "attention": AttentionSpec,
    "mlp": FFNSpec,
    "conv": ConvSpec,
    "norm": NormSpec,
    "residual": ResidualSpec,
    "rope": RopeSpec,
    "logits": TargetLogitSpec,
}
_SCHEMAS: dict[tuple[str, str], type] = {}
_VALIDATORS: dict[tuple[str, str], Callable[..., None]] = {}


def component_type_field(kind: str) -> str:
    """Return the serialized discriminator for a component kind."""
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}")
    return "rope_type" if kind == "rope" else "type"


def component_type(kind: str, node):
    """Read the discriminator, preserving its serialized name."""
    return getattr(node, component_type_field(kind), None)


def component_kind(schema: type) -> str | None:
    """Identify a component annotation or node class, including abstract bases."""
    return next(
        (kind for kind, base in _BASES.items() if issubclass(schema, base)), None
    )


def register_component_schema(
    kind: str, schema: type, *, validate: Callable[..., None] | None = None
) -> None:
    """Register a frozen node under its kind and class-level type name.

    Repeating the same registration is harmless; replacing a schema is an
    error. LayerSpec is a fixed reference schema and is not registered here.
    An optional pure validator accepts the node and caller-supplied context.
    Omitting it on a repeated registration preserves the existing validator.
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
        if field.name in ("type", "rope_type"):
            raise TypeError(
                "component type must be a ClassVar, not a configuration field"
            )
        if field.init and (
            field.default is not MISSING or field.default_factory is not MISSING
        ):
            raise TypeError(
                f"{kind}.{field.name}: structural fields must have no defaults"
            )
    if validate is not None and not callable(validate):
        raise TypeError("component validation must be callable")
    name = component_type(kind, schema)
    if not isinstance(name, str) or not name.strip():
        raise ValueError(f"{kind} schema must declare a nonempty type name")
    key = (kind, name)
    existing = _SCHEMAS.get(key)
    if existing is not None and existing is not schema:
        raise ValueError(
            f"component {kind}/{name} already registered to {existing.__name__}"
        )
    previous = _VALIDATORS.get(key)
    if validate is not None and previous is not None and previous is not validate:
        raise ValueError(f"component validation {kind}/{name} already registered")
    _SCHEMAS[key] = schema
    if validate is not None:
        _VALIDATORS[key] = validate


def component_schema(kind: str, name: str) -> type:
    """Resolve a schema without importing a model, backend, or training code."""
    if not isinstance(kind, str) or kind not in _BASES:
        raise ValueError(f"unknown component kind {kind!r}")
    if not isinstance(name, str) or (kind, name) not in _SCHEMAS:
        available = sorted(key[1] for key in _SCHEMAS if key[0] == kind)
        raise ValueError(f"unknown {kind} type {name!r}; available: {available}")
    return _SCHEMAS[kind, name]


def validate_component(kind: str, node, **context) -> None:
    """Run the registered context validator; node-local checks run at construction."""
    name = component_type(kind, node)
    if type(node) is not component_schema(kind, name):
        raise ValueError(f"{kind}/{name}: expected its registered configuration node")
    validator = _VALIDATORS.get((kind, name))
    if validator is not None:
        validator(node, **context)


for _kind, _schema in (
    ("attention", GQASpec),
    ("attention", MHASpec),
    ("attention", MLASpec),
    ("mlp", DenseFFNSpec),
    ("norm", RMSNormSpec),
    ("residual", StandardResidualSpec),
    ("rope", DefaultRoPESpec),
    ("rope", LinearRoPESpec),
    ("rope", NoRoPESpec),
    ("logits", NoLogitTransformSpec),
    ("logits", MuPLogitTransformSpec),
):
    register_component_schema(_kind, _schema)

register_component_schema(
    "conv", GroupedCausalConvSpec, validate=validate_grouped_causal
)
